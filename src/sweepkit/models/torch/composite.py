"""composite: a model joined from blocks in the settings, so "CNN plus RNN" style combinations need no new class.

    front     none      the sequence's steps as they are
              conv1d    cnn1d's blocks along the steps (sweepkit.models.torch.cnn1d.conv1d_front_end)
              conv2d    cnn2d's blocks over the sequence seen as an image, then one vector per remaining step
                        (row_reduce mean or flatten, as in cnn_transformer)
    input     consecutive steps averaged in groups of time_pool (real steps only), dropout (input_dropout), and an
              optional linear projection to proj_dim with the activation
    block     none         no sequence model: the pooling reads the steps
              rnn          a GRU or an LSTM packed to each row's own steps (sweepkit.models.torch.rnn.run_rnn)
              transformer  a linear layer to d_model and a Transformer encoder, padding masked out
              lif          spiking linear layers with LIF neurons (sweepkit.models.torch.lif) and the leaky readout
    pooling   mean, max, meanmax, stats or attention over the real steps; "last" (rnn only): the final states;
              "readout" (lif only): the spiking block's leaky readout
    classifier  dropout, an optional hidden layer with the activation, and the output layer

Exactly the settings of the chosen front and block are given; the others are None.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.cnn1d import check_conv1d
from sweepkit.models.torch.cnn2d import check_front_end
from sweepkit.models.torch.heads import check_head
from sweepkit.models.torch.lif import check_neuron
from sweepkit.models.torch.rnn import check_recurrence
from sweepkit.models.torch.sequence_heads import check_input_stage
from sweepkit.models.torch.transformer import check_encoder


@dataclasses.dataclass(frozen=True)
class Conv1dBlockSettings:
    channels: tuple[int, ...]
    convs_per_block: int
    kernel_size: int              # odd
    pool: int                     # max-pool factor along the steps after each block (1: none)
    norm: Literal["batch", "group", "none"]
    norm_groups: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]


@dataclasses.dataclass(frozen=True)
class Conv2dBlockSettings:
    channels: tuple[int, ...]
    convs_per_block: int
    kernel_size: int              # square, odd
    pool_rows: int
    pool_cols: int
    norm: Literal["batch", "subspectral", "group", "none"]
    norm_groups: int | None
    norm_subbands: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    row_reduce: Literal["mean", "flatten"]


@dataclasses.dataclass(frozen=True)
class RecurrentBlockSettings:
    rnn: Literal["gru", "lstm"]
    rnn_hidden: int
    rnn_layers: int               # 1 to 4
    bidirectional: bool
    rnn_dropout: float            # between stacked layers; 0 with one layer


@dataclasses.dataclass(frozen=True)
class TransformerBlockSettings:
    d_model: int
    n_heads: int
    n_layers: int
    ff_dim: int
    transformer_dropout: float
    transformer_activation: Literal["relu", "gelu"]
    norm_first: bool
    positional: Literal["sinusoidal", "none"]


@dataclasses.dataclass(frozen=True)
class LIFBlockSettings:
    lif_hidden: tuple[int, ...]   # 1 to 3 spiking layers
    recurrent: bool
    lif_norm: Literal["none", "batch", "layer"]
    lif_dropout: float
    beta: float
    learn_beta: Literal["none", "layer", "neuron"]
    threshold: float
    reset: Literal["subtract", "zero"]
    detach_reset: bool
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]
    surrogate_slope: float
    readout_beta: float
    rate_weight: float
    rate_target: float


@dataclasses.dataclass(frozen=True)
class CompositeSettings:
    front: Literal["none", "conv1d", "conv2d"]
    conv1d: Conv1dBlockSettings | None
    conv2d: Conv2dBlockSettings | None
    time_pool: int                # average consecutive steps in groups of this size after the front (1: none)
    input_dropout: float
    proj_dim: int | None          # a linear layer with the activation after the front, or None
    block: Literal["none", "rnn", "transformer", "lif"]
    rnn: RecurrentBlockSettings | None
    transformer: TransformerBlockSettings | None
    lif: LIFBlockSettings | None
    pooling: Literal["mean", "max", "meanmax", "stats", "attention", "last", "readout"]
    classifier_hidden: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # projection and the classifier's hidden layer
    dropout: float


def check(settings: CompositeSettings, input_shape) -> None:
    check_input_stage(settings, input_shape)
    check_head(settings)
    fronts = {"none": (), "conv1d": ("conv1d",), "conv2d": ("conv2d",)}[settings.front]
    for name in ("conv1d", "conv2d"):
        given = getattr(settings, name) is not None
        if given != (name in fronts):
            raise ValueError(f"model composite: front {settings.front!r} takes {name} "
                             f"{'settings' if name in fronts else 'None'}")
    if settings.conv1d is not None:
        check_conv1d(settings.conv1d)
    if settings.conv2d is not None:
        check_front_end(settings.conv2d, input_shape)
    for name in ("rnn", "transformer", "lif"):
        given = getattr(settings, name) is not None
        if given != (settings.block == name):
            raise ValueError(f"model composite: block {settings.block!r} takes {name} "
                             f"{'settings' if settings.block == name else 'None'}")
    if settings.rnn is not None:
        check_recurrence(settings.rnn)
    if settings.transformer is not None:
        check_encoder(settings.transformer)
    if settings.lif is not None:
        check_neuron(settings.lif)
        if not 1 <= len(settings.lif.lif_hidden) <= 3:
            raise ValueError("model composite: lif.lif_hidden must list 1 to 3 spiking layers")
    if settings.pooling == "last" and settings.block != "rnn":
        raise ValueError("model composite: pooling 'last' needs block rnn")
    if (settings.pooling == "readout") != (settings.block == "lif"):
        raise ValueError("model composite: block lif reads out through pooling 'readout', and only it does")


def _module_class():
    from torch import nn

    from sweepkit.models.torch import lif
    from sweepkit.models.torch.cnn1d import conv1d_front_end
    from sweepkit.models.torch.cnn2d import conv_front_end, rows_after, steps_after
    from sweepkit.models.torch.cnn_transformer import step_vectors
    from sweepkit.models.torch.heads import activation_layer, classifier_head, masked_pool, pooled_width
    from sweepkit.models.torch.rnn import make_rnn, rnn_width, run_rnn
    from sweepkit.models.torch.transformer import encode, transformer_encoder

    _, _, SpikingLinear = lif.classes()

    class Composite(nn.Module):
        def __init__(self, settings: CompositeSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            s = settings
            self.front_kind = s.front
            channels_in = input_shape[0]
            if s.front == "conv1d":
                self.front = conv1d_front_end(s.conv1d, channels_in)
                self.n_blocks, self.pool_steps = len(s.conv1d.channels), s.conv1d.pool
                width = s.conv1d.channels[-1]
            elif s.front == "conv2d":
                self.front = conv_front_end(s.conv2d, 1)
                self.n_blocks, self.pool_steps = len(s.conv2d.channels), s.conv2d.pool_cols
                self.row_reduce = s.conv2d.row_reduce
                width = s.conv2d.channels[-1]
                if s.conv2d.row_reduce == "flatten":
                    width *= rows_after(s.conv2d, channels_in)
            else:
                self.front = None
                width = channels_in
            self.time_pool = s.time_pool
            self.input_dropout = nn.Dropout(s.input_dropout)
            self.project = nn.Linear(width, s.proj_dim) if s.proj_dim is not None else None
            self.project_act = activation_layer(s.activation) if s.proj_dim is not None else None
            width = s.proj_dim if s.proj_dim is not None else width
            self.block_kind = s.block
            self.is_spiking = s.block == "lif"         # record and spike_report count spikes only then
            self.pooling = s.pooling
            self.attention = None
            if s.block == "rnn":
                self.rnn = make_rnn(s.rnn, width)
                self.rnn_settings = s.rnn
                width = rnn_width(s.rnn)
            elif s.block == "transformer":
                self.to_model = nn.Linear(width, s.transformer.d_model)
                self.encoder = transformer_encoder(s.transformer)
                self.positional = s.transformer.positional
                width = s.transformer.d_model
            elif s.block == "lif":
                self.recorder = lif.Recorder()
                layers = []
                for k, hidden in enumerate(s.lif.lif_hidden):
                    layers.append(SpikingLinear(width, hidden, s.lif, f"lif{k + 1}", self.recorder,
                                                real_input=k == 0))
                    width = hidden
                self.spiking = nn.ModuleList(layers)
                self.readout_beta, self.rate_weight = s.lif.readout_beta, s.lif.rate_weight
            if s.pooling == "attention":
                self.attention = nn.Linear(width, 1)
            if s.pooling not in ("last", "readout"):
                width = pooled_width(width, s.pooling)
            self.embedding_dim = width
            self.head_dropout = s.dropout
            self.head = classifier_head(s, width, n_outputs)

        def _front(self, x, lengths):
            """([batch, steps, width] float32, real steps [batch])."""
            if self.front_kind == "none":
                h = x.float().transpose(1, 2)
                return h, lengths.clamp(1, h.shape[1])
            if self.front_kind == "conv1d":
                h = self.front(x).float().transpose(1, 2)
            else:
                h = step_vectors(self.front(x.unsqueeze(1)), self.row_reduce)
            return h, steps_after(lengths, self.n_blocks, self.pool_steps).clamp_max(h.shape[1])

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The summary [batch, embedding_dim]."""
            if self.block_kind == "lif" and self.recorder.on:
                self.recorder.sequences += x.shape[0]
            h, keep = self._front(x, lengths)
            mask = lif.step_mask(keep, h.shape[1], h.device)
            h, mask = lif.pool_time(h, mask, self.time_pool)
            h = self.input_dropout(h)
            if self.project is not None:
                h = self.project_act(self.project(h))
            h = h * mask[..., None].to(h.dtype)
            if self.block_kind == "rnn":
                r = self.rnn_settings
                return run_rnn(self.rnn, h, mask.sum(dim=1), self.pooling, self.attention, r.rnn, r.rnn_layers,
                               2 if r.bidirectional else 1, r.rnn_hidden)
            if self.block_kind == "transformer":
                h = encode(self.encoder, self.to_model(h), mask, self.positional)
            elif self.block_kind == "lif":
                for layer in self.spiking:
                    h = layer(h, mask)
                return lif.readout(h, mask, self.readout_beta)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

        def activity_penalty(self):
            if self.block_kind != "lif":
                return None
            return lif.penalty_of(self, self.rate_weight)

        def record(self, on: bool) -> None:
            if self.block_kind != "lif":
                return
            if on:
                self.recorder.reset()
            self.recorder.on = on

        def spike_report(self) -> dict:
            return lif.report(self.recorder) if self.block_kind == "lif" else {}

    return Composite


@torch_model("composite", settings=CompositeSettings, kinds=("sequence",),
             description="a front end (none, conv1d, conv2d), a sequence block (none, rnn, transformer, lif) and a "
                         "pooling head joined from the settings", example="sequence_models", check=check)
def build_composite(settings: CompositeSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
