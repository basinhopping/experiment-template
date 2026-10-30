"""Models that read a sequence's steps directly ([batch, channels, steps], each step one vector of `channels` values):
the shared input stage, and two of the four heads (the other two, rnn and transformer, have their own modules and use
this input stage).

The input stage (SequenceInput): consecutive steps averaged in groups of time_pool (real steps only), dropout
(input_dropout), and an optional linear projection to proj_dim with the activation. Then:

    sequence_pool  the steps summarized over each row's own steps (mean, max, both, mean and SD, or learned attention:
                   attentive pooling) and the classifier: the simplest head, for example on a frozen encoder's frames
    sequence_lif   spiking layers on the steps, as in cnn_lif (sweepkit.models.torch.lif: linear synapses, recurrence
                   when asked, LIF neurons, a leaky readout over each row's own steps, the activity penalty); the
                   readout feeds dropout and a linear layer
    rnn            sweepkit.models.torch.rnn
    transformer    sweepkit.models.torch.transformer

The heads read any sequence: the frames of an encoder, sensor channels over time, a log-mel's
bands over frames.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.heads import check_head, classifier_head, masked_pool, pooled_width
from sweepkit.models.torch.lif import check_neuron


def check_input_stage(settings, input_shape) -> None:
    if input_shape is not None and (len(input_shape) != 1 or input_shape[0] < 1):
        raise ValueError(f"model: reads sequences (one channel count per step), not input {input_shape}")
    if settings.time_pool < 1:
        raise ValueError("model.time_pool must be at least 1")
    if not 0.0 <= settings.input_dropout < 1.0:
        raise ValueError("model.input_dropout must be in [0, 1)")
    if getattr(settings, "proj_dim", None) is not None and settings.proj_dim < 1:
        raise ValueError("model.proj_dim must be positive or None")


@dataclasses.dataclass(frozen=True)
class SequencePoolSettings:
    time_pool: int                 # average consecutive steps in groups of this size first (1: none)
    input_dropout: float           # dropout on the steps, before the projection
    proj_dim: int | None           # a linear layer to this width with the activation, or None (the steps as they are)
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # projection and classifier
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]   # attention: attentive pooling
    classifier_hidden: int | None  # width of one hidden layer before the output layer, or None
    dropout: float                 # dropout before the classifier's layers


@dataclasses.dataclass(frozen=True)
class SequenceLIFSettings:
    time_pool: int
    input_dropout: float
    proj_dim: int | None           # a linear layer with the activation before the spiking layers, or None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    # spiking layers, as in cnn_lif (sweepkit.models.torch.lif)
    lif_hidden: tuple[int, ...]    # width of each spiking layer, 1 to 3 layers
    recurrent: bool                # each spiking layer also gets its own spikes of the previous step
    lif_norm: Literal["none", "batch", "layer"]   # normalization of each layer's input current (batch: real steps)
    lif_dropout: float             # dropout on each spiking layer's spikes, one mask per sequence
    beta: float                    # leak per step, in [0, 1)
    learn_beta: Literal["none", "layer", "neuron"]
    threshold: float
    reset: Literal["subtract", "zero"]
    detach_reset: bool
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]
    surrogate_slope: float
    readout_beta: float            # leak of the readout's integrator, in [0, 1); 0 reads spike counts
    rate_weight: float             # weight of the activity penalty in the loss (0: none)
    rate_target: float             # the firing rate the penalty pulls each neuron towards, in [0, 1]
    dropout: float                 # dropout before the output layer


def check_pool(settings: SequencePoolSettings, input_shape) -> None:
    check_input_stage(settings, input_shape)
    check_head(settings)


def check_lif(settings: SequenceLIFSettings, input_shape) -> None:
    check_input_stage(settings, input_shape)
    check_neuron(settings)
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if not 1 <= len(settings.lif_hidden) <= 3:
        raise ValueError("model.lif_hidden must list 1 to 3 spiking layers")


def _input_class():
    from torch import nn

    from sweepkit.models.torch import lif
    from sweepkit.models.torch.heads import activation_layer

    class SequenceInput(nn.Module):
        """[batch, width, steps] and the real steps -> ([batch, steps', out], mask [batch, steps']): time pooling over
        real steps, dropout, and the optional projection with its activation."""

        def __init__(self, settings, n_inputs: int, proj_dim):
            super().__init__()
            self.time_pool = settings.time_pool
            self.dropout = nn.Dropout(settings.input_dropout)
            self.project = nn.Linear(n_inputs, proj_dim) if proj_dim is not None else None
            self.act = activation_layer(settings.activation) if proj_dim is not None else None
            self.width = proj_dim if proj_dim is not None else n_inputs
            self.n_inputs = n_inputs

        def forward(self, x, lengths):
            h = x.float().transpose(1, 2)                                   # [batch, steps, width]
            steps = h.shape[1]
            mask = lif.step_mask(lengths.clamp(1, steps), steps, h.device)
            h, mask = lif.pool_time(h, mask, self.time_pool)
            h = self.dropout(h)
            if self.project is not None:
                h = self.act(self.project(h))
            return h * mask[..., None].to(h.dtype), mask

    return SequenceInput


_INPUT = None


def sequence_input(settings, n_inputs: int, proj_dim):
    global _INPUT
    if _INPUT is None:
        _INPUT = _input_class()
    return _INPUT(settings, n_inputs, proj_dim)


def _pool_class():
    from torch import nn

    class SequencePool(nn.Module):
        def __init__(self, settings: SequencePoolSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.input = sequence_input(settings, input_shape[0], settings.proj_dim)
            self.pooling = settings.pooling
            self.attention = nn.Linear(self.input.width, 1) if settings.pooling == "attention" else None
            self.embedding_dim = pooled_width(self.input.width, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_outputs)

        def embed(self, x, lengths):
            h, mask = self.input(x, lengths)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return SequencePool


def _lif_class():
    from torch import nn

    from sweepkit.models.torch import lif

    _, _, SpikingLinear = lif.classes()

    class SequenceLIF(nn.Module):
        def __init__(self, settings: SequenceLIFSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.recorder = lif.Recorder()
            self.input = sequence_input(settings, input_shape[0], settings.proj_dim)
            width, layers = self.input.width, []
            for k, hidden in enumerate(settings.lif_hidden):
                layers.append(SpikingLinear(width, hidden, settings, f"lif{k + 1}", self.recorder, real_input=k == 0))
                width = hidden
            self.spiking = nn.ModuleList(layers)
            self.readout_beta = settings.readout_beta
            self.rate_weight = settings.rate_weight
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_outputs))

        def embed(self, x, lengths):
            h, mask = self.input(x, lengths)
            if self.recorder.on:
                self.recorder.sequences += x.shape[0]
                if self.input.project is not None:
                    self.recorder.add("proj.mac", mask.float().sum().item() * self.input.n_inputs * self.input.width)
            for layer in self.spiking:
                h = layer(h, mask)
            return lif.readout(h, mask, self.readout_beta)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

        def activity_penalty(self):
            return lif.penalty_of(self, self.rate_weight)

        def record(self, on: bool) -> None:
            if on:
                self.recorder.reset()
            self.recorder.on = on

        def spike_report(self) -> dict:
            return lif.report(self.recorder)

    return SequenceLIF


@torch_model("sequence_pool", settings=SequencePoolSettings, kinds=("sequence",),
             description="a sequence's steps summarized by pooling (attentive pooling among the kinds), then the "
                         "classifier", example="sequence_models", check=check_pool)
def build_sequence_pool(settings: SequencePoolSettings, input_shape: tuple, n_outputs: int):
    return _pool_class()(settings, input_shape, n_outputs)


@torch_model("sequence_lif", settings=SequenceLIFSettings, kinds=("sequence",),
             description="spiking (LIF) layers on a sequence's steps, with a leaky readout", example="spiking",
             check=check_lif)
def build_sequence_lif(settings: SequenceLIFSettings, input_shape: tuple, n_outputs: int):
    return _lif_class()(settings, input_shape, n_outputs)
