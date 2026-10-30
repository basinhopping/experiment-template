"""cnn_lif: the hybrid spiking model. cnn2d's convolutional front end, then spiking (LIF) layers across the steps.

A sequence [channels, steps] is seen as an image, as in cnn2d. The front end's output becomes one vector per remaining
step: the rows are averaged ("mean") or every remaining row's channels are concatenated ("flatten"), as in
cnn_transformer. Consecutive steps can then be averaged in groups of time_pool (real steps only). Those real-valued
vectors are the input current of the first spiking layer; lif_hidden gives the width of each spiking layer (1 to 3),
each linear synapses (recurrent when asked), an optional normalization of the current, LIF neurons
(sweepkit.models.torch.lif) and dropout with one mask per sequence. The readout is non-spiking: the last layer's
spikes feed a leaky integrator with leak readout_beta, averaged over each row's own steps
(lif_reference.readout_embedding), then dropout and a linear layer give the outputs. readout_beta 0 makes that a
linear layer on each neuron's spike count per step.

The training loss adds rate_weight times the activity penalty (sweepkit.models.torch.lif) when rate_weight is above
0. efficiency.json gets each spiking layer's firing rate on the validation set and the operations per sequence: the
front end's multiply-accumulates, the first spiking layer's multiply-accumulates (its input is real), and every later
layer's accumulates (spikes in).
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.cnn2d import check_front_end, conv_front_end, rows_after, steps_after
from sweepkit.models.torch.cnn_transformer import check_sequence_only, step_vectors
from sweepkit.models.torch.lif import check_neuron


@dataclasses.dataclass(frozen=True)
class CNNLIFSettings:
    # convolutional front end, as in cnn2d
    channels: tuple[int, ...]
    convs_per_block: int
    kernel_size: int
    pool_rows: int
    pool_cols: int
    norm: Literal["batch", "subspectral", "group", "none"]   # front end, as in cnn2d
    norm_groups: int | None
    norm_subbands: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # front end only
    # from the front end to the spiking layers
    row_reduce: Literal["mean", "flatten"]   # as in cnn_transformer
    time_pool: int                # average consecutive steps in groups of this size before the spiking layers (1: none)
    # spiking layers (sweepkit.models.torch.lif)
    lif_hidden: tuple[int, ...]   # width of each spiking layer, 1 to 3 layers
    recurrent: bool               # each spiking layer also gets its own spikes of the previous step
    lif_norm: Literal["none", "batch", "layer"]   # normalization of each layer's input current (batch: over real steps)
    lif_dropout: float            # dropout on each spiking layer's spikes, one mask per sequence
    beta: float                   # leak per step, in [0, 1)
    learn_beta: Literal["none", "layer", "neuron"]   # fixed leak, one learned per layer, or one per neuron
    threshold: float              # firing threshold
    reset: Literal["subtract", "zero"]   # subtract the threshold after a spike, or set the potential to zero
    detach_reset: bool            # no gradient through the reset
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]   # surrogate gradient of the spike
    surrogate_slope: float        # its sharpness (lif_reference gives each formula)
    # readout and training
    readout_beta: float           # leak of the readout's integrator, in [0, 1); 0 reads spike counts
    rate_weight: float            # weight of the activity penalty in the loss (0: none)
    rate_target: float            # the firing rate the penalty pulls each neuron towards, in [0, 1]
    dropout: float                # dropout before the output layer


def check(settings: CNNLIFSettings, input_shape) -> None:
    check_sequence_only(input_shape)
    check_front_end(settings, input_shape)
    check_neuron(settings)
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if settings.time_pool < 1:
        raise ValueError("model.time_pool must be at least 1")
    if not 1 <= len(settings.lif_hidden) <= 3:
        raise ValueError("model.lif_hidden must list 1 to 3 spiking layers")


def _module_class():
    import torch
    from torch import nn

    from sweepkit.models.torch import lif

    _, _, SpikingLinear = lif.classes()

    class CNNLIF(nn.Module):
        def __init__(self, settings: CNNLIFSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            rows = input_shape[0]
            self.recorder = lif.Recorder()
            self.front = conv_front_end(settings, 1)
            self.n_blocks, self.pool_cols = len(settings.channels), settings.pool_cols
            self.row_reduce, self.time_pool = settings.row_reduce, settings.time_pool
            self.rows, self.channels = rows, settings.channels
            self.kernel_size, self.convs_per_block, self.pool_rows = (settings.kernel_size, settings.convs_per_block,
                                                                      settings.pool_rows)
            width = settings.channels[-1] * (rows_after(settings, rows) if settings.row_reduce == "flatten" else 1)
            layers = []
            for k, hidden in enumerate(settings.lif_hidden):
                layers.append(SpikingLinear(width, hidden, settings, f"lif{k + 1}", self.recorder, real_input=k == 0))
                width = hidden
            self.spiking = nn.ModuleList(layers)
            self.readout_beta = settings.readout_beta
            self.rate_weight = settings.rate_weight
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_outputs))

        def steps_after_blocks(self, lengths):
            return steps_after(lengths, self.n_blocks, self.pool_cols)

        def _front_macs(self, lengths):
            """Multiply-accumulates of the front end's convolutions over each row's own steps."""
            total, c_in, steps = 0.0, 1, lengths.float()
            rows = self.rows
            for c_out in self.channels:
                for _ in range(self.convs_per_block):
                    total += float((steps * rows).sum().item()) * c_out * c_in * self.kernel_size ** 2
                    c_in = c_out
                steps = torch.div(steps, self.pool_cols, rounding_mode="floor").clamp_min(1)
                rows //= self.pool_rows
            return total

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The readout [batch, embedding_dim]."""
            if self.recorder.on:
                self.recorder.sequences += x.shape[0]
                self.recorder.add("front.mac", self._front_macs(lengths))
            h = step_vectors(self.front(x.unsqueeze(1)), self.row_reduce)   # [batch, steps', width]
            steps = h.shape[1]
            mask = lif.step_mask(self.steps_after_blocks(lengths).clamp_max(steps), steps, h.device)
            h, mask = lif.pool_time(h, mask, self.time_pool)
            for layer in self.spiking:
                h = layer(h, mask)
            return lif.readout(h, mask, self.readout_beta)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

        def activity_penalty(self):
            """Added to the loss by sweepkit.train.loop.Step; None when rate_weight is 0."""
            return lif.penalty_of(self, self.rate_weight)

        def record(self, on: bool) -> None:
            if on:
                self.recorder.reset()
            self.recorder.on = on

        def spike_report(self) -> dict:
            return lif.report(self.recorder)

    return CNNLIF


@torch_model("cnn_lif", settings=CNNLIFSettings, kinds=("sequence",),
             description="hybrid spiking model: cnn2d's front end, then LIF layers across the steps and a leaky readout",
             example="spiking", check=check)
def build_cnn_lif(settings: CNNLIFSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
