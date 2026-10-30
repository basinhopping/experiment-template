"""snn: the fully spiking model. Convolution blocks like cnn2d's, with LIF neurons (sweepkit.models.torch.lif) after
every convolution in place of the activation, and max pooling of the spikes (a spike survives if any in the window
fired).

A sequence [channels, steps] is seen as an image (rows = channels, columns = steps). Two ways for it to enter
(encoding):

  "frames"  each column (step) is one time step. The first convolution reads the input values (real numbers as
            input current, as in Bittar and Garner 2022); its neurons, one per channel and row, integrate along the
            steps. Later layers read spikes. Pooling along the steps shortens every later layer's sequence.
  "repeat"  the whole image is shown static_steps times (the vision approach): each neuron is one channel at one row
            and column, and integrates over the repeats.

After the last block, "frames" gives one vector of spikes per remaining step (channels times remaining rows);
"repeat" gives, per repeat, each channel and row's mean over the row's own columns. Then up to two spiking linear
layers (lif_hidden, recurrent when asked) and the readout of cnn_lif: a leaky integrator averaged over the steps
(lif_reference.readout_embedding), dropout and a linear layer.

conv_norm "batch" puts batch norm on each convolution's current (over the batch, rows and columns, as in cnn2d);
"none" gives the convolutions a bias instead. As in cnn2d, a pooling window at the end of a row's steps can see one
padded step. The activity penalty and the efficiency report work as in cnn_lif; here the first convolution is the
only layer with real input (multiply-accumulates; with "repeat" it is computed once, not per repeat), and every other
layer receives spikes (accumulates).
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.cnn2d import pad_steps, steps_after
from sweepkit.models.torch.cnn_transformer import check_sequence_only
from sweepkit.models.torch.lif import check_neuron


@dataclasses.dataclass(frozen=True)
class SNNSettings:
    encoding: Literal["frames", "repeat"]   # see the module docstring
    static_steps: int | None      # repeats of the image for "repeat"; None for "frames"
    # spiking convolution blocks
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + LIF layers per block
    kernel_size: int              # square kernel over rows and steps, odd
    pool_rows: int                # max-pool factor along rows after each block
    pool_cols: int                # max-pool factor along steps after each block
    conv_norm: Literal["batch", "none"]   # batch norm on each convolution's current, or none (then a bias)
    # spiking linear layers after the convolutions
    lif_hidden: tuple[int, ...]   # width of each, 0 to 2 layers
    recurrent: bool               # the linear layers also get their own spikes of the previous step
    lif_norm: Literal["none", "batch", "layer"]   # normalization of the linear layers' input current
    lif_dropout: float            # dropout on the spikes of every spiking layer, one mask per sequence
    # neurons, everywhere (sweepkit.models.torch.lif)
    beta: float
    learn_beta: Literal["none", "layer", "neuron"]
    threshold: float
    reset: Literal["subtract", "zero"]
    detach_reset: bool
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]
    surrogate_slope: float
    # readout and training
    readout_beta: float
    rate_weight: float
    rate_target: float
    dropout: float                # dropout before the output layer


def check(settings: SNNSettings, input_shape) -> None:
    check_sequence_only(input_shape)
    if settings.encoding == "repeat":
        if settings.static_steps is None or settings.static_steps < 1:
            raise ValueError("model: encoding 'repeat' needs static_steps of at least 1")
    elif settings.static_steps is not None:
        raise ValueError("model: encoding 'frames' takes static_steps None")
    if not settings.channels or any(c < 1 for c in settings.channels):
        raise ValueError("model.channels must be a non-empty list of positive numbers")
    if settings.convs_per_block < 1:
        raise ValueError("model.convs_per_block must be at least 1")
    if settings.kernel_size < 1 or settings.kernel_size % 2 == 0:
        raise ValueError("model.kernel_size must be odd")
    if settings.pool_rows < 1 or settings.pool_cols < 1:
        raise ValueError("model.pool_rows and pool_cols must be at least 1")
    if input_shape is not None and input_shape[0] // settings.pool_rows ** len(settings.channels) < 1:
        raise ValueError(f"model: {len(settings.channels)} blocks pooling rows by {settings.pool_rows} leave none of "
                         f"{input_shape[0]}")
    if len(settings.lif_hidden) > 2:
        raise ValueError("model.lif_hidden lists at most 2 spiking linear layers")
    check_neuron(settings)
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")


def _module_class():
    import torch
    from torch import nn

    from sweepkit.models.torch import lif

    Neuron, _, SpikingLinear = lif.classes()

    class SpikingConv(nn.Module):
        """Convolution, optional batch norm of its current, and LIF neurons."""

        def __init__(self, c_in: int, c_out: int, rows: int, settings, name: str, recorder, real_input: bool):
            super().__init__()
            k = settings.kernel_size
            self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=settings.conv_norm == "none")
            self.norm = nn.BatchNorm2d(c_out) if settings.conv_norm == "batch" else None
            beta_shape = (c_out, rows) if settings.encoding == "frames" else (c_out, rows, 1)
            self.neuron = Neuron(beta_shape, settings, name, recorder)
            self.fan_out = c_out * k * k
            self.dropout = settings.lif_dropout
            self.real_input, self.name, self.recorder = real_input, name, recorder

        def current(self, h, step_mask):
            """h [n, c_in, rows, steps] -> the masked current [n, c_out, rows, steps]."""
            cur = self.conv(h)
            if self.norm is not None:
                cur = self.norm(cur)
            return cur * step_mask[:, None, None, :].to(cur.dtype)

        def count_input(self, h, step_mask, repeats: int = 1):
            if not self.recorder.on:
                return
            m = step_mask[:, None, None, :].float()
            if self.real_input:
                self.recorder.add(f"{self.name}.mac", (m.expand_as(h).sum().item()) * self.fan_out * repeats)
            else:
                self.recorder.add(f"{self.name}.ac", (h.float() * m).sum().item() * self.fan_out)

    class SNN(nn.Module):
        def __init__(self, settings: SNNSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.recorder = lif.Recorder()
            self.encoding, self.static_steps = settings.encoding, settings.static_steps
            self.n_blocks, self.pool_cols = len(settings.channels), settings.pool_cols
            self.pool = nn.MaxPool2d((settings.pool_rows, settings.pool_cols))
            convs, c_in, rows = [], 1, input_shape[0]
            for b, c_out in enumerate(settings.channels):
                block = []
                for k in range(settings.convs_per_block):
                    block.append(SpikingConv(c_in, c_out, rows, settings, f"conv{b + 1}.{k + 1}", self.recorder,
                                             real_input=not convs and not block))
                    c_in = c_out
                convs.append(nn.ModuleList(block))
                rows //= settings.pool_rows
            self.blocks = nn.ModuleList(convs)
            width = settings.channels[-1] * rows
            hidden = []
            for k, size in enumerate(settings.lif_hidden):
                real = k == 0 and settings.encoding == "repeat"      # per-repeat rates, not spikes
                hidden.append(SpikingLinear(width, size, settings, f"lif{k + 1}", self.recorder, real_input=real))
                width = size
            self.spiking = nn.ModuleList(hidden)
            self.dropout_p = settings.lif_dropout
            self.readout_beta, self.rate_weight = settings.readout_beta, settings.rate_weight
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_outputs))

        def steps_after_blocks(self, lengths):
            return steps_after(lengths, self.n_blocks, self.pool_cols)

        def _frames(self, x, lengths):
            """encoding "frames": one time step per column. Returns ([batch, steps, width], step mask)."""
            h = pad_steps(x, self.pool_cols ** self.n_blocks).unsqueeze(1)   # [batch, 1, rows, steps]
            valid = lengths
            for block in self.blocks:
                steps = h.shape[-1]
                mask = lif.step_mask(valid.clamp_max(steps), steps, h.device)
                for layer in block:
                    layer.count_input(h, mask)
                    cur = layer.current(h, mask).permute(0, 3, 1, 2)      # [batch, steps, channels, rows]
                    s = layer.neuron(cur, mask[:, :, None, None])
                    s = lif.time_dropout(s, self.dropout_p, self.training)
                    h = s.permute(0, 2, 3, 1)                             # [batch, channels, rows, steps]
                h = self.pool(h)
                valid = torch.div(valid, self.pool_cols, rounding_mode="floor").clamp_min(1)
            b_, c, f, t = h.shape
            out = h.permute(0, 3, 1, 2).reshape(b_, t, c * f)
            return out, lif.step_mask(valid.clamp_max(t), t, h.device)

        def _repeat(self, x, lengths):
            """encoding "repeat": the image static_steps times. Returns ([batch, repeats, width], mask)."""
            n, reps = x.shape[0], self.static_steps
            h = pad_steps(x, self.pool_cols ** self.n_blocks).unsqueeze(1)   # [batch, 1, rows, steps]
            valid = lengths
            first = True
            for block in self.blocks:
                steps = h.shape[-1]
                mask = lif.step_mask(valid.clamp_max(steps), steps, h.device)
                for layer in block:
                    if first:                                             # the same current every repeat
                        layer.count_input(h, mask)
                        cur = layer.current(h, mask)[:, None].expand(-1, reps, -1, -1, -1)
                        first = False
                    else:                                                 # h: [batch, repeats, c, rows, steps]
                        flat = h.reshape((n * reps,) + tuple(h.shape[2:]))
                        layer.count_input(flat, mask.repeat_interleave(reps, dim=0))
                        cur = layer.current(flat, mask.repeat_interleave(reps, dim=0))
                        cur = cur.reshape((n, reps) + tuple(cur.shape[1:]))
                    s = layer.neuron(cur, mask[:, None, None, None, :])
                    h = lif.time_dropout(s, self.dropout_p, self.training)
                flat = self.pool(h.reshape((n * reps,) + tuple(h.shape[2:])))
                h = flat.reshape((n, reps) + tuple(flat.shape[1:]))
                valid = torch.div(valid, self.pool_cols, rounding_mode="floor").clamp_min(1)
            steps = h.shape[-1]
            mask = lif.step_mask(valid.clamp_max(steps), steps, h.device).float()
            rates = (h * mask[:, None, None, None, :]).sum(dim=-1) / mask.sum(dim=1)[:, None, None, None]
            out = rates.reshape(n, reps, -1)                              # [batch, repeats, channels * rows]
            return out, torch.ones(n, reps, dtype=torch.bool, device=x.device)

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The readout [batch, embedding_dim]."""
            if self.recorder.on:
                self.recorder.sequences += x.shape[0]
            if self.encoding == "frames":
                h, mask = self._frames(x.float(), lengths)
            else:
                h, mask = self._repeat(x.float(), lengths)
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

    return SNN


@torch_model("snn", settings=SNNSettings, kinds=("sequence",),
             description="fully spiking model: convolutions with LIF neurons, spiking linear layers and a leaky readout",
             example="spiking", check=check)
def build_snn(settings: SNNSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
