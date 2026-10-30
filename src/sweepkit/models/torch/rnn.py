"""rnn: a recurrent network (GRU or LSTM) over a sequence's steps.

The sequence passes the shared input stage (sweepkit.models.torch.sequence_heads: time pooling, dropout, an optional
projection), then a GRU (Cho et al. 2014) or an LSTM (Hochreiter and Schmidhuber 1997), one to four layers, in one
direction or both. Each row is packed to its own real steps (torch.nn.utils.rnn.pack_padded_sequence), so padding
never enters the recurrence in either direction. The recurrent layers run in float32 outside autocast. The summary is
a pooling over the real steps (mean, max, both, mean and SD, or learned attention) or "last": the last layer's final
state of each direction (the forward direction at the row's last real step, the backward one at its first). Then the
classifier.

run_rnn is shared with cnn_rnn and the composite model.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.heads import check_head, classifier_head, masked_pool, pooled_width
from sweepkit.models.torch.sequence_heads import check_input_stage, sequence_input


@dataclasses.dataclass(frozen=True)
class RNNSettings:
    time_pool: int                 # average consecutive steps in groups of this size first (1: none)
    input_dropout: float
    proj_dim: int | None           # a linear layer with the activation before the recurrence, or None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # projection and classifier
    rnn: Literal["gru", "lstm"]
    rnn_hidden: int                # units per direction in each layer
    rnn_layers: int                # stacked layers, 1 to 4
    bidirectional: bool            # True: a second direction reads each row backwards (outputs concatenated)
    rnn_dropout: float             # between stacked layers (PyTorch's dropout argument); 0 with one layer
    pooling: Literal["mean", "max", "meanmax", "stats", "attention", "last"]
    classifier_hidden: int | None
    dropout: float


def check_recurrence(settings) -> None:
    """The recurrent fields (shared with cnn_rnn and the composite model's block)."""
    if settings.rnn_hidden < 1:
        raise ValueError("model.rnn_hidden must be at least 1")
    if not 1 <= settings.rnn_layers <= 4:
        raise ValueError("model.rnn_layers must be 1 to 4")
    if not 0.0 <= settings.rnn_dropout < 1.0:
        raise ValueError("model.rnn_dropout must be in [0, 1)")
    if settings.rnn_layers == 1 and settings.rnn_dropout > 0:
        raise ValueError("model.rnn_dropout acts between stacked layers, so it must be 0 with one layer")


def check(settings: RNNSettings, input_shape) -> None:
    check_input_stage(settings, input_shape)
    check_head(settings)
    check_recurrence(settings)


def rnn_width(settings) -> int:
    """Width of the recurrent network's output at each step (both directions concatenated)."""
    return settings.rnn_hidden * (2 if settings.bidirectional else 1)


def make_rnn(settings, width_in: int):
    from torch import nn

    recurrent = nn.GRU if settings.rnn == "gru" else nn.LSTM
    return recurrent(width_in, settings.rnn_hidden, num_layers=settings.rnn_layers, batch_first=True,
                     dropout=settings.rnn_dropout, bidirectional=settings.bidirectional)


def run_rnn(rnn, h, keep, pooling: str, attention, kind: str, layers: int, directions: int, hidden: int):
    """h: [batch, steps, width] float32; keep: [batch] real steps (1 to steps). The pooled summary of the recurrent
    network's outputs over each row's real steps, or the final states with pooling "last"."""
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    batch, steps = h.shape[0], h.shape[1]
    with torch.autocast(device_type=h.device.type, enabled=False):
        packed = pack_padded_sequence(h.float(), keep.detach().cpu().to(torch.int64), batch_first=True,
                                      enforce_sorted=False)
        out, state = rnn(packed)
        if pooling == "last":
            final = state[0] if kind == "lstm" else state           # [layers * directions, batch, hidden]
            final = final.view(layers, directions, batch, hidden)[-1]
            return final.transpose(0, 1).reshape(batch, directions * hidden)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=steps)   # [batch, steps, width]
    mask = torch.arange(steps, device=h.device)[None, :] < keep[:, None]
    return masked_pool(out.float().transpose(1, 2), mask, pooling, attention)


def _module_class():
    from torch import nn

    class RNN(nn.Module):
        def __init__(self, settings: RNNSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.input = sequence_input(settings, input_shape[0], settings.proj_dim)
            self.rnn = make_rnn(settings, self.input.width)
            self.rnn_kind, self.rnn_layers, self.rnn_hidden = settings.rnn, settings.rnn_layers, settings.rnn_hidden
            self.directions = 2 if settings.bidirectional else 1
            width = rnn_width(settings)
            self.pooling = settings.pooling
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            self.embedding_dim = width if settings.pooling == "last" else pooled_width(width, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_outputs)

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The summary [batch, embedding_dim]."""
            h, mask = self.input(x, lengths)
            return run_rnn(self.rnn, h, mask.sum(dim=1), self.pooling, self.attention, self.rnn_kind,
                           self.rnn_layers, self.directions, self.rnn_hidden)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return RNN


@torch_model("rnn", settings=RNNSettings, kinds=("sequence",),
             description="recurrent network (GRU or LSTM) over a sequence's steps, packed to each row's own steps",
             example="sequence_models", check=check)
def build_rnn(settings: RNNSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
