"""cnn_rnn: the two-dimensional convolutional front end of cnn2d, then a recurrent network across the steps.

A sequence [channels, steps] is seen as an image, as in cnn2d. The front end's output is turned into one vector per
remaining step as in cnn_transformer (row_reduce "mean" or "flatten"), and a GRU (Cho et al. 2014) or an LSTM
(Hochreiter and Schmidhuber 1997), one to four layers, in one direction or both, reads those steps in order, each row
packed to its own real steps (sweepkit.models.torch.rnn.run_rnn), so padding never enters the recurrence. The summary
is cnn2d's pooling over the real steps or "last" (the final states). The classifier is cnn2d's.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.cnn2d import check_front_end, conv_front_end, rows_after, steps_after
from sweepkit.models.torch.cnn_transformer import check_sequence_only, step_vectors
from sweepkit.models.torch.heads import check_head, classifier_head, pooled_width
from sweepkit.models.torch.rnn import check_recurrence, make_rnn, rnn_width, run_rnn


@dataclasses.dataclass(frozen=True)
class CNNRNNSettings:
    # convolutional front end, as in cnn2d
    channels: tuple[int, ...]
    convs_per_block: int
    kernel_size: int
    pool_rows: int
    pool_cols: int                # max-pool factor along the steps after each block (steps / pool_cols ** blocks remain)
    norm: Literal["batch", "subspectral", "group", "none"]
    norm_groups: int | None
    norm_subbands: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # front end and classifier
    # from the front end to the recurrent network, as in cnn_transformer
    row_reduce: Literal["mean", "flatten"]
    # recurrent network
    rnn: Literal["gru", "lstm"]
    rnn_hidden: int               # units per direction in each layer
    rnn_layers: int               # stacked layers, 1 to 4
    bidirectional: bool
    rnn_dropout: float            # dropout between stacked layers; 0 with one layer
    # summary and classifier
    pooling: Literal["mean", "max", "meanmax", "stats", "attention", "last"]
    classifier_hidden: int | None
    dropout: float


def check(settings: CNNRNNSettings, input_shape) -> None:
    check_sequence_only(input_shape)
    check_front_end(settings, input_shape)
    check_head(settings)
    check_recurrence(settings)


def _module_class():
    from torch import nn

    class CNNRNN(nn.Module):
        def __init__(self, settings: CNNRNNSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            rows = input_shape[0]
            self.front = conv_front_end(settings, 1)
            self.n_blocks = len(settings.channels)
            self.pool_cols = settings.pool_cols
            self.row_reduce = settings.row_reduce
            width_in = settings.channels[-1]
            if settings.row_reduce == "flatten":
                width_in *= rows_after(settings, rows)
            self.rnn = make_rnn(settings, width_in)
            self.rnn_kind, self.rnn_layers = settings.rnn, settings.rnn_layers
            self.directions = 2 if settings.bidirectional else 1
            self.rnn_hidden = settings.rnn_hidden
            width = rnn_width(settings)
            self.pooling = settings.pooling
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            width = width if settings.pooling == "last" else pooled_width(width, settings.pooling)
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_outputs)

        def steps_after_blocks(self, lengths):
            return steps_after(lengths, self.n_blocks, self.pool_cols)

        def sequence(self, h, keep):
            """h: [batch, steps, width] float32, the front end's steps; keep: [batch] real steps (1 to steps). The
            pooled summary [batch, embedding_dim]."""
            return run_rnn(self.rnn, h, keep, self.pooling, self.attention, self.rnn_kind, self.rnn_layers,
                           self.directions, self.rnn_hidden)

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The pooled summary [batch, embedding_dim]."""
            h = step_vectors(self.front(x.unsqueeze(1)), self.row_reduce)
            keep = self.steps_after_blocks(lengths).clamp_max(h.shape[1])
            return self.sequence(h, keep)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return CNNRNN


@torch_model("cnn_rnn", settings=CNNRNNSettings, kinds=("sequence",),
             description="cnn2d's convolutional front end, then a GRU or LSTM across the steps",
             example="audio_logmel", check=check)
def build_cnn_rnn(settings: CNNRNNSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
