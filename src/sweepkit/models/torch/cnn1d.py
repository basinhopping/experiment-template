"""cnn1d: a one-dimensional convolutional network along a sequence's steps.

The sequence's channels are the convolutions' input channels. Blocks of [convolution, normalization, activation] x
convs_per_block, then max pooling along the steps (pool) after each block; the last block's output is summarized by
`pooling` over each row's own steps (padding is masked out), and the classifier gives the outputs.

Normalization after each convolution: batch norm, group norm with norm_groups groups (1 group is layer norm), or none
(the convolutions then get a bias).

The one-dimensional counterpart of cnn2d, with the same blocks, pooling and classifier.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.cnn2d import steps_after
from sweepkit.models.torch.heads import activation_layer, check_head, classifier_head, masked_pool, pooled_width


@dataclasses.dataclass(frozen=True)
class CNN1dSettings:
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + normalization + activation layers per block
    kernel_size: int              # odd
    pool: int                     # max-pool factor along the steps after each block (1: none)
    norm: Literal["batch", "group", "none"]
    norm_groups: int | None       # groups for norm "group", dividing every block's channels; None otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    classifier_hidden: int | None
    dropout: float


def check_conv1d(settings) -> None:
    if not settings.channels or any(c < 1 for c in settings.channels):
        raise ValueError("model.channels must be a non-empty list of positive numbers")
    if settings.convs_per_block < 1:
        raise ValueError("model.convs_per_block must be at least 1")
    if settings.kernel_size < 1 or settings.kernel_size % 2 == 0:
        raise ValueError("model.kernel_size must be odd")
    if settings.pool < 1:
        raise ValueError("model.pool must be at least 1")
    if settings.norm == "group":
        if settings.norm_groups is None or settings.norm_groups < 1:
            raise ValueError("model: norm 'group' needs norm_groups of at least 1")
        bad = [c for c in settings.channels if c % settings.norm_groups]
        if bad:
            raise ValueError(f"model: norm_groups {settings.norm_groups} does not divide channels {bad}")
    elif settings.norm_groups is not None:
        raise ValueError(f"model: norm {settings.norm!r} takes norm_groups None")


def check(settings: CNN1dSettings, input_shape) -> None:
    check_conv1d(settings)
    check_head(settings)
    if input_shape is not None and len(input_shape) != 1:
        raise ValueError(f"model cnn1d reads sequences (one channel count), not input {input_shape}")


def conv1d_front_end(settings, in_channels: int):
    """The one-dimensional blocks (shared with the composite model); the steps are padded first to the length the
    poolings need."""
    from torch import nn

    from sweepkit.models.torch.cnn2d import padded_sequential_class

    layers, c_in = [], in_channels
    for c_out in settings.channels:
        for _ in range(settings.convs_per_block):
            layers.append(nn.Conv1d(c_in, c_out, settings.kernel_size, padding=settings.kernel_size // 2,
                                    bias=settings.norm == "none"))
            if settings.norm == "batch":
                layers.append(nn.BatchNorm1d(c_out))
            elif settings.norm == "group":
                layers.append(nn.GroupNorm(settings.norm_groups, c_out))
            layers.append(activation_layer(settings.activation))
            c_in = c_out
        if settings.pool > 1:
            layers.append(nn.MaxPool1d(settings.pool))
    blocks = padded_sequential_class()(*layers)
    blocks.minimum = settings.pool ** len(settings.channels) if settings.pool > 1 else 1
    return blocks


def _module_class():
    import torch
    from torch import nn

    class CNN1d(nn.Module):
        def __init__(self, settings: CNN1dSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.blocks = conv1d_front_end(settings, input_shape[0])
            self.n_blocks, self.pool = len(settings.channels), settings.pool
            self.pooling = settings.pooling
            width = settings.channels[-1]
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            self.embedding_dim = pooled_width(width, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_outputs)

        def steps_after_blocks(self, lengths):
            return steps_after(lengths, self.n_blocks, self.pool) if self.pool > 1 else lengths.clamp_min(1)

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The pooled summary [batch, embedding_dim]."""
            h = self.blocks(x).float()                                    # [batch, channels, steps']
            steps = h.shape[-1]
            keep = self.steps_after_blocks(lengths).clamp_max(steps)
            mask = torch.arange(steps, device=h.device)[None, :] < keep[:, None]
            return masked_pool(h, mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return CNN1d


@torch_model("cnn1d", settings=CNN1dSettings, kinds=("sequence",),
             description="one-dimensional CNN along a sequence's steps", example="sequence_models", check=check)
def build_cnn1d(settings: CNN1dSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
