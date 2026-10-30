"""cnn2d: a two-dimensional convolutional network over a sequence seen as an image, or over an image.

A sequence [channels, steps] (for example a log-mel spectrogram: mel bands by frames) is one image with one input
channel: its rows are the sequence's channels and its columns are the steps. An image [channels, height, width] keeps
its channels. Blocks of [convolution, normalization, activation] x convs_per_block, then max pooling over rows and
columns (pool_rows, pool_cols) after each block. After the last block:
    a sequence: the rows are averaged, and the columns (steps) are summarized by `pooling` over each row's own steps
                only (padding is masked out);
    an image:   every remaining position is summarized by `pooling`.
Then the classifier (sweepkit.models.torch.heads.classifier_head).

Normalization after each convolution:
    batch        batch norm (Ioffe and Szegedy 2015)
    subspectral  batch norm computed separately in norm_subbands bands of rows (Chang et al. 2021; meant for spectrograms)
    group        group norm with norm_groups groups per layer (Wu and He 2018; 1 group is layer norm)
    none         no normalization, and the convolutions get a bias instead
Batch norm (and subspectral norm) count the padded steps of a sequence batch in their statistics. Training batches
are padded to their longest row and evaluation batches are sorted by length, so with rows of very different lengths
the statistics differ between training and evaluation: use `crop` (with `eval_window`) for such data, or group norm.

The front end (conv_front_end, steps_after, pad_steps) is shared with cnn_transformer, cnn_rnn and cnn_lif and the
composite model. A batch whose rows are all shorter than the poolings need is padded with zeros to that length first
(the step masks leave the padding out).
"""
import dataclasses
import functools
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.heads import activation_layer, check_head, classifier_head, masked_pool, pooled_width

NORMS = ("batch", "subspectral", "group", "none")


@dataclasses.dataclass(frozen=True)
class CNN2dSettings:
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + normalization + activation layers per block
    kernel_size: int              # square kernel, odd
    pool_rows: int                # max-pool factor along rows after each block (a sequence's channels, an image's height)
    pool_cols: int                # max-pool factor along columns after each block (a sequence's steps, an image's width)
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    classifier_hidden: int | None  # width of one hidden layer before the output layer, or None for none
    dropout: float                # dropout before the classifier's layers
    norm: Literal["batch", "subspectral", "group", "none"]
    norm_groups: int | None       # groups for norm "group", dividing every block's channels; None otherwise
    norm_subbands: int | None     # row bands for norm "subspectral", at least 2; None otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]


def rows_of(input_shape) -> int | None:
    """The rows the front end sees: a sequence's channels, an image's height (None when not known)."""
    if input_shape is None:
        return None
    return input_shape[0] if len(input_shape) == 1 else input_shape[1]


def check_front_end(settings, input_shape) -> None:
    """Checks of the convolutional front end, shared by every model that has one. input_shape may be None (not known
    when the sweep is expanded)."""
    if not settings.channels or any(c < 1 for c in settings.channels):
        raise ValueError("model.channels must be a non-empty list of positive numbers")
    if settings.convs_per_block < 1:
        raise ValueError("model.convs_per_block must be at least 1")
    if settings.kernel_size < 1 or settings.kernel_size % 2 == 0:
        raise ValueError("model.kernel_size must be odd")
    if settings.pool_rows < 1 or settings.pool_cols < 1:
        raise ValueError("model.pool_rows and pool_cols must be at least 1")
    if settings.norm not in NORMS:
        raise ValueError(f"model: norm must be one of {NORMS}")
    if settings.norm == "group":
        if settings.norm_groups is None or settings.norm_groups < 1 or settings.norm_subbands is not None:
            raise ValueError("model: norm 'group' needs norm_groups of at least 1 and norm_subbands None")
        bad = [c for c in settings.channels if c % settings.norm_groups]
        if bad:
            raise ValueError(f"model: norm_groups {settings.norm_groups} does not divide channels {bad}")
    elif settings.norm == "subspectral":
        if settings.norm_subbands is None or settings.norm_subbands < 2 or settings.norm_groups is not None:
            raise ValueError("model: norm 'subspectral' needs norm_subbands of at least 2 and norm_groups None")
    elif settings.norm_groups is not None or settings.norm_subbands is not None:
        raise ValueError(f"model: norm {settings.norm!r} takes norm_groups and norm_subbands None")
    rows = rows_of(input_shape)
    if rows is None:
        return
    if len(input_shape) not in (1, 3):
        raise ValueError(f"model: a two-dimensional front end reads sequences or images, not input {input_shape}")
    left = rows
    for _ in settings.channels:
        left //= settings.pool_rows
    if left < 1:
        raise ValueError(f"model: {len(settings.channels)} blocks pooling rows by {settings.pool_rows} leave none of "
                         f"{rows}")
    if len(input_shape) == 3 and input_shape[2] // settings.pool_cols ** len(settings.channels) < 1:
        raise ValueError(f"model: {len(settings.channels)} blocks pooling columns by {settings.pool_cols} leave none "
                         f"of {input_shape[2]}")
    if settings.norm == "subspectral":
        for block in range(len(settings.channels)):
            rows_here = rows // settings.pool_rows ** block
            if rows_here < settings.norm_subbands:
                raise ValueError(f"model: block {block + 1} sees {rows_here} rows, fewer than norm_subbands "
                                 f"{settings.norm_subbands}")


def check(settings: CNN2dSettings, input_shape) -> None:
    check_front_end(settings, input_shape)
    check_head(settings)


def _subspectral_class():
    import torch
    from torch import nn

    class SubSpectralNorm(nn.Module):
        """Batch norm computed separately in each of `subbands` bands of rows (Chang et al. 2021). Input
        [batch, channels, rows, columns]; torch.tensor_split cuts the rows as evenly as possible (a band count that
        does not divide them gives some bands one row more)."""

        def __init__(self, channels: int, subbands: int):
            super().__init__()
            self.subbands = subbands
            self.norms = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(subbands))

        def forward(self, x):
            parts = torch.tensor_split(x, self.subbands, dim=2)
            return torch.cat([norm(part) for norm, part in zip(self.norms, parts)], dim=2)

    return SubSpectralNorm


def pad_steps(x, minimum: int):
    """x with its last axis (the steps) right-padded with zeros to at least `minimum`, so that the max poolings of a
    batch whose rows are all short still leave one step (MaxPool raises on an output of size 0). The padding lies
    beyond every row's length, so the step masks leave it out."""
    import torch.nn.functional as F

    short = minimum - x.shape[-1]
    return F.pad(x, (0, short)) if short > 0 else x


@functools.lru_cache(maxsize=None)
def padded_sequential_class():
    """nn.Sequential that pads the steps to `minimum` first (its parameters keep the plain Sequential's names)."""
    from torch import nn

    class PaddedSequential(nn.Sequential):
        minimum = 1

        def forward(self, x):
            return super().forward(pad_steps(x, self.minimum))

    return PaddedSequential


def conv_front_end(settings, in_channels: int):
    """The convolution blocks: Conv2d (a bias only without normalization), the normalization, the activation, per
    convolution, then MaxPool2d per block; the steps are padded first to the length the poolings need."""
    from torch import nn

    subspectral = _subspectral_class() if settings.norm == "subspectral" else None
    layers, c_in = [], in_channels
    for c_out in settings.channels:
        for _ in range(settings.convs_per_block):
            layers.append(nn.Conv2d(c_in, c_out, settings.kernel_size, padding=settings.kernel_size // 2,
                                    bias=settings.norm == "none"))
            if settings.norm == "batch":
                layers.append(nn.BatchNorm2d(c_out))
            elif settings.norm == "subspectral":
                layers.append(subspectral(c_out, settings.norm_subbands))
            elif settings.norm == "group":
                layers.append(nn.GroupNorm(settings.norm_groups, c_out))
            layers.append(activation_layer(settings.activation))
            c_in = c_out
        layers.append(nn.MaxPool2d((settings.pool_rows, settings.pool_cols)))
    blocks = padded_sequential_class()(*layers)
    blocks.minimum = settings.pool_cols ** len(settings.channels)
    return blocks


def rows_after(settings, rows: int) -> int:
    for _ in settings.channels:
        rows //= settings.pool_rows
    return rows


def steps_after(lengths, n_blocks: int, pool: int):
    """Real steps left after n_blocks max poolings along the steps (at least 1)."""
    import torch

    steps = lengths
    for _ in range(n_blocks):
        steps = torch.div(steps, pool, rounding_mode="floor")
    return steps.clamp_min(1)


def _module_class():
    import torch
    from torch import nn

    class CNN2d(nn.Module):
        def __init__(self, settings: CNN2dSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.image = len(input_shape) == 3
            self.blocks = conv_front_end(settings, input_shape[0] if self.image else 1)
            self.n_blocks = len(settings.channels)
            self.pool_cols = settings.pool_cols
            self.pooling = settings.pooling
            width = settings.channels[-1]
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            width = pooled_width(width, settings.pooling)
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_outputs)

        def steps_after_blocks(self, lengths):
            return steps_after(lengths, self.n_blocks, self.pool_cols)

        def embed(self, x, lengths):
            """x: [batch, channels, steps] with lengths [batch], or [batch, channels, height, width] with lengths
            None. The pooled summary [batch, embedding_dim]."""
            if self.image:
                h = self.blocks(x).float().flatten(2)                    # [batch, channels, positions]
                mask = torch.ones(h.shape[0], h.shape[2], dtype=torch.bool, device=h.device)
                return masked_pool(h, mask, self.pooling, self.attention)
            h = self.blocks(x.unsqueeze(1))                               # [batch, channels, rows, steps']
            h = h.float().mean(dim=2)                                     # [batch, channels, steps']
            steps = h.shape[-1]
            keep = self.steps_after_blocks(lengths).clamp_max(steps)
            mask = torch.arange(steps, device=h.device)[None, :] < keep[:, None]
            return masked_pool(h, mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return CNN2d


@torch_model("cnn2d", settings=CNN2dSettings, kinds=("sequence", "image"),
             description="two-dimensional CNN over a sequence seen as an image (a spectrogram) or over an image",
             example="image_cnn", check=check)
def build_cnn2d(settings: CNN2dSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
