"""cnn_transformer: the two-dimensional convolutional front end of cnn2d, then a Transformer encoder across the steps.

A sequence [channels, steps] is seen as an image (rows = channels, columns = steps), as in cnn2d. The front end's
output is turned into one vector per remaining step: the remaining rows are averaged (row_reduce "mean") or the
channels of every remaining row are concatenated ("flatten"), and a linear layer maps that to d_model. A Transformer
encoder (Vaswani et al. 2017) then attends across the steps; padded steps are masked out of the attention and of the
pooling. The pooling and classifier are cnn2d's. Every size is a setting; nothing is fixed in the code except the
sinusoidal position code's base of 10,000 (Vaswani et al.).
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.cnn2d import check_front_end, conv_front_end, rows_after, steps_after
from sweepkit.models.torch.heads import check_head, classifier_head, masked_pool, pooled_width
from sweepkit.models.torch.transformer import check_encoder, encode, transformer_encoder


@dataclasses.dataclass(frozen=True)
class CNNTransformerSettings:
    # convolutional front end, as in cnn2d
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int
    kernel_size: int              # square kernel, odd
    pool_rows: int                # max-pool factor along rows (the sequence's channels) after each block
    pool_cols: int                # max-pool factor along columns (steps) after each block
    norm: Literal["batch", "subspectral", "group", "none"]
    norm_groups: int | None
    norm_subbands: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # front end and classifier
    # from the front end to the Transformer
    row_reduce: Literal["mean", "flatten"]
    # mean: average the remaining rows (one value per channel); flatten: keep every channel of every remaining row.
    # Either way a linear layer maps the result to d_model.
    # Transformer encoder
    d_model: int
    n_heads: int                  # must divide d_model
    n_layers: int
    ff_dim: int
    transformer_dropout: float
    transformer_activation: Literal["relu", "gelu"]
    norm_first: bool
    positional: Literal["sinusoidal", "none"]
    # summary and classifier, as in cnn2d
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    classifier_hidden: int | None
    dropout: float


def check_sequence_only(input_shape) -> None:
    if input_shape is not None and len(input_shape) != 1:
        raise ValueError(f"model: reads sequences [channels, steps], not input {input_shape}")


def check(settings: CNNTransformerSettings, input_shape) -> None:
    check_sequence_only(input_shape)
    check_front_end(settings, input_shape)
    check_head(settings)
    check_encoder(settings)


def step_vectors(h, row_reduce: str):
    """The front end's output [batch, channels, rows, steps] as [batch, steps, width] float32."""
    if row_reduce == "mean":
        return h.float().mean(dim=2).transpose(1, 2)
    b, c, f, t = h.shape
    return h.float().permute(0, 3, 1, 2).reshape(b, t, c * f)


def _module_class():
    import torch
    from torch import nn

    class CNNTransformer(nn.Module):
        def __init__(self, settings: CNNTransformerSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            rows = input_shape[0]
            self.front = conv_front_end(settings, 1)
            self.n_blocks = len(settings.channels)
            self.pool_cols = settings.pool_cols
            self.row_reduce = settings.row_reduce
            width_in = settings.channels[-1]
            if settings.row_reduce == "flatten":
                width_in *= rows_after(settings, rows)
            self.project = nn.Linear(width_in, settings.d_model)
            self.encoder = transformer_encoder(settings)
            self.positional = settings.positional
            self.pooling = settings.pooling
            self.attention = nn.Linear(settings.d_model, 1) if settings.pooling == "attention" else None
            width = pooled_width(settings.d_model, settings.pooling)
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_outputs)

        def steps_after_blocks(self, lengths):
            return steps_after(lengths, self.n_blocks, self.pool_cols)

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The pooled summary [batch, embedding_dim]."""
            h = step_vectors(self.front(x.unsqueeze(1)), self.row_reduce)   # [batch, steps', width]
            h = self.project(h)                                              # [batch, steps', d_model]
            steps = h.shape[1]
            keep = self.steps_after_blocks(lengths).clamp_max(steps)
            mask = torch.arange(steps, device=h.device)[None, :] < keep[:, None]
            h = encode(self.encoder, h, mask, self.positional)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return CNNTransformer


@torch_model("cnn_transformer", settings=CNNTransformerSettings, kinds=("sequence",),
             description="cnn2d's convolutional front end, then a Transformer encoder across the steps",
             example="audio_logmel", check=check)
def build_cnn_transformer(settings: CNNTransformerSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
