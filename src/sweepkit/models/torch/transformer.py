"""transformer: a Transformer encoder over a sequence's steps.

The sequence passes the shared input stage (sweepkit.models.torch.sequence_heads: time pooling and dropout), a linear
layer to d_model, an optional sinusoidal position code, and a Transformer encoder (Vaswani et al. 2017;
torch.nn.TransformerEncoderLayer) with padded steps masked out of the attention and of the pooling. Then the pooling
(mean, max, both, mean and SD, or learned attention) and the classifier. Every size is a setting.

transformer_encoder is shared with cnn_transformer and the composite model. Ported from the project the template grew
out of (its CNN + Transformer and its Transformer head on an encoder's frames).
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.heads import check_head, classifier_head, masked_pool, pooled_width, sinusoidal
from sweepkit.models.torch.sequence_heads import check_input_stage, sequence_input


@dataclasses.dataclass(frozen=True)
class TransformerSettings:
    time_pool: int                 # average consecutive steps in groups of this size first (1: none)
    input_dropout: float
    d_model: int                   # the linear layer from the steps to the Transformer's width
    n_heads: int                   # attention heads; must divide d_model
    n_layers: int                  # encoder layers
    ff_dim: int                    # width of each layer's feed-forward block
    transformer_dropout: float     # dropout on attention weights, the feed-forward block and each residual branch
    transformer_activation: Literal["relu", "gelu"]   # inside the feed-forward block
    norm_first: bool               # pre-LN (with a final layer norm) or post-LN (the original Transformer)
    positional: Literal["sinusoidal", "none"]
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    classifier_hidden: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # the classifier's hidden layer
    dropout: float


def check_encoder(settings) -> None:
    """The Transformer fields (shared with cnn_transformer and the composite model's block)."""
    if settings.d_model < 1 or settings.n_heads < 1 or settings.d_model % settings.n_heads:
        raise ValueError(f"model: n_heads {settings.n_heads} must divide d_model {settings.d_model}")
    if settings.n_layers < 1 or settings.ff_dim < 1:
        raise ValueError("model: n_layers and ff_dim must be at least 1")
    if not 0.0 <= settings.transformer_dropout < 1.0:
        raise ValueError("model.transformer_dropout must be in [0, 1)")
    if settings.positional == "sinusoidal" and settings.d_model % 2:
        raise ValueError("model: the sinusoidal position code needs an even d_model")


def check(settings: TransformerSettings, input_shape) -> None:
    check_input_stage(settings, input_shape)
    check_head(settings)
    check_encoder(settings)


def transformer_encoder(settings):
    from torch import nn

    layer = nn.TransformerEncoderLayer(settings.d_model, settings.n_heads, dim_feedforward=settings.ff_dim,
                                       dropout=settings.transformer_dropout,
                                       activation=settings.transformer_activation, batch_first=True,
                                       norm_first=settings.norm_first)
    return nn.TransformerEncoder(layer, num_layers=settings.n_layers,
                                 norm=nn.LayerNorm(settings.d_model) if settings.norm_first else None,
                                 enable_nested_tensor=False)


def encode(encoder, h, mask, positional: str):
    """h: [batch, steps, d_model]; mask: [batch, steps] True on real steps. The encoder's output, padding masked out
    of the attention."""
    if positional == "sinusoidal":
        h = h + sinusoidal(h.shape[1], h.shape[-1], h.device, h.dtype)
    return encoder(h, src_key_padding_mask=~mask)


def _module_class():
    from torch import nn

    class Transformer(nn.Module):
        def __init__(self, settings: TransformerSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            self.input = sequence_input(settings, input_shape[0], None)
            self.project = nn.Linear(input_shape[0], settings.d_model)
            self.encoder = transformer_encoder(settings)
            self.positional = settings.positional
            self.pooling = settings.pooling
            self.attention = nn.Linear(settings.d_model, 1) if settings.pooling == "attention" else None
            self.embedding_dim = pooled_width(settings.d_model, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_outputs)

        def embed(self, x, lengths):
            """x: [batch, channels, steps]; lengths: [batch]. The pooled summary [batch, embedding_dim]."""
            h, mask = self.input(x, lengths)
            h = encode(self.encoder, self.project(h), mask, self.positional)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return Transformer


@torch_model("transformer", settings=TransformerSettings, kinds=("sequence",),
             description="Transformer encoder over a sequence's steps, padding masked out", example="sequence_models",
             check=check)
def build_transformer(settings: TransformerSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
