"""mlp: a multilayer perceptron on tabular rows.

The row's columns (after the job's preparation steps) pass through the hidden layers, each a linear layer, an optional
normalization (batch norm or layer norm), the activation and dropout; the last hidden layer's output is the summary
(model.embed), and one linear layer gives the outputs. With no hidden layer the model is a linear (logistic) model on
the columns, and the summary is the columns themselves.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.heads import activation_layer


@dataclasses.dataclass(frozen=True)
class MLPSettings:
    hidden: tuple[int, ...]       # width of each hidden layer (may be empty: a linear model)
    norm: Literal["none", "batch", "layer"]   # after each hidden linear layer, before the activation
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    dropout: float                # after each hidden layer's activation, and before the output layer


def check(settings: MLPSettings, input_shape) -> None:
    if any(h < 1 for h in settings.hidden):
        raise ValueError("model.hidden widths must be positive")
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if input_shape is not None and (len(input_shape) != 1 or input_shape[0] < 1):
        raise ValueError(f"model mlp reads tabular rows (one column count), not input {input_shape}")


def _module_class():
    from torch import nn

    class MLP(nn.Module):
        def __init__(self, settings: MLPSettings, input_shape: tuple, n_outputs: int):
            super().__init__()
            layers, width = [], input_shape[0]
            for size in settings.hidden:
                layers.append(nn.Linear(width, size, bias=settings.norm != "batch"))
                if settings.norm == "batch":
                    layers.append(nn.BatchNorm1d(size))
                elif settings.norm == "layer":
                    layers.append(nn.LayerNorm(size))
                layers.append(activation_layer(settings.activation))
                layers.append(nn.Dropout(settings.dropout))
                width = size
            self.body = nn.Sequential(*layers)
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_outputs))

        def embed(self, x, lengths):
            """x: [batch, columns]; lengths: None. The summary [batch, embedding_dim]."""
            return self.body(x.float())

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return MLP


@torch_model("mlp", settings=MLPSettings, kinds=("tabular",), description="multilayer perceptron on tabular rows",
             example="classify_mlp", check=check)
def build_mlp(settings: MLPSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs)
