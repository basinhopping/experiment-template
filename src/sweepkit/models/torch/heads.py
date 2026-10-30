"""Parts every torch model shares: activations, summaries over a sequence's real steps (pooling), the classifier,
the sinusoidal position code, and the extra heads on a model's pooled summary (model.embed).

Pooling (masked_pool), over the real steps of h [batch, width, steps] with a mask [batch, steps] that is True on real
steps; padded steps never enter:
    mean       the mean over the real steps
    max        the maximum over the real steps
    meanmax    both, concatenated (twice the width)
    stats      the mean and the standard deviation, concatenated (twice the width)
    attention  a learned score per step, softmax over the real steps, and the weighted mean (attentive pooling)

The classifier (classifier_head): dropout, an optional hidden layer with the model's activation and dropout again,
and the output layer with one output per class, label or regression target.

The extra heads (sweepkit.families.torch builds them after the model, each from its own random stream, so a job
without them draws exactly what it drew before):
    auxiliary head   dropout (the model's classifier dropout) and one linear layer from the summary to the auxiliary
                     target's outputs. "multitask" trains it like the main output, so the shared layers learn the
                     auxiliary target too. "adversarial" puts a gradient reversal between the summary and the head
                     (Ganin and Lempitsky 2015): the head still learns to predict the auxiliary target, while the
                     gradient that reaches the shared layers is negated, pushing them to drop it.
    projection head  a linear layer as wide as the summary, ReLU, and a linear layer to the contrastive dimension, the
                     projection network of Khosla et al. (2020, section 3.1). Used in training only.
"""
import math

POOLINGS = ("mean", "max", "meanmax", "stats", "attention")
ACTIVATIONS = ("relu", "leaky_relu", "elu", "gelu", "silu")
AUX_KINDS = ("multitask", "adversarial")


def check_head(settings) -> None:
    """The classifier's settings (dropout, classifier_hidden); no torch needed."""
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if getattr(settings, "classifier_hidden", None) is not None and settings.classifier_hidden < 1:
        raise ValueError("model.classifier_hidden must be positive or None")


def pooled_width(width: int, pooling: str) -> int:
    return 2 * width if pooling in ("meanmax", "stats") else width


def activation_layer(name: str):
    """The activation by name. leaky_relu has slope 0.01 below zero and elu has alpha 1 (PyTorch's values, passed
    explicitly here); gelu is the exact (erf) form."""
    from torch import nn

    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name!r}")


def masked_pool(h, mask, kind: str, attention):
    """h: [batch, width, steps] float32; mask: [batch, steps] bool, True on real steps."""
    import torch

    m = mask[:, None, :].to(h.dtype)
    count = m.sum(dim=-1).clamp_min(1.0)
    mean = (h * m).sum(dim=-1) / count
    if kind == "mean":
        return mean
    lowest = torch.finfo(h.dtype).min
    top = h.masked_fill(~mask[:, None, :], lowest).amax(dim=-1)
    if kind == "max":
        return top
    if kind == "meanmax":
        return torch.cat([mean, top], dim=1)
    if kind == "stats":
        var = (((h - mean[..., None]) ** 2) * m).sum(dim=-1) / count
        return torch.cat([mean, torch.sqrt(var + 1e-6)], dim=1)
    if kind == "attention":
        scores = attention(h.transpose(1, 2)).squeeze(-1).float()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(h.dtype)
        return (h * weights[:, None, :]).sum(dim=-1)
    raise ValueError(kind)


def step_mask(lengths, steps: int, device):
    """[batch, steps] bool, True on each sequence's own steps."""
    import torch

    return torch.arange(steps, device=device)[None, :] < lengths[:, None]


def classifier_head(settings, width: int, n_outputs: int):
    """Dropout, an optional hidden layer with the model's activation, and the output layer."""
    from torch import nn

    head = [nn.Dropout(settings.dropout)]
    if settings.classifier_hidden is not None:
        head += [nn.Linear(width, settings.classifier_hidden), activation_layer(settings.activation),
                 nn.Dropout(settings.dropout)]
        width = settings.classifier_hidden
    head.append(nn.Linear(width, n_outputs))
    return nn.Sequential(*head)


def sinusoidal(length: int, width: int, device, dtype):
    """[length, width]: sin at even columns and cos at odd, frequencies 1 / 10000^(2i / width) (Vaswani et al.
    2017; the base 10,000 is theirs, a fixed fact rather than a setting)."""
    import torch

    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    rate = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    code = torch.zeros(length, width, device=device, dtype=torch.float32)
    code[:, 0::2] = torch.sin(position * rate)
    code[:, 1::2] = torch.cos(position * rate)
    return code.to(dtype)


def _aux_class():
    import torch
    from torch import nn

    class _Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return grad.neg()

    class AuxHead(nn.Module):
        def __init__(self, width: int, n_out: int, dropout: float, reverse: bool):
            super().__init__()
            self.reverse = reverse
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(width, n_out)

        def forward(self, embedding):
            h = _Reverse.apply(embedding) if self.reverse else embedding
            return self.linear(self.dropout(h))

    return AuxHead


def build_aux_head(kind: str, width: int, n_out: int, dropout: float):
    if kind not in AUX_KINDS:
        raise ValueError(f"aux head kind {kind!r} is not one of {AUX_KINDS}")
    return _aux_class()(width, n_out, dropout, reverse=kind == "adversarial")


def _projection_class():
    from torch import nn

    class ProjectionHead(nn.Module):
        def __init__(self, width: int, dim: int):
            super().__init__()
            self.hidden = nn.Linear(width, width)
            self.relu = nn.ReLU()
            self.out = nn.Linear(width, dim)

        def forward(self, embedding):
            return self.out(self.relu(self.hidden(embedding)))

    return ProjectionHead


def build_projection_head(width: int, dim: int):
    if width < 1 or dim < 1:
        raise ValueError(f"a projection head needs width and dim of at least 1, got {width} and {dim}")
    return _projection_class()(width, dim)
