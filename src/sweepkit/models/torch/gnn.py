"""Graph neural networks for node classification, multi-label and regression, in plain PyTorch: GCN (Kipf and
Welling 2017) and GAT (Velickovic et al. 2018). PyTorch Geometric is not used (the graph profile installs it for those
who want its layers).

A graph dataset (the "graph" dataset, or the synthetic "graph" generator) has one row per node, its features, and the
edges. A fold's roles name the nodes whose labels train, validate and test; the job's preparation steps are fitted on
the training nodes' features and applied to every node. Every training step runs the whole graph, and the loss reads
the outputs of the step's training nodes only (batch_size at least the training nodes: one full-graph step per epoch,
as in both papers); evaluation runs the whole graph once per role.

Transductive: messages pass along every edge, so the outputs of the training nodes depend on the features and edges
of validation and test nodes (never their labels), and a test node's output on the training nodes'. A result of these
models is a transductive one, as node classification on one graph usually is; their registry targets say so
(is_transductive), and every report that ranks or judges such a configuration notes it
(sweepkit.analysis.transductive).

gcn  each layer: h' = activation(A_hat h W + b). normalize symmetric: A_hat = D^-1/2 (A + I) D^-1/2 with D the
     in-degree (the paper's propagation rule; for a graph listed in both directions the in-degree is the degree);
     normalize mean: the mean over each node's in-neighbours. self_loops adds each node's own features (the "+ I").
gat  each layer and head: z = h W; a score e_ij = LeakyReLU(a_src . z_j + a_dst . z_i) for each edge j -> i (and
     i -> i with self_loops); softmax over each node's incoming edges, dropout on those coefficients in training, and
     the weighted sum of the z_j; the heads concatenated, then the bias and the activation (elu in the paper).
Weights start from Glorot's uniform initialization (both papers). Dropout on the input features before the first
layer (input_dropout) and between layers (dropout); the classifier (dropout, an optional hidden layer, the output
layer) reads the last layer's output for the batch's nodes, which is the summary (model.embed).

Depth: each width in `hidden` is one graph layer, and each graph layer reads one more hop of neighbours; the
classifier after them is a plain linear layer. So `hidden` [16] reads one hop, while the papers' two-layer models,
whose output layer is itself a graph layer, read two: [16, 16] reads two hops, as they do.
"""
import dataclasses
from typing import Literal

from sweepkit.models.torch.base import torch_model
from sweepkit.models.torch.declare import declare
from sweepkit.models.torch.heads import check_head

ACTIVATION = Literal["relu", "leaky_relu", "elu", "gelu", "silu"]


@dataclasses.dataclass(frozen=True)
class GCNSettings:
    hidden: tuple[int, ...]         # width of each graph convolution (at least one layer)
    normalize: Literal["symmetric", "mean"]
    self_loops: bool                # each node also receives its own features
    activation: ACTIVATION          # after each layer, and in the classifier's hidden layer
    input_dropout: float            # on every node's features before the first layer, in training
    dropout: float                  # between layers, and in the classifier
    classifier_hidden: int | None   # a hidden layer in the classifier, or None


@dataclasses.dataclass(frozen=True)
class GATSettings:
    hidden: tuple[int, ...]         # each attention layer's width per head (at least one layer)
    heads: int                      # attention heads per layer, their outputs concatenated
    negative_slope: float           # of the LeakyReLU on the attention scores (0.2 in the paper)
    attention_dropout: float        # on the attention coefficients, in training
    self_loops: bool                # each node also attends to itself
    activation: ACTIVATION
    input_dropout: float
    dropout: float
    classifier_hidden: int | None


def _check(s, input_shape, name: str) -> None:
    if not s.hidden or any(h < 1 for h in s.hidden):
        raise ValueError(f"model {name}: hidden needs at least one positive width")
    if not 0.0 <= s.input_dropout < 1.0:
        raise ValueError(f"model {name}: input_dropout must be in [0, 1)")
    check_head(s)
    if input_shape is not None and (len(input_shape) != 1 or input_shape[0] < 1):
        raise ValueError(f"model {name} reads a graph's node features (one feature count), not input {input_shape}")


def check_gcn(s: GCNSettings, input_shape) -> None:
    _check(s, input_shape, "gcn")


def check_gat(s: GATSettings, input_shape) -> None:
    _check(s, input_shape, "gat")
    if s.heads < 1 or s.negative_slope < 0:
        raise ValueError("model gat: heads at least 1 and negative_slope not negative")
    if not 0.0 <= s.attention_dropout < 1.0:
        raise ValueError("model gat: attention_dropout must be in [0, 1)")


def edges_with_loops(edges, n: int, self_loops: bool) -> tuple:
    """(source, target) index tensors of the edges, with one self loop per node when asked."""
    import torch

    source, target = edges[0], edges[1]
    if self_loops:
        loops = torch.arange(n, device=edges.device)
        source, target = torch.cat([source, loops]), torch.cat([target, loops])
    return source, target


def gcn_weights(source, target, n: int, normalize: str, dtype):
    """Each edge's weight in A_hat: 1 / sqrt(d_source d_target) (symmetric) or 1 / d_target (mean), d the in-degree
    with the self loops counted; a node with no incoming edge receives nothing."""
    import torch

    ones = torch.ones(target.shape[0], dtype=dtype, device=target.device)
    degree = torch.zeros(n, dtype=dtype, device=target.device).index_add_(0, target, ones).clamp_min(1.0)
    if normalize == "symmetric":
        return degree[source].rsqrt() * degree[target].rsqrt()
    return 1.0 / degree[target]


def _module_class():
    import torch
    from torch import nn

    from sweepkit.models.torch.heads import activation_layer, classifier_head

    class GraphConv(nn.Module):
        def __init__(self, n_in: int, n_out: int):
            super().__init__()
            self.linear = nn.Linear(n_in, n_out, bias=False)
            self.bias = nn.Parameter(torch.zeros(n_out))
            nn.init.xavier_uniform_(self.linear.weight)

        def forward(self, h, source, target, weight):
            z = self.linear(h)
            out = torch.zeros((h.shape[0], z.shape[1]), dtype=z.dtype, device=z.device)
            out.index_add_(0, target, z[source] * weight[:, None].to(z.dtype))
            return out + self.bias

    class GraphAttention(nn.Module):
        def __init__(self, n_in: int, width: int, heads: int, negative_slope: float, attention_dropout: float):
            super().__init__()
            self.heads, self.width, self.negative_slope = heads, width, negative_slope
            self.linear = nn.Linear(n_in, heads * width, bias=False)
            self.att_source = nn.Parameter(torch.empty(1, heads, width))
            self.att_target = nn.Parameter(torch.empty(1, heads, width))
            self.bias = nn.Parameter(torch.zeros(heads * width))
            self.attention_dropout = nn.Dropout(attention_dropout)
            for weight in (self.linear.weight, self.att_source, self.att_target):
                nn.init.xavier_uniform_(weight)

        def forward(self, h, source, target):
            n = h.shape[0]
            z = self.linear(h).view(n, self.heads, self.width)
            from_source = (z * self.att_source).sum(dim=-1)            # [nodes, heads]
            to_target = (z * self.att_target).sum(dim=-1)
            scores = nn.functional.leaky_relu(from_source[source] + to_target[target], self.negative_slope)
            index = target[:, None].expand(-1, self.heads)
            with torch.no_grad():                                      # the softmax's shift; it cancels exactly
                top = torch.full((n, self.heads), float("-inf"), dtype=scores.dtype, device=scores.device)
                top = top.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
            weight = torch.exp(scores - top[target])
            total = torch.zeros((n, self.heads), dtype=weight.dtype, device=weight.device).index_add_(0, target,
                                                                                                     weight)
            alpha = self.attention_dropout(weight / total[target])
            out = torch.zeros((n, self.heads, self.width), dtype=z.dtype, device=z.device)
            out.index_add_(0, target, z[source] * alpha[..., None].to(z.dtype))
            return out.reshape(n, self.heads * self.width) + self.bias

    class GraphNetwork(nn.Module):
        def __init__(self, settings, input_shape: tuple, n_outputs: int, kind: str):
            super().__init__()
            self.kind = kind
            self.self_loops = settings.self_loops
            self.normalize = getattr(settings, "normalize", None)
            width = input_shape[0]
            layers = []
            for size in settings.hidden:
                if kind == "gcn":
                    layers.append(GraphConv(width, size))
                    width = size
                else:
                    layers.append(GraphAttention(width, size, settings.heads, settings.negative_slope,
                                                 settings.attention_dropout))
                    width = size * settings.heads
            self.input_dropout = nn.Dropout(settings.input_dropout)
            self.layers = nn.ModuleList(layers)
            self.activations = nn.ModuleList([activation_layer(settings.activation) for _ in layers])
            self.dropouts = nn.ModuleList([nn.Dropout(settings.dropout) for _ in layers[1:]])
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_outputs)

        def embed(self, x, lengths):
            """x: a GraphBatch (sweepkit.train.batches); lengths: None. The batch nodes' summaries [nodes, width]."""
            device = self.head[-1].weight.device
            features, edges = x.graph.tensors(device)
            n = features.shape[0]
            source, target = edges_with_loops(edges, n, self.self_loops)
            h = self.input_dropout(features)
            weight = gcn_weights(source, target, n, self.normalize, h.dtype) if self.kind == "gcn" else None
            for k, (layer, activation) in enumerate(zip(self.layers, self.activations)):
                if k > 0:
                    h = self.dropouts[k - 1](h)
                h = activation(layer(h, source, target, weight) if self.kind == "gcn" else layer(h, source, target))
            return h[torch.as_tensor(x.nodes, device=device)]

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, lengths):
            return self.classify(self.embed(x, lengths))

    return GraphNetwork


@torch_model("gcn", settings=GCNSettings, kinds=("graph",),
             description="graph convolutional network on a graph's nodes (Kipf and Welling 2017; transductive)",
             example="graph_networks", check=check_gcn)
def build_gcn(settings: GCNSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs, "gcn")


@torch_model("gat", settings=GATSettings, kinds=("graph",),
             description="graph attention network on a graph's nodes (Velickovic et al. 2018; transductive)",
             example="graph_networks", check=check_gat)
def build_gat(settings: GATSettings, input_shape: tuple, n_outputs: int):
    return _module_class()(settings, input_shape, n_outputs, "gat")


declare("gcn", "torch", ("torch",), transductive=True)
declare("gat", "torch", ("torch",), transductive=True)
