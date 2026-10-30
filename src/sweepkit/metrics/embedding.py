"""Embedding (dimensionality reduction) criteria without labels, and with known labels.

Label-free, on the fixed sample, with the task's n_neighbors:
  trustworthiness     are the embedding's neighbours true neighbours? (Venna and Kaski 2001; scikit-learn's)
  continuity          are the true neighbours still neighbours in the embedding? (trustworthiness, spaces swapped)
  knn_preservation    mean share of each row's k nearest neighbours kept in the embedding
  reconstruction_error  mean squared error of the inverse transform (methods that have one; lower is better)
  explained_variance  share of variance the components keep (linear methods)
With known labels: knn_label_agreement (leave-one-out k-nearest-neighbour accuracy of the reference labels in the
embedding) and reference_silhouette (silhouette of the reference labels in the embedding).

Two-dimensional UMAP and t-SNE pictures are for looking, not for measuring distances: these criteria score
neighbourhoods, which those methods aim to keep, not distances, which they do not.

When the data are a precomputed distance matrix, the true neighbours come from those distances: trustworthiness and
knn_preservation use them, and continuity (which needs the original space's coordinates for its swapped spaces) is
undefined.
"""
import numpy as np

from sweepkit.metrics.unsupervised_context import PRECOMPUTED, Undefined
from sweepkit.registry import register_metric

TASK = ("embedding",)


def _sampled(ctx):
    """(the sample's original rows, or their distances to each other when precomputed; its embedding; k)."""
    if ctx.distances is not None:
        x = ctx.distances[np.ix_(ctx.sample, ctx.sample)]
    else:
        x = ctx.x[ctx.sample].astype(np.float64)
    e = ctx.embedding[ctx.sample].astype(np.float64)
    k = ctx.n_neighbors
    if k >= x.shape[0] / 2:
        raise Undefined(f"n_neighbors {k} must be below half the sample ({x.shape[0]} rows)")
    return x, e, k


@register_metric("trustworthiness", tasks=TASK, direction="max", needs=("x", "embedding"), weighted=False,
                 label_free=True, description="whether the embedding's neighbours are true neighbours")
def trustworthiness(ctx) -> float:
    from sklearn.manifold import trustworthiness as tw

    x, e, k = _sampled(ctx)
    return float(tw(x, e, n_neighbors=k, metric="precomputed" if ctx.distances is not None else "euclidean"))


@register_metric("continuity", tasks=TASK, direction="max", needs=("x", "embedding"), weighted=False,
                 label_free=True, description="whether the true neighbours stay neighbours in the embedding")
def continuity(ctx) -> float:
    from sklearn.manifold import trustworthiness as tw

    if ctx.distances is not None:
        raise Undefined(PRECOMPUTED)
    x, e, k = _sampled(ctx)
    return float(tw(e, x, n_neighbors=k))


def _neighbours(points: np.ndarray, k: int, precomputed: bool = False) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    if precomputed:
        index = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(points)
        found = index.kneighbors(points, return_distance=False)
        return np.array([[j for j in row if j != i][:k] for i, row in enumerate(found)], dtype=np.int64)
    index = NearestNeighbors(n_neighbors=k + 1).fit(points)
    found = index.kneighbors(points, return_distance=False)
    out = np.empty((points.shape[0], k), dtype=np.int64)
    for i, row in enumerate(found):
        others = [j for j in row if j != i][:k]
        out[i] = others
    return out


@register_metric("knn_preservation", tasks=TASK, direction="max", needs=("x", "embedding"), weighted=False,
                 label_free=True, description="mean share of each row's k nearest neighbours kept in the embedding")
def knn_preservation(ctx) -> float:
    x, e, k = _sampled(ctx)
    a, b = _neighbours(x, k, ctx.distances is not None), _neighbours(e, k)
    return float(np.mean([np.intersect1d(a[i], b[i]).size / k for i in range(a.shape[0])]))


def _model_value(ctx, key: str) -> float:
    if key not in ctx.model_info:
        raise Undefined(f"the method does not report {key}")
    return float(ctx.model_info[key])


@register_metric("reconstruction_error", tasks=TASK, direction="min", needs=("model",), weighted=False,
                 label_free=True, description="mean squared error of the inverse transform on the evaluated rows")
def reconstruction_error(ctx) -> float:
    return _model_value(ctx, "reconstruction_error")


@register_metric("explained_variance", tasks=TASK, direction="max", needs=("model",), weighted=False,
                 label_free=True, description="share of the variance the components keep (linear methods)")
def explained_variance(ctx) -> float:
    return _model_value(ctx, "explained_variance")


def _reference(ctx):
    if ctx.reference is None:
        raise ValueError("no reference labels")
    reference = np.asarray(ctx.reference).astype(str)[ctx.sample]
    known = reference != ""
    if known.sum() < 3:
        raise Undefined("fewer than 3 sampled rows have a reference label")
    return ctx.embedding[ctx.sample][known].astype(np.float64), reference[known]


@register_metric("knn_label_agreement", tasks=TASK, direction="max", needs=("embedding", "reference"),
                 weighted=False, label_free=False,
                 description="leave-one-out k-nearest-neighbour accuracy of the reference labels in the embedding")
def knn_label_agreement(ctx) -> float:
    e, reference = _reference(ctx)
    k = min(ctx.n_neighbors, e.shape[0] - 1)
    neighbours = _neighbours(e, k)
    correct = 0
    for i in range(e.shape[0]):
        values, counts = np.unique(reference[neighbours[i]], return_counts=True)
        best = values[counts == counts.max()]
        # ties go to the label of the nearest neighbour among the tied labels
        vote = next(reference[j] for j in neighbours[i] if reference[j] in best)
        correct += vote == reference[i]
    return float(correct / e.shape[0])


@register_metric("reference_silhouette", tasks=TASK, direction="max", needs=("embedding", "reference"),
                 weighted=False, label_free=False, description="silhouette of the reference labels in the embedding")
def reference_silhouette(ctx) -> float:
    from sklearn.metrics import silhouette_score

    e, reference = _reference(ctx)
    if np.unique(reference).size < 2:
        raise Undefined("the sampled reference labels hold one class")
    return float(silhouette_score(e, reference))
