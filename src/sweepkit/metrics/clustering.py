"""Clustering criteria without labels, and agreement with known labels (the reveal metrics).

Label-free (choose with these, fixed in the RULE before results exist):
  silhouette          mean silhouette on the fixed sample, noise rows left out (Rousseeuw 1987)
  calinski_harabasz   between- over within-cluster dispersion, noise left out (higher is better)
  davies_bouldin      mean similarity of each cluster to its most similar one, noise left out (lower is better)
  dbcv                density-based clustering validation (Moulavi et al. 2014), the index made for density-based
                      clusters such as HDBSCAN's, on the fixed sample; noise rows count in the total but score 0
  inertia             within-cluster sum of squares (lower is better; compare at a fixed number of clusters)
  bic, aic            the mixture model's own information criteria on the rows it was fitted on (lower is better)
  gap                 the gap statistic (Tibshirani, Walther and Hastie 2001) against uniform reference data in the
                      sample's bounding box, refitting the same configuration criteria_options.gap_references times
Descriptive (recorded, never chosen on): n_clusters, noise_fraction, largest_cluster_fraction.

With known labels (read only after the choice is recorded): ari, ami, nmi, homogeneity, completeness, v_measure,
fowlkes_mallows and purity, over the evaluated rows that have a reference label. A noise label (-1) counts as one
more cluster.

A criterion with no value for a result (one cluster, every row noise) raises Undefined; the run records why.
"""
import numpy as np

from sweepkit.metrics.unsupervised_context import PRECOMPUTED, Undefined
from sweepkit.registry import register_metric
from sweepkit.streams import generator

TASK = ("clustering",)


def _clustered(ctx, positions):
    """The rows' coordinates and labels, noise left out. Undefined when the data are precomputed distances."""
    if ctx.distances is not None:
        raise Undefined(PRECOMPUTED)
    labels = ctx.labels[positions]
    keep = labels >= 0
    return ctx.x[positions][keep], labels[keep]


def _need_two(labels) -> None:
    if np.unique(labels).size < 2:
        raise Undefined(f"{np.unique(labels).size} cluster(s) outside noise; the criterion needs at least 2")


@register_metric("silhouette", tasks=TASK, direction="max", needs=("x", "labels"), weighted=False, label_free=True,
                 description="mean silhouette on the fixed sample, noise left out")
def silhouette(ctx) -> float:
    from sklearn.metrics import silhouette_score

    if ctx.distances is not None:
        labels = ctx.labels[ctx.sample]
        keep = labels >= 0
        rows = ctx.sample[keep]
        labels = labels[keep]
        _need_two(labels)
        if labels.size <= np.unique(labels).size:
            raise Undefined("every sampled row is its own cluster")
        return float(silhouette_score(ctx.distances[np.ix_(rows, rows)], labels, metric="precomputed"))
    x, labels = _clustered(ctx, ctx.sample)
    _need_two(labels)
    if labels.size <= np.unique(labels).size:
        raise Undefined("every sampled row is its own cluster")
    return float(silhouette_score(x, labels))


@register_metric("calinski_harabasz", tasks=TASK, direction="max", needs=("x", "labels"), weighted=False,
                 label_free=True, description="between- over within-cluster dispersion, noise left out")
def calinski_harabasz(ctx) -> float:
    from sklearn.metrics import calinski_harabasz_score

    x, labels = _clustered(ctx, np.arange(ctx.x.shape[0]))
    _need_two(labels)
    return float(calinski_harabasz_score(x, labels))


@register_metric("davies_bouldin", tasks=TASK, direction="min", needs=("x", "labels"), weighted=False,
                 label_free=True, description="mean similarity of each cluster to its most similar one, noise left out")
def davies_bouldin(ctx) -> float:
    from sklearn.metrics import davies_bouldin_score

    x, labels = _clustered(ctx, np.arange(ctx.x.shape[0]))
    _need_two(labels)
    return float(davies_bouldin_score(x, labels))


@register_metric("inertia", tasks=TASK, direction="min", needs=("x", "labels"), weighted=False, label_free=True,
                 description="within-cluster sum of squared distances to each cluster's mean, noise left out")
def inertia(ctx) -> float:
    x, labels = _clustered(ctx, np.arange(ctx.x.shape[0]))
    total = 0.0
    for c in np.unique(labels):
        members = x[labels == c].astype(np.float64)
        total += float(((members - members.mean(axis=0)) ** 2).sum())
    return total


def _model_value(ctx, key: str) -> float:
    if key not in ctx.model_info:
        raise Undefined(f"the model does not report {key} (only mixture models do)")
    return float(ctx.model_info[key])


@register_metric("bic", tasks=TASK, direction="min", needs=("model",), weighted=False, label_free=True,
                 description="the mixture model's Bayesian information criterion on the rows it was fitted on")
def bic(ctx) -> float:
    return _model_value(ctx, "bic")


@register_metric("aic", tasks=TASK, direction="min", needs=("model",), weighted=False, label_free=True,
                 description="the mixture model's Akaike information criterion on the rows it was fitted on")
def aic(ctx) -> float:
    return _model_value(ctx, "aic")


@register_metric("n_clusters", tasks=TASK, direction="max", needs=("labels",), weighted=False, label_free=True,
                 description="number of clusters, noise not counted", descriptive=True)
def n_clusters(ctx) -> float:
    return float(np.unique(ctx.labels[ctx.labels >= 0]).size)


@register_metric("noise_fraction", tasks=TASK, direction="min", needs=("labels",), weighted=False, label_free=True,
                 description="share of evaluated rows labelled noise (-1)", descriptive=True)
def noise_fraction(ctx) -> float:
    return float(np.mean(ctx.labels < 0))


@register_metric("largest_cluster_fraction", tasks=TASK, direction="min", needs=("labels",), weighted=False,
                 label_free=True, description="share of evaluated rows in the largest cluster", descriptive=True)
def largest_cluster_fraction(ctx) -> float:
    labels = ctx.labels[ctx.labels >= 0]
    if labels.size == 0:
        raise Undefined("every row is noise")
    return float(np.bincount(labels).max() / ctx.labels.size)


def _log_dispersion(x: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels[labels >= 0]):
        members = x[labels == c].astype(np.float64)
        total += float(((members - members.mean(axis=0)) ** 2).sum())
    if total <= 0:
        raise Undefined("the within-cluster dispersion is 0")
    return float(np.log(total))


@register_metric("gap", tasks=TASK, direction="max", needs=("x", "labels", "refit"), weighted=False,
                 label_free=True, options=(("gap_references", int),),
                 description="gap statistic against uniform data in the sample's bounding box (Tibshirani 2001)")
def gap(ctx) -> float:
    if ctx.distances is not None:
        raise Undefined(PRECOMPUTED)
    if ctx.refit is None:
        raise Undefined("this method cannot be refitted on reference data")
    b = int(ctx.options["gap_references"])
    if b < 1:
        raise ValueError("criteria_options.gap_references must be at least 1")
    x = ctx.x[ctx.sample].astype(np.float64)
    labels = ctx.labels[ctx.sample]
    observed = _log_dispersion(x, labels)
    low, high = x.min(axis=0), x.max(axis=0)
    rng = generator(ctx.seed, "criterion gap")
    reference = []
    for _ in range(b):
        fake = rng.uniform(low, high, size=x.shape)
        reference.append(_log_dispersion(fake, np.asarray(ctx.refit(fake.astype(np.float32)))))
    return float(np.mean(reference) - observed)


def _log_core_distances(distances: np.ndarray, d: int) -> np.ndarray:
    """log of the all-points core distance of each point of one cluster (Moulavi et al. 2014, definition 2), from
    its distances to the cluster's other points, computed in log space so that a high dimension cannot overflow."""
    n = distances.shape[0]
    with np.errstate(divide="ignore"):
        terms = np.where(distances > 0, -d * np.log(np.where(distances > 0, distances, 1.0)), -np.inf)
    top = terms.max(axis=1)
    finite = np.isfinite(top)
    log_sum = np.full(n, -np.inf)
    log_sum[finite] = np.log(np.exp(terms[finite] - top[finite, None]).sum(axis=1)) + top[finite]
    out = np.full(n, -np.inf)
    out[finite] = (-1.0 / d) * (log_sum[finite] - np.log(n - 1))
    return out


def dbcv_score(x: np.ndarray, labels: np.ndarray) -> tuple:
    """(DBCV, per-cluster validity) with Euclidean distances. Noise (-1) counts in the total row count but scores 0.
    Fewer than two clusters: Undefined."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist

    x = np.asarray(x, dtype=np.float64)
    clusters = [c for c in np.unique(labels) if c >= 0]
    if len(clusters) < 2:
        raise Undefined(f"{len(clusters)} cluster(s) outside noise; DBCV needs at least 2")
    d = x.shape[1]
    core, internal, sparseness, members = {}, {}, {}, {}
    for c in clusters:
        points = x[labels == c]
        members[c] = points
        if points.shape[0] < 2:
            raise Undefined(f"cluster {c} has one row in the sample; DBCV needs at least 2 per cluster")
        distances = cdist(points, points)
        log_core = _log_core_distances(distances, d)
        core_c = np.exp(log_core)
        reach = np.maximum(distances, np.maximum(core_c[:, None], core_c[None, :]))
        np.fill_diagonal(reach, 0.0)
        tree = minimum_spanning_tree(reach).toarray()
        # an edge of weight 0 (duplicate rows) is invisible to minimum_spanning_tree; such rows are interior anyway
        tree = np.maximum(tree, tree.T)
        degree = (tree > 0).sum(axis=1)
        inner = degree > 1
        if not inner.any():
            inner = np.ones(points.shape[0], dtype=bool)
        edges = tree[np.ix_(inner, inner)]
        weights = edges[edges > 0]
        if weights.size == 0:
            weights = tree[tree > 0]
        sparseness[c] = float(weights.max()) if weights.size else 0.0
        core[c] = core_c
        internal[c] = inner
    separation = {c: np.inf for c in clusters}
    for i, a in enumerate(clusters):
        for b in clusters[i + 1:]:
            pa, pb = members[a][internal[a]], members[b][internal[b]]
            reach = np.maximum(cdist(pa, pb), np.maximum(core[a][internal[a]][:, None], core[b][internal[b]][None, :]))
            value = float(reach.min())
            separation[a] = min(separation[a], value)
            separation[b] = min(separation[b], value)
    validity = {}
    total = labels.size
    score = 0.0
    for c in clusters:
        denom = max(separation[c], sparseness[c])
        validity[c] = (separation[c] - sparseness[c]) / denom if denom > 0 else 0.0
        score += (members[c].shape[0] / total) * validity[c]
    return float(score), {int(c): float(v) for c, v in validity.items()}


@register_metric("dbcv", tasks=TASK, direction="max", needs=("x", "labels"), weighted=False, label_free=True,
                 description="density-based clustering validation (Moulavi 2014) on the fixed sample")
def dbcv(ctx) -> float:
    if ctx.distances is not None:
        raise Undefined(PRECOMPUTED)
    return dbcv_score(ctx.x[ctx.sample], ctx.labels[ctx.sample])[0]


# Agreement with known labels: the reveal metrics -------------------------------------------------------------------

def _pair(ctx):
    if ctx.reference is None:
        raise ValueError("no reference labels")
    reference = np.asarray(ctx.reference).astype(str)
    known = reference != ""
    if not known.any():
        raise Undefined("no evaluated row has a reference label")
    return reference[known], ctx.labels[known]


def _reveal(name: str, description: str):
    return register_metric(name, tasks=TASK, direction="max", needs=("labels", "reference"), weighted=False,
                           label_free=False, description=description)


@_reveal("ari", "adjusted Rand index between the clusters and the reference labels")
def ari(ctx) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(*_pair(ctx)))


@_reveal("ami", "adjusted mutual information between the clusters and the reference labels")
def ami(ctx) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    return float(adjusted_mutual_info_score(*_pair(ctx)))


@_reveal("nmi", "normalized mutual information between the clusters and the reference labels")
def nmi(ctx) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(*_pair(ctx)))


@_reveal("homogeneity", "each cluster holds one reference label")
def homogeneity(ctx) -> float:
    from sklearn.metrics import homogeneity_score

    return float(homogeneity_score(*_pair(ctx)))


@_reveal("completeness", "each reference label sits in one cluster")
def completeness(ctx) -> float:
    from sklearn.metrics import completeness_score

    return float(completeness_score(*_pair(ctx)))


@_reveal("v_measure", "harmonic mean of homogeneity and completeness")
def v_measure(ctx) -> float:
    from sklearn.metrics import v_measure_score

    return float(v_measure_score(*_pair(ctx)))


@_reveal("fowlkes_mallows", "geometric mean of pairwise precision and recall")
def fowlkes_mallows(ctx) -> float:
    from sklearn.metrics import fowlkes_mallows_score

    return float(fowlkes_mallows_score(*_pair(ctx)))


@_reveal("purity", "share of rows in their cluster's most common reference label")
def purity(ctx) -> float:
    reference, labels = _pair(ctx)
    total = 0
    for c in np.unique(labels):
        _, counts = np.unique(reference[labels == c], return_counts=True)
        total += counts.max()
    return float(total / reference.size)


def adjusted_rand(a, b) -> float:
    """ARI between two label vectors (stability, CPU and GPU agreement)."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))
