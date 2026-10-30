"""Other clustering methods: BIRCH, self-organizing maps (SOM and FlowSOM style), time-series k-means with dynamic
time warping and k-Shape, spectral biclustering and co-clustering, and topics from latent Dirichlet allocation.

som is the template's own online self-organizing map (Kohonen) in numpy: a grid of nodes, each with a weight vector;
every row, in a seeded order each epoch, pulls its best-matching node and the node's grid neighbours towards itself,
with a learning rate and a Gaussian neighbourhood radius that both shrink exponentially from their first to their
last value. Each row's cluster is its best-matching node, so a grid of r by c nodes gives up to r * c clusters. The
mean distance from the rows to their nodes (the quantization error) is recorded after every epoch, and the node
weights are written as som_weights.npy.

flowsom is FlowSOM's two steps with the template's own pieces: the same map, then the node weights grouped into
n_clusters by agglomerative clustering (SciPy), so each row's cluster is its node's group (node_clusters.npy).

dtw_kmeans and kshape (the tslearn package, profile unsupervised) take each row as one equally long time series, its
columns being the time points in order (the synthetic "series" data). k-Shape compares shapes after z-normalizing
each series, which tslearn expects to be done beforehand: normalize does it inside the model.

spectral_biclustering and spectral_coclustering (scikit-learn) group rows and columns at once; the row groups are
the clusters, and the column groups are written as column_labels.npy. Both take non-negative data, as counts or
expression values are.

lda fits latent Dirichlet allocation (scikit-learn) to counts, such as words per document; each row's cluster is
its most likely topic, its soft probability that topic's share, and the model's perplexity on the fit rows is
recorded. Its variational fit can merge two topics from an unlucky start, so it restarts n_init times from seeded
starts and keeps the fit with the highest approximate log-likelihood bound on the fit rows (scikit-learn's score),
which reads no labels.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, LibraryClusterer, _host, library_params, merged_kwargs, plain_params
from sweepkit.registry import register_model
from sweepkit.settings import Scalar
from sweepkit.streams import generator, int_seed

CLUSTER = ("clustering",)
TAB = ("tabular",)


@dataclasses.dataclass(frozen=True)
class BirchSettings:
    threshold: float                # radius of a subcluster
    branching_factor: int
    n_clusters: int | None          # a final clustering of the subclusters into this many; None: the subclusters
    extra: dict[str, Scalar]


@register_model("birch", settings=BirchSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="BIRCH (scikit-learn; one pass, many rows)",
                example="cluster_birch")
class Birch(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.cluster import Birch as B

        return B(**merged_kwargs({"threshold": s.threshold, "branching_factor": s.branching_factor,
                                  "n_clusters": s.n_clusters, "compute_labels": True}, s.extra, "birch"))

    def info(self):
        return {"subclusters": int(len(self.model.subcluster_centers_))}


def som_train(x: np.ndarray, rows: int, cols: int, epochs: int, learning_rate: float, learning_rate_final: float,
              sigma: float, sigma_final: float, init: str, rng) -> tuple:
    """(weights [rows * cols, features], grid coordinates [rows * cols, 2], quantization error after each epoch)."""
    n, d = x.shape
    k = rows * cols
    grid = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=np.float64)
    if init == "rows":
        if n < k:
            raise ValueError(f"som: init rows needs at least as many rows ({n}) as nodes ({k})")
        weights = x[rng.choice(n, size=k, replace=False)].astype(np.float64).copy()
    else:
        mean = x.mean(axis=0)
        _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
        spread = sv[:2] / np.sqrt(max(n - 1, 1))
        axes = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt[:1], np.zeros((1, d))])
        span_r = np.linspace(-1, 1, rows) if rows > 1 else np.zeros(1)
        span_c = np.linspace(-1, 1, cols) if cols > 1 else np.zeros(1)
        weights = np.array([mean + span_r[r] * spread[0] * axes[0]
                            + (span_c[c] * spread[1] * axes[1] if spread.size > 1 else 0.0)
                            for r in range(rows) for c in range(cols)])
    total = epochs * n
    errors = []
    step = 0
    for _ in range(epochs):
        for i in rng.permutation(n):
            progress = step / max(total - 1, 1)
            lr = learning_rate * (learning_rate_final / learning_rate) ** progress
            radius = sigma * (sigma_final / sigma) ** progress
            row = x[i]
            bmu = int(np.argmin(((weights - row) ** 2).sum(axis=1)))
            pull = np.exp(-((grid - grid[bmu]) ** 2).sum(axis=1) / (2 * radius * radius))
            weights += lr * pull[:, None] * (row - weights)
            step += 1
        errors.append(float(best_matching(x, weights)[1].mean()))
    return weights, grid, errors


def best_matching(x: np.ndarray, weights: np.ndarray) -> tuple:
    """(node index, distance to it) of every row, a block of rows at a time so memory stays small."""
    x = np.asarray(x, dtype=np.float64)
    nodes = np.empty(x.shape[0], dtype=np.int64)
    distances = np.empty(x.shape[0], dtype=np.float64)
    squared_weights = (weights ** 2).sum(axis=1)
    for start in range(0, x.shape[0], 4096):
        block = x[start:start + 4096]
        d = (block ** 2).sum(axis=1)[:, None] - 2 * block @ weights.T + squared_weights[None, :]
        nearest = np.argmin(d, axis=1)
        nodes[start:start + 4096] = nearest
        distances[start:start + 4096] = np.sqrt(np.maximum(d[np.arange(block.shape[0]), nearest], 0.0))
    return nodes, distances


@dataclasses.dataclass(frozen=True)
class SOMSettings:
    grid_rows: int
    grid_cols: int
    epochs: int                     # passes over the fit rows
    learning_rate: float            # at the first step
    learning_rate_final: float      # at the last step (exponential decay between)
    sigma: float                    # neighbourhood radius on the grid at the first step
    sigma_final: float              # at the last step
    init: Literal["rows", "pca"]    # node weights start at random fit rows, or on the plane of the first two
                                    # principal components


def _check_som(job):
    s = job.model
    if s.grid_rows < 1 or s.grid_cols < 1 or s.grid_rows * s.grid_cols < 2 or s.epochs < 1:
        raise ValueError(f"{job.model_name}: a grid of at least two nodes and at least one epoch")
    if not (0 < s.learning_rate_final <= s.learning_rate) or not (0 < s.sigma_final <= s.sigma):
        raise ValueError(f"{job.model_name}: learning rates and radii positive, the final at most the first")


class _SOM(Clusterer):
    def train(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.weights, self.grid, self.errors = som_train(
            x, self.s.grid_rows, self.s.grid_cols, self.s.epochs, self.s.learning_rate, self.s.learning_rate_final,
            self.s.sigma, self.s.sigma_final, self.s.init, generator(self.seed, "som"))
        self.nodes, self.distances = best_matching(x, self.weights)

    def can_assign(self):
        return True

    def soft(self):
        return {"node_distance": self.distances.astype(np.float64)}

    def params(self):
        return plain_params(dataclasses.asdict(self.s))


@register_model("som", settings=SOMSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=(), description="self-organizing map; clusters are its nodes (own)",
                example="cluster_som")
class SOM(_SOM):
    check_job = staticmethod(_check_som)

    def fit(self, x):
        self.train(x)

    def labels(self):
        return self.nodes

    def assign(self, x_new):
        return best_matching(x_new, self.weights)[0]

    def info(self):
        return {"quantization_error": self.errors[-1], "quantization_error_first_epoch": self.errors[0],
                "nodes_used": int(np.unique(self.nodes).size)}

    def extra_files(self):
        return {"som_weights.npy": self.weights.reshape(self.s.grid_rows, self.s.grid_cols, -1).astype(np.float32)}


@dataclasses.dataclass(frozen=True)
class FlowSOMSettings:
    grid_rows: int
    grid_cols: int
    epochs: int
    learning_rate: float
    learning_rate_final: float
    sigma: float
    sigma_final: float
    init: Literal["rows", "pca"]
    n_clusters: int                 # groups of nodes (metaclusters)
    meta_linkage: Literal["ward", "average", "complete"]


@register_model("flowsom", settings=FlowSOMSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("scipy",),
                description="FlowSOM style: a self-organizing map, then its nodes grouped into clusters (own)",
                example="cluster_som")
class FlowSOM(_SOM):
    @staticmethod
    def check_job(job):
        _check_som(job)
        if not 1 <= job.model.n_clusters <= job.model.grid_rows * job.model.grid_cols:
            raise ValueError("flowsom: n_clusters between 1 and the number of nodes")

    def fit(self, x):
        from scipy.cluster.hierarchy import fcluster, linkage

        self.train(x)
        z = linkage(self.weights, method=self.s.meta_linkage, metric="euclidean")
        raw = fcluster(z, t=self.s.n_clusters, criterion="maxclust")
        _, groups = np.unique(raw, return_inverse=True)
        self.node_clusters = groups.astype(np.int64)

    def labels(self):
        return self.node_clusters[self.nodes]

    def assign(self, x_new):
        return self.node_clusters[best_matching(x_new, self.weights)[0]]

    def info(self):
        return {"quantization_error": self.errors[-1], "nodes_used": int(np.unique(self.nodes).size),
                "clusters_found": int(np.unique(self.node_clusters).size)}

    def extra_files(self):
        return {"som_weights.npy": self.weights.reshape(self.s.grid_rows, self.s.grid_cols, -1).astype(np.float32),
                "node_clusters.npy": self.node_clusters.reshape(self.s.grid_rows, self.s.grid_cols)}


@dataclasses.dataclass(frozen=True)
class DTWKMeansSettings:
    n_clusters: int
    metric: Literal["dtw", "softdtw", "euclidean"]
    max_iter: int
    n_init: int
    max_iter_barycenter: int        # iterations of the barycenter (DBA or soft-DTW) per step
    extra: dict[str, Scalar]        # other TimeSeriesKMeans parameters (tol, init, dtw_inertia)


def _series(x) -> np.ndarray:
    """tslearn's shape: [series, time points, 1]."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], x.shape[1], 1)


@register_model("dtw_kmeans", settings=DTWKMeansSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="unsupervised", packages=("tslearn",),
                description="time-series k-means with dynamic time warping (tslearn; rows are series)",
                example="cluster_series")
class DTWKMeans(LibraryClusterer):
    def make(self, s, seed, device):
        # tslearn 0.10: TimeSeriesKMeans(n_clusters, max_iter, tol, n_init, metric, max_iter_barycenter,
        # metric_params, n_jobs, dtw_inertia, verbose, random_state, init)
        from tslearn.clustering import TimeSeriesKMeans

        return TimeSeriesKMeans(**merged_kwargs({"n_clusters": s.n_clusters, "metric": s.metric,
                                                 "max_iter": s.max_iter, "n_init": s.n_init,
                                                 "max_iter_barycenter": s.max_iter_barycenter,
                                                 "random_state": seed}, s.extra, "dtw_kmeans"))

    def fit(self, x):
        self.model = self.make(self.s, self.seed, self.device)
        self.model.fit(_series(x))

    def assign(self, x_new):
        return np.asarray(self.model.predict(_series(x_new)), dtype=np.int64)

    def info(self):
        return {"inertia": float(self.model.inertia_)}


@dataclasses.dataclass(frozen=True)
class KShapeSettings:
    n_clusters: int
    max_iter: int
    n_init: int
    normalize: bool                 # z-normalize each series first (k-Shape assumes it)
    extra: dict[str, Scalar]        # other KShape parameters (tol, init)


@register_model("kshape", settings=KShapeSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="unsupervised", packages=("tslearn",),
                description="k-Shape: clusters of time series by shape (tslearn; rows are series)",
                example="cluster_series")
class KShape(LibraryClusterer):
    def make(self, s, seed, device):
        # tslearn 0.10: KShape(n_clusters, max_iter, tol, n_init, verbose, random_state, init)
        from tslearn.clustering import KShape as K

        return K(**merged_kwargs({"n_clusters": s.n_clusters, "max_iter": s.max_iter, "n_init": s.n_init,
                                  "random_state": seed}, s.extra, "kshape"))

    def _prepare(self, x):
        series = _series(x)
        if self.s.normalize:
            # tslearn 0.10: TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(X)
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance

            series = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(series)
        return series

    def fit(self, x):
        self.model = self.make(self.s, self.seed, self.device)
        self.model.fit(self._prepare(x))

    def assign(self, x_new):
        return np.asarray(self.model.predict(self._prepare(x_new)), dtype=np.int64)

    def info(self):
        return {"inertia": float(self.model.inertia_)}


@dataclasses.dataclass(frozen=True)
class SpectralBiclusteringSettings:
    n_clusters: int                 # row groups (and as many column groups)
    method: Literal["bistochastic", "scale", "log"]
    n_components: int               # singular vectors computed
    n_best: int                     # of which the best are used
    n_init: int
    extra: dict[str, Scalar]


@register_model("spectral_biclustering", settings=SpectralBiclusteringSettings, family="clusterer", tasks=CLUSTER,
                kinds=TAB, gpu="none", profile="core", packages=("sklearn",),
                description="spectral biclustering of rows and columns (scikit-learn; checkerboard structure)",
                example="cluster_bicluster")
class SpectralBiclustering(Clusterer):
    def fit(self, x):
        from sklearn.cluster import SpectralBiclustering as S

        self.model = S(**merged_kwargs({"n_clusters": self.s.n_clusters, "method": self.s.method,
                                        "n_components": self.s.n_components, "n_best": self.s.n_best,
                                        "n_init": self.s.n_init, "random_state": self.seed}, self.s.extra,
                                       "spectral_biclustering"))
        self.model.fit(np.asarray(x, dtype=np.float64))

    def labels(self):
        return np.asarray(self.model.row_labels_, dtype=np.int64)

    def params(self):
        return library_params(self.model)

    def extra_files(self):
        return {"column_labels.npy": np.asarray(self.model.column_labels_, dtype=np.int64)}


@dataclasses.dataclass(frozen=True)
class SpectralCoclusteringSettings:
    n_clusters: int                 # biclusters (each a group of rows with a group of columns)
    n_init: int
    extra: dict[str, Scalar]


@register_model("spectral_coclustering", settings=SpectralCoclusteringSettings, family="clusterer", tasks=CLUSTER,
                kinds=TAB, gpu="none", profile="core", packages=("sklearn",),
                description="spectral co-clustering of rows and columns (scikit-learn; block-diagonal structure)",
                example="cluster_bicluster")
class SpectralCoclustering(Clusterer):
    def fit(self, x):
        from sklearn.cluster import SpectralCoclustering as S

        x = np.asarray(x, dtype=np.float64)
        if (x < 0).any():
            raise ValueError("spectral_coclustering takes non-negative data; leave out scaling to z-scores")
        self.model = S(**merged_kwargs({"n_clusters": self.s.n_clusters, "n_init": self.s.n_init,
                                        "random_state": self.seed}, self.s.extra, "spectral_coclustering"))
        self.model.fit(x)

    def labels(self):
        return np.asarray(self.model.row_labels_, dtype=np.int64)

    def params(self):
        return library_params(self.model)

    def extra_files(self):
        return {"column_labels.npy": np.asarray(self.model.column_labels_, dtype=np.int64)}


@dataclasses.dataclass(frozen=True)
class LDASettings:
    n_topics: int
    doc_topic_prior: float | None   # alpha; None: 1 / n_topics (scikit-learn)
    topic_word_prior: float | None  # eta; None: 1 / n_topics
    learning_method: Literal["batch", "online"]
    max_iter: int
    n_init: int                     # restarts; the one with the highest likelihood bound is kept
    extra: dict[str, Scalar]


@register_model("lda", settings=LDASettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",),
                description="topics of counts by latent Dirichlet allocation; clusters are the likeliest topics "
                            "(scikit-learn)", example="cluster_topics")
class LDA(Clusterer):
    def fit(self, x):
        from sklearn.decomposition import LatentDirichletAllocation

        x = np.asarray(x, dtype=np.float64)
        if (x < 0).any():
            raise ValueError("lda takes counts (non-negative); leave out scaling and centring")
        if self.s.n_init < 1:
            raise ValueError("lda: n_init at least 1")
        best = None
        self.bounds = []
        for r in range(self.s.n_init):
            model = LatentDirichletAllocation(**merged_kwargs({
                "n_components": self.s.n_topics, "doc_topic_prior": self.s.doc_topic_prior,
                "topic_word_prior": self.s.topic_word_prior, "learning_method": self.s.learning_method,
                "max_iter": self.s.max_iter, "random_state": int_seed(self.seed, f"lda restart {r}")},
                self.s.extra, "lda"))
            shares = model.fit_transform(x)
            bound = float(model.score(x))
            self.bounds.append(bound)
            if best is None or bound > best[0]:
                best = (bound, model, shares)
        _, self.model, self.shares = best
        self.perplexity = float(self.model.perplexity(x))

    def labels(self):
        return np.argmax(self.shares, axis=1).astype(np.int64)

    def soft(self):
        return {"probability": self.shares.max(axis=1).astype(np.float64)}

    def can_assign(self):
        return True

    def assign(self, x_new):
        return np.argmax(self.model.transform(np.asarray(x_new, dtype=np.float64)), axis=1).astype(np.int64)

    def info(self):
        return {"perplexity": self.perplexity, "likelihood_bound": max(self.bounds),
                "restart_bounds": [round(b, 3) for b in self.bounds]}

    def params(self):
        return library_params(self.model)

    def extra_files(self):
        topics = self.model.components_ / self.model.components_.sum(axis=1, keepdims=True)
        return {"topic_words.npy": topics.astype(np.float32)}
