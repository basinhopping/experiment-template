"""Centroid and model-based clustering: k-means (also mini-batch and bisecting), k-medoids, fuzzy c-means, Gaussian
mixtures and Bayesian Gaussian mixtures.

k-means has a GPU path in cuML. k-medoids is the template's own alternating implementation (the GPL package kmedoids,
with FasterPAM, is an optional extra: method "fasterpam"). Fuzzy c-means is the template's own (Bezdek 1981):
memberships sum to 1 per row; the label is the largest membership, and soft() gives that membership.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, LibraryClusterer, merged_kwargs, plain_params
from sweepkit.models.distances import DISTANCE_OR_PRECOMPUTED, matrix
from sweepkit.registry import register_model
from sweepkit.settings import Scalar
from sweepkit.streams import generator

CLUSTER = ("clustering",)
TAB = ("tabular",)


@dataclasses.dataclass(frozen=True)
class KMeansSettings:
    n_clusters: int
    init: Literal["k-means++", "random"]
    n_init: int                     # runs with different centroid seeds; the best by inertia is kept
    max_iter: int
    tol: float
    extra: dict[str, Scalar]


@register_model("kmeans", settings=KMeansSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="cuml",
                profile="core", packages=("sklearn",), description="k-means (scikit-learn; cuML on the GPU)",
                example="cluster_kmeans")
class KMeans(LibraryClusterer):
    def make(self, s, seed, device):
        kwargs = {"n_clusters": s.n_clusters, "init": s.init, "n_init": s.n_init, "max_iter": s.max_iter,
                  "tol": s.tol, "random_state": seed}
        if device == "cuda":
            from cuml.cluster import KMeans as K
        else:
            from sklearn.cluster import KMeans as K
        return K(**merged_kwargs(kwargs, s.extra, "kmeans"))

    def info(self):
        return {"inertia": float(self.model.inertia_), "n_iter": int(getattr(self.model, "n_iter_", -1))}


@dataclasses.dataclass(frozen=True)
class MiniBatchKMeansSettings:
    n_clusters: int
    batch_size: int
    n_init: int
    max_iter: int
    extra: dict[str, Scalar]


@register_model("minibatch_kmeans", settings=MiniBatchKMeansSettings, family="clusterer", tasks=CLUSTER, kinds=TAB,
                gpu="none", profile="core", packages=("sklearn",), description="mini-batch k-means for many rows",
                example="cluster_kmeans")
class MiniBatchKMeans(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.cluster import MiniBatchKMeans as K

        return K(**merged_kwargs({"n_clusters": s.n_clusters, "batch_size": s.batch_size, "n_init": s.n_init,
                                  "max_iter": s.max_iter, "random_state": seed}, s.extra, "minibatch_kmeans"))

    def info(self):
        return {"inertia": float(self.model.inertia_)}


@dataclasses.dataclass(frozen=True)
class BisectingKMeansSettings:
    n_clusters: int
    bisecting_strategy: Literal["biggest_inertia", "largest_cluster"]
    n_init: int
    extra: dict[str, Scalar]


@register_model("bisecting_kmeans", settings=BisectingKMeansSettings, family="clusterer", tasks=CLUSTER, kinds=TAB,
                gpu="none", profile="core", packages=("sklearn",), description="bisecting k-means (divisive)",
                example="cluster_kmeans")
class BisectingKMeans(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.cluster import BisectingKMeans as K

        return K(**merged_kwargs({"n_clusters": s.n_clusters, "bisecting_strategy": s.bisecting_strategy,
                                  "n_init": s.n_init, "random_state": seed}, s.extra, "bisecting_kmeans"))

    def info(self):
        return {"inertia": float(self.model.inertia_)}


@dataclasses.dataclass(frozen=True)
class KMedoidsSettings:
    n_clusters: int
    metric: DISTANCE_OR_PRECOMPUTED  # any distance of sweepkit.models.distances (gower for mixed columns), or
                                     # precomputed: the data are a distance matrix
    method: Literal["alternate", "fasterpam"]   # alternate: the template's own; fasterpam: the GPL package kmedoids
    max_iter: int
    n_init: int                     # restarts from different medoid seeds; the lowest total distance is kept


@register_model("kmedoids", settings=KMedoidsSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("scipy",),
                description="k-medoids on any distance (own alternating algorithm; FasterPAM with the extras)",
                example="cluster_kmedoids")
class KMedoids(Clusterer):
    def fit(self, x):
        d, self.distances = matrix(np.asarray(x, dtype=np.float64), self.s.metric, self.categorical)
        n, k = d.shape[0], self.s.n_clusters
        if k >= n:
            raise ValueError(f"kmedoids: n_clusters {k} must be below the rows ({n})")
        if self.s.method == "fasterpam":
            import kmedoids

            best = None
            for r in range(self.s.n_init):
                result = kmedoids.fasterpam(d, k, max_iter=self.s.max_iter, random_state=self.seed + r)
                if best is None or result.loss < best[0]:
                    best = (float(result.loss), np.asarray(result.medoids), np.asarray(result.labels))
        else:
            rng = generator(self.seed, "kmedoids")
            best = None
            for _ in range(self.s.n_init):
                medoids = _kpp_medoids(d, k, rng)
                for _ in range(self.s.max_iter):
                    labels = np.argmin(d[:, medoids], axis=1)
                    new = medoids.copy()
                    for c in range(k):
                        members = np.flatnonzero(labels == c)
                        if members.size:
                            new[c] = members[np.argmin(d[np.ix_(members, members)].sum(axis=0))]
                    if np.array_equal(np.sort(new), np.sort(medoids)):
                        break
                    medoids = new
                labels = np.argmin(d[:, medoids], axis=1)
                loss = float(d[np.arange(n), medoids[labels]].sum())
                if best is None or loss < best[0]:
                    best = (loss, medoids, labels)
        self.loss, self.medoid_index, self._labels = best
        self.medoids = np.asarray(x)[self.medoid_index]

    def labels(self):
        return np.asarray(self._labels, dtype=np.int64)

    def can_assign(self):
        return True

    def assign(self, x_new):
        """Each new row to its nearest medoid. With precomputed distances, x_new holds the new rows' distances to the
        fit rows (the runner's columns), so the medoids' columns are read directly."""
        x_new = np.asarray(x_new, dtype=np.float64)
        if self.s.metric == "precomputed":
            return np.argmin(x_new[:, self.medoid_index], axis=1)
        return np.argmin(self.distances.between(x_new, self.medoids), axis=1)

    def info(self):
        return {"total_distance": self.loss}

    def params(self):
        return plain_params({**dataclasses.asdict(self.s), "medoid_rows": [int(i) for i in self.medoid_index]})


def _kpp_medoids(d: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ style seeding on a distance matrix."""
    n = d.shape[0]
    chosen = [int(rng.integers(n))]
    for _ in range(1, k):
        nearest = d[:, chosen].min(axis=1)
        p = nearest ** 2
        total = p.sum()
        chosen.append(int(rng.choice(n, p=p / total)) if total > 0 else int(rng.integers(n)))
    return np.array(chosen, dtype=np.int64)


@dataclasses.dataclass(frozen=True)
class FuzzyCMeansSettings:
    n_clusters: int
    m: float                        # fuzzifier, above 1 (2 is usual)
    max_iter: int
    tol: float                      # stop when memberships change less than this


@register_model("fuzzy_cmeans", settings=FuzzyCMeansSettings, family="clusterer", tasks=CLUSTER, kinds=TAB,
                gpu="none", profile="core", packages=(), description="fuzzy c-means (own implementation)",
                example="cluster_kmeans")
class FuzzyCMeans(Clusterer):
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        n, k, m = x.shape[0], self.s.n_clusters, self.s.m
        if not m > 1:
            raise ValueError("fuzzy_cmeans: m must be above 1")
        rng = generator(self.seed, "fuzzy cmeans")
        u = rng.dirichlet(np.ones(k), size=n)
        for it in range(1, self.s.max_iter + 1):
            weights = u ** m
            centers = (weights.T @ x) / weights.sum(axis=0)[:, None]
            dist = np.maximum(np.sqrt(((x[:, None, :] - centers[None]) ** 2).sum(axis=2)), 1e-12)
            power = 2.0 / (m - 1)
            new = 1.0 / ((dist[:, :, None] / dist[:, None, :]) ** power).sum(axis=2)
            change = np.abs(new - u).max()
            u = new
            if change < self.s.tol:
                break
        self.u, self.centers, self.n_iter = u, centers, it

    def labels(self):
        return np.argmax(self.u, axis=1).astype(np.int64)

    def soft(self):
        return {"probability": self.u.max(axis=1)}

    def can_assign(self):
        return True

    def assign(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        dist = ((x_new[:, None, :] - self.centers[None]) ** 2).sum(axis=2)
        return np.argmin(dist, axis=1)

    def info(self):
        return {"n_iter": self.n_iter}

    def params(self):
        return plain_params(dataclasses.asdict(self.s))

    def extra_files(self):
        return {"memberships.npy": self.u.astype(np.float32)}


@dataclasses.dataclass(frozen=True)
class GMMSettings:
    n_components: int
    covariance_type: Literal["full", "tied", "diag", "spherical"]
    n_init: int
    max_iter: int
    reg_covar: float
    extra: dict[str, Scalar]


@register_model("gmm", settings=GMMSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="Gaussian mixture model (BIC and AIC reported)",
                example="cluster_gmm")
class GMM(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(**merged_kwargs({"n_components": s.n_components, "covariance_type": s.covariance_type,
                                                "n_init": s.n_init, "max_iter": s.max_iter, "reg_covar": s.reg_covar,
                                                "random_state": seed}, s.extra, "gmm"))

    def fit(self, x):
        super().fit(x)
        self._x = np.asarray(x)
        self._labels = self.model.predict(self._x)

    def labels(self):
        return np.asarray(self._labels, dtype=np.int64)

    def soft(self):
        return {"probability": self.model.predict_proba(self._x).max(axis=1)}

    def info(self):
        return {"bic": float(self.model.bic(self._x)), "aic": float(self.model.aic(self._x)),
                "converged": bool(self.model.converged_)}


@dataclasses.dataclass(frozen=True)
class BayesianGMMSettings:
    n_components: int               # the most components; unneeded ones get weight near 0
    covariance_type: Literal["full", "tied", "diag", "spherical"]
    weight_concentration_prior_type: Literal["dirichlet_process", "dirichlet_distribution"]
    weight_concentration_prior: float | None   # None: 1 / n_components
    max_iter: int
    extra: dict[str, Scalar]


@register_model("bayesian_gmm", settings=BayesianGMMSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",),
                description="Bayesian Gaussian mixture (finds how many components it needs)", example="cluster_gmm")
class BayesianGMM(GMM):
    def make(self, s, seed, device):
        from sklearn.mixture import BayesianGaussianMixture

        return BayesianGaussianMixture(**merged_kwargs({
            "n_components": s.n_components, "covariance_type": s.covariance_type,
            "weight_concentration_prior_type": s.weight_concentration_prior_type,
            "weight_concentration_prior": s.weight_concentration_prior, "max_iter": s.max_iter,
            "random_state": seed}, s.extra, "bayesian_gmm"))

    def info(self):
        return {"effective_components": int((self.model.weights_ > 0.01).sum()),
                "converged": bool(self.model.converged_)}
