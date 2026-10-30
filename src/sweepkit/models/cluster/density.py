"""Density-based clustering: DBSCAN; HDBSCAN from scikit-learn, from the hdbscan package and from fast_hdbscan (with
PLSCAN, which picks the cluster resolution itself, and FLASC branch detection); OPTICS; mean shift.

GPU paths: DBSCAN and HDBSCAN in cuML (cuML's HDBSCAN takes Euclidean distances only). Noise rows get label -1. HDBSCAN
gives each row a membership probability (soft "probability") and, in the hdbscan package, a GLOSH outlier score
(soft "outlier_score").

HDBSCAN slows down and finds less structure as the number of columns grows; reduce to about 10 dimensions first
(a reduce step with UMAP or PCA; docs/UNSUPERVISED.md).
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, LibraryClusterer, _host, library_params, merged_kwargs
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

CLUSTER = ("clustering",)
TAB = ("tabular",)


@dataclasses.dataclass(frozen=True)
class DBSCANSettings:
    eps: float                      # neighbourhood radius
    min_samples: int                # rows within eps (the row itself included) for a core row
    metric: Literal["euclidean", "manhattan", "cosine", "chebyshev", "precomputed"]   # precomputed: the data are a
                                                                                       # distance matrix
    extra: dict[str, Scalar]


@register_model("dbscan", settings=DBSCANSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="cuml",
                profile="core", packages=("sklearn",), description="DBSCAN (scikit-learn; cuML on the GPU)",
                example="cluster_density")
class DBSCAN(LibraryClusterer):
    @staticmethod
    def gpu_unsupported(s):
        """Why cuML's DBSCAN cannot take these settings, or None (fit.device auto then runs them on the CPU)."""
        if s.metric not in ("euclidean", "cosine", "precomputed"):
            return f"cuML's DBSCAN takes euclidean, cosine or precomputed distances, not {s.metric}"
        return None

    @staticmethod
    def check_job(job):
        reason = DBSCAN.gpu_unsupported(job.model)
        if job.fit.device == "cuda" and reason:
            raise ValueError(f"dbscan: {reason}; use one of them, or fit.device cpu or auto")

    def make(self, s, seed, device):
        if device == "cuda":
            from cuml.cluster import DBSCAN as D
        else:
            from sklearn.cluster import DBSCAN as D
        return D(**merged_kwargs({"eps": s.eps, "min_samples": s.min_samples, "metric": s.metric}, s.extra,
                                 "dbscan"))

    def can_assign(self):
        return False


@dataclasses.dataclass(frozen=True)
class HDBSCANSettings:
    min_cluster_size: int
    min_samples: int | None         # None: min_cluster_size
    cluster_selection_epsilon: float
    cluster_selection_method: Literal["eom", "leaf"]
    allow_single_cluster: bool
    metric: Literal["euclidean", "manhattan", "cosine", "chebyshev", "precomputed"]   # precomputed: the data are a
                                                                                       # distance matrix
    extra: dict[str, Scalar]


def _hdbscan_kwargs(s: HDBSCANSettings) -> dict:
    return {"min_cluster_size": s.min_cluster_size, "min_samples": s.min_samples,
            "cluster_selection_epsilon": s.cluster_selection_epsilon,
            "cluster_selection_method": s.cluster_selection_method, "allow_single_cluster": s.allow_single_cluster,
            "metric": s.metric}


def _hdbscan_gpu_unsupported(s) -> str | None:
    """Why cuML's HDBSCAN cannot take these settings, or None (fit.device auto then runs them on the CPU)."""
    if s.metric != "euclidean":
        return f"cuML's HDBSCAN takes Euclidean distances only, not {s.metric}"
    return None


def _check_cuda_euclidean(job) -> None:
    reason = _hdbscan_gpu_unsupported(job.model)
    if job.fit.device == "cuda" and reason:
        raise ValueError(f"{job.model_name}: {reason}; use metric euclidean, or fit.device cpu or auto")


@register_model("hdbscan_sklearn", settings=HDBSCANSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="cuml",
                profile="core", packages=("sklearn",),
                description="HDBSCAN, scikit-learn's implementation (cuML's on the GPU)", example="cluster_hdbscan")
class HDBSCANSklearn(LibraryClusterer):
    check_job = staticmethod(_check_cuda_euclidean)
    gpu_unsupported = staticmethod(_hdbscan_gpu_unsupported)

    def make(self, s, seed, device):
        if device == "cuda":
            from cuml.cluster.hdbscan import HDBSCAN as H

            return H(**merged_kwargs({**_hdbscan_kwargs(s), "prediction_data": True}, s.extra, "hdbscan_sklearn"))
        from sklearn.cluster import HDBSCAN as H

        return H(**merged_kwargs({**_hdbscan_kwargs(s), "copy": True}, s.extra, "hdbscan_sklearn"))

    def soft(self):
        return {"probability": _host(self.model.probabilities_).astype(np.float64)}

    def can_assign(self):
        return self.device == "cuda"

    def assign(self, x_new):
        from cuml.cluster.hdbscan import approximate_predict

        labels, _ = approximate_predict(self.model, x_new)
        return _host(labels).astype(np.int64)


@register_model("hdbscan", settings=HDBSCANSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="cuml",
                profile="unsupervised", packages=("hdbscan",),
                description="HDBSCAN, the hdbscan package (outlier scores, approximate_predict; cuML on the GPU)",
                example="cluster_hdbscan")
class HDBSCANPackage(LibraryClusterer):
    check_job = staticmethod(_check_cuda_euclidean)
    gpu_unsupported = staticmethod(_hdbscan_gpu_unsupported)

    def make(self, s, seed, device):
        if device == "cuda":
            from cuml.cluster.hdbscan import HDBSCAN as H
        else:
            from hdbscan import HDBSCAN as H
        # prediction data (for approximate_predict) cannot be made from precomputed distances
        return H(**merged_kwargs({**_hdbscan_kwargs(s), "prediction_data": s.metric != "precomputed"}, s.extra,
                                 "hdbscan"))

    def soft(self):
        out = {"probability": _host(self.model.probabilities_).astype(np.float64)}
        if self.device != "cuda":
            out["outlier_score"] = np.nan_to_num(np.asarray(self.model.outlier_scores_, dtype=np.float64))
        return out

    def can_assign(self):
        return self.s.metric != "precomputed"

    def assign(self, x_new):
        if self.device == "cuda":
            from cuml.cluster.hdbscan import approximate_predict
        else:
            from hdbscan import approximate_predict
        labels, _ = approximate_predict(self.model, x_new)
        return _host(labels).astype(np.int64)


@dataclasses.dataclass(frozen=True)
class FastHDBSCANSettings:
    min_cluster_size: int
    min_samples: int | None
    cluster_selection_method: Literal["eom", "leaf"]
    cluster_selection_epsilon: float
    allow_single_cluster: bool      # fast_hdbscan (0.3.2) finds no cluster, with or without this, when the rows never
                                    # split into two parts of min_cluster_size rows each
    extra: dict[str, Scalar]


@register_model("fast_hdbscan",settings=FastHDBSCANSettings, family="clusterer", tasks=CLUSTER, kinds=TAB,
                gpu="none", profile="unsupervised", packages=("fast_hdbscan",),
                description="fast_hdbscan: multicore HDBSCAN for low-dimensional Euclidean data",
                example="cluster_hdbscan")
class FastHDBSCAN(LibraryClusterer):
    def make(self, s, seed, device):
        import fast_hdbscan

        return fast_hdbscan.HDBSCAN(**merged_kwargs({
            "min_cluster_size": s.min_cluster_size, "min_samples": s.min_samples,
            "cluster_selection_method": s.cluster_selection_method,
            "cluster_selection_epsilon": s.cluster_selection_epsilon,
            "allow_single_cluster": s.allow_single_cluster}, s.extra, "fast_hdbscan"))

    def soft(self):
        return {"probability": np.asarray(self.model.probabilities_, dtype=np.float64)}

    def can_assign(self):
        return False


@dataclasses.dataclass(frozen=True)
class PLSCANSettings:
    base_min_cluster_size: int      # the smallest cluster size considered (fast_hdbscan's documented 10 works widely)
    min_samples: int
    max_layers: int                 # resolutions kept
    layer_similarity_threshold: float   # how alike two kept resolutions may be
    extra: dict[str, Scalar]


@register_model("plscan", settings=PLSCANSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="unsupervised", packages=("fast_hdbscan",),
                description="PLSCAN (fast_hdbscan): HDBSCAN that picks the cluster resolution itself",
                example="cluster_hdbscan")
class PLSCAN(Clusterer):
    def fit(self, x):
        import fast_hdbscan

        self.model = fast_hdbscan.PLSCAN(**merged_kwargs({
            "base_min_cluster_size": self.s.base_min_cluster_size, "min_samples": self.s.min_samples,
            "max_layers": self.s.max_layers, "layer_similarity_threshold": self.s.layer_similarity_threshold},
            self.s.extra, "plscan"))
        self.model.fit(x)

    def labels(self):
        return np.asarray(self.model.labels_, dtype=np.int64)

    def info(self):
        scores = getattr(self.model, "layer_persistence_scores_", None)
        return {"layers": int(len(scores)) if scores is not None else None}

    def params(self):
        return library_params(self.model)

    def extra_files(self):
        layers = getattr(self.model, "cluster_layers_", None)
        if layers is None:
            return {}
        return {"plscan_layers.npy": np.stack([np.asarray(layer, dtype=np.int64) for layer in layers])}


@dataclasses.dataclass(frozen=True)
class FLASCSettings:
    min_cluster_size: int           # of the HDBSCAN clusters searched for branches
    min_samples: int | None
    allow_single_cluster: bool      # HDBSCAN may keep all the rows as one cluster. fast_hdbscan (0.3.2) does so only
                                    # when they split into two parts of min_cluster_size rows each, and keeps the rows
                                    # of those parts; without such a split it finds no cluster, with or without this
    min_branch_size: int            # smallest branch kept (the BranchDetector's min_cluster_size)
    label_sides_as_branches: bool
    extra: dict[str, Scalar]        # passed to the BranchDetector


@register_model("flasc_branches", settings=FLASCSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="unsupervised", packages=("fast_hdbscan",),
                description="HDBSCAN clusters split into their branches (FLASC, fast_hdbscan's BranchDetector)",
                example="cluster_hdbscan")
class FLASCBranches(Clusterer):
    def fit(self, x):
        import fast_hdbscan

        # fast_hdbscan 0.3.2: HDBSCAN(*, min_cluster_size, min_samples, cluster_selection_method,
        # allow_single_cluster, ...)
        self.clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=self.s.min_cluster_size,
                                              min_samples=self.s.min_samples,
                                              allow_single_cluster=self.s.allow_single_cluster).fit(x)
        if not np.any(np.asarray(self.clusterer.labels_) >= 0):
            # the BranchDetector would raise "Input contains only noise points"
            raise ValueError("flasc_branches: HDBSCAN left every row as noise, so there is no cluster to split into "
                             "branches. fast_hdbscan finds a cluster only when the rows split into two parts of "
                             "min_cluster_size rows each, with or without allow_single_cluster; lower "
                             "min_cluster_size")
        self.detector = fast_hdbscan.BranchDetector(**merged_kwargs({
            "min_cluster_size": self.s.min_branch_size, "label_sides_as_branches": self.s.label_sides_as_branches},
            self.s.extra, "flasc_branches")).fit(self.clusterer)

    def labels(self):
        return np.asarray(self.detector.labels_, dtype=np.int64)

    def info(self):
        return {"hdbscan_clusters": int(np.unique(self.clusterer.labels_[self.clusterer.labels_ >= 0]).size)}

    def params(self):
        return {"hdbscan": library_params(self.clusterer), "branches": library_params(self.detector)}

    def extra_files(self):
        return {"hdbscan_labels.npy": np.asarray(self.clusterer.labels_, dtype=np.int64)}


@dataclasses.dataclass(frozen=True)
class OPTICSSettings:
    min_samples: int
    max_eps: float | None           # None: no limit
    cluster_method: Literal["xi", "dbscan"]
    xi: float | None                # xi method only
    eps: float | None               # dbscan method only
    metric: Literal["euclidean", "manhattan", "cosine", "chebyshev", "precomputed"]
    extra: dict[str, Scalar]


@register_model("optics", settings=OPTICSSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="OPTICS (scikit-learn)", example="cluster_density")
class OPTICS(LibraryClusterer):
    @staticmethod
    def check_job(job):
        s = job.model
        if (s.cluster_method == "xi") != (s.xi is not None) or (s.cluster_method == "dbscan") != (s.eps is not None):
            raise ValueError("optics: xi with cluster_method xi, eps with cluster_method dbscan, the other None")

    def make(self, s, seed, device):
        import math

        from sklearn.cluster import OPTICS as O

        kwargs = {"min_samples": s.min_samples, "max_eps": math.inf if s.max_eps is None else s.max_eps,
                  "cluster_method": s.cluster_method, "metric": s.metric}
        if s.xi is not None:
            kwargs["xi"] = s.xi
        if s.eps is not None:
            kwargs["eps"] = s.eps
        return O(**merged_kwargs(kwargs, s.extra, "optics"))

    def can_assign(self):
        return False


@dataclasses.dataclass(frozen=True)
class MeanShiftSettings:
    bandwidth: float | None         # None: estimated from the data (scikit-learn's estimate_bandwidth)
    bin_seeding: bool
    cluster_all: bool               # False: rows outside every kernel are noise (-1)
    max_iter: int
    extra: dict[str, Scalar]


@register_model("mean_shift", settings=MeanShiftSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="mean shift (scikit-learn)",
                example="cluster_density")
class MeanShift(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.cluster import MeanShift as M

        return M(**merged_kwargs({"bandwidth": s.bandwidth, "bin_seeding": s.bin_seeding,
                                  "cluster_all": s.cluster_all, "max_iter": s.max_iter}, s.extra, "mean_shift"))

    def can_assign(self):
        return self.s.cluster_all
