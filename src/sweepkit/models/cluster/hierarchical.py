"""Hierarchical clustering: agglomerative (Ward, complete, average or single linkage) and Genie.

agglomerative builds the whole merge tree with SciPy's linkage on the fit rows, writes it as linkage.npy (SciPy's
linkage matrix: the two clusters merged, their distance, the new cluster's size), and cuts it into a given number
of clusters or at a given distance; the cut and the merge heights on either side of it are recorded in info. SciPy
holds every pairwise distance, so memory grows with the square of the fit rows (20,000 rows: 1.6 GB); screen on
subsamples. Ward linkage takes Euclidean distances only.

On the GPU (fit.device cuda), cuML's AgglomerativeClustering does single linkage only, cut into a number of clusters;
it builds the tree on a nearest-neighbour graph for Euclidean distances and on every pair for the other metrics, and
writes its merge pairs as children.npy, since cuML gives no merge distances by default.

Rows the tree was not built on are labelled with the label of their nearest fit row (the same metric): a choice the
template makes, since a hierarchy has no rule of its own for new rows.

genie is Gagolewski's Genie (the genieclust package, profile extras, AGPL): single linkage on a minimum spanning tree
with a correction that keeps the cluster sizes from becoming too unequal (gini_threshold), robust to outliers and
fast on many rows.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, LibraryClusterer, _host, library_params, merged_kwargs, plain_params
from sweepkit.models.distances import DISTANCE_OR_PRECOMPUTED, matrix
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

CLUSTER = ("clustering",)
TAB = ("tabular",)
# metrics SciPy's linkage computes from the rows themselves (the rest go through sweepkit.models.distances)
SCIPY_DIRECT = ("euclidean", "manhattan", "cosine", "chebyshev", "correlation")


def nearest_fit_labels(x_fit: np.ndarray, labels: np.ndarray, x_new: np.ndarray, metric: str) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=1, metric=metric).fit(x_fit).kneighbors(np.asarray(x_new, dtype=np.float64),
                                                                                return_distance=False)
    return np.asarray(labels)[idx[:, 0]].astype(np.int64)


@dataclasses.dataclass(frozen=True)
class AgglomerativeSettings:
    linkage: Literal["ward", "complete", "average", "single"]
    metric: DISTANCE_OR_PRECOMPUTED  # any distance of sweepkit.models.distances, or precomputed (the data are a
                                     # distance matrix); ward: euclidean only
    cut: Literal["n_clusters", "distance"]
    n_clusters: int | None          # cut n_clusters: how many clusters; None otherwise
    distance: float | None          # cut distance: the merge height the tree is cut at; None otherwise
    extra: dict[str, Scalar]        # passed to SciPy's linkage (optimal_ordering) or, on the GPU, to cuML (c,
                                    # connectivity)


@register_model("agglomerative", settings=AgglomerativeSettings, family="clusterer", tasks=CLUSTER, kinds=TAB,
                gpu="cuml", profile="core", packages=("scipy",),
                description="agglomerative clustering with its merge tree saved (SciPy; cuML single linkage on the "
                            "GPU)", example="cluster_hierarchical")
class Agglomerative(Clusterer):
    @staticmethod
    def gpu_unsupported(s):
        """Why cuML cannot take these settings, or None (fit.device auto then runs them on the CPU)."""
        if s.metric == "precomputed":
            return "precomputed distances run on the CPU"
        if s.linkage != "single" or s.cut != "n_clusters" or s.metric not in ("euclidean", "manhattan", "cosine"):
            return ("cuML does single linkage only, cut n_clusters, metric euclidean, manhattan or cosine (here "
                    f"{s.linkage} linkage, cut {s.cut}, metric {s.metric})")
        return None

    @staticmethod
    def check_job(job):
        s = job.model
        if (s.cut == "n_clusters") != (s.n_clusters is not None) or (s.cut == "distance") != (s.distance is not None):
            raise ValueError("agglomerative: n_clusters with cut n_clusters, distance with cut distance, the other "
                             "None")
        if s.n_clusters is not None and s.n_clusters < 1:
            raise ValueError("agglomerative: n_clusters at least 1")
        if s.distance is not None and not s.distance > 0:
            raise ValueError("agglomerative: distance above 0")
        if s.linkage == "ward" and s.metric != "euclidean":
            raise ValueError("agglomerative: ward linkage takes metric euclidean")
        reason = Agglomerative.gpu_unsupported(s)
        if job.fit.device == "cuda" and reason:
            raise ValueError(f"agglomerative: {reason}; use fit.device cpu or auto")

    def fit(self, x):
        self.x_fit = np.asarray(x, dtype=np.float64)
        n = self.x_fit.shape[0]
        self.files = {}
        if self.device == "cuda":
            # cuML 26.x: AgglomerativeClustering(n_clusters, *, metric, connectivity, linkage="single", c, ...);
            # labels_ and children_ after fit
            from cuml.cluster import AgglomerativeClustering

            # cuML's knn connectivity takes Euclidean distances only, so other metrics use pairwise connectivity
            # (every pair on the GPU) unless extra names the connectivity
            fixed = {"n_clusters": self.s.n_clusters, "metric": self.s.metric, "linkage": "single"}
            if "connectivity" not in self.s.extra:
                fixed["connectivity"] = "knn" if self.s.metric == "euclidean" else "pairwise"
            self.model = AgglomerativeClustering(**merged_kwargs(fixed, self.s.extra, "agglomerative"))
            self.model.fit(self.x_fit.astype(np.float32))
            self._labels = _host(self.model.labels_).astype(np.int64)
            self.files["children.npy"] = _host(self.model.children_).astype(np.int64)
            self.cut_info = {"cut": "n_clusters", "n_clusters": self.s.n_clusters, "library": "cuml"}
            return
        from scipy.cluster.hierarchy import fcluster, linkage

        if self.s.cut == "n_clusters" and self.s.n_clusters > n:
            raise ValueError(f"agglomerative: n_clusters {self.s.n_clusters} is above the fit rows ({n})")
        if self.s.metric in SCIPY_DIRECT:
            metric = "cityblock" if self.s.metric == "manhattan" else self.s.metric
            self.linkage_args = merged_kwargs({"method": self.s.linkage, "metric": metric}, self.s.extra,
                                              "agglomerative")
            z = linkage(self.x_fit, **self.linkage_args)
        else:
            # every other distance (and precomputed ones) as a condensed matrix (sweepkit.models.distances)
            from scipy.spatial.distance import squareform

            d, self.distances = matrix(self.x_fit, self.s.metric, self.categorical)
            self.linkage_args = merged_kwargs({"method": self.s.linkage}, self.s.extra, "agglomerative")
            z = linkage(squareform(d, checks=False), **self.linkage_args)
        self.files["linkage.npy"] = z.astype(np.float64)
        if self.s.cut == "n_clusters":
            raw = fcluster(z, t=self.s.n_clusters, criterion="maxclust")
            k = self.s.n_clusters
            heights = z[:, 2]
            below = float(heights[n - k - 1]) if n - k - 1 >= 0 else None
            above = float(heights[n - k]) if 0 <= n - k < heights.size else None
            self.cut_info = {"cut": "n_clusters", "n_clusters": k, "last_merge_kept": below,
                             "first_merge_undone": above, "library": "scipy"}
        else:
            raw = fcluster(z, t=self.s.distance, criterion="distance")
            self.cut_info = {"cut": "distance", "distance": self.s.distance, "library": "scipy"}
        _, labels = np.unique(raw, return_inverse=True)
        self._labels = labels.astype(np.int64)

    def labels(self):
        return self._labels

    def can_assign(self):
        return True

    def assign(self, x_new):
        """Each new row takes the cluster of its nearest fit row (with precomputed distances, x_new holds the new
        rows' distances to the fit rows)."""
        if self.s.metric == "precomputed":
            return self._labels[np.argmin(np.asarray(x_new, dtype=np.float64), axis=1)].astype(np.int64)
        if self.s.metric in SCIPY_DIRECT:
            metric = "cityblock" if self.s.metric == "manhattan" else self.s.metric
            return nearest_fit_labels(self.x_fit, self._labels, x_new, metric)
        if not hasattr(self, "distances"):
            from sweepkit.models.distances import Distances

            self.distances = Distances(self.s.metric, self.categorical).fit(self.x_fit)
        return self._labels[np.argmin(self.distances.between(x_new, self.x_fit), axis=1)].astype(np.int64)

    def info(self):
        found = int(np.unique(self._labels).size)
        return {**{k: v for k, v in self.cut_info.items() if k != "library"}, "clusters_found": found}

    def params(self):
        base = {k: v for k, v in dataclasses.asdict(self.s).items() if k != "extra"}
        if self.device == "cuda":
            return {**plain_params(base), "cuml": library_params(self.model)}
        return plain_params({**base, **{f"linkage_{k}": v for k, v in self.linkage_args.items()}})

    def extra_files(self):
        return dict(self.files)


@dataclasses.dataclass(frozen=True)
class GenieSettings:
    n_clusters: int
    gini_threshold: float           # in [0, 1]; 1 turns the correction off (single linkage); 0.1 to 0.7 are usual
    M: int                          # smoothing factor of the mutual reachability distance; 0 or 1: plain distances
    metric: Literal["euclidean", "manhattan", "cosine"]
    extra: dict[str, Scalar]        # other Genie parameters (coarser, quitefastmst_params)


@register_model("genie", settings=GenieSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="extras", packages=("genieclust",),
                description="Genie: robust hierarchical clustering on a minimum spanning tree (genieclust, AGPL)",
                example="cluster_hierarchical")
class Genie(LibraryClusterer):
    def make(self, s, seed, device):
        # genieclust 1.3.0: Genie(n_clusters, *, gini_threshold, M, metric, coarser, quitefastmst_params, verbose);
        # the metric names "euclidean", "manhattan" and "cosine" are synonyms deadwood accepts
        import genieclust

        return genieclust.Genie(**merged_kwargs({"n_clusters": s.n_clusters, "gini_threshold": s.gini_threshold,
                                                 "M": s.M, "metric": s.metric}, s.extra, "genie"))

    def fit(self, x):
        self.x_fit = np.asarray(x, dtype=np.float64)
        super().fit(self.x_fit)

    def can_assign(self):
        return True

    def assign(self, x_new):
        metric = "cityblock" if self.s.metric == "manhattan" else self.s.metric
        return nearest_fit_labels(self.x_fit, self.labels(), x_new, metric)

    def info(self):
        return {"clusters_found": int(self.model.n_clusters_)}

    def extra_files(self):
        children = getattr(self.model, "children_", None)
        distances = getattr(self.model, "distances_", None)
        counts = getattr(self.model, "counts_", None)
        if children is None or distances is None or counts is None:
            return {}
        z = np.column_stack([np.asarray(children, dtype=np.float64), np.asarray(distances, dtype=np.float64),
                             np.asarray(counts, dtype=np.float64)])
        return {"linkage.npy": z}
