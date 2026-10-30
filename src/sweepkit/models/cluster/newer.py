"""Newer clustering methods that ship as packages.

lumbermark is Gagolewski's Lumbermark (2026, arXiv 2604.07143; the lumbermark package with its dependency deadwood,
both AGPL, profile extras): a divisive method that cuts a mutual reachability minimum spanning tree into a given
number of clusters, chopping off large limbs joined by thin segments. It is meant as a robust alternative to HDBSCAN
that is told how many clusters to find. Rows it was not fitted on are labelled with their nearest fit row's label.

Torque Clustering (Yang and Lin, 2025) has no official Python package; its official code is MATLAB under a
non-commercial license. docs/EXTENDING.md shows how to wrap an unofficial Python port with scaffold, as a recipe the
template does not ship.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import LibraryClusterer, merged_kwargs
from sweepkit.models.cluster.hierarchical import nearest_fit_labels
from sweepkit.registry import register_model
from sweepkit.settings import Scalar


@dataclasses.dataclass(frozen=True)
class LumbermarkSettings:
    n_clusters: int
    min_cluster_factor: float       # in [0, 1]: no cluster smaller than this times rows / n_clusters
    M: int                          # smoothing factor of the mutual reachability distance; 0 or 1: plain distances
    nested: bool                    # partitions for fewer clusters nest inside this one
    skip_leaves: bool | None        # leave the tree's leaves out of the size counts; None: when M > 0 (the
                                    # package's "auto")
    min_cluster_size: int           # a floor on cluster sizes for small data
    metric: Literal["euclidean", "manhattan", "cosine"]
    extra: dict[str, Scalar]        # other Lumbermark parameters (quitefastmst_params)


@register_model("lumbermark", settings=LumbermarkSettings, family="clusterer", tasks=("clustering",),
                kinds=("tabular",), gpu="none", profile="extras", packages=("lumbermark",),
                description="Lumbermark: divisive clustering of a mutual reachability spanning tree (AGPL)",
                example="cluster_lumbermark")
class Lumbermark(LibraryClusterer):
    def make(self, s, seed, device):
        # lumbermark 0.9.1: Lumbermark(n_clusters, *, min_cluster_factor, M, nested, skip_leaves, min_cluster_size,
        # metric, quitefastmst_params, verbose); labels_ and n_clusters_ after fit
        import lumbermark

        return lumbermark.Lumbermark(**merged_kwargs({
            "n_clusters": s.n_clusters, "min_cluster_factor": s.min_cluster_factor, "M": s.M, "nested": s.nested,
            "skip_leaves": "auto" if s.skip_leaves is None else s.skip_leaves,
            "min_cluster_size": s.min_cluster_size, "metric": s.metric}, s.extra, "lumbermark"))

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
