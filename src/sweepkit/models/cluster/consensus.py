"""Consensus clustering (Monti et al. 2003), the template's own: how often two rows land in the same cluster when a
base clusterer is run again on many subsamples.

The base clusterer is any registered clusterer, given as a nested section with its own settings, for example
{"name": "kmeans", "n_clusters": 4, ...}. It is fitted n_resamples times, each time on a seeded subsample of a
fraction of the fit rows (without replacement). For every pair of rows, the co-association is the number of
subsamples that put both in one cluster divided by the number that held both (noise, label -1, joins no cluster);
a pair never sampled together counts 0, and every row's co-association with itself is 1. The final clusters are an
average-linkage (or complete or single) clustering of 1 minus the co-association, cut into n_clusters.

The co-association matrix is written as co_association.npy (float32). It holds every pair of fit rows, so memory
grows with the square of the rows: 10,000 rows need 400 MB for the matrix and as much again while counting; run it on
subsamples of large data. Rows outside the fit set cannot be labelled (use fit.assign fit_rows).
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, build_clusterer, plain_params
from sweepkit.registry import register_model
from sweepkit.settings import section_type, to_plain
from sweepkit.streams import generator, int_seed


@dataclasses.dataclass(frozen=True)
class ConsensusSettings:
    base: section_type("model", ("clusterer",))   # the clusterer run on every subsample, with its settings
    n_resamples: int
    fraction: float                 # share of the fit rows in each subsample
    n_clusters: int                 # clusters of the final cut
    linkage: Literal["average", "complete", "single"]


@register_model("consensus", settings=ConsensusSettings, family="clusterer", tasks=("clustering",),
                kinds=("tabular",), gpu="none", profile="core", packages=("scipy",),
                description="consensus clustering: co-association over subsamples of a base clusterer (own)",
                example="cluster_consensus")
class Consensus(Clusterer):
    @staticmethod
    def check_job(job):
        s = job.model
        if s.base.name == "consensus":
            raise ValueError("consensus: the base clusterer cannot be consensus itself")
        if s.n_resamples < 2 or not 0 < s.fraction <= 1 or s.n_clusters < 1:
            raise ValueError("consensus: n_resamples at least 2, fraction in (0, 1], n_clusters at least 1")

    def fit(self, x):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        x = np.asarray(x, dtype=np.float32)
        n = x.shape[0]
        size = int(round(self.s.fraction * n))
        if size < 2 or self.s.n_clusters > n:
            raise ValueError(f"consensus: {n} fit rows give subsamples of {size} rows; too few for the cut")
        rng = generator(self.seed, "consensus subsamples")
        together = np.zeros((n, n), dtype=np.float32)
        sampled = np.zeros((n, n), dtype=np.float32)
        found = []
        for r in range(self.s.n_resamples):
            rows = np.sort(rng.choice(n, size=size, replace=False))
            base = build_clusterer(self.s.base, int_seed(self.seed, f"consensus resample {r}"), self.device)
            base.set_columns(self.feature_names, self.categorical)
            base.fit(x[rows])
            labels = np.asarray(base.labels(), dtype=np.int64)
            sampled[np.ix_(rows, rows)] += 1
            groups = np.unique(labels[labels >= 0])
            for g in groups:
                members = rows[labels == g]
                together[np.ix_(members, members)] += 1
            found.append(int(groups.size))
        with np.errstate(divide="ignore", invalid="ignore"):
            co = np.where(sampled > 0, together / sampled, 0.0).astype(np.float32)
        np.fill_diagonal(co, 1.0)
        self.co_association = co
        distance = np.clip(1.0 - co.astype(np.float64), 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        z = linkage(squareform(distance, checks=False), method=self.s.linkage)
        raw = fcluster(z, t=self.s.n_clusters, criterion="maxclust")
        _, labels = np.unique(raw, return_inverse=True)
        self._labels = labels.astype(np.int64)
        same = self._labels[:, None] == self._labels[None, :]
        off = ~np.eye(n, dtype=bool)
        self._info = {"base_clusters_found": {"min": min(found), "max": max(found)},
                      "within_co_association": float(co[same & off].mean()) if (same & off).any() else None,
                      "between_co_association": float(co[~same].mean()) if (~same).any() else None,
                      "pairs_never_sampled_together": int(((sampled == 0) & off).sum() // 2)}

    def labels(self):
        return self._labels

    def info(self):
        return self._info

    def params(self):
        out = {k: v for k, v in dataclasses.asdict(self.s).items() if k != "base"}
        return {**plain_params(out), "base": {"name": self.s.base.name, **to_plain(self.s.base.settings)}}

    def extra_files(self):
        return {"co_association.npy": self.co_association}
