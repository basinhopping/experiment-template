"""reduce: run a registered embedder (PCA, UMAP, ...) as a preparation step, fitted on the fit rows, and replace the
columns with its components. Rows outside the fit set go through the embedder's transform, so methods without one
(t-SNE, MDS, spectral embedding) work only when the evaluated rows are the fit rows.

For HDBSCAN on many columns, UMAP's documentation recommends reducing to about 10 dimensions with more neighbours than
for pictures and min_dist 0 first, and warns that UMAP can tear clusters, so clusters found that way need checking
(docs/UNSUPERVISED.md).

subsample: keep a fixed share (or number) of the fit rows, at random or within strata of a column, by SHA-256 of the
row IDs: choose settings on a subsample, then run once on everything.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.prep.base import PrepStep
from sweepkit.registry import register_prep
from sweepkit.settings import section_type
from sweepkit.splits.base import stable_keys


@dataclasses.dataclass(frozen=True)
class ReduceSettings:
    method: section_type("model", ("embedder",))   # a registered embedder with its settings


@register_prep("reduce", settings=ReduceSettings, kinds=("tabular",), profile="core", packages=(),
               description="replace the columns with an embedder's components (PCA, UMAP, ...)",
               example="cluster_umap_hdbscan")
class Reduce(PrepStep):
    def fit(self, x, positions):
        from sweepkit.models.base import build_embedder

        self.embedder = build_embedder(self.s.method, self.ctx.seed, self.ctx.device)
        self.fit_positions = np.asarray(positions)
        self.fit_out = np.asarray(self.embedder.fit_transform(np.asarray(x, dtype=np.float32)), dtype=np.float64)
        self.width = self.fit_out.shape[1]

    def transform(self, x, positions):
        positions = np.asarray(positions)
        if positions.shape == self.fit_positions.shape and np.array_equal(positions, self.fit_positions):
            return self.fit_out
        if not self.embedder.can_transform():
            raise ValueError(f"prep reduce: {self.s.method.name} cannot embed rows it was not fitted on; evaluate the "
                             "fit rows only (split all or subsamples)")
        return np.asarray(self.embedder.transform(np.asarray(x, dtype=np.float32)), dtype=np.float64)

    def names(self, names):
        return tuple(f"{self.s.method.name}_{k + 1}" for k in range(self.width))

    def categorical_after(self, categorical):
        return {}

    def record(self):
        return {"method": self.s.method.name, "components": self.width, **self.embedder.info()}


@dataclasses.dataclass(frozen=True)
class SubsampleSettings:
    method: Literal["random", "stratified"]
    fraction: float | None          # share of the fit rows kept (or n_rows)
    n_rows: int | None              # number of fit rows kept (or fraction)
    stratify: str | None            # stratified: the column whose values keep their shares
    salt: str                       # fixes the subsample


@register_prep("subsample", settings=SubsampleSettings, kinds=("tabular",), profile="core", packages=(),
               removes_rows=True, description="fit on a fixed subsample of the fit rows", example="prep_clean")
class Subsample(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name != "subsample":
                continue
            if (s.fraction is None) == (s.n_rows is None):
                raise ValueError("prep subsample: set fraction or n_rows, not both")
            if (s.method == "stratified") != (s.stratify is not None):
                raise ValueError("prep subsample: stratified needs a stratify column; random takes None")

    def fit(self, x, positions):
        positions = np.asarray(positions)
        n = positions.size
        ids = self.ctx.table.ids[positions]
        keys = stable_keys(ids, self.s.salt)
        keep = np.zeros(n, dtype=bool)
        if self.s.method == "random":
            size = self.s.n_rows if self.s.n_rows is not None else int(round(self.s.fraction * n))
            keep[np.lexsort((ids, keys))[:min(size, n)]] = True
        else:
            strata = self.ctx.table.column(self.s.stratify).astype(str)[positions]
            share = self.s.fraction if self.s.fraction is not None else self.s.n_rows / n
            for stratum in np.unique(strata):
                members = np.flatnonzero(strata == stratum)
                members = members[np.lexsort((ids[members], keys[members]))]
                keep[members[:max(1, int(round(share * members.size)))]] = True
        self.drop = ~keep

    def removed(self):
        return self.drop

    def transform(self, x, positions):
        return np.asarray(x)

    def record(self):
        return {"method": self.s.method, "kept": int((~self.drop).sum())}
