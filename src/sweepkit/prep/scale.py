"""Scaling, fitted on the fit rows (per group when a group column is named).

  zscore    (x - mean) / standard deviation (a column with SD 0 is centred only)
  minmax    to [0, 1] with the fit rows' minimum and maximum
  robust    (x - median) / interquartile range
  maxabs    x / largest absolute value
  l1_rows, l2_rows   each row divided by its L1 or L2 norm over the named columns (needs no fitting)
per_group: the statistics are computed within each value of a column (for example a batch, a subject or a plate),
and a row is scaled with its own group's statistics; a group with no fit rows is refused.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.prep.base import PrepStep, columns_matching
from sweepkit.registry import register_prep


@dataclasses.dataclass(frozen=True)
class ScaleSettings:
    method: Literal["zscore", "minmax", "robust", "maxabs", "l1_rows", "l2_rows"]
    columns: tuple[str, ...]        # names, "re:<regex>", or ["*"] for every numeric column
    per_group: str | None           # a column of the rows: statistics within each of its values; None: all fit rows


def _stats(part: np.ndarray, method: str) -> tuple:
    if method == "zscore":
        center, spread = np.nanmean(part, axis=0), np.nanstd(part, axis=0)
    elif method == "minmax":
        center, spread = np.nanmin(part, axis=0), np.nanmax(part, axis=0) - np.nanmin(part, axis=0)
    elif method == "robust":
        q1, median, q3 = np.nanquantile(part, [0.25, 0.5, 0.75], axis=0)
        center, spread = median, q3 - q1
    else:
        center, spread = np.zeros(part.shape[1]), np.nanmax(np.abs(part), axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    return center, spread


@register_prep("scale", settings=ScaleSettings, kinds=("tabular",), profile="core", packages=(),
               description="z-score, min-max, robust, max-abs, per-row L1 or L2, optionally within groups",
               example="prep_transform")
class Scale(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name == "scale" and s.method in ("l1_rows", "l2_rows") and s.per_group is not None:
                raise ValueError("prep scale: per-row scaling takes per_group None")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        chosen = columns_matching(self.ctx.feature_names, self.s.columns, "prep scale columns")
        self.cols = np.array([j for j in chosen if j not in self.ctx.categorical], dtype=np.int64)
        if self.cols.size == 0:
            raise ValueError("prep scale: no numeric column selected")
        if self.s.method in ("l1_rows", "l2_rows"):
            return
        part = x[:, self.cols]
        if self.s.per_group is None:
            self.stats = {None: _stats(part, self.s.method)}
        else:
            groups = self.ctx.table.column(self.s.per_group).astype(str)[positions]
            self.stats = {g: _stats(part[groups == g], self.s.method) for g in np.unique(groups)}

    def transform(self, x, positions):
        x = np.array(x, dtype=np.float64, copy=True)
        part = x[:, self.cols]
        if self.s.method in ("l1_rows", "l2_rows"):
            norm = np.abs(part).sum(axis=1) if self.s.method == "l1_rows" else np.sqrt((part ** 2).sum(axis=1))
            norm = np.where(norm > 0, norm, 1.0)
            x[:, self.cols] = part / norm[:, None]
            return x
        if self.s.per_group is None:
            center, spread = self.stats[None]
            x[:, self.cols] = (part - center) / spread
            return x
        groups = self.ctx.table.column(self.s.per_group).astype(str)[positions]
        unseen = sorted(set(groups.tolist()) - set(self.stats))
        if unseen:
            raise ValueError(f"prep scale per {self.s.per_group}: groups {unseen[:5]} have no fit rows")
        for g in np.unique(groups):
            mask = groups == g
            center, spread = self.stats[g]
            x[np.ix_(mask, self.cols)] = (part[mask] - center) / spread
        return x

    def record(self):
        return {"method": self.s.method, "per_group": self.s.per_group,
                "columns": [self.ctx.feature_names[j] for j in self.cols]}
