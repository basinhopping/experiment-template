"""Choosing columns: by name, by a statistic of the fit rows, and one-hot encoding of categorical code columns.

  select_columns    keep the columns named (names, "re:<regex>", "*")
  select_features   variance_top: the n columns with the largest fit-row variance; variance_threshold: columns whose
                    variance is above threshold; highly_variable: the n most variable features of count data after
                    log1p, by normalized dispersion within bins of mean expression (the single-cell "seurat" method);
                    correlation: drop a column whose absolute correlation with an earlier kept column is above
                    threshold
  onehot            categorical code columns become one 0/1 column per category (the dataset's category list)
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.prep.base import PrepStep, columns_matching
from sweepkit.registry import register_prep


@dataclasses.dataclass(frozen=True)
class SelectColumnsSettings:
    keep: tuple[str, ...]           # names, "re:<regex>", or "*"


@register_prep("select_columns", settings=SelectColumnsSettings, kinds=("tabular",), profile="core", packages=(),
               description="keep the columns named", example="prep_transform")
class SelectColumns(PrepStep):
    def fit(self, x, positions):
        self.keep_idx = columns_matching(self.ctx.feature_names, self.s.keep, "prep select_columns keep")

    def transform(self, x, positions):
        return np.asarray(x)[:, self.keep_idx]

    def names(self, names):
        return tuple(names[j] for j in self.keep_idx)

    def categorical_after(self, categorical):
        new_index = {int(old): i for i, old in enumerate(self.keep_idx)}
        return {new_index[j]: v for j, v in categorical.items() if j in new_index}

    def record(self):
        return {"kept": [self.ctx.feature_names[j] for j in self.keep_idx]}


@dataclasses.dataclass(frozen=True)
class SelectFeaturesSettings:
    method: Literal["variance_top", "variance_threshold", "highly_variable", "correlation"]
    n: int | None                   # variance_top, highly_variable: how many columns to keep
    threshold: float | None         # variance_threshold: minimum variance; correlation: maximum absolute correlation
    n_bins: int | None              # highly_variable: bins of mean expression


@register_prep("select_features", settings=SelectFeaturesSettings, kinds=("tabular",), profile="core", packages=(),
               description="keep columns by variance, dispersion (highly variable genes) or low correlation",
               example="prep_transform")
class SelectFeatures(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        need = {"variance_top": ("n",), "variance_threshold": ("threshold",), "highly_variable": ("n", "n_bins"),
                "correlation": ("threshold",)}
        for name, s in job.prep:
            if name != "select_features":
                continue
            for field in ("n", "threshold", "n_bins"):
                if (getattr(s, field) is not None) != (field in need[s.method]):
                    raise ValueError(f"prep select_features {s.method}: set {list(need[s.method])} and leave the "
                                     "others None")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        num = self.numeric_columns(x.shape[1])
        m = self.s.method
        if m == "variance_top":
            variance = np.nanvar(x[:, num], axis=0)
            order = np.argsort(-variance, kind="stable")[:self.s.n]
            chosen = num[np.sort(order)]
        elif m == "variance_threshold":
            chosen = num[np.nanvar(x[:, num], axis=0) > self.s.threshold]
        elif m == "highly_variable":
            counts = np.expm1(x[:, num])
            mean = counts.mean(axis=0)
            var = counts.var(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dispersion = np.where(mean > 0, var / mean, 0.0)
                log_dispersion = np.log(np.where(dispersion > 0, dispersion, np.nan))
                log_mean = np.log1p(mean)
            edges = np.linspace(log_mean.min(), log_mean.max() + 1e-12, self.s.n_bins + 1)
            bins = np.clip(np.searchsorted(edges, log_mean, side="right") - 1, 0, self.s.n_bins - 1)
            normalized = np.full(num.size, -np.inf)
            for b in np.unique(bins):
                members = (bins == b) & np.isfinite(log_dispersion)
                if members.sum() < 2:
                    normalized[members] = 0.0
                    continue
                values = log_dispersion[members]
                sd = values.std(ddof=1)
                normalized[members] = (values - values.mean()) / (sd if sd > 0 else 1.0)
            order = np.argsort(-normalized, kind="stable")[:self.s.n]
            chosen = num[np.sort(order)]
            self.dispersion = dict(zip([self.ctx.feature_names[j] for j in num], normalized.tolist()))
        else:
            corr = np.corrcoef(x[:, num], rowvar=False)
            keep = []
            for k in range(num.size):
                if all(abs(corr[k, kk]) <= self.s.threshold for kk in keep):
                    keep.append(k)
            chosen = num[keep]
        cat = [j for j in range(x.shape[1]) if j in self.ctx.categorical]
        self.keep_idx = np.array(sorted(set(chosen.tolist()) | set(cat)), dtype=np.int64)
        if self.keep_idx.size == 0:
            raise ValueError("prep select_features: no column kept")

    def transform(self, x, positions):
        return np.asarray(x)[:, self.keep_idx]

    def names(self, names):
        return tuple(names[j] for j in self.keep_idx)

    def categorical_after(self, categorical):
        new_index = {int(old): i for i, old in enumerate(self.keep_idx)}
        return {new_index[j]: v for j, v in categorical.items() if j in new_index}

    def record(self):
        return {"method": self.s.method, "kept": [self.ctx.feature_names[j] for j in self.keep_idx]}


@dataclasses.dataclass(frozen=True)
class OneHotSettings:
    drop_first: bool                # leave out each column's first category (a linear model's reference level)


@register_prep("onehot", settings=OneHotSettings, kinds=("tabular",), profile="core", packages=(),
               description="categorical code columns to one 0/1 column per category", example="prep_mixed")
class OneHot(PrepStep):
    def fit(self, x, positions):
        self.cat = sorted(self.ctx.categorical)
        if not self.cat:
            raise ValueError("prep onehot: the data have no categorical column (see the dataset's categorical "
                             "setting)")

    def transform(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        blocks = []
        for j in range(x.shape[1]):
            if j not in self.ctx.categorical:
                blocks.append(x[:, [j]])
                continue
            categories = self.ctx.categorical[j]
            codes = x[:, j]
            known = np.isin(codes, np.arange(len(categories)))
            if not known.all():
                raise ValueError(f"prep onehot: column {self.ctx.feature_names[j]} holds a code outside its "
                                 "categories")
            start = 1 if self.s.drop_first else 0
            blocks.append(np.stack([(codes == c).astype(np.float64) for c in range(start, len(categories))], axis=1))
        return np.concatenate(blocks, axis=1)

    def names(self, names):
        out = []
        start = 1 if self.s.drop_first else 0
        for j, n in enumerate(names):
            if j in self.ctx.categorical:
                out.extend(f"{n}={c}" for c in self.ctx.categorical[j][start:])
            else:
                out.append(n)
        return tuple(out)

    def categorical_after(self, categorical):
        return {}

    def record(self):
        return {"expanded": [self.ctx.feature_names[j] for j in self.cat], "drop_first": self.s.drop_first}
