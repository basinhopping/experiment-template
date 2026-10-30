"""Cleaning steps: missing values, duplicate rows, constant columns, outliers."""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.prep.base import PrepStep
from sweepkit.registry import register_prep
from sweepkit.settings import section_type
from sweepkit.streams import int_seed

TABULAR = ("tabular",)


@dataclasses.dataclass(frozen=True)
class ImputeSettings:
    method: Literal["drop_rows", "mean", "median", "constant", "knn", "iterative"]
    # drop_rows: fit rows with a missing value leave the fit (rows outside it must have none); mean, median: each
    # column's statistic over the fit rows; constant: fill_value; knn: the mean of the n_neighbors nearest fit rows
    # (scikit-learn's KNNImputer); iterative: each column modelled from the others (IterativeImputer, max_iter rounds).
    # Categorical code columns always take their most frequent fit value, whatever the method.
    fill_value: float | None        # constant only
    n_neighbors: int | None         # knn only
    max_iter: int | None            # iterative only


@register_prep("impute", settings=ImputeSettings, kinds=TABULAR, profile="core", packages=("sklearn",),
               removes_rows=True, description="fill or drop missing values", example="prep_clean")
class Impute(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name != "impute":
                continue
            wanted = {"constant": "fill_value", "knn": "n_neighbors", "iterative": "max_iter"}.get(s.method)
            what = wanted if wanted else "none of fill_value, n_neighbors, max_iter"
            for field in ("fill_value", "n_neighbors", "max_iter"):
                if (getattr(s, field) is not None) != (field == wanted):
                    raise ValueError(f"prep impute {s.method}: set {what} and leave the others None")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        missing = np.isnan(x)
        self.missing_counts = missing.sum(axis=0)
        self.drop = None
        cat = [j for j in range(x.shape[1]) if j in self.ctx.categorical]
        num = self.numeric_columns(x.shape[1])
        self.fill = np.zeros(x.shape[1])
        for j in cat:
            values = x[~missing[:, j], j]
            if values.size == 0:
                raise ValueError(f"prep impute: column {self.ctx.feature_names[j]} has no value in the fit rows")
            codes, counts = np.unique(values, return_counts=True)
            self.fill[j] = codes[np.argmax(counts)]
        if self.s.method == "drop_rows":
            self.drop = missing.any(axis=1)
            return
        if self.s.method in ("mean", "median", "constant"):
            for j in num:
                column = x[~missing[:, j], j]
                if self.s.method == "constant":
                    self.fill[j] = self.s.fill_value
                elif column.size == 0:
                    raise ValueError(f"prep impute: column {self.ctx.feature_names[j]} has no value in the fit rows")
                else:
                    self.fill[j] = column.mean() if self.s.method == "mean" else np.median(column)
            return
        empty = [self.ctx.feature_names[j] for j in num if missing[:, j].all()]
        if empty:
            raise ValueError(f"prep impute: column {empty[0]} has no value in the fit rows")
        if self.s.method == "knn":
            from sklearn.impute import KNNImputer

            self.model = KNNImputer(n_neighbors=self.s.n_neighbors)
        else:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            self.model = IterativeImputer(max_iter=self.s.max_iter, random_state=int_seed(self.ctx.seed, "prep impute"))
        self.num = num
        self.complete_row = np.array([x[~missing[:, j], j].mean() for j in num])
        self.model.fit(x[:, num])

    def removed(self):
        return self.drop

    def transform(self, x, positions):
        x = np.array(x, dtype=np.float64, copy=True)
        if self.s.method == "drop_rows":
            return x
        missing = np.isnan(x)
        for j in range(x.shape[1]):
            if j in self.ctx.categorical or self.s.method in ("mean", "median", "constant"):
                x[missing[:, j], j] = self.fill[j]
        if self.s.method in ("knn", "iterative") and missing[:, self.num].any():
            # scikit-learn's IterativeImputer gives its starting (mean) imputation when every value it is handed is
            # missing; one complete row (the fit means), transformed with the rows and then dropped, keeps each row's
            # result the same whichever other rows are transformed with it
            part = np.vstack([x[:, self.num], self.complete_row])
            x[:, self.num] = self.model.transform(part)[:-1]
        return x

    def record(self):
        return {"method": self.s.method, "missing_per_column": dict(zip(self.ctx.feature_names,
                                                                          self.missing_counts.tolist()))}


@dataclasses.dataclass(frozen=True)
class DropDuplicatesSettings:
    decimals: int | None            # round to this many decimals before comparing; None: exact


@register_prep("drop_duplicates", settings=DropDuplicatesSettings, kinds=TABULAR, profile="core", packages=(),
               removes_rows=True, description="take repeated rows out of the fit, keeping the first",
               example="prep_clean")
class DropDuplicates(PrepStep):
    def fit(self, x, positions):
        values = np.asarray(x, dtype=np.float64)
        if self.s.decimals is not None:
            values = np.round(values, self.s.decimals)
        _, first = np.unique(values, axis=0, return_index=True)
        keep = np.zeros(values.shape[0], dtype=bool)
        keep[first] = True
        self.drop = ~keep

    def removed(self):
        return self.drop

    def transform(self, x, positions):
        return np.asarray(x)

    def record(self):
        return {"decimals": self.s.decimals}


@dataclasses.dataclass(frozen=True)
class DropConstantSettings:
    min_variance: float             # drop columns whose fit-row variance is at most this (0: only constant columns)
    max_frequency_ratio: float | None    # near-constant: the most common value at least this many times as common as
                                         # the second (with min_unique_fraction); None: no near-constant rule
    min_unique_fraction: float | None    # ... and distinct values below this share of the rows


@register_prep("drop_constant", settings=DropConstantSettings, kinds=TABULAR, profile="core", packages=(),
               description="drop constant and near-constant columns", example="prep_clean")
class DropConstant(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name == "drop_constant" and (s.max_frequency_ratio is None) != (s.min_unique_fraction is None):
                raise ValueError("prep drop_constant: set max_frequency_ratio and min_unique_fraction together")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        variance = np.nanvar(x, axis=0)
        drop = variance <= self.s.min_variance
        if self.s.max_frequency_ratio is not None:
            for j in range(x.shape[1]):
                values = x[~np.isnan(x[:, j]), j]
                uniq, counts = np.unique(values, return_counts=True)
                if uniq.size < 2:
                    continue
                top = np.sort(counts)[::-1]
                if top[0] / top[1] >= self.s.max_frequency_ratio and uniq.size / n < self.s.min_unique_fraction:
                    drop[j] = True
        self.keep = np.flatnonzero(~drop)
        if self.keep.size == 0:
            raise ValueError("prep drop_constant: every column would be dropped")
        self.dropped = [self.ctx.feature_names[j] for j in np.flatnonzero(drop)]

    def transform(self, x, positions):
        return np.asarray(x)[:, self.keep]

    def names(self, names):
        return tuple(names[j] for j in self.keep)

    def categorical_after(self, categorical):
        new_index = {int(old): i for i, old in enumerate(self.keep)}
        return {new_index[j]: v for j, v in categorical.items() if j in new_index}

    def record(self):
        return {"dropped_columns": self.dropped}


@dataclasses.dataclass(frozen=True)
class OutliersSettings:
    method: Literal["clip_quantiles", "clip_sd", "remove_by_detector"]
    # clip_quantiles: each column clipped to its fit rows' low and high quantiles (winsorizing); clip_sd: to the fit
    # mean plus or minus n_sd standard deviations; remove_by_detector: fit rows the detector flags leave the fit
    low_quantile: float | None
    high_quantile: float | None
    n_sd: float | None
    detector: section_type("model", ("detector",)) | None


@register_prep("outliers", settings=OutliersSettings, kinds=TABULAR, profile="core", packages=(),
               removes_rows=True, description="clip extreme values, or remove rows a detector flags",
               example="prep_clean")
class Outliers(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name != "outliers":
                continue
            need = {"clip_quantiles": ("low_quantile", "high_quantile"), "clip_sd": ("n_sd",),
                    "remove_by_detector": ("detector",)}[s.method]
            for field in ("low_quantile", "high_quantile", "n_sd", "detector"):
                if (getattr(s, field) is not None) != (field in need):
                    raise ValueError(f"prep outliers {s.method}: set {list(need)} and leave the others None")
            if s.method == "clip_quantiles" and not 0 <= s.low_quantile < s.high_quantile <= 1:
                raise ValueError("prep outliers: 0 <= low_quantile < high_quantile <= 1")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        self.num = self.numeric_columns(x.shape[1])
        self.drop = None
        if self.s.method == "clip_quantiles":
            self.low = np.nanquantile(x[:, self.num], self.s.low_quantile, axis=0)
            self.high = np.nanquantile(x[:, self.num], self.s.high_quantile, axis=0)
        elif self.s.method == "clip_sd":
            mean, sd = np.nanmean(x[:, self.num], axis=0), np.nanstd(x[:, self.num], axis=0)
            self.low, self.high = mean - self.s.n_sd * sd, mean + self.s.n_sd * sd
        else:
            from sweepkit.models.base import build_detector

            detector = build_detector(self.s.detector, self.ctx.seed, "cpu")
            detector.fit(x)
            self.drop = np.asarray(detector.flag(x), dtype=bool)

    def removed(self):
        return self.drop

    def transform(self, x, positions):
        x = np.array(x, dtype=np.float64, copy=True)
        if self.s.method != "remove_by_detector":
            x[:, self.num] = np.clip(x[:, self.num], self.low, self.high)
        return x

    def record(self):
        if self.s.method == "remove_by_detector":
            return {"method": self.s.method, "detector": self.s.detector.name}
        return {"method": self.s.method, "low": self.low.tolist(), "high": self.high.tolist()}
