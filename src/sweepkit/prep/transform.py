"""Value transformations, each on the columns named (patterns: a name, "re:<regular expression>", or "*" for every
numeric column). Fitted transformations (Box-Cox, Yeo-Johnson, quantile, rank) learn from the fit rows only.

  log          log(x + offset); every value must be above -offset
  log1p        log(1 + x)
  arcsinh      arcsinh(x / cofactor) (cytometry: cofactor 5 for mass, 150 for flow cytometry, by convention)
  sqrt         square root (values must not be negative)
  boxcox       Box-Cox with each column's lambda fitted on the fit rows (values must be positive)
  yeojohnson   Yeo-Johnson, lambda fitted per column
  clr          centred log-ratio per row over the named columns: log(x + pseudocount) minus the row's mean of those logs
               (compositional data such as microbiome proportions)
  alr          additive log-ratio: log((x + pseudocount) / (x_reference + pseudocount)); the reference column is dropped
  total_count  each row scaled so that the named columns sum to target_sum (single-cell counts)
  quantile     each column mapped through its fit rows' quantiles to a uniform or normal distribution
  rank         each column's empirical distribution function on the fit rows (a value's share of fit values at or
               below it)
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.prep.base import PrepStep, columns_matching
from sweepkit.registry import register_prep

METHODS = ("log", "log1p", "arcsinh", "sqrt", "boxcox", "yeojohnson", "clr", "alr", "total_count", "quantile", "rank")
NEEDS = {"log": ("offset",), "arcsinh": ("cofactor",), "clr": ("pseudocount",), "alr": ("pseudocount", "reference"),
         "total_count": ("target_sum",), "quantile": ("n_quantiles", "output")}
OPTIONAL = ("offset", "cofactor", "pseudocount", "reference", "target_sum", "n_quantiles", "output")


@dataclasses.dataclass(frozen=True)
class TransformSettings:
    method: Literal["log", "log1p", "arcsinh", "sqrt", "boxcox", "yeojohnson", "clr", "alr", "total_count",
                    "quantile", "rank"]
    columns: tuple[str, ...]        # which columns: names, "re:<regex>", or ["*"] for every numeric column
    offset: float | None            # log
    cofactor: float | None          # arcsinh
    pseudocount: float | None       # clr, alr
    reference: str | None           # alr: the denominator column
    target_sum: float | None        # total_count
    n_quantiles: int | None         # quantile
    output: Literal["uniform", "normal"] | None   # quantile


@register_prep("transform", settings=TransformSettings, kinds=("tabular",), profile="core", packages=("sklearn",),
               description="log, arcsinh, Box-Cox, Yeo-Johnson, centred log-ratio, total-count, quantile, rank",
               example="prep_transform")
class Transform(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name != "transform":
                continue
            need = NEEDS.get(s.method, ())
            for field in OPTIONAL:
                if (getattr(s, field) is not None) != (field in need):
                    raise ValueError(f"prep transform {s.method}: set {list(need) or 'none of the options'} and "
                                     "leave the others None")
            for field in ("cofactor", "pseudocount", "target_sum"):
                value = getattr(s, field)
                if value is not None and not value > 0:
                    raise ValueError(f"prep transform: {field} must be positive")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        chosen = columns_matching(self.ctx.feature_names, self.s.columns, "prep transform columns")
        self.cols = np.array([j for j in chosen if j not in self.ctx.categorical], dtype=np.int64)
        if self.cols.size == 0:
            raise ValueError("prep transform: no numeric column selected")
        part = x[:, self.cols]
        m = self.s.method
        self.ref = None
        if m == "alr":
            names = [self.ctx.feature_names[j] for j in self.cols]
            if self.s.reference not in names:
                raise ValueError(f"prep transform alr: reference {self.s.reference!r} is not among the columns")
            self.ref = int(self.cols[names.index(self.s.reference)])
        if m in ("boxcox", "yeojohnson"):
            from sklearn.preprocessing import PowerTransformer

            if m == "boxcox" and not (part > 0).all():
                raise ValueError("prep transform boxcox: every value must be positive")
            self.model = PowerTransformer(method="box-cox" if m == "boxcox" else "yeo-johnson", standardize=False)
            self.model.fit(part)
        elif m == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            self.model = QuantileTransformer(n_quantiles=min(self.s.n_quantiles, part.shape[0]),
                                             output_distribution=self.s.output, subsample=part.shape[0])
            self.model.fit(part)
        elif m == "rank":
            self.sorted = np.sort(part, axis=0)

    def transform(self, x, positions):
        x = np.array(x, dtype=np.float64, copy=True)
        part = x[:, self.cols]
        m, s = self.s.method, self.s
        if m == "log":
            if not (part + s.offset > 0).all():
                raise ValueError(f"prep transform log: a value is at or below -offset ({-s.offset})")
            part = np.log(part + s.offset)
        elif m == "log1p":
            if not (part > -1).all():
                raise ValueError("prep transform log1p: a value is at or below -1")
            part = np.log1p(part)
        elif m == "arcsinh":
            part = np.arcsinh(part / s.cofactor)
        elif m == "sqrt":
            if (part < 0).any():
                raise ValueError("prep transform sqrt: a value is negative")
            part = np.sqrt(part)
        elif m in ("boxcox", "yeojohnson", "quantile"):
            if m == "boxcox" and not (part > 0).all():
                raise ValueError("prep transform boxcox: every value must be positive")
            part = self.model.transform(part)
        elif m == "rank":
            n = self.sorted.shape[0]
            part = np.stack([np.searchsorted(self.sorted[:, k], part[:, k], side="right") / n
                             for k in range(part.shape[1])], axis=1)
        elif m == "clr":
            if (part + s.pseudocount <= 0).any():
                raise ValueError("prep transform clr: a value is at or below -pseudocount")
            logs = np.log(part + s.pseudocount)
            part = logs - logs.mean(axis=1, keepdims=True)
        elif m == "alr":
            if (part + s.pseudocount <= 0).any():
                raise ValueError("prep transform alr: a value is at or below -pseudocount")
            denominator = np.log(x[:, self.ref] + s.pseudocount)
            part = np.log(part + s.pseudocount) - denominator[:, None]
        elif m == "total_count":
            sums = part.sum(axis=1, keepdims=True)
            if (sums <= 0).any():
                raise ValueError(f"prep transform total_count: {int((sums <= 0).sum())} rows have no counts; remove "
                                 "them first")
            part = part / sums * s.target_sum
        x[:, self.cols] = part
        if m == "alr":
            keep = [j for j in range(x.shape[1]) if j != self.ref]
            return x[:, keep]
        return x

    def names(self, names):
        if self.s.method == "alr":
            return tuple(n for j, n in enumerate(names) if j != self.ref)
        return tuple(names)

    def categorical_after(self, categorical):
        if self.s.method != "alr":
            return dict(categorical)
        return {(j if j < self.ref else j - 1): v for j, v in categorical.items()}

    def record(self):
        out = {"method": self.s.method, "columns": [self.ctx.feature_names[j] for j in self.cols]}
        if self.s.method in ("boxcox", "yeojohnson"):
            out["lambdas"] = self.model.lambdas_.tolist()
        return out
