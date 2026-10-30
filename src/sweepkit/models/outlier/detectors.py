"""Outlier detectors: isolation forest (cuML on the GPU), local outlier factor, one-class SVM, elliptic envelope, and
HDBSCAN's GLOSH outlier scores (the hdbscan package). Scores are higher for more unusual rows; flag() marks the rows
each method calls outliers (its own threshold, set by contamination where it has one)."""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Detector, LibraryDetector, _host, library_params, merged_kwargs
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

OUT = ("outlier",)
TAB = ("tabular",)


def _contamination(value: float | None):
    return "auto" if value is None else value


@dataclasses.dataclass(frozen=True)
class IsolationForestSettings:
    n_estimators: int
    max_samples: int | None         # rows per tree; None: min(256, rows)
    max_features: float             # share of columns per tree
    contamination: float | None     # share of rows to flag; None: scikit-learn's "auto" threshold
    extra: dict[str, Scalar]


@register_model("isolation_forest", settings=IsolationForestSettings, family="detector", tasks=OUT, kinds=TAB,
                gpu="cuml", profile="core", packages=("sklearn",),
                description="isolation forest (scikit-learn; cuML on the GPU)", example="outlier_detectors")
class IsolationForest(LibraryDetector):
    def make(self, s, seed, device):
        if device == "cuda":
            from cuml.ensemble import IsolationForest as F
        else:
            from sklearn.ensemble import IsolationForest as F
        return F(**merged_kwargs({"n_estimators": s.n_estimators,
                                  "max_samples": "auto" if s.max_samples is None else s.max_samples,
                                  "max_features": s.max_features, "contamination": _contamination(s.contamination),
                                  "random_state": seed}, s.extra, "isolation_forest"))


@dataclasses.dataclass(frozen=True)
class LOFSettings:
    n_neighbors: int
    metric: Literal["euclidean", "manhattan", "cosine", "chebyshev"]
    contamination: float | None
    extra: dict[str, Scalar]


@register_model("lof", settings=LOFSettings, family="detector", tasks=OUT, kinds=TAB, gpu="none", profile="core",
                packages=("sklearn",), description="local outlier factor (novelty mode, so it scores new rows too)",
                example="outlier_detectors")
class LOF(LibraryDetector):
    def make(self, s, seed, device):
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(**merged_kwargs({"n_neighbors": s.n_neighbors, "metric": s.metric,
                                                   "contamination": _contamination(s.contamination),
                                                   "novelty": True}, s.extra, "lof"))


@dataclasses.dataclass(frozen=True)
class OneClassSVMSettings:
    kernel: Literal["rbf", "poly", "sigmoid", "linear"]
    nu: float                       # an upper bound on the share of rows flagged
    gamma: float | None             # None: scikit-learn's "scale"
    extra: dict[str, Scalar]


@register_model("one_class_svm", settings=OneClassSVMSettings, family="detector", tasks=OUT, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="one-class SVM", example="outlier_detectors")
class OneClassSVM(LibraryDetector):
    def make(self, s, seed, device):
        from sklearn.svm import OneClassSVM as O

        return O(**merged_kwargs({"kernel": s.kernel, "nu": s.nu, "gamma": "scale" if s.gamma is None else s.gamma},
                                 s.extra, "one_class_svm"))


@dataclasses.dataclass(frozen=True)
class EllipticEnvelopeSettings:
    contamination: float
    support_fraction: float | None  # None: the minimum covariance determinant's default
    extra: dict[str, Scalar]


@register_model("elliptic_envelope", settings=EllipticEnvelopeSettings, family="detector", tasks=OUT, kinds=TAB,
                gpu="none", profile="core", packages=("sklearn",),
                description="elliptic envelope (robust covariance; Gaussian-shaped data)", example="outlier_detectors")
class EllipticEnvelope(LibraryDetector):
    def make(self, s, seed, device):
        from sklearn.covariance import EllipticEnvelope as E

        return E(**merged_kwargs({"contamination": s.contamination, "support_fraction": s.support_fraction,
                                  "random_state": seed}, s.extra, "elliptic_envelope"))


@dataclasses.dataclass(frozen=True)
class HDBSCANOutlierSettings:
    min_cluster_size: int
    min_samples: int | None
    flag_quantile: float            # rows whose GLOSH score is above this quantile of the fit rows' scores are flagged
    extra: dict[str, Scalar]


@register_model("hdbscan_outliers", settings=HDBSCANOutlierSettings, family="detector", tasks=OUT, kinds=TAB,
                gpu="none", profile="unsupervised", packages=("hdbscan",),
                description="HDBSCAN's GLOSH outlier scores (the hdbscan package)", example="outlier_detectors")
class HDBSCANOutliers(Detector):
    def fit(self, x):
        import hdbscan

        self.model = hdbscan.HDBSCAN(**merged_kwargs({"min_cluster_size": self.s.min_cluster_size,
                                                      "min_samples": self.s.min_samples, "prediction_data": True},
                                                     self.s.extra, "hdbscan_outliers"))
        self.model.fit(x)
        self.fit_scores = np.nan_to_num(np.asarray(self.model.outlier_scores_, dtype=np.float64))
        self.threshold = float(np.quantile(self.fit_scores, self.s.flag_quantile))
        self.fit_x = np.asarray(x)

    def score(self, x):
        x = np.asarray(x)
        if x.shape == self.fit_x.shape and np.array_equal(x, self.fit_x):
            return self.fit_scores
        from hdbscan.prediction import approximate_predict_scores

        return np.nan_to_num(_host(approximate_predict_scores(self.model, x)).astype(np.float64))

    def flag(self, x):
        return self.score(x) > self.threshold

    def params(self):
        return library_params(self.model)
