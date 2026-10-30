"""Distances between rows, for the methods that take one: k-medoids, agglomerative clustering, PCoA and MDS (any of
these), and DBSCAN and scikit-learn's HDBSCAN (precomputed).

    euclidean, manhattan, chebyshev     the usual geometric distances
    cosine, correlation                 one minus the cosine, or the correlation, between two rows
    canberra                            a weighted Manhattan distance, sensitive near zero
    braycurtis                          counts and abundances (ecology, microbiome)
    mahalanobis                         Euclidean after whitening by the fit rows' covariance (its pseudo-inverse, so
                                        constant or collinear columns are allowed); learned from the fit rows only
    jaccard, hamming                    binary data: a value is on when it is not 0. jaccard: the share of the
                                        columns on in either row that disagree; hamming: the share of all columns
                                        that differ
    gower                               mixed numeric and categorical columns (sweepkit.models.cluster.mixed): numeric
                                        columns scaled by their range on the fit rows, categories equal or not
    precomputed                         the data are already a distance matrix: one row per row ID and one column per
                                        row ID (docs/DATA.md). The unsupervised runners then give each method the
                                        block of the rows it fits (and, to place other rows, their distances to the
                                        fit rows), and refuse preparation steps, which would change the distances.

Every distance here is computed in float64 by SciPy's cdist (scipy.spatial.distance) or by the template's own Gower
function, so a method's result does not depend on which library computed its distances.
"""
from typing import Literal

import numpy as np

NAMES = ("euclidean", "manhattan", "chebyshev", "cosine", "correlation", "canberra", "braycurtis", "mahalanobis",
         "jaccard", "hamming", "gower")
DISTANCE = Literal[NAMES]
DISTANCE_OR_PRECOMPUTED = Literal[NAMES + ("precomputed",)]
BINARY = ("jaccard", "hamming")


def scipy_name(metric: str) -> str:
    return "cityblock" if metric == "manhattan" else metric


class Distances:
    """Distances of one metric between rows, with what the metric learns from the fit rows."""

    def __init__(self, metric: str, categorical: dict):
        if metric not in NAMES:
            raise ValueError(f"distance {metric!r} is not one of {NAMES} (precomputed distances are not computed)")
        self.metric = metric
        self.categorical = dict(categorical or {})
        self.inverse_covariance = None
        self.ranges = None

    def fit(self, x_fit: np.ndarray) -> "Distances":
        x_fit = np.asarray(x_fit, dtype=np.float64)
        if self.metric == "mahalanobis":
            if x_fit.shape[0] < 2:
                raise ValueError("mahalanobis: the covariance needs at least 2 fit rows")
            self.inverse_covariance = np.linalg.pinv(np.cov(x_fit, rowvar=False).reshape(x_fit.shape[1],
                                                                                          x_fit.shape[1]))
        if self.metric == "gower":
            self.ranges = np.ptp(x_fit, axis=0)
        return self

    def between(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """[rows of a, rows of b] distances, float64."""
        from scipy.spatial.distance import cdist

        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if self.metric == "gower":
            if self.ranges is None:
                raise ValueError("gower: fit the distance on the fit rows first")
            return _gower_between(a, b, self.categorical, self.ranges)
        if self.metric == "mahalanobis":
            if self.inverse_covariance is None:
                raise ValueError("mahalanobis: fit the distance on the fit rows first")
            return cdist(a, b, metric="mahalanobis", VI=self.inverse_covariance)
        if self.metric in BINARY:
            return cdist(a != 0, b != 0, metric=self.metric)
        return cdist(a, b, metric=scipy_name(self.metric))


def _gower_between(a: np.ndarray, b: np.ndarray, categorical: dict, ranges: np.ndarray) -> np.ndarray:
    """Gower (1971) distances between the rows of a and b: the mean over columns of |difference| / range (numeric
    columns, ranges from the fit rows) or 0 when equal and 1 when not (categorical columns)."""
    out = np.zeros((a.shape[0], b.shape[0]))
    for j in range(a.shape[1]):
        if j in categorical:
            out += (a[:, j][:, None] != b[:, j][None, :]).astype(np.float64)
        elif ranges[j] > 0:
            out += np.abs(a[:, j][:, None] - b[:, j][None, :]) / ranges[j]
    return out / max(1, a.shape[1])


def matrix(x: np.ndarray, metric: str, categorical: dict) -> tuple:
    """(all distances between the rows of x, the fitted Distances to place other rows), for the fit rows. With
    precomputed distances, x itself (checked) and None."""
    if metric == "precomputed":
        return check_square(x), None
    distances = Distances(metric, categorical).fit(x)
    if metric == "gower":
        from sweepkit.models.cluster.mixed import gower_matrix

        return np.asarray(gower_matrix(x, categorical or {}), dtype=np.float64), distances
    return distances.between(x, x), distances


def check_square(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"precomputed distances: expected a square block of the fit rows, got {d.shape}")
    if not np.all(np.isfinite(d)) or (d < 0).any():
        raise ValueError("precomputed distances: every distance must be finite and at least 0")
    return d
