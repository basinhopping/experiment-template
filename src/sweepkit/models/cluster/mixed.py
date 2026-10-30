"""Clustering rows that mix numeric and categorical columns.

gower_matrix   Gower's (1971) distance between every pair of rows: for a numeric column, the absolute difference
               divided by the column's range over the rows given (a constant column adds 0); for a categorical
               column (integer codes; the dataset's categorical columns, kept by the preparation steps), 0 when the
               codes are equal and 1 otherwise; then the mean over columns. It holds every pair, so memory grows with
               the square of the rows (10,000 rows: 400 MB in float32); screen on subsamples. kmedoids uses it with
               metric gower.
kprototypes    Huang's (1997) k-prototypes: squared Euclidean distance on the numeric columns plus gamma times the
               number of categorical mismatches; prototypes are the numeric means and the categorical modes of their
               members. The template's own implementation (the kmodes package is an alternative).
"""
import dataclasses

import numpy as np

from sweepkit.models.base import Clusterer, plain_params
from sweepkit.registry import register_model
from sweepkit.streams import generator


def _split(x: np.ndarray, categorical: dict) -> tuple:
    cat = np.array(sorted(categorical), dtype=np.int64)
    num = np.array([j for j in range(x.shape[1]) if j not in categorical], dtype=np.int64)
    return num, cat


def gower_matrix(x: np.ndarray, categorical: dict) -> np.ndarray:
    """[rows, rows] Gower distances in float32."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("gower: expected a 2-D table with at least one column")
    num, cat = _split(x, categorical)
    n = x.shape[0]
    total = np.zeros((n, n), dtype=np.float32)
    for j in num:
        column = x[:, j]
        spread = float(column.max() - column.min())
        if spread > 0:
            total += (np.abs(column[:, None] - column[None, :]) / spread).astype(np.float32)
    for j in cat:
        column = x[:, j]
        total += (column[:, None] != column[None, :]).astype(np.float32)
    total /= x.shape[1]
    return total


@dataclasses.dataclass(frozen=True)
class KPrototypesSettings:
    n_clusters: int
    gamma: float | None             # weight of a categorical mismatch; None: half the mean standard deviation of
                                    # the numeric columns on the fit rows (Huang 1997)
    n_init: int                     # restarts from different random prototypes; the lowest cost is kept
    max_iter: int


@register_model("kprototypes", settings=KPrototypesSettings, family="clusterer", tasks=("clustering",),
                kinds=("tabular",), gpu="none", profile="core", packages=(),
                description="k-prototypes for mixed numeric and categorical columns (own implementation)",
                example="cluster_mixed")
class KPrototypes(Clusterer):
    def _cost(self, xn, xc, centers_n, centers_c, gamma):
        cost = np.zeros((xn.shape[0], centers_n.shape[0]))
        if xn.shape[1]:
            cost += ((xn[:, None, :] - centers_n[None]) ** 2).sum(axis=2)
        if xc.shape[1]:
            cost += gamma * (xc[:, None, :] != centers_c[None]).sum(axis=2)
        return cost

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        num, cat = _split(x, self.categorical)
        if cat.size == 0:
            raise ValueError("kprototypes: the data have no categorical column; use kmeans, or keep the dataset's "
                             "categorical columns as codes (data.categorical codes) without onehot")
        xn, xc = x[:, num], x[:, cat].astype(np.int64)
        n, k = x.shape[0], self.s.n_clusters
        if k >= n:
            raise ValueError(f"kprototypes: n_clusters {k} must be below the rows ({n})")
        gamma = self.s.gamma
        if gamma is None:
            gamma = 0.5 * float(xn.std(axis=0).mean()) if xn.shape[1] else 1.0
        rng = generator(self.seed, "kprototypes")
        best = None
        for _ in range(self.s.n_init):
            start = rng.choice(n, size=k, replace=False)
            centers_n, centers_c = xn[start].copy(), xc[start].copy()
            labels = None
            for it in range(1, self.s.max_iter + 1):
                new = np.argmin(self._cost(xn, xc, centers_n, centers_c, gamma), axis=1)
                if labels is not None and np.array_equal(new, labels):
                    break
                labels = new
                for c in range(k):
                    members = labels == c
                    if not members.any():
                        far = int(np.argmax(self._cost(xn, xc, centers_n, centers_c, gamma).min(axis=1)))
                        centers_n[c], centers_c[c] = xn[far], xc[far]
                        continue
                    if xn.shape[1]:
                        centers_n[c] = xn[members].mean(axis=0)
                    for j in range(xc.shape[1]):
                        values, counts = np.unique(xc[members, j], return_counts=True)
                        centers_c[c, j] = values[np.argmax(counts)]
            cost_matrix = self._cost(xn, xc, centers_n, centers_c, gamma)
            labels = np.argmin(cost_matrix, axis=1)
            cost = float(cost_matrix[np.arange(n), labels].sum())
            if best is None or cost < best[0]:
                best = (cost, labels, centers_n, centers_c, it)
        self.cost, self._labels, self.centers_n, self.centers_c, self.n_iter = best
        self.gamma_used = gamma
        self.num, self.cat = num, cat

    def labels(self):
        return np.asarray(self._labels, dtype=np.int64)

    def can_assign(self):
        return True

    def assign(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        return np.argmin(self._cost(x_new[:, self.num], x_new[:, self.cat].astype(np.int64), self.centers_n,
                                    self.centers_c, self.gamma_used), axis=1)

    def info(self):
        return {"cost": self.cost, "n_iter": self.n_iter}

    def params(self):
        return plain_params({**dataclasses.asdict(self.s), "gamma_used": self.gamma_used,
                             "categorical_columns": [self.feature_names[j] for j in self.cat]
                             if self.feature_names else [int(j) for j in self.cat]})
