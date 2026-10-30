"""Regression metrics, each with optional row weights (a bootstrap draw's counts).

A whole-number weight counts its row that many times: every metric here gives, under weights, the value it gives on
the rows repeated by their weights (the paired bootstrap relies on it). So median_ae is the median of the repeated
rows (the mean of the two middle errors when the total weight is even, as numpy.median gives without weights), and
spearman ranks the repeated rows (ties share the mean of their positions) before the weighted Pearson correlation.
"""
import numpy as np

from sweepkit.registry import register_metric

TASK = ("regression",)


def _arrays(o, w):
    y = np.asarray(o.y_true, dtype=float)
    p = np.asarray(o.y_pred, dtype=float)
    weights = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("no rows")
    return y, p, weights


def _wmean(values, weights) -> float:
    return float((weights * values).sum() / weights.sum())


@register_metric("mse", tasks=TASK, direction="min", needs=("y_true", "y_pred"), weighted=True, label_free=False,
                 description="mean squared error")
def mse(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return _wmean((y - p) ** 2, weights)


@register_metric("rmse", tasks=TASK, direction="min", needs=("y_true", "y_pred"), weighted=True, label_free=False,
                 description="root mean squared error")
def rmse(o, w=None) -> float:
    return float(np.sqrt(mse(o, w)))


@register_metric("mae", tasks=TASK, direction="min", needs=("y_true", "y_pred"), weighted=True, label_free=False,
                 description="mean absolute error")
def mae(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return _wmean(np.abs(y - p), weights)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The median of the values with each counted as often as its weight: for whole-number weights, the median of
    the repeated values (the mean of the two middle ones when the total weight is even)."""
    keep = weights > 0
    order = np.argsort(values[keep], kind="stable")
    v, cumulative = values[keep][order], np.cumsum(weights[keep][order])
    half = 0.5 * cumulative[-1]
    i = int(np.searchsorted(cumulative, half))          # the first value whose cumulative weight reaches half
    if cumulative[i] == half and i + 1 < v.size:
        return float((v[i] + v[i + 1]) / 2)
    return float(v[i])


def weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each row's rank among the rows counted as often as their weights, ties sharing the mean of their positions
    (without weights, scipy.stats.rankdata's average ranks)."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    _, start = np.unique(v, return_index=True)
    tied_weight = np.add.reduceat(w, start)
    below = np.concatenate([[0.0], np.cumsum(tied_weight)[:-1]])
    ranks = np.empty(v.size)
    ranks[order] = np.repeat(below + (tied_weight + 1) / 2, np.diff(np.append(start, v.size)))
    return ranks


@register_metric("median_ae", tasks=TASK, direction="min", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="median absolute error (the median of the rows repeated by their "
                                               "weights)")
def median_ae(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return weighted_median(np.abs(y - p), weights)


@register_metric("r2", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True, label_free=False,
                 description="coefficient of determination")
def r2(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    mean = _wmean(y, weights)
    total = (weights * (y - mean) ** 2).sum()
    if total == 0:
        raise ValueError("r2 is undefined when the true values do not vary")
    return float(1 - (weights * (y - p) ** 2).sum() / total)


def _weighted_pearson(a, b, weights) -> float:
    ma, mb = _wmean(a, weights), _wmean(b, weights)
    cov = (weights * (a - ma) * (b - mb)).sum()
    va, vb = (weights * (a - ma) ** 2).sum(), (weights * (b - mb) ** 2).sum()
    if va == 0 or vb == 0:
        raise ValueError("a correlation is undefined when one side does not vary")
    return float(cov / np.sqrt(va * vb))


@register_metric("pearson", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="Pearson correlation of predictions with the truth")
def pearson(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return _weighted_pearson(y, p, weights)


@register_metric("spearman", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="Spearman rank correlation (the rows repeated by their weights are "
                                               "ranked, then correlated)")
def spearman(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return _weighted_pearson(weighted_ranks(y, weights), weighted_ranks(p, weights), weights)


@register_metric("ccc", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True, label_free=False,
                 description="Lin's concordance correlation coefficient")
def ccc(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    my, mp = _wmean(y, weights), _wmean(p, weights)
    vy, vp = _wmean((y - my) ** 2, weights), _wmean((p - mp) ** 2, weights)
    cov = _wmean((y - my) * (p - mp), weights)
    denom = vy + vp + (my - mp) ** 2
    if denom == 0:
        raise ValueError("ccc is undefined when neither side varies and they agree")
    return float(2 * cov / denom)


@register_metric("max_error", tasks=TASK, direction="min", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="largest absolute error over rows with weight")
def max_error(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return float(np.abs(y - p)[weights > 0].max())


# Batched forms for the paired bootstrap (sweepkit.eval.stats): weighted sums for every draw and seed at once
# (weights [draws, rows], y_pred [seeds, rows]). tests/test_metrics.py checks each against the functions above.

def _sums(fold: dict, weights: np.ndarray) -> dict:
    y = np.asarray(fold["y_true"], dtype=float)
    p = np.asarray(fold["y_pred"], dtype=float)                  # [seeds, rows]
    w = np.asarray(weights, dtype=float)
    n = w.sum(axis=1, keepdims=True)                             # [draws, 1]
    if (n <= 0).any():
        raise ValueError("no rows")
    e = y[None, :] - p
    return {"n": n, "y": (w @ y)[:, None], "yy": (w @ (y * y))[:, None], "p": w @ p.T, "pp": w @ (p * p).T,
            "yp": w @ (p * y[None, :]).T, "ee": w @ (e * e).T, "ae": w @ np.abs(e).T}


def _b_mse(f, w):
    s = _sums(f, w)
    return s["ee"] / s["n"]


def _b_mae(f, w):
    s = _sums(f, w)
    return s["ae"] / s["n"]


def _b_r2(f, w):
    s = _sums(f, w)
    total = s["yy"] - s["y"] * s["y"] / s["n"]
    if (total <= 0).any():
        raise ValueError("r2 is undefined when the true values do not vary")
    return 1 - s["ee"] / total


def _b_pearson(f, w):
    s = _sums(f, w)
    cov = s["yp"] - s["y"] * s["p"] / s["n"]
    vy = s["yy"] - s["y"] * s["y"] / s["n"]
    vp = s["pp"] - s["p"] * s["p"] / s["n"]
    if (vy <= 0).any() or (vp <= 0).any():
        raise ValueError("a correlation is undefined when one side does not vary")
    return cov / np.sqrt(vy * vp)


def _b_ccc(f, w):
    s = _sums(f, w)
    my, mp = s["y"] / s["n"], s["p"] / s["n"]
    vy = s["yy"] / s["n"] - my * my
    vp = s["pp"] / s["n"] - mp * mp
    cov = s["yp"] / s["n"] - my * mp
    denom = vy + vp + (my - mp) ** 2
    if (denom <= 0).any():
        raise ValueError("ccc is undefined when neither side varies and they agree")
    return 2 * cov / denom


def _add_batched():
    from sweepkit.registry import METRICS

    METRICS.add_batched("mse", _b_mse)
    METRICS.add_batched("rmse", lambda f, w: np.sqrt(_b_mse(f, w)))
    METRICS.add_batched("mae", _b_mae)
    METRICS.add_batched("r2", _b_r2)
    METRICS.add_batched("pearson", _b_pearson)
    METRICS.add_batched("ccc", _b_ccc)


_add_batched()
