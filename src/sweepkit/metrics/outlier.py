"""Outlier detection criteria without labels, and with known outliers.

Label-free:
  em_auc   area under the excess-mass curve (higher is better), and
  mv_auc   area under the mass-volume curve (lower is better), Goix 2016, "How to evaluate the quality of
           unsupervised anomaly detection algorithms?". Both compare how much of the data and how much of the volume
           of the data's bounding box the method calls normal at each level, using uniform points in that box. Above
           em_mv_max_features dimensions, they average over em_mv_feature_draws random subsets of that many features,
           refitting the method on each subset, as the paper does. Every parameter is in the task's
           criteria_options (Goix's values: 100000 uniform points, t_max 0.9, alpha from 0.9 to 0.999, 5 features,
           50 draws).
  flagged_fraction  share of evaluated rows the method flags (descriptive)
With known outliers (reference 1 = outlier, 0 = inlier; read only after the choice): auroc,
outlier_average_precision and
precision_at_n (precision among the n highest-scored rows, n the number of known outliers).
"""
import numpy as np

from sweepkit.metrics.unsupervised_context import Undefined
from sweepkit.registry import register_metric
from sweepkit.streams import generator

TASK = ("outlier",)
EM_MV_OPTIONS = (("em_mv_uniform_points", int), ("em_mv_max_features", int), ("em_mv_feature_draws", int),
                 ("em_t_max", float), ("em_t_points", int), ("mv_alpha_min", float), ("mv_alpha_max", float),
                 ("mv_alpha_points", int))


def _trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def excess_mass(t: np.ndarray, t_max: float, volume: float, s_uniform: np.ndarray, s_data: np.ndarray) -> float:
    """AUC of the excess-mass curve EM(t) = max_u P(s(X) > u) - t Leb(s > u), on t up to where EM falls to t_max.
    s is a normality score (higher is more normal)."""
    em = np.zeros(t.shape[0])
    em[0] = 1.0
    n, m = s_data.shape[0], s_uniform.shape[0]
    for u in np.unique(s_data):
        em = np.maximum(em, (s_data > u).sum() / n - t * (s_uniform > u).sum() / m * volume)
    stop = int(np.argmax(em <= t_max)) + 1
    if stop == 1:
        raise Undefined("the excess-mass curve never fell to em_t_max on the t grid; raise em_t_points")
    return _trapezoid(t[:stop], em[:stop])


def mass_volume(alphas: np.ndarray, volume: float, s_uniform: np.ndarray, s_data: np.ndarray) -> float:
    """AUC of the mass-volume curve MV(alpha) = Leb(s >= u_alpha), u_alpha the level holding mass alpha."""
    n, m = s_data.shape[0], s_uniform.shape[0]
    order = np.argsort(s_data)
    mv = np.zeros(alphas.shape[0])
    count, u = 0, s_data[order[-1]]
    for i, alpha in enumerate(alphas):
        while count / n < alpha:
            count += 1
            u = s_data[order[-count]]
        mv[i] = (s_uniform >= u).sum() / m * volume
    return _trapezoid(alphas, mv)


def _em_mv(ctx, which: str) -> float:
    if ctx.score_new is None:
        raise Undefined("this method cannot score new points")
    o = ctx.options
    x = ctx.x[ctx.sample].astype(np.float64)
    d = x.shape[1]
    rng = generator(ctx.seed, f"criterion {which}")
    subsets = [None] if d <= o["em_mv_max_features"] else [
        np.sort(rng.choice(d, size=o["em_mv_max_features"], replace=False)) for _ in range(o["em_mv_feature_draws"])]
    values = []
    for features in subsets:
        part = x if features is None else x[:, features]
        low, high = part.min(axis=0), part.max(axis=0)
        widths = high - low
        if (widths <= 0).any():
            continue
        volume = float(np.prod(widths))
        uniform = rng.uniform(low, high, size=(o["em_mv_uniform_points"], part.shape[1]))
        s_data = -np.asarray(ctx.score_new(features, part, part), dtype=np.float64)
        s_uniform = -np.asarray(ctx.score_new(features, part, uniform), dtype=np.float64)
        if which == "em":
            t = np.linspace(0.0, 100.0 / volume, o["em_t_points"], endpoint=False)
            values.append(excess_mass(t, o["em_t_max"], volume, s_uniform, s_data))
        else:
            alphas = np.linspace(o["mv_alpha_min"], o["mv_alpha_max"], o["mv_alpha_points"])
            values.append(mass_volume(alphas, volume, s_uniform, s_data))
    if not values:
        raise Undefined("every feature subset had a feature that does not vary")
    return float(np.mean(values))


@register_metric("em_auc", tasks=TASK, direction="max", needs=("x", "score_new"), weighted=False, label_free=True,
                 options=EM_MV_OPTIONS, description="area under the excess-mass curve (Goix 2016)")
def em_auc(ctx) -> float:
    return _em_mv(ctx, "em")


@register_metric("mv_auc", tasks=TASK, direction="min", needs=("x", "score_new"), weighted=False, label_free=True,
                 options=EM_MV_OPTIONS, description="area under the mass-volume curve (Goix 2016)")
def mv_auc(ctx) -> float:
    return _em_mv(ctx, "mv")


@register_metric("flagged_fraction", tasks=TASK, direction="min", needs=("flags",), weighted=False, label_free=True,
                 description="share of evaluated rows the method flags as outliers", descriptive=True)
def flagged_fraction(ctx) -> float:
    if "flagged" not in ctx.model_info:
        raise Undefined("the method does not flag rows")
    return float(ctx.model_info["flagged"])


def _pair(ctx):
    if ctx.reference is None:
        raise ValueError("no reference labels")
    reference = np.asarray(ctx.reference).astype(str)
    known = np.isin(reference, ("0", "1", "0.0", "1.0"))
    if not known.any():
        raise Undefined("no evaluated row has a 0/1 reference")
    truth = reference[known].astype(float).astype(int)
    if truth.min() == truth.max():
        raise Undefined("the reference holds only outliers or only inliers")
    return truth, ctx.scores[known]


@register_metric("auroc", tasks=TASK, direction="max", needs=("scores", "reference"), weighted=False, label_free=False,
                 description="area under the ROC curve of the scores against the known outliers")
def auroc(ctx) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(*_pair(ctx)))


@register_metric("outlier_average_precision", tasks=TASK, direction="max", needs=("scores", "reference"),
                 weighted=False, label_free=False,
                 description="area under the precision-recall curve against the known outliers")
def average_precision_outlier(ctx) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(*_pair(ctx)))


@register_metric("precision_at_n", tasks=TASK, direction="max", needs=("scores", "reference"), weighted=False,
                 label_free=False, description="precision among the n highest-scored rows, n = known outliers")
def precision_at_n(ctx) -> float:
    truth, scores = _pair(ctx)
    n = int(truth.sum())
    top = np.argsort(-scores, kind="stable")[:n]
    return float(truth[top].mean())
