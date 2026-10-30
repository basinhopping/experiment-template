"""Multi-label metrics: y_true and y_pred are 0/1 arrays [rows, labels]; scores (when given) one column per label.

A label never predicted and never true has F1 0 in label_f1_macro; a row with no true and no predicted label has F1 1
in sample_f1. label_roc_auc is undefined for a label that is on in every row or in none, and label_average_precision
for a label on in no row: both raise, where scikit-learn would return nan or 0 with a warning. A row of weight 0
counts as no row.
"""
import numpy as np

from sweepkit.registry import register_metric

TASK = ("multilabel",)


def _arrays(o, w):
    y = np.asarray(o.y_true).astype(bool)
    p = np.asarray(o.y_pred).astype(bool)
    weights = np.ones(y.shape[0]) if w is None else np.asarray(w, dtype=float)
    return y, p, weights


@register_metric("subset_accuracy", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="share of rows with every label right")
def subset_accuracy(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return float((weights * (y == p).all(axis=1)).sum() / weights.sum())


@register_metric("hamming_loss", tasks=TASK, direction="min", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="share of wrong labels")
def hamming_loss(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    return float((weights * (y != p).mean(axis=1)).sum() / weights.sum())


def _counts(y, p, weights):
    tp = (weights[:, None] * (y & p)).sum(axis=0)
    fp = (weights[:, None] * (~y & p)).sum(axis=0)
    fn = (weights[:, None] * (y & ~p)).sum(axis=0)
    return tp, fp, fn


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


@register_metric("label_f1_micro", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="F1 over every label decision")
def label_f1_micro(o, w=None) -> float:
    tp, fp, fn = _counts(*_arrays(o, w))
    return float(_f1(tp.sum(), fp.sum(), fn.sum()))


@register_metric("label_f1_macro", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="mean over labels of each label's F1")
def label_f1_macro(o, w=None) -> float:
    tp, fp, fn = _counts(*_arrays(o, w))
    return float(_f1(tp, fp, fn).mean())


@register_metric("sample_f1", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="mean over rows of each row's F1 (a row with no true and no predicted "
                                               "label counts 1)")
def sample_f1(o, w=None) -> float:
    y, p, weights = _arrays(o, w)
    tp = (y & p).sum(axis=1)
    denom = y.sum(axis=1) + p.sum(axis=1)
    per_row = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 1.0)
    return float((weights * per_row).sum() / weights.sum())


def _scores(o):
    if o.scores is None:
        raise ValueError("this metric needs label scores, and the model gave none")
    return np.asarray(o.scores, dtype=float)


@register_metric("label_average_precision", tasks=TASK, direction="max", needs=("y_true", "scores"), weighted=True,
                 label_free=False, description="mean over labels of the area under each label's precision-recall "
                                               "curve")
def label_average_precision(o, w=None) -> float:
    from sklearn.metrics import average_precision_score

    y, _, weights = _arrays(o, w)
    s = _scores(o)
    values = []
    for j in range(y.shape[1]):
        if weights[y[:, j]].sum() <= 0:
            raise ValueError(f"label_average_precision: label {o.names[j]} is never on")
        values.append(average_precision_score(y[:, j], s[:, j], sample_weight=weights))
    return float(np.mean(values))


@register_metric("label_roc_auc", tasks=TASK, direction="max", needs=("y_true", "scores"), weighted=True,
                 label_free=False, description="mean over labels of the area under each label's ROC curve")
def label_roc_auc(o, w=None) -> float:
    from sklearn.metrics import roc_auc_score

    y, _, weights = _arrays(o, w)
    s = _scores(o)
    values = []
    for j in range(y.shape[1]):
        if weights[y[:, j]].sum() <= 0 or weights[~y[:, j]].sum() <= 0:
            raise ValueError(f"label_roc_auc: label {o.names[j]} is always or never on")
        values.append(roc_auc_score(y[:, j], s[:, j], sample_weight=weights))
    return float(np.mean(values))
