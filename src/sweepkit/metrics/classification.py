"""Classification metrics. Each takes the role's Outputs and optional row weights (a bootstrap draw's counts) and
returns one number; every count-based metric is computed from one weighted confusion matrix (true classes as rows,
predicted as columns), so all metrics of one run agree with each other by construction.

A class that is never predicted has precision 0 and F1 0. balanced_accuracy (the mean of per-class recall, also
called unweighted accuracy) is undefined when a class has no rows, and raises. So do roc_auc when a class has no rows
or every row (binary: either class has none), and average_precision when a class has no rows (binary: the second
class), where scikit-learn would return nan or 0 with a warning. A row of weight 0 counts as no row.
"""
import numpy as np

from sweepkit.registry import register_metric

TASK = ("classification",)


def confusion_matrix(y_true, y_pred, n_classes: int, weights=None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"expected two 1-D arrays of equal length, got {y_true.shape} and {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no predictions")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must hold class indices, got dtype {arr.dtype}")
        if arr.min() < 0 or arr.max() >= n_classes:
            raise ValueError(f"{name} holds indices outside 0..{n_classes - 1}")
    w = np.ones(y_true.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    cm = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(cm, (y_true, y_pred), w)
    return cm


def _recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    if (support == 0).any():
        raise ValueError(f"classes {np.flatnonzero(support == 0).tolist()} have no rows; recall is undefined")
    return np.diag(cm) / support


def _precision(cm: np.ndarray) -> np.ndarray:
    predicted = cm.sum(axis=0)
    out = np.zeros(cm.shape[0])
    nonzero = predicted > 0
    out[nonzero] = np.diag(cm)[nonzero] / predicted[nonzero]
    return out


def _f1(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    p = _precision(cm)
    r = np.zeros(cm.shape[0])
    has = support > 0
    r[has] = np.diag(cm)[has] / support[has]
    out = np.zeros_like(p)
    nonzero = (p + r) > 0
    out[nonzero] = 2 * p[nonzero] * r[nonzero] / (p[nonzero] + r[nonzero])
    return out


def per_class(cm: np.ndarray, names: tuple) -> dict:
    support = cm.sum(axis=1)
    recall = {n: (float(cm[i, i] / support[i]) if support[i] else None) for i, n in enumerate(names)}
    return {"recall": recall, "precision": dict(zip(names, _precision(cm).tolist())),
            "f1": dict(zip(names, _f1(cm).tolist()))}


def _cm(o, w) -> np.ndarray:
    return confusion_matrix(o.y_true, o.y_pred, len(o.names), w)


@register_metric("accuracy", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="share of rows predicted correctly (weighted accuracy, WA)")
def accuracy(o, w=None) -> float:
    cm = _cm(o, w)
    return float(np.trace(cm) / cm.sum())


@register_metric("balanced_accuracy", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="mean of per-class recall (unweighted accuracy, UA)")
def balanced_accuracy(o, w=None) -> float:
    return float(_recall(_cm(o, w)).mean())


@register_metric("f1_macro", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="mean of per-class F1")
def f1_macro(o, w=None) -> float:
    return float(_f1(_cm(o, w)).mean())


@register_metric("f1_weighted", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="per-class F1 weighted by class support")
def f1_weighted(o, w=None) -> float:
    cm = _cm(o, w)
    support = cm.sum(axis=1)
    return float((_f1(cm) * support).sum() / support.sum())


@register_metric("f1_micro", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="F1 over all rows (equals accuracy for single-label classification)")
def f1_micro(o, w=None) -> float:
    return accuracy(o, w)


@register_metric("precision_macro", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="mean of per-class precision")
def precision_macro(o, w=None) -> float:
    return float(_precision(_cm(o, w)).mean())


@register_metric("recall_macro", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="mean of per-class recall (the same as balanced_accuracy)")
def recall_macro(o, w=None) -> float:
    return balanced_accuracy(o, w)


@register_metric("mcc", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True, label_free=False,
                 description="Matthews correlation coefficient (multi-class form)")
def mcc(o, w=None) -> float:
    cm = _cm(o, w)
    t = cm.sum(axis=1)
    p = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    denom = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return float((c * s - (t * p).sum()) / denom) if denom > 0 else 0.0


@register_metric("cohen_kappa", tasks=TASK, direction="max", needs=("y_true", "y_pred"), weighted=True,
                 label_free=False, description="Cohen's kappa: agreement beyond chance")
def cohen_kappa(o, w=None) -> float:
    cm = _cm(o, w)
    s = cm.sum()
    observed = np.trace(cm) / s
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (s * s)
    return float((observed - expected) / (1 - expected)) if expected < 1 else 0.0


def _scores(o) -> np.ndarray:
    if o.scores is None or o.scores_kind == "none":
        raise ValueError("this metric needs class scores (probabilities or decision values), and the model gave none")
    return np.asarray(o.scores, dtype=float)


def _weight_of(mask: np.ndarray, w) -> float:
    """The total weight of the rows in mask (their number without weights)."""
    return float(mask.sum()) if w is None else float(np.asarray(w, dtype=float)[mask].sum())


def _need_rows(name: str, o, c: int, w, both_sides: bool) -> np.ndarray:
    """The rows of class c, after checking that the class has rows (and, with both_sides, that other rows exist)."""
    positive = o.y_true == c
    if _weight_of(positive, w) <= 0 or (both_sides and _weight_of(~positive, w) <= 0):
        raise ValueError(f"{name}: class {o.names[c]} has no rows" + (", or only its rows" if both_sides else ""))
    return positive


@register_metric("roc_auc", tasks=TASK, direction="max", needs=("y_true", "scores"), weighted=True, label_free=False,
                 description="area under the ROC curve; one class against the rest, averaged over classes "
                             "(binary: the second class's)")
def roc_auc(o, w=None) -> float:
    from sklearn.metrics import roc_auc_score

    scores = _scores(o)
    n = len(o.names)
    if n == 2:
        column = scores[:, 1] if scores.ndim == 2 else scores
        return float(roc_auc_score(_need_rows("roc_auc", o, 1, w, True), column, sample_weight=w))
    values = [roc_auc_score(_need_rows("roc_auc", o, c, w, True), scores[:, c], sample_weight=w) for c in range(n)]
    return float(np.mean(values))


@register_metric("average_precision", tasks=TASK, direction="max", needs=("y_true", "scores"), weighted=True,
                 label_free=False, description="area under the precision-recall curve, averaged over classes "
                                               "(binary: the second class's)")
def average_precision(o, w=None) -> float:
    from sklearn.metrics import average_precision_score

    scores = _scores(o)
    n = len(o.names)
    if n == 2:
        column = scores[:, 1] if scores.ndim == 2 else scores
        return float(average_precision_score(_need_rows("average_precision", o, 1, w, False), column,
                                             sample_weight=w))
    return float(np.mean([average_precision_score(_need_rows("average_precision", o, c, w, False), scores[:, c],
                                                  sample_weight=w) for c in range(n)]))


def _probabilities(o) -> np.ndarray:
    if o.scores_kind != "probability":
        raise ValueError("this metric needs probabilities, and the model gave decision values")
    p = np.asarray(o.scores, dtype=float)
    return p / p.sum(axis=1, keepdims=True)


@register_metric("log_loss", tasks=TASK, direction="min", needs=("y_true", "scores"), weighted=True,
                 label_free=False, description="mean negative log probability of the true class")
def log_loss(o, w=None) -> float:
    p = np.clip(_probabilities(o)[np.arange(o.y_true.shape[0]), o.y_true], 1e-12, None)
    weights = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    return float(-(weights * np.log(p)).sum() / weights.sum())


@register_metric("brier", tasks=TASK, direction="min", needs=("y_true", "scores"), weighted=True, label_free=False,
                 description="mean squared difference between the probabilities and the one-hot truth")
def brier(o, w=None) -> float:
    p = _probabilities(o)
    onehot = np.eye(p.shape[1])[o.y_true]
    per_row = ((p - onehot) ** 2).sum(axis=1)
    weights = np.ones_like(per_row) if w is None else np.asarray(w, dtype=float)
    return float((weights * per_row).sum() / weights.sum())


# Batched forms for the paired bootstrap (sweepkit.eval.stats): every draw's confusion matrices at once. Each takes
# confusion matrices [..., classes, classes] and gives the metric over the leading axes; the registered functions above
# are the reference, and tests/test_metrics.py checks each form against them under random weights.

def weighted_confusions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, weights: np.ndarray) -> np.ndarray:
    """[draws, seeds, classes, classes]: the confusion matrix of every seed's predictions (y_pred [seeds, rows])
    under every draw's row weights (weights [draws, rows])."""
    from scipy import sparse

    seeds, rows = y_pred.shape
    out = np.empty((weights.shape[0], seeds, n_classes * n_classes))
    for s in range(seeds):
        index = np.asarray(y_true) * n_classes + np.asarray(y_pred[s])
        onehot = sparse.csr_matrix((np.ones(rows), (np.arange(rows), index)), shape=(rows, n_classes * n_classes))
        out[:, s, :] = np.asarray(onehot.T @ weights.T).T
    return out.reshape(weights.shape[0], seeds, n_classes, n_classes)


def _diag(cm):
    return np.diagonal(cm, axis1=-2, axis2=-1)


def _b_accuracy(cm):
    return _diag(cm).sum(-1) / cm.sum(axis=(-2, -1))


def _b_recall(cm):
    support = cm.sum(-1)
    if (support == 0).any():
        raise ValueError("a draw holds no row of some class; recall is undefined")
    return _diag(cm) / support


def _b_precision(cm):
    predicted = cm.sum(-2)
    return np.divide(_diag(cm), predicted, out=np.zeros_like(predicted), where=predicted > 0)


def _b_f1(cm):
    support = cm.sum(-1)
    p = _b_precision(cm)
    r = np.divide(_diag(cm), support, out=np.zeros_like(support), where=support > 0)
    total = p + r
    return np.divide(2 * p * r, total, out=np.zeros_like(total), where=total > 0)


def _b_f1_weighted(cm):
    support = cm.sum(-1)
    return (_b_f1(cm) * support).sum(-1) / support.sum(-1)


def _b_mcc(cm):
    t = cm.sum(-1)
    p = cm.sum(-2)
    c = _diag(cm).sum(-1)
    s = cm.sum(axis=(-2, -1))
    denom = np.sqrt((s * s - (p * p).sum(-1)) * (s * s - (t * t).sum(-1)))
    return np.divide(c * s - (t * p).sum(-1), denom, out=np.zeros_like(denom), where=denom > 0)


def _b_kappa(cm):
    s = cm.sum(axis=(-2, -1))
    observed = _diag(cm).sum(-1) / s
    expected = (cm.sum(-1) * cm.sum(-2)).sum(-1) / (s * s)
    gap = 1 - expected
    return np.divide(observed - expected, gap, out=np.zeros_like(gap), where=expected < 1)


_FROM_CONFUSION = {
    "accuracy": _b_accuracy, "f1_micro": _b_accuracy,
    "balanced_accuracy": lambda cm: _b_recall(cm).mean(-1), "recall_macro": lambda cm: _b_recall(cm).mean(-1),
    "f1_macro": lambda cm: _b_f1(cm).mean(-1), "f1_weighted": _b_f1_weighted,
    "precision_macro": lambda cm: _b_precision(cm).mean(-1), "mcc": _b_mcc, "cohen_kappa": _b_kappa,
}


def _confusion_form(form):
    def batched(fold: dict, weights: np.ndarray) -> np.ndarray:
        cms = weighted_confusions(fold["y_true"], np.asarray(fold["y_pred"]), len(fold["names"]), weights)
        return form(cms)
    return batched


def _per_row_mean_form(per_row):
    """A metric that is a weighted mean of one value per row (log loss, Brier)."""
    def batched(fold: dict, weights: np.ndarray) -> np.ndarray:
        scores = fold.get("scores")
        if scores is None:
            raise ValueError("this metric needs class scores, and the runs wrote none")
        values = np.stack([per_row(np.asarray(fold["y_true"]), np.asarray(scores[s]), fold["scores_kind"])
                           for s in range(np.asarray(fold["y_pred"]).shape[0])])          # [seeds, rows]
        return (weights @ values.T) / weights.sum(axis=1, keepdims=True)
    return batched


def _log_loss_rows(y_true, scores, kind):
    if kind != "probability":
        raise ValueError("this metric needs probabilities, and the model gave decision values")
    p = scores / scores.sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[np.arange(y_true.shape[0]), y_true], 1e-12, None))


def _brier_rows(y_true, scores, kind):
    if kind != "probability":
        raise ValueError("this metric needs probabilities, and the model gave decision values")
    p = scores / scores.sum(axis=1, keepdims=True)
    return ((p - np.eye(p.shape[1])[y_true]) ** 2).sum(axis=1)


def _add_batched():
    from sweepkit.registry import METRICS

    for name, form in _FROM_CONFUSION.items():
        METRICS.add_batched(name, _confusion_form(form))
    METRICS.add_batched("log_loss", _per_row_mean_form(_log_loss_rows))
    METRICS.add_batched("brier", _per_row_mean_form(_brier_rows))


_add_batched()
