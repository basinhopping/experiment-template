"""Supervised tasks: classification (binary or multi-class), multi-label classification and regression.

Each names its target, the metrics recorded for every role, the one metric that chooses (its direction comes from
the metric's registration), and group columns whose groups get their own metrics (for example a site or a sex).
A classification task lists its classes in output order, so the order never depends on which classes a fold holds.
"""
import dataclasses

import numpy as np

from sweepkit.registry import METRICS, register_task
from sweepkit.splits.base import SUPERVISED_ROLES


@dataclasses.dataclass
class Outputs:
    """What a supervised model produced for one role's rows."""

    y_true: np.ndarray          # class indices [n]; floats [n]; or 0/1 [n, labels]
    y_pred: np.ndarray          # the same shape as y_true
    scores: np.ndarray | None   # classification: [n, classes] probabilities or decision values; multi-label: [n,
                                # labels]; regression: None
    scores_kind: str            # "probability", "decision" or "none"
    names: tuple                # class or label names; () for regression


def _check_metrics(task: str, metrics: tuple, select: str) -> None:
    if not metrics:
        raise ValueError(f"task {task}: list at least one metric")
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"task {task}: metrics repeat")
    for name in metrics:
        entry = METRICS.get(name)
        if task not in entry["tasks"]:
            raise ValueError(f"task {task}: metric {name!r} scores {list(entry['tasks'])}")
    if select not in metrics:
        raise ValueError(f"task {task}: select {select!r} must be one of its metrics {list(metrics)}")


def score(outputs: Outputs, metrics: tuple, weights=None) -> dict:
    return {name: float(METRICS.get(name)["fn"](outputs, weights)) for name in metrics}


def report(outputs: Outputs, metrics: tuple, extra: dict) -> dict:
    out = {"n": int(outputs.y_true.shape[0]), "metrics": score(outputs, metrics)}
    out.update(extra)
    return out


def by_group(outputs: Outputs, metrics: tuple, groups: dict, detail) -> dict:
    """{column: {group: metrics or None where a metric is undefined in that group}}."""
    out = {}
    for column, values in groups.items():
        out[column] = {}
        for group in sorted(set(values.tolist())):
            mask = values == group
            part = Outputs(outputs.y_true[mask], outputs.y_pred[mask],
                           None if outputs.scores is None else outputs.scores[mask], outputs.scores_kind,
                           outputs.names)
            try:
                out[column][group] = {"n": int(mask.sum()), "metrics": score(part, metrics), **detail(part)}
            except ValueError as err:
                out[column][group] = {"n": int(mask.sum()), "metrics": None, "undefined_because": str(err)}
    return out


@dataclasses.dataclass(frozen=True)
class ClassificationSettings:
    target: str                     # the column holding each row's class
    classes: tuple[str, ...]        # every class, in output order (binary: two)
    metrics: tuple[str, ...]        # recorded for every role, for example ["balanced_accuracy", "accuracy", "f1_macro"]
    select: str                     # the metric that chooses (one of metrics); its direction is the metric's
    groups: tuple[str, ...]         # columns whose groups get their own metrics (may be empty)


@register_task("classification", settings=ClassificationSettings, roles=SUPERVISED_ROLES,
               description="binary or multi-class classification")
class Classification:
    @staticmethod
    def check_job(job) -> None:
        s = job.task
        if len(s.classes) < 2 or len(set(s.classes)) != len(s.classes):
            raise ValueError("task classification: list at least two distinct classes")
        _check_metrics("classification", s.metrics, s.select)

    @staticmethod
    def targets(s: ClassificationSettings, table) -> np.ndarray:
        raw = table.column(s.target).astype(str)
        index = {c: i for i, c in enumerate(s.classes)}
        unknown = sorted(set(raw.tolist()) - set(index))
        if unknown:
            raise ValueError(f"task classification: column {s.target!r} holds values {unknown[:5]} that are not in "
                             f"classes {list(s.classes)}; filter those rows in the data settings or list them")
        return np.array([index[v] for v in raw.tolist()], dtype=np.int64)

    @staticmethod
    def names(s) -> tuple:
        return tuple(s.classes)

    @staticmethod
    def detail(outputs: Outputs) -> dict:
        from sweepkit.metrics.classification import confusion_matrix, per_class

        n = len(outputs.names)
        cm = confusion_matrix(outputs.y_true, outputs.y_pred, n)
        return {"support": dict(zip(outputs.names, cm.sum(axis=1).tolist())), **per_class(cm, outputs.names),
                "confusion_true_by_predicted": cm.tolist()}


@dataclasses.dataclass(frozen=True)
class MultilabelSettings:
    targets: tuple[str, ...]        # one 0/1 column per label
    metrics: tuple[str, ...]
    select: str
    groups: tuple[str, ...]


@register_task("multilabel", settings=MultilabelSettings, roles=SUPERVISED_ROLES,
               description="several yes/no labels per row")
class Multilabel:
    @staticmethod
    def check_job(job) -> None:
        s = job.task
        if len(s.targets) < 2 or len(set(s.targets)) != len(s.targets):
            raise ValueError("task multilabel: list at least two distinct label columns")
        _check_metrics("multilabel", s.metrics, s.select)

    @staticmethod
    def targets_of(s: MultilabelSettings, table) -> np.ndarray:
        columns = []
        for name in s.targets:
            values = np.asarray(table.column(name)).astype(float)
            if not np.isin(values, (0.0, 1.0)).all():
                raise ValueError(f"task multilabel: column {name!r} must hold 0 or 1 only")
            columns.append(values.astype(np.int64))
        return np.stack(columns, axis=1)

    targets = targets_of

    @staticmethod
    def names(s) -> tuple:
        return tuple(s.targets)

    @staticmethod
    def detail(outputs: Outputs) -> dict:
        positives = outputs.y_true.sum(axis=0).tolist()
        return {"positives": dict(zip(outputs.names, positives))}


@dataclasses.dataclass(frozen=True)
class RegressionSettings:
    target: str                     # the column holding each row's number
    metrics: tuple[str, ...]        # for example ["rmse", "mae", "r2"]
    select: str
    groups: tuple[str, ...]


@register_task("regression", settings=RegressionSettings, roles=SUPERVISED_ROLES,
               description="a number per row")
class Regression:
    @staticmethod
    def check_job(job) -> None:
        _check_metrics("regression", job.task.metrics, job.task.select)

    @staticmethod
    def targets(s: RegressionSettings, table) -> np.ndarray:
        values = np.asarray(table.column(s.target), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"task regression: column {s.target!r} holds missing or infinite values; filter those "
                             "rows in the data settings")
        return values

    @staticmethod
    def names(s) -> tuple:
        return ()

    @staticmethod
    def detail(outputs: Outputs) -> dict:
        return {"mean_true": float(np.mean(outputs.y_true)), "mean_pred": float(np.mean(outputs.y_pred))}
