"""The files a supervised run writes for each role (shared by the estimator and torch families).

    predictions_<role>.csv   row_id, the task's group columns, the truth, the prediction, and the scores
    metrics_<role>.json      {"n", "metrics": {...}, "detail": {...}, "by_group": {...}, "select": {...}}

Roles: val (chooses; analyze reads it), test (read only after a selection is recorded), and train when the fit
section asks for it (over- and underfitting). A log line never carries a test number (logs travel with the
validation files).
"""
import numpy as np

from sweepkit.families.common import csv_text, number
from sweepkit.registry import METRICS, TASKS
from sweepkit.tasks.supervised import Outputs, by_group, score


def task_targets(job, table) -> np.ndarray:
    return TASKS.get(job.task_name).target.targets(job.task, table)


def class_names(job) -> tuple:
    return TASKS.get(job.task_name).target.names(job.task)


def selection(job, metrics: dict) -> dict:
    name = job.task.select
    return {"metric": name, "direction": METRICS.get(name)["direction"], "value": metrics[name]}


def role_files(job, role: str, table, positions: np.ndarray, outputs: Outputs, extra: dict) -> dict:
    """{predictions_<role>.csv, metrics_<role>.json} for one role."""
    task = TASKS.get(job.task_name).target
    metrics = score(outputs, job.task.metrics)
    groups = {c: table.column(c).astype(str)[positions] for c in job.task.groups}
    body = {"role": role, "n": int(positions.size), "metrics": metrics, "detail": task.detail(outputs),
            "by_group": by_group(outputs, job.task.metrics, groups, task.detail), **extra}
    if role != "test":
        body["select"] = selection(job, metrics)
    return {f"predictions_{role}.csv": predictions_csv(job, table, positions, outputs, groups),
            f"metrics_{role}.json": body}


def predictions_csv(job, table, positions, outputs: Outputs, groups: dict) -> str:
    ids = table.ids[positions]
    names = outputs.names
    group_cols = list(groups)
    if job.task_name == "classification":
        header = ["row_id"] + group_cols + ["label", "pred"] + [f"score_{c}" for c in names]
        rows = []
        for k in range(positions.size):
            row = [ids[k]] + [groups[c][k] for c in group_cols] + [names[outputs.y_true[k]], names[outputs.y_pred[k]]]
            if outputs.scores is not None:
                row += [number(v, 6) for v in outputs.scores[k]]
            else:
                row += [""] * len(names)
            rows.append(row)
        return csv_text(header + ["scores_kind"], [r + [outputs.scores_kind] for r in rows])
    if job.task_name == "regression":
        header = ["row_id"] + group_cols + ["target", "pred"]
        rows = [[ids[k]] + [groups[c][k] for c in group_cols] + [repr(float(outputs.y_true[k])),
                                                                 repr(float(outputs.y_pred[k]))]
                for k in range(positions.size)]
        return csv_text(header, rows)
    header = (["row_id"] + group_cols + [f"true_{n}" for n in names] + [f"pred_{n}" for n in names]
              + [f"score_{n}" for n in names])
    rows = []
    for k in range(positions.size):
        row = [ids[k]] + [groups[c][k] for c in group_cols] + outputs.y_true[k].tolist() + outputs.y_pred[k].tolist()
        row += [number(v, 6) for v in outputs.scores[k]] if outputs.scores is not None else [""] * len(names)
        rows.append(row)
    return csv_text(header, rows)


def read_predictions(path, task_name: str) -> dict:
    """A predictions file back as arrays (analysis, compare and test-compare read through this)."""
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no rows")
    ids = [r["row_id"] for r in rows]
    if task_name == "classification":
        names = [k[len("score_"):] for k in rows[0] if k.startswith("score_")]
        index = {c: i for i, c in enumerate(names)}
        kind = rows[0]["scores_kind"]
        scores = None
        if kind != "none" and rows[0][f"score_{names[0]}"] != "":
            scores = np.array([[float(r[f"score_{c}"]) for c in names] for r in rows])
        return {"row_ids": ids, "names": tuple(names), "y_true": np.array([index[r["label"]] for r in rows]),
                "y_pred": np.array([index[r["pred"]] for r in rows]), "scores": scores, "scores_kind": kind}
    if task_name == "regression":
        return {"row_ids": ids, "names": (), "y_true": np.array([float(r["target"]) for r in rows]),
                "y_pred": np.array([float(r["pred"]) for r in rows]), "scores": None, "scores_kind": "none"}
    names = [k[len("true_"):] for k in rows[0] if k.startswith("true_")]
    scores = None
    if rows[0][f"score_{names[0]}"] != "":
        scores = np.array([[float(r[f"score_{n}"]) for n in names] for r in rows])
    return {"row_ids": ids, "names": tuple(names),
            "y_true": np.array([[int(r[f"true_{n}"]) for n in names] for r in rows]),
            "y_pred": np.array([[int(r[f"pred_{n}"]) for n in names] for r in rows]),
            "scores": scores, "scores_kind": "probability" if scores is not None else "none"}


def outputs_of(read: dict) -> Outputs:
    return Outputs(read["y_true"], read["y_pred"], read["scores"], read["scores_kind"], read["names"])
