"""Reading a sweep's record and its run folders, with the rule about which files each step may open in one place.

Files that choose (opened by status, analyze, select, smoke and compare-runs):
  spec.json, metrics_val.json, metrics_train.json, predictions_val.csv, predictions_train.csv,
  criteria.json, assignments.csv, embedding.npy, embedding_rows.csv, removed_rows.csv, scores.csv,
  efficiency.json, provenance.json, model_params.json, prep_record.json, log.txt
Files that judge a choice (opened only by test-report, reveal-report, test-compare and export, for configurations a
recorded selection or a fixed RULE names):
  metrics_test.json, predictions_test.csv, reveal.json, reveal_reference.csv, and a torch run's external datasets:
  metrics_external_<name>.json, predictions_external_<name>.csv

Runs on the identity check fold (sweepkit.splits.base.CHECK_FOLD) validate on test rows, so every result file of
theirs judges: a choosing step may open only their spec.json, provenance.json and efficiency.json, and leaves the
fold out of its fold lists (result_folds).

Every analysis step opens run files through read_json, read_rows and read_npy with its purpose ("choose" or "judge");
a choosing step that asks for a judging file raises, and tests/test_analysis.py checks that analyze, select, status
and smoke open no judging file at all.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

from sweepkit import paths
from sweepkit.splits.base import CHECK_FOLD

JUDGING = ("metrics_test.json", "predictions_test.csv", "reveal.json", "reveal_reference.csv")
# files named per external dataset of a torch run (fit.external): they judge too
JUDGING_PREFIXES = ("metrics_external_", "predictions_external_")
SUPERVISED_TASKS = ("classification", "multilabel", "regression")
# what each family writes as its main output for the evaluated (choosing) rows
OUTPUT_FILE = {"torch": "predictions_val.csv", "estimator": "predictions_val.csv", "clusterer": "assignments.csv",
               "embedder": "embedding.npy", "detector": "scores.csv"}


# files of a check-fold run that hold no number computed on its evaluated rows
CHECK_RUN_READABLE = ("spec.json", "provenance.json", "efficiency.json")
_CHECK_RUN = re.compile(rf"[0-9a-f]{{12}}_{CHECK_FOLD}_seed\d+")


class JudgingFileRefused(PermissionError):
    pass


def is_check_run(folder) -> bool:
    """Whether a run folder (named by its run ID, <hash12>_<fold>_seed<k>) is a run on the identity check fold."""
    return _CHECK_RUN.fullmatch(Path(folder).name) is not None


def result_folds(folds) -> list:
    """The folds a choosing step reads: every fold but the identity check fold."""
    return [f for f in folds if f != CHECK_FOLD]


def _check(folder, name: str, purpose: str) -> None:
    if purpose not in ("choose", "judge"):
        raise ValueError(f"purpose must be choose or judge, got {purpose!r}")
    if purpose == "choose" and (name in JUDGING or name.startswith(JUDGING_PREFIXES)):
        raise JudgingFileRefused(f"{name} judges a choice; a step that chooses never opens it (read it with "
                                 "test-report, reveal-report or test-compare after select)")
    if purpose == "choose" and name not in CHECK_RUN_READABLE and is_check_run(folder):
        raise JudgingFileRefused(f"{Path(folder).name} ran on the {CHECK_FOLD} fold, whose validation rows are test "
                                 f"rows: a step that chooses never opens its {name} (test-report checks that fold)")


def read_json(folder: Path, name: str, purpose: str):
    _check(folder, name, purpose)
    with open(Path(folder) / name, encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(folder: Path, name: str, purpose: str) -> list:
    _check(folder, name, purpose)
    with open(Path(folder) / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_npy(folder: Path, name: str, purpose: str) -> np.ndarray:
    _check(folder, name, purpose)
    return np.load(Path(folder) / name, allow_pickle=False)


def file_path(folder: Path, name: str, purpose: str) -> Path:
    """The path of a run file, for hashing: the same rule applies."""
    _check(folder, name, purpose)
    return Path(folder) / name


def _read_plain_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sweep_record(sweep_id: str) -> dict:
    """The sweep as enqueued (artifacts/sweeps/<sweep_id>/): its file, summary and job list."""
    folder = paths.sweeps_dir() / sweep_id
    if not (folder / "sweep_file.json").is_file():
        raise FileNotFoundError(f"no record of sweep {sweep_id!r} in {paths.sweeps_dir()}; enqueue it (or run it "
                                "with run --local), or pull its artifacts from where it ran")
    with open(folder / "jobs.csv", newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle))
    return {"sweep_id": sweep_id, "sweep_file": _read_plain_json(folder / "sweep_file.json"),
            "summary": _read_plain_json(folder / "summary.json"), "jobs": jobs}


def run_folders(sweep_id: str) -> dict:
    """{"complete": [...], "failed": [...], "in_progress": [...]} folders of the sweep's runs."""
    root = paths.runs_dir(sweep_id)
    out = {"complete": [], "failed": [], "in_progress": []}
    if not root.is_dir():
        return out
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.endswith(".tmp"):
            out["in_progress"].append(folder)
        elif ".failed-" in folder.name:
            out["failed"].append(folder)
        elif (folder / "spec.json").is_file():
            out["complete"].append(folder)
    return out


def complete_runs(sweep_id: str) -> list:
    """[{"folder", "spec"}] for every complete run folder of the sweep."""
    return [{"folder": f, "spec": read_json(f, "spec.json", "choose")} for f in run_folders(sweep_id)["complete"]]


def task_of(sweep_file: dict) -> tuple:
    """(task name, task settings as a plain dict) of the sweep's protocol (the same in every configuration)."""
    task = dict(sweep_file["base"]["task"])
    return task.pop("name"), task


def is_supervised(task_name: str) -> bool:
    return task_name in SUPERVISED_TASKS


def configurations(record: dict, runs: list) -> dict:
    """config_hash -> {"config_hash", "stage", "kind", "label", "control", "family", "settings", "overrides",
    "runs": {seed: {fold: run}}}, in the order the sweep lists them."""
    kinds = {s["name"]: s["kind"] for s in record["sweep_file"]["stages"]}
    order = {}
    for row in record["jobs"]:
        order.setdefault(row["config_hash"], len(order))
    out = {}
    for run in sorted(runs, key=lambda r: (order.get(r["spec"]["config_hash"], len(order)), r["spec"]["seed"],
                                           r["spec"]["fold"])):
        spec = run["spec"]
        config = out.setdefault(spec["config_hash"], {
            "config_hash": spec["config_hash"], "stage": spec["stage"], "kind": kinds.get(spec["stage"], "unknown"),
            "label": spec["label"], "control": spec["settings"]["fit"].get("control", "none"),
            "family": spec["family"], "settings": spec["settings"], "overrides": spec.get("overrides", {}),
            "runs": {}})
        config["runs"].setdefault(int(spec["seed"]), {})[spec["fold"]] = run
    return out


def complete_seeds(config: dict, folds: list) -> list:
    """Seeds of the configuration with a complete run on every fold."""
    return sorted(s for s, by_fold in config["runs"].items() if all(f in by_fold for f in folds))


def fold_means(config: dict, folds: list, value_of) -> dict:
    """seed -> mean over folds of value_of(run), for complete seeds; None where some fold's value is None."""
    out = {}
    for seed in complete_seeds(config, folds):
        values = [value_of(config["runs"][seed][f]) for f in folds]
        out[seed] = None if any(v is None for v in values) else float(np.mean(values))
    return out


def choosing_body(run: dict, task_name: str) -> dict:
    """The run's choosing file, read once and kept on the run: metrics_val.json (supervised) or criteria.json."""
    key = "_choosing"
    if key not in run:
        name = "metrics_val.json" if is_supervised(task_name) else "criteria.json"
        run[key] = read_json(run["folder"], name, "choose")
    return run[key]


def choosing_value(run: dict, task_name: str, metric: str):
    body = choosing_body(run, task_name)
    values = body["metrics"] if is_supervised(task_name) else body["criteria"]
    if metric not in values:
        raise KeyError(f"{run['folder'].name}: no {metric!r} in its {'metrics' if is_supervised(task_name) else 'criteria'}"
                       f" {sorted(values)}")
    return values[metric]


def find_selection(name: str) -> dict:
    path = paths.selections_dir() / name / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"no selection named {name!r} (artifacts/selections/{name}/selection.json); record "
                                "one with select")
    return _read_plain_json(path)


def latest(parent: Path, prefix: str, required_file: str):
    """(folder, parsed required_file) of the newest complete folder <prefix>_<utc> under parent, or (None, None)."""
    if not parent.is_dir():
        return None, None
    found = []
    for p in parent.iterdir():
        if not p.is_dir() or p.name.endswith(".tmp") or ".failed-" in p.name or not p.name.startswith(prefix + "_"):
            continue
        rest = p.name[len(prefix) + 1:]
        if rest[:1].isdigit() and (p / required_file).is_file():     # not another name that shares the prefix
            found.append(p)
    if not found:
        return None, None
    folder = sorted(found)[-1]
    return folder, _read_plain_json(folder / required_file)
