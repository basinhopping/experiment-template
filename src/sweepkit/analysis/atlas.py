"""atlas: every sweep, configuration, setting and value tried, in two tables.

    configs.csv   one row per configuration of every recorded sweep: sweep, purpose, stage, label, configuration hash,
                  family, runs expected, complete and failed, the choosing metric and its fold mean over complete seeds
                  (choosing files only), then one column per setting path (data.*, split.*, task.*, prep.<i>.*,
                  features.*, model.*, fit.*)
    values.csv    one row per (setting path, value): how many configurations and sweeps tried it, and the sweeps
    atlas.xlsx    both tables as sheets, when openpyxl is installed and --format xlsx asks for it

Settings come from the runs' spec.json; a configuration none of whose runs has a folder yet is re-expanded from its
recorded sweep file (and listed with "settings unknown" if the current code refuses that file).
"""
import json

import numpy as np

from sweepkit import paths
from sweepkit.analysis.records import (choosing_value, complete_seeds, configurations, read_json, result_folds,
                                       run_folders, sweep_record, task_of)
from sweepkit.families.common import csv_text
from sweepkit.tasks.unsupervised import STABILITY


def flatten(settings: dict) -> dict:
    out = {}

    def walk(value, prefix):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(value, list) and prefix == "prep":
            for i, step in enumerate(value):
                walk(step, f"prep.{i}")
        else:
            out[prefix] = json.dumps(value, sort_keys=True)

    walk(settings, "")
    return out


def _sweep_ids() -> list:
    root = paths.sweeps_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "sweep_file.json").is_file() and not p.name.endswith(".tmp"))


def build() -> dict:
    rows, keys = [], set()
    for sweep_id in _sweep_ids():
        record = sweep_record(sweep_id)
        sweep_file = record["sweep_file"]
        folds = result_folds(sweep_file["folds"])
        task_name, task = task_of(sweep_file)
        folders = run_folders(sweep_id)
        runs = [{"folder": f, "spec": read_json(f, "spec.json", "choose")} for f in folders["complete"]]
        configs = configurations(record, runs)
        failed = {}
        for f in folders["failed"]:
            try:
                h = read_json(f, "spec.json", "choose")["config_hash"]
            except (OSError, ValueError):
                continue
            failed[h] = failed.get(h, 0) + 1
        expanded = None
        by_hash = {}
        for job in record["jobs"]:
            by_hash.setdefault(job["config_hash"], []).append(job)
        for h, jobs in by_hash.items():
            config = configs.get(h)
            settings = config["settings"] if config else None
            if settings is None:
                if expanded is None:
                    try:
                        from sweepkit.sweep.design import expand

                        expanded = {j["config_hash"]: j["settings"] for j in expand(sweep_file)[0]}
                    except (ValueError, TypeError):
                        expanded = {}
                settings = expanded.get(h)
            value = None
            if config and task["select"] != STABILITY:
                per_seed = []
                for seed in complete_seeds(config, folds):
                    values = [choosing_value(config["runs"][seed][f], task_name, task["select"]) for f in folds]
                    per_seed.append(None if None in values else float(np.mean(values)))
                value = float(np.mean(per_seed)) if per_seed and None not in per_seed else None
            flat = flatten(settings) if settings else {}
            keys |= set(flat)
            rows.append({"sweep_id": sweep_id, "purpose": sweep_file["purpose"], "stage": jobs[0]["stage"],
                         "label": jobs[0]["label"], "config_hash": h,
                         "family": (config or {}).get("family", ""), "runs_expected": len(jobs),
                         "runs_complete": sum(len(v) for v in config["runs"].values()) if config else 0,
                         "runs_failed": failed.get(h, 0), "select": task["select"],
                         "select_value": "" if value is None else repr(value),
                         "settings_known": settings is not None, "settings": flat})
    return {"rows": rows, "keys": sorted(keys)}


def tables(atlas: dict) -> tuple:
    fixed = ["sweep_id", "purpose", "stage", "label", "config_hash", "family", "runs_expected", "runs_complete",
             "runs_failed", "select", "select_value", "settings_known"]
    config_rows = [[r[k] for k in fixed] + [r["settings"].get(key, "") for key in atlas["keys"]] for r in atlas["rows"]]
    tried = {}
    for r in atlas["rows"]:
        for key, value in r["settings"].items():
            entry = tried.setdefault((key, value), {"configs": 0, "sweeps": set()})
            entry["configs"] += 1
            entry["sweeps"].add(r["sweep_id"])
    value_rows = [[k, v, e["configs"], len(e["sweeps"]), " ".join(sorted(e["sweeps"]))]
                  for (k, v), e in sorted(tried.items())]
    return (fixed + atlas["keys"], config_rows), (["setting", "value", "configs", "sweeps", "sweep_ids"], value_rows)


def write(folder, atlas: dict, fmt: str) -> None:
    (config_header, config_rows), (value_header, value_rows) = tables(atlas)
    folder.write_text("configs.csv", csv_text(config_header, config_rows))
    folder.write_text("values.csv", csv_text(value_header, value_rows))
    if fmt == "xlsx":
        import io

        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.title = "configs"
        for row in [config_header] + config_rows:
            sheet.append([str(v) for v in row])
        values = book.create_sheet("values")
        for row in [value_header] + value_rows:
            values.append([str(v) for v in row])
        buffer = io.BytesIO()
        book.save(buffer)
        folder.write_bytes("atlas.xlsx", buffer.getvalue())
