"""status: the check after a night or a batch. Queue counts, stop files, failures, and the ranking of complete
(configuration, seed) pairs by the choosing value, read from the choosing files only (never a test or reveal file).

A SWAP_FAILED marker (node/swap_and_restart.sh writes queue/SWAP_FAILED when a swap was refused, could not install
the new code, or did not enqueue its sweeps) is printed first, because nothing else in the report can be trusted until
it is understood.

--since limits the runs to those completed at or after a UTC time (as written in each run's provenance.json), so a
morning check can look at last night only.
"""
import statistics

from sweepkit import paths
from sweepkit.analysis.records import (choosing_body, is_supervised, read_json, result_folds, run_folders,
                                       sweep_record, task_of)
from sweepkit.analysis.validation import select_of
from sweepkit.sweep.queue import Queue
from sweepkit.tasks.unsupervised import STABILITY


def _error_line(folder) -> str:
    path = folder / "error.txt"
    if not path.is_file():
        return "(no error.txt)"
    lines = [line for line in path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def summarise(sweep_id: str, since: str | None) -> dict:
    record = sweep_record(sweep_id)
    sweep_file = record["sweep_file"]
    folds = result_folds(sweep_file["folds"])
    task_name, task = task_of(sweep_file)
    select, direction = select_of(task)
    base_stages = {s["name"] for s in sweep_file["stages"] if s["kind"] == "base"}
    folders = run_folders(sweep_id)
    complete = []
    identity = 0
    for folder in folders["complete"]:
        provenance = read_json(folder, "provenance.json", "choose")
        if since is not None and provenance["utc"] < since:
            continue
        spec = read_json(folder, "spec.json", "choose")
        if spec["fold"] not in folds:
            identity += 1                # the identity check fold: complete, but its numbers are test numbers
            continue
        run = {"folder": folder, "spec": spec}
        value = None
        if select != STABILITY:
            body = choosing_body(run, task_name)
            value = (body["metrics"] if is_supervised(task_name) else body["criteria"])[select]
        efficiency = read_json(folder, "efficiency.json", "choose")
        complete.append({"run_id": spec["run_id"], "stage": spec["stage"], "label": spec["label"],
                         "config_hash": spec["config_hash"], "fold": spec["fold"], "seed": spec["seed"],
                         "control": spec["settings"]["fit"].get("control", "none"), "value": value,
                         "seconds": sum(v for k, v in efficiency.items() if k.endswith("_seconds") and v is not None),
                         "device_used": provenance.get("device_used"), "utc": provenance["utc"]})
    groups = {}
    for r in complete:
        g = groups.setdefault((r["config_hash"], r["seed"]), {"stage": r["stage"], "label": r["label"],
                                                               "control": r["control"], "seed": r["seed"],
                                                               "config_hash": r["config_hash"], "folds": {}})
        g["folds"][r["fold"]] = r["value"]
    for g in groups.values():
        g["complete"] = all(f in g["folds"] for f in folds)
        values = [g["folds"].get(f) for f in folds]
        g["fold_mean"] = statistics.fmean(values) if g["complete"] and None not in values else None
    sign = -1 if direction == "max" else 1
    ranked = sorted((g for g in groups.values() if g["fold_mean"] is not None and g["control"] == "none"),
                    key=lambda g: sign * g["fold_mean"])
    base = [g for g in groups.values() if g["stage"] in base_stages and g["fold_mean"] is not None]
    base_by_seed = {g["seed"]: g["fold_mean"] for g in base}
    for g in ranked:
        paired = base_by_seed.get(g["seed"])
        g["minus_base_same_seed"] = g["fold_mean"] - paired if paired is not None else None
    baseline = None
    if base:
        values = [g["fold_mean"] for g in base]
        baseline = {"seeds": len(values), "mean": statistics.fmean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values)}
    chance = None
    if task_name == "classification" and select in ("balanced_accuracy", "recall_macro"):
        chance = 1 / len(task["classes"])
    queue_root = paths.queue_dir()
    swap_failed = (queue_root / "SWAP_FAILED").read_text(encoding="utf-8").strip() \
        if (queue_root / "SWAP_FAILED").is_file() else None
    stops = sorted(p.name for p in queue_root.glob("STOP*")) if queue_root.is_dir() else []
    seconds = [r["seconds"] for r in complete]
    devices = sorted({str(r["device_used"]) for r in complete})
    return {"sweep_id": sweep_id, "since": since, "select": {"metric": select, "direction": direction},
            "folds": folds, "queue": Queue(queue_root).counts(sweep_id), "stop_files": stops,
            "swap_failed": swap_failed, "runs_expected": len(record["jobs"]), "runs_complete": len(complete) + identity,
            "runs_identity": identity,
            "runs_failed": [{"folder": f.name, "error": _error_line(f)} for f in folders["failed"]],
            "runs_in_progress": [f.name for f in folders["in_progress"]],
            "median_seconds": statistics.median(seconds) if seconds else None, "devices_used": devices,
            "baseline": baseline, "chance": chance,
            "controls": [g for g in groups.values() if g["control"] != "none" and g["fold_mean"] is not None],
            "ranked": ranked, "partial": sum(1 for g in groups.values() if not g["complete"])}


def render(s: dict, top: int) -> str:
    lines = []
    if s["swap_failed"] is not None:
        lines.append(f"ALERT SWAP_FAILED: {s['swap_failed']} (queue/SWAP_FAILED, from node/swap_and_restart.sh: "
                     f"read it and the swap's log, then move the file into logs/)")
    lines.append(f"sweep {s['sweep_id']}" + (f", runs completed since {s['since']}" if s["since"] else ""))
    lines.append(f"queue: {s['queue']}" + (f"; stop files: {', '.join(s['stop_files'])}" if s["stop_files"] else ""))
    runs = (f"runs: {s['runs_complete']} complete of {s['runs_expected']}"
            + (f" ({s['runs_identity']} on the identity check fold, whose numbers are not shown)"
               if s["runs_identity"] else "")
            + f", {len(s['runs_failed'])} failed, {len(s['runs_in_progress'])} in progress")
    if s["median_seconds"] is not None:
        runs += f"; median time per run {s['median_seconds']:.1f}s; devices used {', '.join(s['devices_used'])}"
    lines.append(runs)
    for f in s["runs_failed"][:10]:
        lines.append(f"failed {f['folder']}: {f['error']}")
    sel = s["select"]
    if sel["metric"] == STABILITY:
        lines.append("chosen by stability, which analyze computes; run analyze for the ranking")
        return "\n".join(lines)
    b = s["baseline"]
    if b:
        sd = f"{b['sd']:.4f}" if b["sd"] is not None else "n/a"
        lines.append(f"baseline: {b['seeds']} seeds, fold-mean {sel['metric']} {b['mean']:.4f}, SD {sd}, range "
                     f"{b['min']:.4f} to {b['max']:.4f}")
    for g in s["controls"]:
        chance = f" (chance {s['chance']:.4f})" if s["chance"] is not None else ""
        lines.append(f"control {g['label']} seed {g['seed']}: fold-mean {sel['metric']} {g['fold_mean']:.4f}{chance}")
    lines.append(f"top {min(top, len(s['ranked']))} of {len(s['ranked'])} complete (configuration, seed) pairs "
                 f"({s['partial']} partial) by fold-mean {sel['metric']}:")
    for g in s["ranked"][:top]:
        delta = f"{g['minus_base_same_seed']:+.4f}" if g["minus_base_same_seed"] is not None else "n/a"
        lines.append(f"  {g['fold_mean']:.4f}  vs base {delta}  seed {g['seed']}  {g['config_hash'][:12]} "
                     f"{g['stage']}: {g['label']}")
    return "\n".join(lines)
