"""Choose first, then read what judges the choice.

    select         records the chosen configuration of an analyzed sweep in artifacts/selections/<name>/
                   selection.json: the configuration, the reason, its choosing values, the analysis it came from and
                   the UTC time. It refuses without a finished analyze report that covers every complete run of the
                   sweep, never overwrites a record, and never opens a test or reveal file.
    test_report    supervised: the test files (metrics_test.json, predictions_test.csv) of the selected
                   configuration's runs only: per seed the fold mean of every task metric, t intervals over seeds,
                   per-class detail and group breakdowns.
    reveal_report  unsupervised: the reveal files (reveal.json) of the selected configuration's runs only: agreement
                   with the reference labels per seed, with t intervals over seeds.
Both reports check the discipline first (sweepkit.analysis.rules): a RULE covering the sweep, fixed before the
selection was recorded.

Nothing here retrains. Test predictions come from the models the runs saved them from (the fit section's refit
setting says which).
"""
import numpy as np

from sweepkit.analysis.records import (complete_runs, complete_seeds, configurations, find_selection, is_supervised,
                                       latest, read_json, sweep_record, task_of)
from sweepkit.analysis.rules import covering_rule, enforce, late_note, now_utc
from sweepkit.analysis.transductive import note as transductive_note
from sweepkit.analysis.transductive import parts as transductive_parts
from sweepkit.splits.base import CHECK_FOLD
from sweepkit.eval.stats import shown, t_interval
from sweepkit.hashing import sha256_file
from sweepkit import paths


def selection_record(sweep_id: str, config_prefix: str, name: str, reason: str) -> dict:
    """What select writes. Reads the analysis report and the sweep's record, nothing that judges."""
    if len(config_prefix) < 8:
        raise ValueError("give at least 8 characters of the configuration hash")
    if not reason.strip():
        raise ValueError("give the reason for the choice in one sentence")
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"selection name {name!r}: letters, digits, _ and - only")
    folder, report = latest(paths.analysis_dir(), sweep_id, "report.json")
    if report is None:
        raise ValueError(f"no finished analyze report for {sweep_id}: run analyze first, then choose")
    record = sweep_record(sweep_id)
    runs = complete_runs(sweep_id)
    if len(runs) != report["n_runs"]:
        raise ValueError(f"{folder.name} analyzed {report['n_runs']} runs, and {len(runs)} are complete now: run "
                         "analyze again so the choice sees every result")
    configs = configurations(record, runs)
    matches = [h for h in configs if h.startswith(config_prefix)]
    if len(matches) != 1:
        raise ValueError(f"{config_prefix!r} matches {len(matches)} configurations of {sweep_id}")
    chosen = configs[matches[0]]
    summary = next((s for s in report["ranked"] if s["config_hash"] == matches[0]), None)
    if summary is None:
        raise ValueError(f"{matches[0][:12]} is not ranked in {folder.name} (a control, no complete seed, or no "
                         "defined choosing value)")
    _, report_sha = sha256_file(folder / "report.json")
    task_name, _ = task_of(record["sweep_file"])
    keep = ("value", "interval", "seeds", "metrics", "stability", "by_group", "versus_baseline")
    return {"name": name, "sweep_id": sweep_id, "config_hash": matches[0], "stage": chosen["stage"],
            "label": chosen["label"], "family": chosen["family"], "settings": chosen["settings"],
            "reason": reason.strip(), "selected_utc": now_utc(), "task": task_name, "select": report["select"],
            "transductive": transductive_parts(chosen["settings"]),
            "choosing": {k: summary[k] for k in keep if k in summary},
            "rank": 1 + report["ranked"].index(summary), "n_ranked": len(report["ranked"]),
            "analysis_report": {"folder": folder.name, "sha256": report_sha},
            "baseline": report["baseline"], "folds": report["folds"]}


def _discipline(selection: dict, what: str) -> tuple:
    folder, rule = covering_rule(selection["sweep_id"], selection["selected_utc"])
    problems = [] if rule else [f"no RULE covering sweep {selection['sweep_id']} was fixed before selection "
                                f"{selection['name']} was recorded ({selection['selected_utc']}); a RULE covers "
                                "only choices recorded after it: fix one with rule-fix, then record the choice again "
                                "with select under a new name (a selection is written once); rule-fix asks for "
                                "a reason (--late-reason) when runs of the sweep exist already"]
    notes = [late_note(rule)] if rule and late_note(rule) else []
    return enforce(problems, what) + notes, (folder.name if folder else None)


def _selected_runs(selection: dict) -> tuple:
    record = sweep_record(selection["sweep_id"])
    config = configurations(record, complete_runs(selection["sweep_id"]))[selection["config_hash"]]
    folds = list(selection["folds"])
    seeds = complete_seeds(config, folds)
    if not seeds:
        raise ValueError("no seed of the selected configuration has every fold complete")
    incomplete = sorted(set(config["runs"]) - set(seeds))
    return config, folds, seeds, incomplete


def _summaries(per_seed: dict, names: list, confidence: float) -> dict:
    out = {}
    seeds = sorted(per_seed)
    for m in names:
        values = [per_seed[s][m] for s in seeds]
        if any(v is None for v in values):
            out[m] = {"mean": None, "interval": None, "undefined_in_seeds": [s for s in seeds if per_seed[s][m] is None]}
            continue
        out[m] = {"mean": float(np.mean(values)), "interval": t_interval(values, confidence) if len(values) >= 2 else None}
    return out


def test_report(selection_name: str, confidence: float) -> dict:
    selection = find_selection(selection_name)
    if not is_supervised(selection["task"]):
        raise ValueError(f"{selection_name} is a {selection['task']} selection: use reveal-report")
    warnings, rule = _discipline(selection, "test-report")
    config, folds, seeds, incomplete = _selected_runs(selection)
    task = task_of(sweep_record(selection["sweep_id"])["sweep_file"])[1]
    names = list(task["metrics"])
    per_seed, by_fold, details, groups = {}, {}, {}, {}
    for seed in seeds:
        bodies = [read_json(config["runs"][seed][f]["folder"], "metrics_test.json", "judge") for f in folds]
        per_seed[seed] = {m: float(np.mean([b["metrics"][m] for b in bodies])) for m in names}
        by_fold[seed] = {f: b["metrics"][selection["select"]["metric"]] for f, b in zip(folds, bodies)}
        details[seed] = [b["detail"] for b in bodies]
        for column in task.get("groups", []):
            for f, b in zip(folds, bodies):
                for group, entry in b["by_group"].get(column, {}).items():
                    value = None if entry["metrics"] is None else entry["metrics"][selection["select"]["metric"]]
                    groups.setdefault(column, {}).setdefault(group, {}).setdefault(seed, []).append(value)
    out = {"selection": {k: selection[k] for k in ("name", "sweep_id", "config_hash", "label", "reason",
                                                   "selected_utc")},
           "warnings": warnings, "rule": rule, "confidence": confidence, "seeds": seeds,
           "transductive": transductive_parts(selection["settings"]),
           "incomplete_seeds": incomplete, "metrics": _summaries(per_seed, names, confidence),
           "per_seed": per_seed, "select_by_fold": by_fold, "select": selection["select"]}
    if selection["task"] == "classification":
        recalls = [d["recall"] for seed in seeds for d in details[seed]]
        classes = list(recalls[0])
        out["recall"] = {c: (None if any(r[c] is None for r in recalls) else float(np.mean([r[c] for r in recalls])))
                         for c in classes}
    out["by_group"] = {}
    for column, by_group in groups.items():
        out["by_group"][column] = {}
        for group, by_seed in sorted(by_group.items()):
            means = [None if len(v) != len(folds) or None in v else float(np.mean(v)) for v in
                     (by_seed.get(s, []) for s in seeds)]
            out["by_group"][column][group] = None if None in means else float(np.mean(means))
    out["identity_check"] = identity_check(config, names)
    return out


def identity_check(config: dict, names: list):
    """The identity check fold's runs of the configuration, if the sweep ran it: on that fold the validation rows ARE
    the test rows, so with refit train the validation and test numbers must agree exactly (the two evaluation paths
    give the same numbers). None when the sweep did not run the fold."""
    runs = {seed: by_fold[CHECK_FOLD] for seed, by_fold in config["runs"].items() if CHECK_FOLD in by_fold}
    if not runs:
        return None
    if config["settings"]["fit"].get("refit", "train") != "train":
        return {"runs": len(runs), "compared": False,
                "note": "refit is not train: the test rows are predicted by another model, so the numbers differ"}
    differing = []
    for seed, run in sorted(runs.items()):
        val = read_json(run["folder"], "metrics_val.json", "judge")["metrics"]
        test = read_json(run["folder"], "metrics_test.json", "judge")["metrics"]
        for m in names:
            a, b = val.get(m), test.get(m)
            if not (a == b or (a is not None and b is not None and abs(a - b) <= 1e-12 * max(1.0, abs(a)))):
                differing.append(f"seed {seed} {m}: validation {a!r}, test {b!r}")
    return {"runs": len(runs), "compared": True, "agree": not differing, "differing": differing}


def reveal_report(selection_name: str, confidence: float) -> dict:
    selection = find_selection(selection_name)
    if is_supervised(selection["task"]):
        raise ValueError(f"{selection_name} is a {selection['task']} selection: use test-report")
    warnings, rule = _discipline(selection, "reveal-report")
    config, folds, seeds, incomplete = _selected_runs(selection)
    task = task_of(sweep_record(selection["sweep_id"])["sweep_file"])[1]
    if task["reference"] is None:
        raise ValueError(f"sweep {selection['sweep_id']} names no reference column, so its runs wrote no reveal file")
    names = list(task["reveal_metrics"])
    per_seed, by_fold, n_reference = {}, {}, {}
    for seed in seeds:
        bodies = [read_json(config["runs"][seed][f]["folder"], "reveal.json", "judge") for f in folds]
        per_seed[seed] = {}
        for m in names:
            values = [b["metrics"][m] for b in bodies]
            per_seed[seed][m] = None if None in values else float(np.mean(values))
        by_fold[seed] = {f: b["metrics"] for f, b in zip(folds, bodies)}
        n_reference[seed] = {f: b["n_with_reference"] for f, b in zip(folds, bodies)}
    return {"selection": {k: selection[k] for k in ("name", "sweep_id", "config_hash", "label", "reason",
                                                    "selected_utc")},
            "warnings": warnings, "rule": rule, "reference": task["reference"], "confidence": confidence,
            "transductive": transductive_parts(selection["settings"]),
            "seeds": seeds, "incomplete_seeds": incomplete, "metrics": _summaries(per_seed, names, confidence),
            "per_seed": per_seed, "by_fold": by_fold, "n_with_reference": n_reference,
            "choosing": selection["choosing"].get("value"), "select": selection["select"]}


def _line(name: str, m: dict, confidence: float) -> str:
    if m["mean"] is None:
        return f"  {name} undefined (seeds {m['undefined_in_seeds']})"
    ci = m["interval"]
    # an SD of identical values comes out of the floating sums as about 1e-17: shown against the mean, it is 0
    return f"  {name} {shown(m['mean'])}" + (f" [{shown(ci['low'])}, {shown(ci['high'])}] ({round(confidence * 100)}% "
                                            f"t interval over seeds), SD {shown(ci['sd'], scale=m['mean'])}"
                                            if ci else " (one seed)")


def render(report: dict, kind: str) -> str:
    sel = report["selection"]
    lines = list(report["warnings"])
    lines.append(f"{kind} results of selection {sel['name']}: {sel['sweep_id']} / {sel['config_hash'][:12]} "
                 f"({sel['label']}), seeds {report['seeds']}")
    lines.append(f"chosen because: {sel['reason']} (recorded {sel['selected_utc']}; RULE {report['rule']})")
    if report.get("transductive"):
        lines.append("NOTE the chosen configuration is " + transductive_note(report["transductive"]))
    if report["incomplete_seeds"]:
        lines.append(f"left out, not every fold complete: seeds {report['incomplete_seeds']}")
    for name, m in report["metrics"].items():
        lines.append(_line(name, m, report["confidence"]))
    if report.get("recall"):
        lines.append("  recall " + "  ".join(f"{c} {shown(v, 3)}"
                                            for c, v in report["recall"].items()))
    for column, groups in (report.get("by_group") or {}).items():
        lines.append(f"  {report['select']['metric']} by {column}: "
                     + "  ".join(f"{g} {shown(v)}" for g, v in groups.items()))
    if "select_by_fold" in report:
        for seed in report["seeds"]:
            lines.append(f"  seed {seed}: " + "  ".join(f"{f} {shown(v)}"
                                                        for f, v in report["select_by_fold"][seed].items()))
    check = report.get("identity_check")
    if check is not None:
        if not check["compared"]:
            lines.append(f"identity check fold: {check['runs']} runs, not compared ({check['note']})")
        elif check["agree"]:
            lines.append(f"identity check fold: validation and test numbers agree in all {check['runs']} runs")
        else:
            lines.append(f"IDENTITY CHECK FAILED: validation and test numbers differ on the fold whose validation "
                         f"rows are its test rows, so the two evaluation paths disagree: "
                         + "; ".join(check["differing"][:5]))
    return "\n".join(lines)
