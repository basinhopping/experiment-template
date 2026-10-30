"""compare-runs: the runs of one sweep against every other copy of the same runs, for exact reproduction.

A run ID is the configuration hash, the fold and the seed, so two folders with the same run ID in different sweeps
fitted the same settings on the same data from the same seed. Deterministic code must give identical files; a sweep
that repeats earlier runs with new code shows whether the new code reproduces the old.

For every complete run folder of the sweep, against every complete folder of the same name under
artifacts/runs/<other sweep>/:
  choosing outputs        predictions_val.csv, predictions_train.csv, assignments.csv, embedding.npy,
                          embedding_rows.csv, removed_rows.csv, scores.csv: byte for byte (SHA-256)
  JSON records            metrics_val.json, metrics_train.json, criteria.json, model_params.json, prep_record.json:
                          as JSON, after removing every key ending in _seconds (wall-clock times differ between runs)
  judging files           metrics_test.json, predictions_test.csv, reveal.json, reveal_reference.csv: by SHA-256 only,
                          never parsed, so no test or reveal value is read or shown
spec.json, efficiency.json, provenance.json and log.txt are not compared (they hold the sweep's name, times and
versions). Each copy's code hash (provenance.json) is reported; a run counts as reproduced only when at least one copy
was made by other code.

Verdicts: SAME (every run of the sweep complete, identical to every copy, each with a copy made by other code; exit
code 0), DIFFERENT (some copy differs, or some run has no copy made by other code; exit code 1), INCOMPLETE (no
difference so far, but some runs are not finished: run it again later; exit code 3), NO COPIES (every run complete and
none has a copy in another sweep, so nothing was compared; exit code 0).
"""
from pathlib import Path

from sweepkit import paths
from sweepkit.analysis.records import (CHECK_RUN_READABLE, JUDGING, JUDGING_PREFIXES, is_check_run, read_json,
                                       sweep_record)
from sweepkit.hashing import sha256_file

HASHED = ("predictions_val.csv", "predictions_train.csv", "assignments.csv", "embedding.npy", "embedding_rows.csv",
          "removed_rows.csv", "scores.csv") + JUDGING
AS_JSON = ("metrics_val.json", "metrics_train.json", "criteria.json", "model_params.json", "prep_record.json")


def _without_times(value):
    if isinstance(value, dict):
        return {k: _without_times(v) for k, v in value.items() if not k.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_times(v) for v in value]
    return value


def differences(a, b, where: str) -> list:
    """Where two parsed JSON values differ. Numbers must match exactly, in value and type."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in b:
                out.append(f"{where}.{key} only in this run")
            elif key not in a:
                out.append(f"{where}.{key} only in the copy")
            else:
                out += differences(a[key], b[key], f"{where}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} entries against {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{where}[{i}]")
        return out
    if type(a) is type(b) and a == b:
        return []
    return [f"{where}: {a!r} against {b!r}"]


def _code_of(folder: Path):
    try:
        return read_json(folder, "provenance.json", "choose").get("code_sha256")
    except (OSError, ValueError):
        return None


def compare_run(sweep_id: str, run_id: str, others: list, shown: int) -> dict:
    mine = paths.runs_dir(sweep_id) / run_id
    code = _code_of(mine)
    check = is_check_run(mine)           # the identity check fold: every result file compared by SHA-256, never read
    hashed = [n for n in HASHED if (mine / n).is_file()] + ([n for n in AS_JSON if (mine / n).is_file()
                                                              and n not in CHECK_RUN_READABLE] if check else [])
    # a torch run's external datasets: judging files named per dataset, compared by SHA-256
    hashed += sorted(p.name for p in mine.iterdir() if p.is_file() and p.name.startswith(JUDGING_PREFIXES))
    as_json = [n for n in AS_JSON if (mine / n).is_file() and (not check or n in CHECK_RUN_READABLE)]
    hashes = {n: sha256_file(mine / n)[1] for n in hashed}
    parsed = {n: _without_times(read_json(mine, n, "choose")) for n in as_json}
    copies = []
    for other in others:
        folder = paths.runs_dir(other) / run_id
        if not folder.is_dir():
            continue
        problems = []
        for n in hashed:
            if not (folder / n).is_file():
                problems.append(f"{n} missing in the copy")
            elif sha256_file(folder / n)[1] != hashes[n]:
                problems.append(f"{n} differs" + (" (compared by SHA-256; not read)"
                                                  if n in JUDGING or n.startswith(JUDGING_PREFIXES) or check else ""))
        found = []
        for n in as_json:
            if not (folder / n).is_file():
                problems.append(f"{n} missing in the copy")
                continue
            found += differences(parsed[n], _without_times(read_json(folder, n, "choose")), n)
        problems += found[:shown] + ([f"... and {len(found) - shown} more differences"] if len(found) > shown else [])
        other_code = _code_of(folder)
        copies.append({"sweep": other, "code_sha256": other_code, "same_code": other_code == code,
                       "identical": not problems, "problems": problems})
    return {"run_id": run_id, "code_sha256": code, "copies": copies, "n_copies": len(copies),
            "n_other_code": sum(not c["same_code"] for c in copies), "identical": all(c["identical"] for c in copies)}


def compare_sweep(sweep_id: str, shown: int) -> dict:
    expected = [row["run_id"] for row in sweep_record(sweep_id)["jobs"]]
    root = paths.all_runs_dir()
    others = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != sweep_id) if root.is_dir() else []
    results, missing = [], []
    for run_id in expected:
        if (paths.runs_dir(sweep_id) / run_id).is_dir():
            results.append(compare_run(sweep_id, run_id, others, shown))
        else:
            missing.append(run_id)
    differing = [r["run_id"] for r in results if not r["identical"]]
    without_other_code = [r["run_id"] for r in results if r["n_other_code"] == 0]
    no_copies = bool(results) and not any(r["n_copies"] for r in results)
    if differing:
        verdict = "DIFFERENT"
    elif no_copies:
        verdict = "INCOMPLETE" if missing else "NO COPIES"
    elif without_other_code:
        verdict = "DIFFERENT"
    elif missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "SAME"
    versions = {}
    for r in results:
        for c in r["copies"]:
            versions.setdefault(str(c["code_sha256"]), set()).add(c["sweep"])
    return {"sweep_id": sweep_id, "verdict": verdict, "n_expected": len(expected), "n_complete": len(results),
            "missing": missing, "differing": differing, "without_other_code": without_other_code,
            "n_copies": sum(r["n_copies"] for r in results), "n_other_code": sum(r["n_other_code"] for r in results),
            "code_versions": {k: sorted(v) for k, v in versions.items()}, "runs": results}


def render(report: dict) -> str:
    lines = [f"compare-runs {report['sweep_id']}: {report['n_complete']} of {report['n_expected']} runs complete, "
             f"{report['n_copies']} copies in other sweeps ({report['n_other_code']} made by other code)"]
    for code, sweeps in sorted(report["code_versions"].items()):
        lines.append(f"  copies made by code {code[:12]}: {', '.join(sweeps)}")
    shown = 0
    for r in report["runs"] if report["verdict"] != "NO COPIES" else []:
        differ = [c for c in r["copies"] if not c["identical"]]
        if not differ and r["n_other_code"] > 0:
            continue
        shown += 1
        if shown > 10:
            continue
        state = "DIFFERENT" if differ else ("identical" if r["n_copies"] else "no copy")
        lines.append(f"  {r['run_id']}: {r['n_copies']} copies, {r['n_other_code']} by other code, {state}")
        if differ:
            first = differ[0]
            for problem in first["problems"]:
                lines.append(f"    vs {first['sweep']} (code {str(first['code_sha256'])[:12]}): {problem}")
            if len(differ) > 1:
                lines.append(f"    also differs from the copies in {', '.join(c['sweep'] for c in differ[1:])}")
    if shown > 10:
        lines.append(f"  ... and {shown - 10} more runs that differ or have no copy made by other code")
    if report["missing"]:
        lines.append(f"  not finished yet: {len(report['missing'])} runs, for example {report['missing'][0]}")
    if report["without_other_code"] and report["verdict"] not in ("NO COPIES", "INCOMPLETE"):
        lines.append(f"  {len(report['without_other_code'])} runs have no copy made by other code, so nothing shows "
                     f"they reproduce earlier code (for example {report['without_other_code'][0]})")
    if report["verdict"] == "SAME":
        lines.append(f"SAME: all {report['n_complete']} runs are identical to every copy ({report['n_copies']} copies, "
                     f"{report['n_other_code']} made by other code). Judging files compared by SHA-256 only.")
    elif report["verdict"] == "NO COPIES":
        lines.append(f"NO COPIES: none of the {report['n_complete']} runs has a copy in another sweep, so nothing was "
                     "compared. compare-runs checks a sweep that repeats earlier runs (the same configurations, folds "
                     "and seeds, so the same run IDs).")
    elif report["verdict"] == "INCOMPLETE":
        lines.append(f"INCOMPLETE: no difference so far, but {len(report['missing'])} runs are not finished; run the "
                     "same command again later")
    else:
        lines.append(f"DIFFERENT: {len(report['differing'])} runs differ from a copy; {len(report['without_other_code'])}"
                     " have no copy made by other code")
    return "\n".join(lines)
