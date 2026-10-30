"""smoke: run before enqueueing a sweep, so a setting that crashes fails in minutes, not overnight.

1. Expands the sweep file exactly as enqueue would (nothing is queued, nothing is recorded under artifacts/sweeps/).
2. Runs every distinct configuration once, through the same family runner a real job uses, on its first fold and
   first seed, with each role cut to --rows rows (a fixed choice of rows by SHA-256), and with the fit section's own
   shortening where the family has one (the torch family: two epochs of a few batches). Run folders go to
   artifacts/smoke/<sweep_id>_<utc>/runs/, never among the sweep's real runs.
3. Flags configurations that built identical models and preparation and gave identical outputs (the same
   model_params.json, prep_record.json and validation predictions, assignments, embedding or scores): a setting that
   changes nothing usually means a typo or a setting the method ignores. A control that only shuffles the
   training labels builds the same model but gives other outputs, so it is not flagged.
4. Writes artifacts/smoke/<sweep_id>_<utc>/table.csv (one row per configuration: status, seconds, error) and
   summary.json, and ends with SMOKE OK or SMOKE FAILED.

It never opens a test or reveal file (it only reads model_params.json and prep_record.json back).
"""
import copy
import time
import traceback

from sweepkit import paths
from sweepkit.analysis.records import OUTPUT_FILE, file_path, read_json
from sweepkit.families.common import csv_text
from sweepkit.hashing import canonical_json, sha256_file, sha256_text
from sweepkit.registry import FITS
from sweepkit.runs.spec import config_hash, plain, run_id, validate_job
from sweepkit.runs.store import OutputFolder, stamp
from sweepkit.splits.base import CHECK_FOLD
from sweepkit.sweep.design import expand
from sweepkit.sweep.enqueue import load_sweep_file
from sweepkit.sweep.runner import describe, execute


def smoke_spec(job: dict, rows: int) -> dict:
    """The job as smoke runs it: the fit section's shortening (if its family has one) and smoke_rows."""
    settings = copy.deepcopy(job["settings"])
    shorten = getattr(FITS.get(settings["fit"]["name"]).target, "smoke_settings", None)
    if shorten is not None:
        settings = shorten(settings)
    validated = validate_job(settings)
    return dict(job, settings=plain(validated), config_hash=config_hash(validated),
                run_id=run_id(validated, job["fold"], job["seed"]), smoke_rows=rows, stage=job["stage"],
                label=job["label"])


def smoke(sweep_file, offered: str, rows: int, log) -> dict:
    if rows < 10:
        raise ValueError("--rows: at least 10 rows per role")
    sweep = load_sweep_file(sweep_file)
    jobs, summary = expand(sweep)
    log(f"sweep {summary['sweep_id']}: {summary['configs']} configurations, {summary['jobs']} jobs")
    for stage in summary["stages"]:
        log(f"  stage {stage['name']} ({stage['kind']}): {stage['configs']} configurations x {len(stage['seeds'])} "
            f"seeds, priority {stage['priority']}")
    distinct = {}
    for job in jobs:
        # one job per configuration, on a fold other than the identity check fold when the sweep has one
        kept = distinct.get(job["config_hash"])
        if kept is None or (kept["fold"] == CHECK_FOLD and job["fold"] != CHECK_FOLD):
            distinct[job["config_hash"]] = job
    folder = OutputFolder(paths.smoke_dir(), f"{summary['sweep_id']}_{stamp()}").begin()
    runs_root = folder.tmp / "runs"
    table, digests, failed = [], {}, 0
    for n, (h, job) in enumerate(distinct.items(), 1):
        started = time.perf_counter()
        try:
            spec = smoke_spec(job, rows)
            chosen = execute(spec, offered, lambda m: None, runs_root=runs_root)
            seconds = time.perf_counter() - started
            run_folder = runs_root / spec["run_id"]
            if job["fold"] != CHECK_FOLD:    # a check-fold run's files are never read by a step that chooses
                params = read_json(run_folder, "model_params.json", "choose")
                prep = read_json(run_folder, "prep_record.json", "choose")
                _, output = sha256_file(file_path(run_folder, OUTPUT_FILE[job["family"]], "choose"))
                digest = sha256_text(canonical_json({"params": params, "prep": prep["steps"],
                                                     "columns": prep["columns"], "output": output}))
                digests.setdefault(digest, []).append(h)
            table.append([h, job["stage"], job["label"], job["family"], "ok", f"{seconds:.2f}", describe(chosen), ""])
            if n <= 3 or n == len(distinct) or n % 25 == 0:
                log(f"  {n:4d} ok  {seconds:6.2f}s  {job['stage']}: {job['label']} ({describe(chosen)})")
        except Exception as err:
            failed += 1
            seconds = time.perf_counter() - started
            table.append([h, job["stage"], job["label"], job["family"], "FAILED", f"{seconds:.2f}", "",
                          f"{type(err).__name__}: {err}"])
            log(f"  {n:4d} FAILED  {job['stage']}: {job['label']}")
            log(traceback.format_exc())
    same = [sorted(g) for g in digests.values() if len(g) > 1]
    for group in same:
        log("NOTE identical models, preparation and outputs from: " + ", ".join(x[:12] for x in group)
            + " (a setting that changes nothing?)")
    result = {"sweep_id": summary["sweep_id"], "configs": len(distinct), "failed": failed, "rows": rows,
              "device": offered, "identical": same, "folder": None}
    try:
        folder.write_text("table.csv", csv_text(["config_hash", "stage", "label", "family", "status", "seconds",
                                                 "choosing", "error"], table))
        folder.write_json("summary.json", {k: v for k, v in result.items() if k != "folder"})
    except BaseException as err:
        folder.fail(err)
        raise
    result["folder"] = str(folder.complete())
    slowest = sorted(table, key=lambda r: -float(r[5]))[:3]
    log("slowest: " + "; ".join(f"{r[0][:12]} {r[5]}s" for r in slowest))
    log(f"saved {result['folder']}")
    log("SMOKE OK" if failed == 0 else f"SMOKE FAILED ({failed} of {len(distinct)} configurations)")
    return result
