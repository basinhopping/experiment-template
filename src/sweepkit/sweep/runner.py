"""Run one job into its run folder: artifacts/runs/<sweep_id>/<run_id>/, written once.

The job's family runner (the fit section's registered procedure) returns every file of the folder except spec.json
and log.txt; this writes them and completes the folder, or keeps the folder as evidence, renamed .failed-<utc>, when
anything fails. The settings must hash to the job's configuration hash, so an edited queue file cannot run under an
old name.
"""
from sweepkit import paths
from sweepkit.registry import FITS
from sweepkit.runs.spec import config_hash, validate_job
from sweepkit.runs.store import OutputFolder


def execute(spec: dict, offered: str, log, runs_root=None) -> dict:
    """Run one job. Returns the choosing entry of its validation metrics or criteria ({"metric", "value", ...})."""
    job = validate_job(spec["settings"])
    if config_hash(job) != spec["config_hash"]:
        raise ValueError("the settings do not hash to the job's config_hash; the job file was edited")
    parent = paths.runs_dir(spec["sweep_id"]) if runs_root is None else runs_root
    folder = OutputFolder(parent, spec["run_id"]).begin()
    lines = []

    def tee(message: str) -> None:
        lines.append(message)
        log(message)

    try:
        folder.write_json("spec.json", spec)
        files = FITS.get(job.fit_name).target.run(job, spec, offered, tee)
        for name, content in files.items():
            if name.endswith(".json"):
                folder.write_json(name, content)
            elif name.endswith(".npy"):
                folder.write_npy(name, content)
            elif isinstance(content, bytes):
                folder.write_bytes(name, content)
            else:
                folder.write_text(name, content)
        folder.write_text("log.txt", "\n".join(lines) + "\n")
        folder.complete()
    except BaseException as err:
        try:
            folder.write_text("log.txt", "\n".join(lines) + "\n")
        except FileExistsError:
            pass
        folder.fail(err)
        raise
    chosen = files.get("metrics_val.json", files.get("criteria.json", {})).get("select", {})
    return chosen


def describe(chosen: dict) -> str:
    if not chosen:
        return ""
    value = chosen.get("value")
    if value is None:
        return f"{chosen.get('metric')} " + ("from analyze" if chosen.get("computed_by") else "undefined")
    return f"{chosen['metric']} {value:.4f}"
