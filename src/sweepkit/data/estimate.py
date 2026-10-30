"""estimate: measure what one configuration costs before a large run is queued.

Runs one job of a sweep (its first fold and seed) through the real runner on growing parts of the data (each role cut
to a fraction of the fold's fitting rows, chosen by SHA-256 as smoke does), from the smallest fraction up, and measures
for each: wall time, peak memory of this process, and the bytes of the run folder. A first run on the smallest part
is not measured: it pays for imports, compilation and caches, which would otherwise make the smallest part look slow.
Then it fits
    time   = a * rows^b       (a straight line on log scales; b near 1 is linear, near 2 quadratic)
    time   = c + d * rows     (a fixed cost plus a cost per row)
    memory = c + d * rows     (c is what Python, the libraries and the loaded data hold anyway)
    disk   = e + g * rows
and predicts one job on the full fold (time: the larger of the two time models; neither prediction falls below what
the largest part measured), the whole sweep (the jobs times one job: configurations that cost more, such as more
clusters or more epochs, are not modelled), and what --workers jobs at once need, against this machine's free disk
and free memory, with a margin of 1.5 times the predicted memory. The predictions are approximate, and the memory
tends to be low when it extrapolates far (in three runs of tests/test_large_data.py at a million rows, from parts of
up to 50,000, the full run took 1.35 to 1.4 times the predicted memory and 0.8 to 1.5 times the predicted time), so
the report says how far it extrapolates, flags a prediction more than 20 times beyond the largest part, and says to
plan with the margin.

Peak memory per part: on Linux the kernel's high-water mark is reset before each part (/proc/self/clear_refs), so
each part's peak is its own; elsewhere the process's peak so far is used (parts run from small to large, so each
value is an upper bound for its part).

Writes artifacts/estimates/<sweep_id>_<utc>/: report.json, report.txt and the parts' run folders.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from sweepkit import paths
from sweepkit.families.common import folds_of, load_data, peak_memory_mb
from sweepkit.runs.spec import validate_job
from sweepkit.runs.store import OutputFolder, stamp
from sweepkit.sweep.design import expand
from sweepkit.sweep.enqueue import load_sweep_file
from sweepkit.sweep.runner import execute


def _reset_peak() -> str:
    """Reset the peak-memory counter where the system allows it; returns how peaks are measured."""
    clear = Path("/proc/self/clear_refs")
    if sys.platform.startswith("linux") and clear.exists():
        try:
            clear.write_text("5", encoding="ascii")
            return "linux high-water mark, reset before each part"
        except OSError:
            pass
    return "the process's peak so far (an upper bound for each part)"


def _peak_mb() -> float | None:
    status = Path("/proc/self/status")
    if sys.platform.startswith("linux") and status.exists():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return peak_memory_mb()


def _free_memory_mb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 2 ** 20
    except ImportError:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return None


def _folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def fit(rows: list, values: list, kind: str) -> dict:
    x, y = np.asarray(rows, dtype=float), np.asarray(values, dtype=float)
    if kind == "power":
        b, log_a = np.polyfit(np.log(x), np.log(np.maximum(y, 1e-6)), 1)
        return {"form": "a * rows^b", "a": float(np.exp(log_a)), "b": float(b)}
    d, c = np.polyfit(x, y, 1)
    return {"form": "c + d * rows", "c": float(c), "d": float(d)}


def predict(model: dict, rows: float) -> float:
    if model["form"] == "a * rows^b":
        return model["a"] * rows ** model["b"]
    return max(model["c"] + model["d"] * rows, 0.0)



def estimate(sweep_file: Path, config: str, offered: str, fractions: list, workers: int, log) -> dict:
    if len(fractions) < 2 or sorted(fractions) != list(fractions) or len(set(fractions)) != len(fractions) \
            or not all(0 < f <= 1 for f in fractions):
        raise ValueError("--fractions: at least two increasing fractions in (0, 1], for example 0.01 0.02 0.05")
    if workers < 1:
        raise ValueError("--workers: at least 1")
    sweep = load_sweep_file(sweep_file)
    jobs, summary = expand(sweep)
    wanted = summary["base_config_hash"] if config == "base" else config
    matches = sorted({j["config_hash"] for j in jobs if j["config_hash"].startswith(wanted)})
    if len(matches) != 1 or len(wanted) < 8:
        raise ValueError(f"--config {config!r}: base, or 8 or more characters of one configuration hash "
                         f"({len(matches)} match)")
    job = next(j for j in jobs if j["config_hash"] == matches[0])
    validated = validate_job(job["settings"])
    data = load_data(validated, log)
    folds, _ = folds_of(validated, data)
    roles = folds[job["fold"]].roles
    fit_role = "train" if "train" in roles else "fit"
    full = int(roles[fit_role].size)
    folder = OutputFolder(paths.estimates_dir(), f"{summary['sweep_id']}_{stamp()}").begin()
    parts = []
    method = None
    try:
        warm = max(10, int(round(fractions[0] * full)))
        execute(dict(job, smoke_rows=warm), offered, lambda m: None, runs_root=folder.tmp / "runs" / "warm_up")
        log(f"  warm-up on {warm} rows (not measured): imports, compilation and caches")
        for k, fraction in enumerate(fractions):
            n = max(10, int(round(fraction * full)))
            spec = dict(job, smoke_rows=n)
            method = _reset_peak()
            runs_root = folder.tmp / "runs" / f"part{k + 1}_{n}_rows"
            started = time.perf_counter()
            execute(spec, offered, lambda m: None, runs_root=runs_root)
            seconds = time.perf_counter() - started
            part = {"fraction": fraction, "rows": n, "seconds": seconds, "peak_memory_mb": _peak_mb(),
                    "folder_bytes": _folder_bytes(runs_root)}
            parts.append(part)
            log(f"  {n} rows: {seconds:.2f}s, peak memory {part['peak_memory_mb']} MB, "
                f"{part['folder_bytes'] / 1e6:.2f} MB written")
        rows = [p["rows"] for p in parts]
        seconds = [p["seconds"] for p in parts]
        time_model = fit(rows, seconds, "power")
        time_linear = fit(rows, seconds, "linear")
        out = {"sweep_id": summary["sweep_id"], "config_hash": matches[0], "label": job["label"],
               "fold": job["fold"], "seed": job["seed"], "fit_role": fit_role, "full_rows": full,
               "jobs_in_sweep": summary["jobs"], "device": offered, "workers": workers, "parts": parts,
               "memory_measured_as": method, "time_model": time_model, "time_linear_model": time_linear,
               "extrapolation": full / max(rows)}
        per_job = {"seconds": max(predict(time_model, full), predict(time_linear, full), max(seconds))}
        if all(p["peak_memory_mb"] is not None for p in parts):
            out["memory_model"] = fit(rows, [p["peak_memory_mb"] for p in parts], "linear")
            per_job["peak_memory_mb"] = max(predict(out["memory_model"], full),
                                            max(p["peak_memory_mb"] for p in parts))
        out["disk_model"] = fit(rows, [p["folder_bytes"] for p in parts], "linear")
        per_job["bytes"] = max(predict(out["disk_model"], full), max(p["folder_bytes"] for p in parts))
        out["per_job"] = per_job
        out["sweep"] = {"seconds_one_worker": per_job["seconds"] * summary["jobs"],
                        "seconds_with_workers": per_job["seconds"] * summary["jobs"] / workers,
                        "bytes": per_job["bytes"] * summary["jobs"]}
        free_disk = shutil.disk_usage(paths.repo_root()).free
        free_memory = _free_memory_mb()
        out["machine"] = {"free_disk_bytes": free_disk, "free_memory_mb": free_memory}
        problems = []
        if out["extrapolation"] > 20:        # a prediction this far beyond the largest part is flagged
            problems.append(f"the full fold has {out['extrapolation']:.0f} times the rows of the largest part: the "
                            "prediction extrapolates far; measure larger fractions (for example 0.05 0.1 0.2) when "
                            "the job is long")
        if out["sweep"]["bytes"] > free_disk:
            problems.append(f"the sweep would write about {out['sweep']['bytes'] / 1e9:.1f} GB and "
                            f"{free_disk / 1e9:.1f} GB are free")
        margin = 1.5                 # the predictions run low when they extrapolate (module docstring)
        out["memory_margin"] = margin
        if free_memory is not None and "peak_memory_mb" in per_job \
                and per_job["peak_memory_mb"] * margin * workers > free_memory:
            problems.append(f"{workers} jobs at once would need about "
                            f"{per_job['peak_memory_mb'] * margin * workers:.0f} MB with the margin of {margin} times "
                            f"the prediction, and {free_memory:.0f} MB are free")
        out["problems"] = problems
        folder.write_json("report.json", out)
        text = render(out)
        folder.write_text("report.txt", text + "\n")
    except BaseException as err:
        folder.fail(err)
        raise
    out["folder"] = str(folder.complete())
    return out


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min" if seconds < 7200 else f"{seconds / 3600:.1f} h"


def render(r: dict) -> str:
    t = r["time_model"]
    lines = [f"estimate for {r['sweep_id']} / {r['config_hash'][:12]} ({r['label']}), fold {r['fold']}, seed "
             f"{r['seed']}, on {r['device']}: {r['full_rows']} {r['fit_role']} rows in the full fold"]
    for p in r["parts"]:
        mem = "n/a" if p["peak_memory_mb"] is None else f"{p['peak_memory_mb']:.0f} MB"
        lines.append(f"  {p['rows']} rows: {p['seconds']:.2f}s, peak memory {mem}, {p['folder_bytes'] / 1e6:.2f} MB "
                     "written")
    lines.append(f"time grows as rows^{t['b']:.2f} (and {r['time_linear_model']['d'] * 1e6:.1f} s per million rows "
                 f"after a fixed {r['time_linear_model']['c']:.2f} s); the full fold is {r['extrapolation']:.1f} "
                 f"times the largest part; memory measured as {r['memory_measured_as']}")
    j = r["per_job"]
    lines.append(f"one full job: about {_duration(j['seconds'])}"
                 + (f", {j['peak_memory_mb']:.0f} MB peak memory" if "peak_memory_mb" in j else "")
                 + f", {j['bytes'] / 1e6:.1f} MB written")
    if "peak_memory_mb" in j:
        margin = r.get("memory_margin", 1.5)
        lines.append(f"these are approximate, and the memory tends to be low when it extrapolates far: plan workers "
                     f"with {margin} times the memory, about {j['peak_memory_mb'] * margin:.0f} MB a job")
    s = r["sweep"]
    lines.append(f"the sweep's {r['jobs_in_sweep']} jobs: about {s['seconds_one_worker'] / 3600:.2f} h with one worker, "
                 f"{s['seconds_with_workers'] / 3600:.2f} h with {r['workers']}, {s['bytes'] / 1e9:.2f} GB written "
                 "(every job costed as this configuration)")
    m = r["machine"]
    lines.append(f"this machine: {m['free_disk_bytes'] / 1e9:.1f} GB free disk"
                 + (f", {m['free_memory_mb']:.0f} MB free memory" if m["free_memory_mb"] is not None else ""))
    for problem in r["problems"]:
        lines.append(f"PROBLEM {problem}")
    if not r["problems"]:
        lines.append("no problem found with disk or memory")
    return "\n".join(lines)
