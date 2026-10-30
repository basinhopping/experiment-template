"""What every family's runner shares: the dataset and its folds (cached in the process, so a worker running many jobs
of one protocol reads the data once), the preparation steps, timing and memory, and the files every run folder
gets."""
import csv
import io
import os
import sys
import time

try:
    import resource
except ImportError:  # Windows has no resource module; peak memory is then not recorded
    resource = None

import numpy as np

from sweepkit.hashing import canonical_json, sha256_text
from sweepkit.prep.base import PrepContext, run_prep
from sweepkit.registry import DATASETS, SPLITS
from sweepkit.runs.spec import Job, plain
from sweepkit.settings import to_plain
from sweepkit.splits.base import folds_hash

_DATA_CACHE = {}
_FOLD_CACHE = {}


def data_key(job: Job) -> str:
    return sha256_text(canonical_json(plain(job)["data"]))


def load_data(job: Job, log):
    key = data_key(job)
    if key not in _DATA_CACHE:
        start = time.perf_counter()
        data = DATASETS.get(job.data_name).target().load(job.data)
        if len(_DATA_CACHE) >= 2:
            _DATA_CACHE.pop(next(iter(_DATA_CACHE)))
        _DATA_CACHE[key] = data
        log(f"data {job.data_name}: {len(data.table)} rows, kind {data.kind}"
            + (f", {data.x.shape[1]} columns" if data.x is not None else "")
            + f", loaded in {time.perf_counter() - start:.1f}s")
    return _DATA_CACHE[key]


def folds_of(job: Job, data) -> dict:
    """name -> Fold for the job's split, checked against the scheme's declared fold names."""
    key = (data_key(job), sha256_text(canonical_json(plain(job)["split"])))
    if key not in _FOLD_CACHE:
        scheme = SPLITS.get(job.split_name).target
        folds = scheme.make(job.split, data.table)
        names = tuple(f.name for f in folds)
        if names != tuple(scheme.fold_names(job.split)):
            raise RuntimeError(f"split {job.split_name} made folds {names}, but declares {scheme.fold_names(job.split)}")
        _FOLD_CACHE.clear()
        _FOLD_CACHE[key] = ({f.name: f for f in folds}, folds_hash(data.table.ids, list(folds)))
    return _FOLD_CACHE[key]


def clear_caches() -> None:
    _DATA_CACHE.clear()
    _FOLD_CACHE.clear()


def prepare(job: Job, data, positions_by_role: dict, fit_role: str, seed: int, device: str, log) -> tuple:
    """Tabular inputs of each role after the job's preparation steps (fitted on fit_role)."""
    x_by_role = {role: np.asarray(data.x[p], dtype=np.float32) for role, p in positions_by_role.items()}
    context = PrepContext(table=data.table, feature_names=tuple(data.feature_names), categorical=dict(data.categorical),
                          seed=seed, device=device)
    x_by_role, positions_by_role, names, categorical, records = run_prep(job.prep, x_by_role, positions_by_role,
                                                                         fit_role, context, log)
    x_by_role = {role: np.asarray(x, dtype=np.float32) for role, x in x_by_role.items()}
    return x_by_role, positions_by_role, names, categorical, records


def require_finite(x: np.ndarray, where: str) -> None:
    bad = ~np.isfinite(x)
    if bad.any():
        rows = int(bad.any(axis=1).sum()) if x.ndim == 2 else int(bad.sum())
        raise ValueError(f"{where}: {rows} rows hold missing or infinite values; add an impute step (or drop_rows) "
                         "to the prep list")


def peak_memory_mb() -> float | None:
    """The process's peak resident memory so far (Linux and Mac; None on Windows)."""
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, ValueError):
        return None
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


class Clock:
    def __init__(self):
        self.parts = {}
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self, name: str) -> float:
        seconds = time.perf_counter() - self._start
        self.parts[name] = round(self.parts.get(name, 0.0) + seconds, 3)
        return seconds


def csv_text(header: list, rows) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def number(value: float, digits: int) -> str:
    """A float as fixed text with the given decimals (CSV files of predictions and scores)."""
    return f"{value:.{digits}f}" if np.isfinite(value) else "nan"


def run_provenance(job: Job, spec: dict, data, fold_hash: str, device_requested: str, device_used: str,
                   fallbacks: list) -> dict:
    from sweepkit.runs.provenance import collect

    out = collect(f"sweepkit {job.family} run")
    out.update({"run_id": spec["run_id"], "folds_hash": fold_hash, "data_fingerprint": data.fingerprint,
                "device_requested": device_requested, "device_used": device_used, "fallbacks": fallbacks,
                "pid": os.getpid()})
    return out


def settings_record(job: Job) -> dict:
    return {"model": {"name": job.model_name, **to_plain(job.model)}, "fit": {"name": job.fit_name, **to_plain(job.fit)}}


def limit_roles(roles: dict, ids: np.ndarray, spec: dict) -> dict:
    """For smoke tests only: each role cut to at most spec["smoke_rows"] rows (a fixed choice by SHA-256), so every
    configuration of a sweep can be tried through the real runner in seconds. Real jobs never carry smoke_rows."""
    from sweepkit.splits.base import stable_keys

    n = spec.get("smoke_rows")
    if n is None:
        return dict(roles)
    out = {}
    for role, positions in roles.items():
        if positions.size <= n:
            out[role] = positions
            continue
        keys = stable_keys(ids[positions], "smoke")
        out[role] = np.sort(positions[np.lexsort((ids[positions], keys))[:n]])
    if set(out) == {"fit", "eval"} and np.array_equal(roles["fit"], roles["eval"]):
        out["eval"] = out["fit"].copy()
    return out
