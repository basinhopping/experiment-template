"""make-synthetic: write a synthetic table with known structure as a CSV under data/synthetic/, in chunks, with a
small sample beside it. The template ships no data; this is how its examples, the large-data test and a new user's
first trial get a file that behaves like real data (row IDs, numeric and categorical columns, a planted truth).

    data/synthetic/<name>.csv          row_id, the generator's columns (categorical ones as text), then its per-row
                                       columns (truth, target, group, ...)
    data/synthetic/<name>_sample.csv   sample_rows rows chosen by SHA-256 of their IDs, in file order
    data/synthetic/<name>.json         the settings, row and column counts, and the SHA-256 of both files

Written once: an existing name is refused. The same settings give the same bytes on every machine (named random
streams; numbers written with 7 significant digits).
"""
import csv
import json

import numpy as np

from sweepkit import paths
from sweepkit.data.synthetic import TABULAR_GENERATORS, make
from sweepkit.hashing import sha256_file
from sweepkit.splits.base import stable_keys


def _frame(made: dict, start: int, stop: int, ids: np.ndarray, names: list, extra: list):
    import pandas as pd

    x = made["x"][start:stop]
    data = {"row_id": ids[start:stop]}
    for j, name in enumerate(names):
        if j in made["categorical"]:
            data[name] = np.asarray(made["categorical"][j], dtype=object)[x[:, j].astype(np.int64)]
        else:
            data[name] = x[:, j].astype(np.float64)
    for c in extra:
        data[c] = np.asarray(made["columns"][c])[start:stop].astype(str)
    return pd.DataFrame(data)


def write_synthetic(generator: str, n_rows: int, n_features: int, n_groups: int, noise: float, seed: int, name: str,
                    chunk_rows: int, sample_rows: int, log) -> dict:
    if generator not in TABULAR_GENERATORS:
        raise ValueError(f"make-synthetic writes tables; generator {generator!r} is not one of {TABULAR_GENERATORS}")
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"name {name!r}: letters, digits, _ and - only")
    if chunk_rows < 1 or not 0 < sample_rows <= n_rows:
        raise ValueError("chunk_rows at least 1; sample_rows between 1 and n_rows")
    folder = paths.synthetic_dir()
    target, sample, record_path = folder / f"{name}.csv", folder / f"{name}_sample.csv", folder / f"{name}.json"
    for p in (target, sample, record_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; synthetic files are written once (choose another name)")
    folder.mkdir(parents=True, exist_ok=True)
    made = make(generator, n_rows, n_features, n_groups, noise, seed)
    if made["x"] is None:
        raise ValueError(f"generator {generator} makes no table")
    names = list(made["names"]) if made["names"] else [f"f{j}" for j in range(made["x"].shape[1])]
    extra = sorted(made["columns"])
    width = len(str(n_rows - 1))
    ids = np.array([f"r{i:0{width}d}" for i in range(n_rows)])
    header = ["row_id"] + names + extra
    in_sample = np.zeros(n_rows, dtype=bool)
    in_sample[np.lexsort((ids, stable_keys(ids, f"sample {name}")))[:sample_rows]] = True
    with open(target, "x", newline="", encoding="utf-8") as handle, \
            open(sample, "x", newline="", encoding="utf-8") as sample_handle:
        csv.writer(handle, lineterminator="\n").writerow(header)
        csv.writer(sample_handle, lineterminator="\n").writerow(header)
        for start in range(0, n_rows, chunk_rows):
            stop = min(n_rows, start + chunk_rows)
            frame = _frame(made, start, stop, ids, names, extra)
            frame.to_csv(handle, header=False, index=False, float_format="%.7g", lineterminator="\n")
            frame[in_sample[start:stop]].to_csv(sample_handle, header=False, index=False, float_format="%.7g",
                                                lineterminator="\n")
            log(f"  {stop} of {n_rows} rows")
    record = {"generator": generator, "n_rows": n_rows, "n_features": n_features, "n_groups": n_groups,
              "noise": noise, "seed": seed, "columns": header, "categorical": [names[j] for j in made["categorical"]],
              "csv": {"path": f"synthetic/{name}.csv", **dict(zip(("bytes", "sha256"), sha256_file(target)))},
              "sample": {"path": f"synthetic/{name}_sample.csv", "rows": sample_rows,
                         **dict(zip(("bytes", "sha256"), sha256_file(sample)))}}
    with open(record_path, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    return record
