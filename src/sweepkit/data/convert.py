"""convert: turn one large CSV (or TSV) under data/ into a store, once, in chunks, so later jobs memory-map it.

A store is a folder data/stores/<name>/ written once:

    values.npy     float32 [rows, input columns], memory-mapped by the store dataset
    row_ids.txt    one row ID per line, in row order
    columns.json   {"features": [...], "categorical": {index: [categories]}}
    meta.csv       the kept columns (targets, groups, reference labels), as text, with row_id first
    record.json    the source file's SHA-256 and size, the settings, row and column counts, missing values per column,
                   and the time the conversion took

With --to parquet, the same columns go to data/stores/<name>/table.parquet instead (needs pyarrow), for users who
want to open the converted data in other tools.

The CSV is read twice: once to count rows and check the header of every chunk, once to fill the memory-mapped
array, so the whole file is never in memory at once.
"""
import csv
import json
import time

import numpy as np

from sweepkit import paths
from sweepkit.data.readers import match_columns, text_column
from sweepkit.datasets.table import file_fingerprint
from sweepkit.runs.store import OutputFolder


def convert(source: str, name: str, sep: str, row_id: str | None, features: tuple, exclude: tuple, keep: tuple,
            chunk_rows: int, to: str, log) -> dict:
    import pandas as pd

    if to not in ("store", "parquet"):
        raise ValueError("--to store or parquet")
    if chunk_rows < 1000:
        raise ValueError("--chunk-rows must be at least 1000")
    path = paths.data_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist (paths are relative to data/)")
    started = time.perf_counter()
    header = pd.read_csv(path, sep=sep, nrows=0).columns.astype(str).tolist()
    kept = match_columns(header, keep, set(), "convert keep") if keep else []
    excluded = set(match_columns(header, exclude, set(), "convert exclude")) if exclude else set()
    taken = set(kept) | excluded | ({row_id} if row_id else set())
    inputs = [c for c in match_columns(header, features, taken, "convert features") if c not in taken]
    if row_id is not None and row_id not in header:
        raise ValueError(f"convert: row_id {row_id!r} is not a column")
    n_rows = 0
    for chunk in pd.read_csv(path, sep=sep, usecols=[inputs[0]], chunksize=chunk_rows):
        n_rows += chunk.shape[0]
    log(f"convert {source}: {n_rows} rows, {len(inputs)} input columns, {len(kept)} kept columns")
    folder = OutputFolder(paths.data_dir() / "stores", name).begin()
    try:
        missing = np.zeros(len(inputs), dtype=np.int64)
        usecols = inputs + kept + ([row_id] if row_id else [])
        if to == "store":
            values = np.lib.format.open_memmap(folder.path("values.npy"), mode="w+", dtype=np.float32,
                                               shape=(n_rows, len(inputs)))
            ids_handle = open(folder.path("row_ids.txt"), "x", encoding="utf-8", newline="\n")
            meta_handle = open(folder.path("meta.csv"), "x", encoding="utf-8", newline="")
            meta = csv.writer(meta_handle, lineterminator="\n")
            meta.writerow(["row_id"] + kept)
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            writer = None
        start = 0
        seen_ids = set()
        for chunk in pd.read_csv(path, sep=sep, usecols=usecols, chunksize=chunk_rows, low_memory=False):
            chunk.columns = chunk.columns.astype(str)
            block = np.empty((chunk.shape[0], len(inputs)), dtype=np.float32)
            for j, c in enumerate(inputs):
                if not pd.api.types.is_numeric_dtype(chunk[c]):
                    raise ValueError(f"convert: input column {c!r} is not numeric in rows {start} to "
                                     f"{start + chunk.shape[0]}; exclude it or keep it as text")
                block[:, j] = chunk[c].to_numpy(dtype=np.float32, na_value=np.nan)
            missing += np.isnan(block).sum(axis=0)
            ids = (text_column(chunk[row_id]) if row_id
                   else np.array([str(i) for i in range(start, start + chunk.shape[0])]))
            repeated = seen_ids.intersection(ids.tolist())
            if repeated or np.unique(ids).size != ids.size:
                raise ValueError(f"convert: row IDs repeat, for example {sorted(repeated)[:3] or ids[:3].tolist()}")
            seen_ids.update(ids.tolist())
            if to == "store":
                values[start:start + chunk.shape[0]] = block
                ids_handle.write("\n".join(ids.tolist()) + "\n")
                kept_text = {c: text_column(chunk[c]) for c in kept}
                for i in range(chunk.shape[0]):
                    meta.writerow([ids[i]] + [kept_text[c][i] for c in kept])
            else:
                table = pa.table({"row_id": ids, **{c: block[:, j] for j, c in enumerate(inputs)},
                                  **{c: text_column(chunk[c]) for c in kept}})
                if writer is None:
                    writer = pq.ParquetWriter(folder.path("table.parquet"), table.schema)
                writer.write_table(table)
            start += chunk.shape[0]
            log(f"  {start} of {n_rows} rows")
        if to == "store":
            values.flush()
            del values
            ids_handle.close()
            meta_handle.close()
            folder.write_json("columns.json", {"features": inputs, "categorical": {}})
        elif writer is not None:
            writer.close()
        record = {"source": source, **file_fingerprint(path), "to": to, "rows": n_rows, "features": len(inputs),
                  "kept": kept, "row_id": row_id, "missing_per_column": dict(zip(inputs, missing.tolist())),
                  "seconds": round(time.perf_counter() - started, 1)}
        folder.write_json("record.json", record)
    except BaseException as err:
        folder.fail(err)
        raise
    done = folder.complete()
    log(f"written {done}")
    return record


def read_store(name: str) -> dict:
    """The store's arrays and records: values memory-mapped, row IDs, columns, kept columns."""
    import pandas as pd

    root = paths.data_dir() / "stores" / name
    if not (root / "record.json").is_file():
        raise FileNotFoundError(f"no finished store {root}; run convert first")
    record = json.loads((root / "record.json").read_text(encoding="utf-8"))
    if record["to"] != "store":
        raise ValueError(f"{root} holds Parquet; read it with the table dataset (format parquet)")
    values = np.load(root / "values.npy", mmap_mode="r")
    ids = np.array((root / "row_ids.txt").read_text(encoding="utf-8").splitlines())
    columns = json.loads((root / "columns.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(root / "meta.csv", dtype=str, keep_default_na=False)
    if meta.shape[0] != ids.size or not np.array_equal(meta["row_id"].to_numpy(dtype=str), ids):
        raise ValueError(f"{root}: meta.csv and row_ids.txt disagree")
    return {"values": values, "ids": ids, "features": columns["features"], "record": record,
            "meta": {c: meta[c].to_numpy(dtype=str) for c in meta.columns if c != "row_id"}}
