"""The table dataset: one table file under data/ (CSV, TSV, Parquet, Excel, HDF5, Feather), read whole.

For a file larger than memory, convert it once into a store (`convert`, then the store dataset), which memory-maps
float32 values and reads only what a job needs.
"""
import dataclasses
import os
from typing import Literal

from sweepkit import paths
from sweepkit.data.base import Data, Dataset, RowTable
from sweepkit.data.readers import FORMATS, read_frame, split_frame
from sweepkit.hashing import sha256_file
from sweepkit.registry import register_dataset

_FILE_HASHES = {}


def file_fingerprint(path) -> dict:
    """{bytes, sha256} of a data file, hashed once per process for each size and modification time."""
    stat = os.stat(path)
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_HASHES:
        _FILE_HASHES[key] = sha256_file(path)
    n_bytes, digest = _FILE_HASHES[key]
    return {"bytes": n_bytes, "sha256": digest}


@dataclasses.dataclass(frozen=True)
class TableSettings:
    path: str                       # the file, relative to data/ (for example "cells/cells_v1.csv")
    format: Literal[FORMATS]        # csv, tsv, parquet, excel, hdf5, feather
    sheet_or_key: str | None        # the Excel sheet or the HDF5 key; None otherwise
    row_id: str | None              # a column of unique row IDs; None: the row number in the file (0, 1, 2, ...)
    features: tuple[str, ...]       # input columns: names, "re:<regex>", or "*" (every column not used otherwise)
    exclude: tuple[str, ...]        # never inputs (names or patterns)
    keep: tuple[str, ...]           # per-row columns carried along as text: targets, groups, reference labels, metadata
    categorical: Literal["error", "drop", "codes", "onehot"]   # what non-numeric input columns become
    filter_column: str | None       # keep only rows whose value here is in filter_values (None: every row)
    filter_values: tuple[str, ...]  # [] when filter_column is None


@register_dataset("table", settings=TableSettings, kinds=("tabular",), profile="core", packages=("pandas",),
                  description="one table file under data/: CSV, TSV, Parquet, Excel, HDF5 or Feather",
                  example="data_table")
class Table(Dataset):
    @staticmethod
    def check_job(job) -> None:
        s = job.data
        paths.data_path(s.path)
        if (s.filter_column is None) != (len(s.filter_values) == 0):
            raise ValueError("data table: name filter_column together with filter_values, or neither")
        if s.format == "hdf5" and s.sheet_or_key is None:
            raise ValueError("data table: format hdf5 needs sheet_or_key (the table's key)")
        if s.format not in ("excel", "hdf5") and s.sheet_or_key is not None:
            raise ValueError(f"data table: format {s.format} takes sheet_or_key None")

    def load(self, s: TableSettings) -> Data:
        path = paths.data_path(s.path)
        if not path.is_file():
            raise FileNotFoundError(f"data table: {path} does not exist (paths are relative to data/)")
        frame = read_frame(path, s.format, s.sheet_or_key)
        parts = split_frame(frame, s.row_id, s.features, s.exclude, s.keep, s.categorical, s.filter_column,
                            s.filter_values, f"data table {s.path}")
        table = RowTable(parts["ids"], parts["kept"])
        return Data("tabular", table, x=parts["x"], feature_names=parts["names"], categorical=parts["categorical"],
                    fingerprint={"path": s.path, **file_fingerprint(path)})
