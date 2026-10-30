"""The store dataset: data converted once by `convert` into data/stores/<name>/, memory-mapped, so data larger than
memory work: a job reads only the rows and columns its fold and settings need."""
import dataclasses

import numpy as np

from sweepkit.data.base import Data, Dataset, RowTable
from sweepkit.data.convert import read_store
from sweepkit.data.readers import match_columns
from sweepkit.registry import register_dataset


@dataclasses.dataclass(frozen=True)
class StoreSettings:
    store: str                      # the store's name (data/stores/<store>/)
    features: tuple[str, ...]       # input columns of the store: names, "re:<regex>", or "*"
    keep: tuple[str, ...]           # kept columns of the store to carry along (targets, groups, references)
    filter_column: str | None
    filter_values: tuple[str, ...]


class _Columns:
    """A memory-mapped array seen through a list of columns, without copying it: rows are read when indexed."""

    def __init__(self, values, columns: np.ndarray):
        self.values = values
        self.columns = columns
        self.all_columns = columns.size == values.shape[1] and np.array_equal(columns, np.arange(values.shape[1]))
        self.shape = (values.shape[0], columns.size)
        self.ndim = 2

    def __getitem__(self, rows):
        block = self.values[np.sort(np.asarray(rows))] if not isinstance(rows, slice) else self.values[rows]
        out = block if self.all_columns else block[:, self.columns]
        if not isinstance(rows, slice):
            order = np.argsort(np.argsort(np.asarray(rows)))
            out = out[order]
        return np.asarray(out, dtype=np.float32)


@register_dataset("store", settings=StoreSettings, kinds=("tabular",), profile="core", packages=("pandas",),
                  description="a store made by convert: memory-mapped float32, for data larger than memory",
                  example="data_store")
class Store(Dataset):
    @staticmethod
    def check_job(job) -> None:
        s = job.data
        if (s.filter_column is None) != (len(s.filter_values) == 0):
            raise ValueError("data store: name filter_column together with filter_values, or neither")
        if s.filter_column is not None and s.filter_column not in s.keep:
            raise ValueError("data store: filter_column must be one of the kept columns")

    def load(self, s: StoreSettings) -> Data:
        store = read_store(s.store)
        names = store["features"]
        chosen = match_columns(names, s.features, set(), f"data store {s.store} features")
        index = np.array([names.index(c) for c in chosen], dtype=np.int64)
        kept = {c: store["meta"][c] for c in (match_columns(list(store["meta"]), s.keep, set(), "data store keep")
                                              if s.keep else [])}
        ids = store["ids"]
        x = _Columns(store["values"], index)
        if s.filter_column is not None:
            rows = np.flatnonzero(np.isin(kept[s.filter_column], list(s.filter_values)))
            if rows.size == 0:
                raise ValueError(f"data store: no row has {s.filter_column} in {list(s.filter_values)}")
            x = np.asarray(store["values"][rows][:, index], dtype=np.float32)
            ids = ids[rows]
            kept = {c: v[rows] for c, v in kept.items()}
        table = RowTable(ids, kept)
        fingerprint = {"store": s.store, "source_sha256": store["record"]["sha256"], "rows": int(len(ids))}
        return Data("tabular", table, x=x, feature_names=tuple(chosen), fingerprint=fingerprint)
