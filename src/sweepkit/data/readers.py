"""Reading tables of any size into the template's form: row IDs, input columns as float32, and kept columns as text.

Formats: CSV and TSV (pandas; chunked for large files), Parquet (pyarrow), Excel (openpyxl), HDF5 (pandas with
PyTables, or h5py for a plain array), Feather. Columns are chosen by name or pattern:
    "name"           that column
    "re:<regex>"     every column whose name matches the regular expression
    "*"              every column not otherwise used (not the row ID, not kept, not excluded)
Non-numeric input columns are refused, dropped, turned into integer codes (categorical, for k-prototypes and Gower)
or one-hot columns, as the settings say. Missing input values stay NaN (an impute step handles them); missing kept
values become "".
"""
import re

import numpy as np

FORMATS = ("csv", "tsv", "parquet", "excel", "hdf5", "feather")


def read_frame(path, fmt: str, sheet_or_key: str | None):
    import pandas as pd

    if fmt not in FORMATS:
        raise ValueError(f"format {fmt!r} is not one of {FORMATS}")
    if fmt == "csv":
        return pd.read_csv(path, low_memory=False)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "excel":
        return pd.read_excel(path, sheet_name=sheet_or_key if sheet_or_key is not None else 0)
    if fmt == "hdf5":
        if sheet_or_key is None:
            raise ValueError("format hdf5 needs the key of the table (sheet_or_key)")
        return pd.read_hdf(path, key=sheet_or_key)
    return pd.read_feather(path)


def match_columns(columns: list, patterns: tuple, taken: set, where: str) -> list:
    """Columns matching the patterns, in the file's order, without repeats."""
    chosen = []
    for pattern in patterns:
        if pattern == "*":
            hits = [c for c in columns if c not in taken]
        elif pattern.startswith("re:"):
            regex = re.compile(pattern[3:])
            hits = [c for c in columns if regex.search(str(c))]
        else:
            hits = [c for c in columns if str(c) == pattern]
        if not hits:
            raise ValueError(f"{where}: {pattern!r} matches no column; the file's columns start {list(columns)[:10]}")
        chosen.extend(hits)
    seen, out = set(), []
    for c in chosen:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def text_column(series) -> np.ndarray:
    values = series.astype(object).where(series.notna(), "")
    return np.array([str(v) for v in values.tolist()], dtype=str)


def split_frame(frame, row_id: str | None, features: tuple, exclude: tuple, keep: tuple, categorical: str,
                filter_column: str | None, filter_values: tuple, where: str) -> dict:
    """{ids, x (float32), feature names, categorical {index: categories}, kept columns}."""
    import pandas as pd

    columns = [str(c) for c in frame.columns]
    frame.columns = columns
    if filter_column is not None:
        if filter_column not in columns:
            raise ValueError(f"{where}: filter_column {filter_column!r} is not a column")
        mask = text_column(frame[filter_column])
        frame = frame[np.isin(mask, list(filter_values))].reset_index(drop=True)
        if frame.shape[0] == 0:
            raise ValueError(f"{where}: no row has {filter_column} in {list(filter_values)}")
    if row_id is not None:
        if row_id not in columns:
            raise ValueError(f"{where}: row_id {row_id!r} is not a column")
        ids = text_column(frame[row_id])
    else:
        ids = np.array([str(i) for i in range(frame.shape[0])])
    kept = match_columns(columns, keep, set(), f"{where} keep") if keep else []
    excluded = set(match_columns(columns, exclude, set(), f"{where} exclude")) if exclude else set()
    taken = set(kept) | excluded | ({row_id} if row_id else set())
    inputs = [c for c in match_columns(columns, features, taken, f"{where} features") if c not in taken]
    if not inputs:
        raise ValueError(f"{where}: no input column left after keep and exclude")
    blocks, names, cat_map = [], [], {}
    for c in inputs:
        series = frame[c]
        if pd.api.types.is_bool_dtype(series):
            blocks.append(series.astype(np.float32).to_numpy()[:, None])
            names.append(c)
        elif pd.api.types.is_numeric_dtype(series):
            blocks.append(series.to_numpy(dtype=np.float32, na_value=np.nan)[:, None])
            names.append(c)
        else:
            text = text_column(series)
            categories = sorted(set(text.tolist()) - {""})
            if categorical == "error":
                raise ValueError(f"{where}: input column {c!r} is not numeric (for example {categories[:3]}); set "
                                 "categorical to drop, codes or onehot, or exclude it")
            if categorical == "drop":
                continue
            if categorical == "codes":
                index = {v: i for i, v in enumerate(categories)}
                codes = np.array([index.get(v, np.nan) if v != "" else np.nan for v in text.tolist()], dtype=np.float32)
                cat_map[len(names)] = categories
                blocks.append(codes[:, None])
                names.append(c)
            else:
                for v in categories:
                    blocks.append((text == v).astype(np.float32)[:, None])
                    names.append(f"{c}={v}")
    if not blocks:
        raise ValueError(f"{where}: every input column was dropped")
    x = np.hstack(blocks).astype(np.float32)
    return {"ids": ids, "x": x, "names": tuple(names), "categorical": cat_map,
            "kept": {c: text_column(frame[c]) for c in kept}}
