"""profile-data: look at data before any model code runs (an empty file, a label file that does not join its rows,
and editor leftovers were each found late once in the project this template grew out of).

For a folder under data/: every file with its size, counts by extension, empty files, files that cannot be read,
leftovers of notebooks, editors and archives (.ipynb_checkpoints, .DS_Store, Thumbs.db, ~$ lock files, __MACOSX,
.~lock, swap and backup files), names that differ only in case (they collide on Windows and on a Mac), names with
characters Windows refuses, and very long paths.

For a table under data/ (CSV, TSV, Parquet, Excel, HDF5, AnnData; CSV and TSV read in chunks): rows and columns;
per column its type, missing values, and either the range, mean and SD (numbers) or the number of distinct values
with the most frequent ones (text); constant columns; text columns that are nearly all numbers (a stray character
in a numeric column); and, with a row ID column, missing and repeated IDs.

Columns named in `hide` (reference labels, and any column that judges a choice: CLAUDE.md, rule 3) show only how
many values are present and missing: no values, no counts of values, no range, no number of distinct values, in the
printed report and in report.json alike.

With a label table: the join between labels and rows on the row ID: rows without a label, labels without a row,
and labels given twice. Only the key column of the label table is reported.

Writes artifacts/checks/profile_<name>_<utc>/report.json and report.txt; reads nothing outside data/.
"""
import os
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from sweepkit import paths

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls", ".h5", ".hdf5", ".h5ad", ".npy")
LEFTOVER_DIRS = (".ipynb_checkpoints", "__MACOSX", "__pycache__", ".Trash", ".Spotlight-V100", ".fseventsd")
LEFTOVER_FILES = (".DS_Store", "Thumbs.db", "desktop.ini", "Icon\r")
LEFTOVER_PREFIXES = ("~$", ".~lock", "._")
LEFTOVER_SUFFIXES = (".tmp", ".swp", ".swo", ".bak", "~", ".crdownload", ".part")
WINDOWS_BAD = set('<>:"|?*')


def profile_folder(root: Path) -> dict:
    files, by_ext, empty, unreadable, leftovers, long_paths, bad_names = [], Counter(), [], [], [], [], []
    names_seen = {}
    total = 0
    for folder, dirs, names in os.walk(root):
        rel_folder = Path(folder).relative_to(root)
        for d in list(dirs):
            if d in LEFTOVER_DIRS:
                leftovers.append((rel_folder / d).as_posix() + "/")
        for name in names:
            rel = (rel_folder / name).as_posix()
            path = Path(folder) / name
            try:
                size = path.stat().st_size
                with open(path, "rb") as handle:
                    handle.read(1)
            except OSError as err:
                unreadable.append(f"{rel}: {type(err).__name__}")
                continue
            files.append(rel)
            total += size
            by_ext[path.suffix.lower() or "(none)"] += 1
            if size == 0:
                empty.append(rel)
            if (name in LEFTOVER_FILES or name.startswith(LEFTOVER_PREFIXES) or name.endswith(LEFTOVER_SUFFIXES)
                    or any(part in LEFTOVER_DIRS for part in Path(rel).parts)):
                leftovers.append(rel)
            if len(str(path)) > 240:
                long_paths.append(rel)
            if WINDOWS_BAD & set(name) or name.endswith((" ", ".")) or any(ord(c) < 32 for c in name):
                bad_names.append(rel)
            key = unicodedata.normalize("NFC", rel).lower()
            names_seen.setdefault(key, []).append(rel)
    collisions = [v for v in names_seen.values() if len(v) > 1]
    return {"kind": "folder", "n_files": len(files), "bytes": total, "by_extension": dict(by_ext.most_common()),
            "empty_files": empty, "unreadable": unreadable, "leftovers": sorted(leftovers),
            "case_collisions": collisions, "names_windows_refuses": bad_names, "very_long_paths": long_paths,
            "tables": sorted(f for f in files if Path(f).suffix.lower() in TABLE_SUFFIXES)[:200]}


def _chunks(path: Path, max_rows: int | None, chunk_rows: int):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix in (".tsv",) else ","
        reader = pd.read_csv(path, sep=sep, chunksize=chunk_rows, nrows=max_rows, dtype=str, keep_default_na=False,
                             na_values=[""], on_bad_lines="skip")
        yield from reader
        return
    from sweepkit.data.readers import read_frame

    formats = {".parquet": "parquet", ".pq": "parquet", ".xlsx": "excel", ".xls": "excel", ".feather": "feather"}
    if suffix not in formats:
        raise ValueError(f"{path.name}: profile-data reads CSV, TSV, Parquet, Excel and Feather tables; profile the "
                         "folder instead, or convert the file first")
    frame = read_frame(path, formats[suffix], None)
    if max_rows is not None:
        frame = frame.iloc[:max_rows]
    yield frame


class _Column:
    """Running statistics of one column over chunks."""

    def __init__(self, distinct_cap: int):
        self.n = self.missing = self.numeric = 0
        self.values = []                      # numbers, per chunk
        self.counts = Counter()
        self.capped = False
        self.cap = distinct_cap

    def add(self, series) -> None:
        import pandas as pd

        self.n += len(series)
        missing = series.isna()
        self.missing += int(missing.sum())
        present = series[~missing]
        as_number = pd.to_numeric(present, errors="coerce")
        numeric = ~as_number.isna()
        self.numeric += int(numeric.sum())
        self.values.append(as_number[numeric].to_numpy(dtype=float))
        if not self.capped:
            self.counts.update(present.astype(str).tolist())
            if len(self.counts) > self.cap:
                self.capped = True

    def summary(self, top: int) -> dict:
        present = self.n - self.missing
        out = {"missing": self.missing, "present": present}
        numbers = np.concatenate(self.values) if self.values else np.array([])
        if present and self.numeric == present:
            finite = numbers[np.isfinite(numbers)]
            out.update({"type": "number", "min": float(finite.min()) if finite.size else None,
                        "max": float(finite.max()) if finite.size else None,
                        "mean": float(finite.mean()) if finite.size else None,
                        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
                        "not_finite": int(numbers.size - finite.size)})
        else:
            out["type"] = "text" if present else "empty"
            if present and self.numeric >= 0.9 * present:
                out["nearly_all_numbers"] = {"numbers": self.numeric, "other": present - self.numeric,
                                             "examples": [v for v, _ in self.counts.most_common()
                                                          if not _is_number(v)][:5]}
        out["distinct"] = f"more than {self.cap}" if self.capped else len(self.counts)
        if out["type"] != "number":
            out["most_frequent"] = [[v, c] for v, c in self.counts.most_common(top)]
        out["constant"] = present > 0 and not self.capped and len(self.counts) == 1
        return out


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def bad_lines(path: Path, max_rows: int | None) -> dict:
    """Lines of a CSV or TSV whose number of fields differs from the header's (an unquoted separator inside a
    value, a truncated line). They are skipped when the table is profiled, and listed here."""
    import csv

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    found, count = [], 0
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=sep)
        header = next(reader, [])
        for k, row in enumerate(reader, 2):
            if max_rows is not None and k - 1 > max_rows:
                break
            if row and len(row) != len(header):
                count += 1
                if len(found) < 10:
                    found.append({"line": k, "fields": len(row)})
    return {"expected_fields": len(header), "count": count, "first": found}


def profile_table(path: Path, row_id: str | None, max_rows: int | None, top: int, hide: tuple = ()) -> dict:
    columns, ids, n_rows, header = {}, Counter(), 0, None
    for chunk in _chunks(path, max_rows, 100_000):
        if header is None:
            header = [str(c) for c in chunk.columns]
            if row_id is not None and row_id not in header:
                raise ValueError(f"{path.name}: no column {row_id!r}; columns are {header[:20]}")
            unknown = [c for c in hide if c not in header]
            if unknown:
                raise ValueError(f"{path.name}: --hide names {unknown}, which are not columns; columns are "
                                 f"{header[:20]}")
            if row_id is not None and row_id in hide:
                raise ValueError(f"{path.name}: the row ID column {row_id!r} cannot be hidden")
            if len(set(header)) != len(header):
                raise ValueError(f"{path.name}: repeated column names {[c for c, k in Counter(header).items() if k > 1]}")
            columns = {c: _Column(100_000) for c in header}
        n_rows += len(chunk)
        for c in header:
            columns[c].add(chunk[c])
        if row_id is not None:
            ids.update(chunk[row_id].astype(str).tolist())
    out = {"kind": "table", "file": path.name, "rows": n_rows, "columns": len(header or []),
           "read_rows_limit": max_rows,
           "by_column": {c: ({"hidden": True, "missing": col.missing, "present": col.n - col.missing,
                              "constant": False} if c in hide else col.summary(top))
                         for c, col in columns.items()},
           "hidden_columns": list(hide)}
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        out["bad_lines"] = bad_lines(path, max_rows)
    out["constant_columns"] = [c for c, s in out["by_column"].items() if s["constant"]]
    out["nearly_numeric_text_columns"] = [c for c, s in out["by_column"].items() if "nearly_all_numbers" in s]
    if row_id is not None:
        repeated = [i for i, k in ids.items() if k > 1]
        out["row_id"] = {"column": row_id, "missing": columns[row_id].missing, "repeated": len(repeated),
                         "repeated_examples": repeated[:10]}
    out["_ids"] = set(ids)
    return out


def join(rows_ids: set, labels_path: Path, label_key: str, max_rows: int | None) -> dict:
    keys = Counter()
    for chunk in _chunks(labels_path, max_rows, 100_000):
        if label_key not in chunk.columns:
            raise ValueError(f"{labels_path.name}: no column {label_key!r}")
        keys.update(chunk[label_key].astype(str).tolist())
    label_ids = set(keys)
    without_label = sorted(rows_ids - label_ids)
    without_row = sorted(label_ids - rows_ids)
    twice = sorted(k for k, n in keys.items() if n > 1)
    return {"labels_file": labels_path.name, "key": label_key, "rows_without_label": len(without_label),
            "labels_without_row": len(without_row), "labels_given_twice": len(twice),
            "examples": {"rows_without_label": without_label[:10], "labels_without_row": without_row[:10],
                         "labels_given_twice": twice[:10]}}


def profile(target: str, row_id: str | None, labels: str | None, label_key: str | None, max_rows: int | None,
            top: int, hide: tuple = ()) -> dict:
    path = paths.data_path(target)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (paths are relative to data/)")
    if path.is_dir():
        report = profile_folder(path)
    else:
        report = profile_table(path, row_id, max_rows, top, tuple(hide))
        ids = report.pop("_ids")
        if labels is not None:
            if row_id is None or label_key is None:
                raise ValueError("a label file needs --row-id (the rows' key) and --label-key (the labels' key)")
            report["join"] = join(ids, paths.data_path(labels), label_key, max_rows)
    report["target"] = target
    return report


def render(report: dict) -> str:
    lines = [f"profile of data/{report['target']}"]
    if report["kind"] == "folder":
        lines.append(f"  {report['n_files']} files, {report['bytes'] / 1e6:.1f} MB; by extension: "
                     + ", ".join(f"{k} {v}" for k, v in report["by_extension"].items()))
        for key, title in (("empty_files", "empty files"), ("unreadable", "unreadable"), ("leftovers", "leftovers"),
                           ("case_collisions", "names that differ only in case"),
                           ("names_windows_refuses", "names Windows refuses"), ("very_long_paths", "very long paths")):
            if report[key]:
                lines.append(f"  {title}: {len(report[key])}, for example {report[key][:3]}")
        if report["tables"]:
            lines.append(f"  tables: {len(report['tables'])}, for example {report['tables'][:5]}")
        return "\n".join(lines)
    lines.append(f"  {report['rows']} rows, {report['columns']} columns"
                 + (f" (first {report['read_rows_limit']} rows read)" if report["read_rows_limit"] else ""))
    bad = report.get("bad_lines")
    if bad and bad["count"]:
        lines.append(f"  {bad['count']} lines have another number of fields than the header's {bad['expected_fields']} "
                     f"and were left out, for example lines {[b['line'] for b in bad['first'][:5]]}")
    for name, s in report["by_column"].items():
        if s.get("hidden"):
            lines.append(f"  {name}: values hidden (--hide); {s['present']} present, {s['missing']} missing")
            continue
        if s["type"] == "number":
            def fmt(v):
                return "n/a" if v is None else f"{v:.4g}"
            text = (f"number, range {fmt(s['min'])} to {fmt(s['max'])}, mean {fmt(s['mean'])}, SD {fmt(s['sd'])}, "
                    f"{s['distinct']} distinct")
        else:
            text = f"{s['type']}, {s['distinct']} distinct" + (f", most frequent {s['most_frequent'][:3]}"
                                                               if s.get("most_frequent") else "")
        flags = []
        if s["missing"]:
            flags.append(f"{s['missing']} missing")
        if s.get("not_finite"):
            flags.append(f"{s['not_finite']} not finite")
        if s["constant"]:
            flags.append("constant")
        if "nearly_all_numbers" in s:
            flags.append(f"text but nearly all numbers (others: {s['nearly_all_numbers']['examples']})")
        lines.append(f"  {name}: {text}" + (f"  [{'; '.join(flags)}]" if flags else ""))
    if "row_id" in report:
        r = report["row_id"]
        lines.append(f"  row ID {r['column']}: {r['missing']} missing, {r['repeated']} repeated"
                     + (f", for example {r['repeated_examples'][:3]}" if r["repeated"] else ""))
    if "join" in report:
        j = report["join"]
        lines.append(f"  join with {j['labels_file']} on {j['key']}: {j['rows_without_label']} rows without a label, "
                     f"{j['labels_without_row']} labels without a row, {j['labels_given_twice']} labels given twice")
    return "\n".join(lines)
