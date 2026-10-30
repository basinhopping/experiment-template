"""Preparation steps: fitted on the fit (training) rows only, then applied to every role's rows.

A step is a class registered with register_prep. It is built with its settings and a PrepContext, then:

    fit(x, positions)      learn what it needs from the fit rows (x holds only those rows; positions are their row
                           positions in the dataset, for steps that read a column such as a batch or a group)
    transform(x, positions) apply it to any rows
    removed()              steps declared removes_rows: a boolean mask over the rows given to fit, True where the step
                           takes the row out of fitting (duplicates, missing values, outliers, a subsample)
    names(names)           the feature names after the step
    record()               what it learned and changed, written to prep_record.json

Nothing a step learns can come from a validation, test or evaluation-only row: run_prep fits each step on the fit
rows alone, so a value in a held-out row never changes a learned statistic (every step's test checks this). A step
that must see other rows (unlabelled) is declared transductive and every report that uses it says so.
"""
import dataclasses

import numpy as np

from sweepkit.registry import PREP


@dataclasses.dataclass
class PrepContext:
    table: object               # the dataset's RowTable (for steps that read a column of the rows)
    feature_names: tuple        # names of the columns of x as the step receives it
    categorical: dict           # column index -> category names: integer-coded categorical columns
    seed: int                   # the run's seed (for steps with randomness, through sweepkit.streams)
    device: str                 # "cpu" or "cuda" (steps with a GPU path)


class PrepStep:
    def __init__(self, settings, context: PrepContext):
        self.s = settings
        self.ctx = context

    def fit(self, x: np.ndarray, positions: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def removed(self):
        return None

    def names(self, names: tuple) -> tuple:
        return tuple(names)

    def categorical_after(self, categorical: dict) -> dict:
        """The categorical columns after the step (steps that drop or add columns override this)."""
        return dict(categorical)

    def record(self) -> dict:
        return {}

    # helpers -------------------------------------------------------------------------------------------------

    def numeric_columns(self, n_columns: int) -> np.ndarray:
        """Indices of the columns that are not categorical codes: the ones numeric transforms act on."""
        return np.array([j for j in range(n_columns) if j not in self.ctx.categorical], dtype=np.int64)


def run_prep(steps: tuple, x_by_role: dict, positions_by_role: dict, fit_role: str, context: PrepContext,
             log) -> tuple:
    """Fit every step on the fit role's rows and apply it to every role.

    steps: ((name, settings), ...). Returns (x_by_role, positions_by_role, feature names, categorical, records).
    Rows a step removes leave the fit role only; records name them by row ID."""
    x_by_role = {role: np.asarray(x) for role, x in x_by_role.items()}
    positions_by_role = {role: np.asarray(p) for role, p in positions_by_role.items()}
    names, categorical = tuple(context.feature_names), dict(context.categorical)
    records = []
    for index, (name, settings) in enumerate(steps):
        entry = PREP.get(name)
        step_context = dataclasses.replace(context, feature_names=names, categorical=categorical)
        step = entry.target(settings, step_context)
        fit_x, fit_positions = x_by_role[fit_role], positions_by_role[fit_role]
        step.fit(fit_x, fit_positions)
        removed = step.removed()
        record = {"step": index, "name": name, "rows_in": int(fit_positions.shape[0]),
                  "columns_in": len(names), **step.record()}
        if removed is not None:
            if not entry.meta.get("removes_rows", False):
                raise RuntimeError(f"prep step {name} removed rows but is not declared removes_rows")
            removed = np.asarray(removed, dtype=bool)
            if removed.shape != (fit_positions.shape[0],):
                raise RuntimeError(f"prep step {name}: removed() must give one flag per fit row")
            ids = context.table.ids[fit_positions[removed]]
            record["rows_removed"] = int(removed.sum())
            record["removed_row_ids"] = ids.tolist()
            x_by_role[fit_role] = fit_x[~removed]
            positions_by_role[fit_role] = fit_positions[~removed]
            if positions_by_role[fit_role].size == 0:
                raise ValueError(f"prep step {name} removed every fit row")
        for role in x_by_role:
            x_by_role[role] = np.asarray(step.transform(x_by_role[role], positions_by_role[role]))
        new_names = step.names(names)
        categorical = step.categorical_after(categorical)
        names = tuple(new_names)
        widths = {role: x.shape[1] for role, x in x_by_role.items() if x.ndim == 2}
        if any(w != len(names) for w in widths.values()):
            raise RuntimeError(f"prep step {name} gave {widths} columns but {len(names)} names")
        record["columns_out"] = len(names)
        records.append(record)
        log(f"  prep {index} {name}: {record['rows_in']} fit rows, {record['columns_in']} -> {len(names)} columns"
            + (f", {record['rows_removed']} rows removed" if "rows_removed" in record else ""))
    return x_by_role, positions_by_role, names, categorical, records


def columns_matching(names: tuple, patterns: tuple, where: str) -> np.ndarray:
    """Indices of the columns whose name equals a pattern or matches "re:<regular expression>"; "*" matches all."""
    import re

    chosen = []
    for pattern in patterns:
        if pattern == "*":
            hits = list(range(len(names)))
        elif pattern.startswith("re:"):
            regex = re.compile(pattern[3:])
            hits = [j for j, n in enumerate(names) if regex.search(n)]
        else:
            hits = [j for j, n in enumerate(names) if n == pattern]
        if not hits:
            raise ValueError(f"{where}: {pattern!r} matches no column")
        chosen.extend(hits)
    return np.array(sorted(set(chosen)), dtype=np.int64)
