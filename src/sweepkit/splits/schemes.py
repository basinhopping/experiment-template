"""The registered split schemes.

Supervised (roles train, val, test):
  kfold               rows dealt into k groups (stratified by a column if named); fold i tests group i, validates on
                      group i+1 (the last on the first) and trains on the rest
  group_kfold         the same, but whole groups (a column: patients, subjects, sites, sessions) are dealt, so no
                      group is in two roles
  leave_one_group_out one fold per value of a group column, in the order given; the next value validates
  fixed               a column says train, val or test for every row (a published split)
  holdout             one fold: stratified validation and test fractions
Each takes identity_fold: true adds the labelled fold "identity" whose validation set is its test set; its
validation and test numbers must come out equal, a check of the two evaluation paths, never a result. No step that
chooses reads it (its validation numbers are test numbers); test-report shows the check.

Unsupervised (roles fit, eval):
  all                 one fold: fit on every row, evaluate every row
  subsamples          k folds, each fitting and evaluating a fixed fraction of the rows: choose settings on them and
                      measure stability across them; then one run on all rows
  fit_holdout         one fold: fit on a fraction, evaluate (assign) the rest with the method's predict

Row and group orders come from SHA-256 (stable_order), never from a random generator, so the folds are the same on
every machine.
"""
import dataclasses
import re

import numpy as np

from sweepkit.registry import register_split
from sweepkit.splits.base import (CHECK_FOLD, SUPERVISED_ROLES, UNSUPERVISED_ROLES, Fold, deal_rows, identity_fold,
                                  next_group_folds, stable_keys, stable_order)


def _group_values(table, column: str) -> np.ndarray:
    return table.column(column).astype(str)


def _with_identity(folds: list, wanted: bool) -> tuple:
    return tuple(folds + [identity_fold(folds[0])]) if wanted else tuple(folds)


@dataclasses.dataclass(frozen=True)
class KFoldSettings:
    k: int                          # number of groups (folds); at least 3 (train, validation and test)
    stratify: str | None            # a column whose classes every group shares equally, or None
    salt: str                       # text that fixes the deal; change it for a different but fixed split
    identity_fold: bool             # add the labelled check fold "identity"


@register_split("kfold", settings=KFoldSettings, roles=SUPERVISED_ROLES,
                description="rows dealt into k groups; fold i tests group i and validates on the next")
class KFold:
    @staticmethod
    def fold_names(s: KFoldSettings) -> tuple:
        return tuple(f"fold{i}" for i in range(1, s.k + 1)) + ((CHECK_FOLD,) if s.identity_fold else ())

    @staticmethod
    def make(s: KFoldSettings, table) -> tuple:
        if s.k < 3:
            raise ValueError("split kfold: k must be at least 3")
        strata = table.column(s.stratify) if s.stratify else None
        group = deal_rows(table.ids, strata, s.k, s.salt)
        members = {g: np.flatnonzero(group == g) for g in range(1, s.k + 1)}
        folds = next_group_folds(list(range(1, s.k + 1)), members, lambda g: f"fold{g}",
                                 lambda g: f"validation group {g}")
        return _with_identity(folds, s.identity_fold)

    @staticmethod
    def check_job(job) -> None:
        if job.split.k < 3:
            raise ValueError("split kfold: k must be at least 3")


@dataclasses.dataclass(frozen=True)
class GroupKFoldSettings:
    group: str                      # the column whose values stay together (a subject, patient, site, session)
    k: int                          # number of folds; at least 3
    salt: str
    identity_fold: bool


@register_split("group_kfold", settings=GroupKFoldSettings, roles=SUPERVISED_ROLES,
                description="whole groups dealt into k parts, so no group is in two roles")
class GroupKFold:
    @staticmethod
    def fold_names(s: GroupKFoldSettings) -> tuple:
        return tuple(f"fold{i}" for i in range(1, s.k + 1)) + ((CHECK_FOLD,) if s.identity_fold else ())

    @staticmethod
    def make(s: GroupKFoldSettings, table) -> tuple:
        values = _group_values(table, s.group)
        groups = stable_order(sorted(set(values.tolist())), s.salt)
        if len(groups) < s.k:
            raise ValueError(f"split group_kfold: {len(groups)} groups in {s.group!r}, fewer than k = {s.k}")
        part = {g: i % s.k + 1 for i, g in enumerate(groups)}
        assigned = np.array([part[v] for v in values.tolist()], dtype=np.int64)
        members = {g: np.flatnonzero(assigned == g) for g in range(1, s.k + 1)}
        folds = next_group_folds(list(range(1, s.k + 1)), members, lambda g: f"fold{g}",
                                 lambda g: f"validation part {g}")
        return _with_identity(folds, s.identity_fold)

    @staticmethod
    def check_job(job) -> None:
        if job.split.k < 3:
            raise ValueError("split group_kfold: k must be at least 3")


def _fold_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9-]+", "_", value).strip("_")
    return label.replace("__", "_") or "blank"


@dataclasses.dataclass(frozen=True)
class LeaveOneGroupOutSettings:
    group: str                      # the column held out one value at a time
    order: tuple[str, ...]          # every value of the column, in the order folds are made (the next validates)
    identity_fold: bool


@register_split("leave_one_group_out", settings=LeaveOneGroupOutSettings, roles=SUPERVISED_ROLES,
                description="one fold per group value, in the order given; the next value validates")
class LeaveOneGroupOut:
    @staticmethod
    def fold_names(s: LeaveOneGroupOutSettings) -> tuple:
        names = tuple(f"test_{_fold_label(v)}" for v in s.order)
        if len(set(names)) != len(names):
            raise ValueError(f"split leave_one_group_out: group values give repeated fold names {names}")
        return names + ((CHECK_FOLD,) if s.identity_fold else ())

    @staticmethod
    def make(s: LeaveOneGroupOutSettings, table) -> tuple:
        values = _group_values(table, s.group)
        present = set(values.tolist())
        if present != set(s.order):
            raise ValueError(f"split leave_one_group_out: the column {s.group!r} holds {sorted(present)}, the order "
                             f"lists {list(s.order)}; list every value once")
        members = {v: np.flatnonzero(values == v) for v in s.order}
        folds = next_group_folds(list(s.order), members, lambda v: f"test_{_fold_label(v)}",
                                 lambda v: f"validation group {v}")
        return _with_identity(folds, s.identity_fold)

    @staticmethod
    def check_job(job) -> None:
        if len(job.split.order) < 3 or len(set(job.split.order)) != len(job.split.order):
            raise ValueError("split leave_one_group_out: order lists at least 3 distinct values")
        LeaveOneGroupOut.fold_names(job.split)


@dataclasses.dataclass(frozen=True)
class FixedSettings:
    column: str                     # a column holding train, val or test for every row
    identity_fold: bool


@register_split("fixed", settings=FixedSettings, roles=SUPERVISED_ROLES,
                description="a column names each row's role (a published split)")
class Fixed:
    @staticmethod
    def fold_names(s: FixedSettings) -> tuple:
        return ("fixed",) + ((CHECK_FOLD,) if s.identity_fold else ())

    @staticmethod
    def make(s: FixedSettings, table) -> tuple:
        values = table.column(s.column).astype(str)
        unknown = sorted(set(values.tolist()) - set(SUPERVISED_ROLES))
        if unknown:
            raise ValueError(f"split fixed: column {s.column!r} holds {unknown}; only train, val and test")
        fold = Fold("fixed", {r: np.flatnonzero(values == r) for r in SUPERVISED_ROLES}, f"column {s.column}", False)
        return _with_identity([fold], s.identity_fold)


@dataclasses.dataclass(frozen=True)
class HoldoutSettings:
    val_fraction: float             # share of rows for validation
    test_fraction: float            # share of rows for test
    stratify: str | None
    salt: str
    identity_fold: bool


@register_split("holdout", settings=HoldoutSettings, roles=SUPERVISED_ROLES,
                description="one fold with stratified validation and test fractions")
class Holdout:
    @staticmethod
    def fold_names(s: HoldoutSettings) -> tuple:
        return ("holdout",) + ((CHECK_FOLD,) if s.identity_fold else ())

    @staticmethod
    def make(s: HoldoutSettings, table) -> tuple:
        n = len(table)
        strata = table.column(s.stratify).astype(str) if s.stratify else np.zeros(n, dtype=str)
        keys = stable_keys(table.ids, s.salt)
        role = np.empty(n, dtype=object)
        for stratum in np.unique(strata):
            members = np.flatnonzero(strata == stratum)
            members = members[np.lexsort((table.ids[members], keys[members]))]
            n_test = int(round(s.test_fraction * members.size))
            n_val = int(round(s.val_fraction * members.size))
            role[members[:n_test]] = "test"
            role[members[n_test:n_test + n_val]] = "val"
            role[members[n_test + n_val:]] = "train"
        fold = Fold("holdout", {r: np.flatnonzero(role == r) for r in SUPERVISED_ROLES},
                    f"validation {s.val_fraction}, test {s.test_fraction}", False)
        return _with_identity([fold], s.identity_fold)

    @staticmethod
    def check_job(job) -> None:
        s = job.split
        if not (0 < s.val_fraction < 1 and 0 < s.test_fraction < 1 and s.val_fraction + s.test_fraction < 1):
            raise ValueError("split holdout: fractions above 0, adding to less than 1")


@dataclasses.dataclass(frozen=True)
class AllSettings:
    pass


@register_split("all", settings=AllSettings, roles=UNSUPERVISED_ROLES,
                description="one fold: fit on every row and evaluate every row")
class All:
    @staticmethod
    def fold_names(s: AllSettings) -> tuple:
        return ("all",)

    @staticmethod
    def make(s: AllSettings, table) -> tuple:
        rows = np.arange(len(table))
        return (Fold("all", {"fit": rows, "eval": rows.copy()}, "every row", False),)


@dataclasses.dataclass(frozen=True)
class SubsamplesSettings:
    k: int                          # number of subsamples (folds)
    fraction: float                 # share of the rows in each subsample
    salt: str


@register_split("subsamples", settings=SubsamplesSettings, roles=UNSUPERVISED_ROLES,
                description="k fixed subsamples, each fitted and evaluated: choose settings, measure stability")
class Subsamples:
    @staticmethod
    def fold_names(s: SubsamplesSettings) -> tuple:
        return tuple(f"sub{i}" for i in range(1, s.k + 1))

    @staticmethod
    def make(s: SubsamplesSettings, table) -> tuple:
        n = len(table)
        size = int(round(s.fraction * n))
        if size < 2:
            raise ValueError(f"split subsamples: {s.fraction} of {n} rows leaves fewer than 2")
        folds = []
        for i in range(1, s.k + 1):
            keys = stable_keys(table.ids, f"{s.salt}|sub{i}")
            chosen = np.lexsort((table.ids, keys))[:size]
            folds.append(Fold(f"sub{i}", {"fit": chosen, "eval": chosen.copy()}, f"{size} of {n} rows", False))
        return tuple(folds)

    @staticmethod
    def check_job(job) -> None:
        if job.split.k < 1 or not 0 < job.split.fraction <= 1:
            raise ValueError("split subsamples: k at least 1, fraction in (0, 1]")


@dataclasses.dataclass(frozen=True)
class FitHoldoutSettings:
    fit_fraction: float             # share of rows the method is fitted on; the rest are assigned by its predict
    salt: str


@register_split("fit_holdout", settings=FitHoldoutSettings, roles=UNSUPERVISED_ROLES,
                description="fit on a fraction, evaluate the other rows through the method's predict")
class FitHoldout:
    @staticmethod
    def fold_names(s: FitHoldoutSettings) -> tuple:
        return ("holdout",)

    @staticmethod
    def make(s: FitHoldoutSettings, table) -> tuple:
        n = len(table)
        order = np.lexsort((table.ids, stable_keys(table.ids, s.salt)))
        n_fit = int(round(s.fit_fraction * n))
        return (Fold("holdout", {"fit": order[:n_fit], "eval": order[n_fit:]},
                     f"fit {n_fit}, evaluate {n - n_fit}", False),)

    @staticmethod
    def check_job(job) -> None:
        if not 0 < job.split.fit_fraction < 1:
            raise ValueError("split fit_holdout: fit_fraction in (0, 1)")
        if job.fit.assign != "predict":
            raise ValueError("split fit_holdout evaluates rows outside the fit set: the fit section needs assign "
                             "predict")
