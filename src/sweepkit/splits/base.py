"""Folds: which rows play which role. Defined once, hashed into every run, read by every model.

Supervised schemes give each fold the roles train, val and test; unsupervised schemes give fit and eval. A split
scheme is registered with the roles it gives, and a task with the roles it needs; check_job matches them.

Where a scheme needs a random-looking order it uses stable_order, which sorts by SHA-256 of a fixed text and the
item, so the folds are the same on every machine and Python version. folds_hash, stamped into each run's
provenance, is the SHA-256 of every fold's row IDs per role.
"""
import dataclasses
import hashlib

import numpy as np

SUPERVISED_ROLES = ("train", "val", "test")
UNSUPERVISED_ROLES = ("fit", "eval")
# The one fold whose validation rows are its test rows (identity_fold below). Its numbers are test numbers, so no step
# that chooses ever reads its results: analyze, status, select, atlas and propose leave it out, and test-report checks
# that its validation and test numbers agree (a check of the two evaluation paths, never a result).
CHECK_FOLD = "identity"


@dataclasses.dataclass(frozen=True)
class Fold:
    """A fold: role -> sorted row positions. selection_on_test marks the labelled identity fold whose validation set
    IS its test set (a check that the validation and test paths give the same numbers; never a result)."""

    name: str
    roles: dict
    note: str
    selection_on_test: bool

    def __post_init__(self):
        for role, positions in self.roles.items():
            positions = np.asarray(positions, dtype=np.int64)
            if positions.ndim != 1:
                raise ValueError(f"{self.name}: role {role} must be one-dimensional")
            if positions.size == 0:
                raise ValueError(f"{self.name}: no rows in role {role}")
            if np.unique(positions).size != positions.size:
                raise ValueError(f"{self.name}: role {role} lists a row twice")
            self.roles[role] = np.sort(positions)
        if self.selection_on_test != (self.name == CHECK_FOLD):
            raise ValueError(f"{self.name}: the fold whose validation rows are its test rows is named {CHECK_FOLD!r}, "
                             "and no other fold has that name (so every step that chooses can leave it out)")
        if set(self.roles) == set(SUPERVISED_ROLES):
            pairs = [("train", "val"), ("train", "test"), ("val", "test")]
            if self.selection_on_test:
                if not np.array_equal(self.roles["val"], self.roles["test"]):
                    raise ValueError(f"{self.name}: selection_on_test needs the validation rows to be the test rows")
                pairs = [("train", "val")]
            for a, b in pairs:
                both = np.intersect1d(self.roles[a], self.roles[b])
                if both.size:
                    raise ValueError(f"{self.name}: {both.size} rows in both {a} and {b}")
        elif set(self.roles) != set(UNSUPERVISED_ROLES):
            raise ValueError(f"{self.name}: roles must be {SUPERVISED_ROLES} or {UNSUPERVISED_ROLES}, got "
                             f"{sorted(self.roles)}")

    def train_final(self) -> np.ndarray:
        """Training plus validation rows, for a refit after the choice (never the test rows)."""
        if self.selection_on_test:
            return self.roles["train"]
        return np.union1d(self.roles["train"], self.roles["val"])


def stable_order(items, salt: str) -> list:
    """The items sorted by SHA-256 of salt and item: a fixed shuffle, the same on every machine."""
    return sorted(items, key=lambda item: (hashlib.sha256(f"{salt}|{item}".encode("utf-8")).hexdigest(), str(item)))


def stable_keys(ids: np.ndarray, salt: str) -> np.ndarray:
    """One 64-bit key per row ID from SHA-256 of salt and ID: argsort gives a fixed shuffle (fast for many rows)."""
    out = np.empty(len(ids), dtype=np.uint64)
    prefix = f"{salt}|".encode("utf-8")
    for i, rid in enumerate(ids.tolist()):
        out[i] = int.from_bytes(hashlib.sha256(prefix + str(rid).encode("utf-8")).digest()[:8], "big")
    return out


def deal(units: list, stratum_of, k: int, salt: str) -> dict:
    """Unit -> group number 1..k, dealt round-robin within each stratum.

    Strata are taken in sorted order and the units of each in stable_order; the deal continues where the previous
    stratum stopped, so every group gets an equal share of every stratum (within one) and the group sizes differ by
    at most one.
    """
    if k < 2:
        raise ValueError(f"need at least 2 groups, got {k}")
    if len(set(units)) != len(units):
        raise ValueError("units repeat")
    by_stratum = {}
    for unit in units:
        by_stratum.setdefault(stratum_of(unit), []).append(unit)
    group, position = {}, 0
    for stratum in sorted(by_stratum, key=str):
        for unit in stable_order(by_stratum[stratum], f"{salt}|{stratum}"):
            group[unit] = position % k + 1
            position += 1
    return group


def deal_rows(ids: np.ndarray, strata: np.ndarray | None, k: int, salt: str) -> np.ndarray:
    """Row position -> group number 1..k, like deal but vectorised for many rows."""
    if k < 2:
        raise ValueError(f"need at least 2 groups, got {k}")
    n = len(ids)
    keys = stable_keys(ids, salt)
    strata = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata).astype(str)
    out = np.empty(n, dtype=np.int64)
    position = 0
    for stratum in sorted(np.unique(strata).tolist(), key=str):
        members = np.flatnonzero(strata == stratum)
        members = members[np.lexsort((ids[members], keys[members]))]
        out[members] = (np.arange(position, position + members.size) % k) + 1
        position += members.size
    return out


def next_group_folds(order: list, members: dict, fold_name, note) -> list:
    """Leave one group out, the next group in `order` validating (the last uses the first).

    members: group -> row positions. fold_name(test group) names each fold; note(val group) describes its
    validation set.
    """
    if len(order) < 3:
        raise ValueError(f"need at least 3 groups for train, validation and test, got {len(order)}")
    folds = []
    for i, test in enumerate(order):
        val = order[(i + 1) % len(order)]
        train = np.concatenate([members[g] for g in order if g not in (test, val)])
        folds.append(Fold(fold_name(test), {"train": train, "val": members[val], "test": members[test]},
                          note(val), False))
    return folds


def identity_fold(first: Fold) -> Fold:
    """The labelled check fold: the first fold's training rows, its test rows as both validation and test."""
    return Fold(CHECK_FOLD, {"train": first.roles["train"], "val": first.roles["test"].copy(),
                             "test": first.roles["test"]}, "validation set = test set (a check, never a result)",
                True)


def folds_hash(ids: np.ndarray, folds: list) -> str:
    """SHA-256 of every fold's role assignment, by row ID."""
    digest = hashlib.sha256()
    for fold in folds:
        digest.update(f"fold {fold.name} {fold.selection_on_test}\n".encode("utf-8"))
        for role in sorted(fold.roles):
            digest.update(f"role {role}\n".encode("utf-8"))
            digest.update("\n".join(ids[fold.roles[role]].tolist()).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def split_report(folds: list, table, target: str | None, groups: tuple) -> dict:
    """Per fold and role: row counts, class counts of a classification target, and the overlap of each declared
    group column across roles (rows of one group in two roles)."""
    out = {}
    for fold in folds:
        entry = {"note": fold.note, "roles": {}}
        for role, positions in fold.roles.items():
            role_entry = {"rows": int(positions.size)}
            if target is not None:
                values, counts = np.unique(table.column(target)[positions].astype(str), return_counts=True)
                role_entry["classes"] = dict(zip(values.tolist(), counts.tolist()))
            entry["roles"][role] = role_entry
        overlaps = {}
        for column in groups:
            values = table.column(column).astype(str)
            sets = {role: set(values[p].tolist()) for role, p in fold.roles.items()}
            names = sorted(sets)
            shared = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]
                      if not (fold.selection_on_test and {a, b} == {"val", "test"})}
            overlaps[column] = shared
        entry["group_overlap"] = overlaps
        out[fold.name] = entry
    return out
