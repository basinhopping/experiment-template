"""Stability: how much a configuration's result changes when only the randomness or the rows change.

Computed by analyze from the label-free output files (assignments.csv, embedding.npy, scores.csv), never from
reference labels:
  across seeds   for each fold, every pair of seeds, on the fold's evaluated rows
  across folds   for each seed, every pair of folds, on the rows both evaluated (subsamples overlap)
Agreement between two results:
  clustering     adjusted Rand index of the cluster labels (noise, -1, counts as one more group)
  outlier        Spearman correlation of the anomaly scores
  embedding      mean Jaccard overlap of each row's n_neighbors nearest neighbours (the task's n_neighbors),
                 among the task's fixed sample of the shared rows (all shared rows when sample_rows is None)
Rows a preparation step removed are left out. A pair with fewer than two shared rows is skipped and counted.

The value that ranks is the mean over every pair, across seeds and across folds. For comparisons, each fold's mean
over its seed pairs is the unit (sweepkit.eval.stats.paired_units_bootstrap): folds are the independent parts.
"""
import itertools

import numpy as np

from sweepkit.analysis.records import complete_seeds, read_npy, read_rows
from sweepkit.splits.base import stable_keys


def _clusters(run) -> dict:
    rows = read_rows(run["folder"], "assignments.csv", "choose")
    return {r["row_id"]: int(r["cluster"]) for r in rows if r["status"] != "removed_by_prep"}


def _scores(run) -> dict:
    rows = read_rows(run["folder"], "scores.csv", "choose")
    return {r["row_id"]: float(r["score"]) for r in rows if r["status"] != "removed_by_prep"}


def _embedding(run) -> dict:
    ids = [r["row_id"] for r in read_rows(run["folder"], "embedding_rows.csv", "choose")]
    coords = read_npy(run["folder"], "embedding.npy", "choose")
    return {"ids": ids, "coords": coords, "index": {i: k for k, i in enumerate(ids)}}


READERS = {"clustering": _clusters, "outlier": _scores, "embedding": _embedding}


def _shared(a, b, task_name: str) -> list:
    keys_a = a["index"].keys() if task_name == "embedding" else a.keys()
    keys_b = b["index"].keys() if task_name == "embedding" else b.keys()
    return sorted(set(keys_a) & set(keys_b))


def _knn_sets(coords: np.ndarray, k: int) -> list:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, coords.shape[0] - 1)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)
    return [set(row[row != i][:k].tolist()) for i, row in enumerate(neighbours)]


def agreement(a, b, task_name: str, task: dict) -> tuple:
    """(value or None, number of shared rows)."""
    shared = _shared(a, b, task_name)
    if len(shared) < 2:
        return None, len(shared)
    if task_name == "clustering":
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score([a[i] for i in shared], [b[i] for i in shared])), len(shared)
    if task_name == "outlier":
        from scipy.stats import spearmanr

        x, y = np.array([a[i] for i in shared]), np.array([b[i] for i in shared])
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            return None, len(shared)
        return float(spearmanr(x, y).statistic), len(shared)
    ids = np.array(shared)
    n = task["sample_rows"]
    if n is not None and n < ids.size:
        ids = ids[np.sort(np.lexsort((ids, stable_keys(ids, task["sample_salt"])))[:n])]
    if ids.size < 3:
        return None, int(ids.size)
    ca = a["coords"][[a["index"][i] for i in ids]]
    cb = b["coords"][[b["index"][i] for i in ids]]
    sets_a, sets_b = _knn_sets(ca, task["n_neighbors"]), _knn_sets(cb, task["n_neighbors"])
    overlap = [len(x & y) / len(x | y) for x, y in zip(sets_a, sets_b)]
    return float(np.mean(overlap)), int(ids.size)


def stability(config: dict, folds: list, task_name: str, task: dict) -> dict:
    seeds = complete_seeds(config, folds)
    reader = READERS[task_name]
    loaded = {(s, f): reader(config["runs"][s][f]) for s in seeds for f in folds}
    by_fold, across_seeds, across_folds, skipped, smallest = {}, [], [], 0, None
    for f in folds:
        values = []
        for s1, s2 in itertools.combinations(seeds, 2):
            value, n = agreement(loaded[(s1, f)], loaded[(s2, f)], task_name, task)
            if value is None:
                skipped += 1
                continue
            values.append(value)
            smallest = n if smallest is None else min(smallest, n)
        if values:
            by_fold[f] = float(np.mean(values))
            across_seeds += values
    for s in seeds:
        for f1, f2 in itertools.combinations(folds, 2):
            value, n = agreement(loaded[(s, f1)], loaded[(s, f2)], task_name, task)
            if value is None:
                skipped += 1
                continue
            across_folds.append(value)
            smallest = n if smallest is None else min(smallest, n)
    every = across_seeds + across_folds
    measure = {"clustering": "adjusted Rand index", "outlier": "Spearman correlation of scores",
               "embedding": f"Jaccard overlap of {task.get('n_neighbors')} nearest neighbours"}[task_name]
    return {"value": float(np.mean(every)) if every else None, "measure": measure,
            "across_seeds": {"mean": float(np.mean(across_seeds)) if across_seeds else None,
                             "n_pairs": len(across_seeds), "by_fold": by_fold},
            "across_folds": {"mean": float(np.mean(across_folds)) if across_folds else None,
                             "n_pairs": len(across_folds)},
            "pairs_skipped": skipped, "fewest_shared_rows": smallest, "seeds": seeds}
