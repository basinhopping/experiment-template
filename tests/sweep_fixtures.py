"""Sweep files used by several tests: small synthetic data, fast models."""
import copy


def cluster_base() -> dict:
    return {
        "data": {"name": "synthetic", "generator": "blobs", "n_rows": 300, "n_features": 4, "n_groups": 3,
                 "noise": 0.6, "seed": 3},
        "split": {"name": "subsamples", "k": 2, "fraction": 0.8, "salt": "t"},
        "task": {"name": "clustering", "criteria": ["silhouette", "dbcv", "n_clusters"], "select": "silhouette",
                 "sample_rows": 200, "sample_salt": "s", "reference": "truth", "reveal_metrics": ["ari"],
                 "criteria_options": {}},
        "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
        "features": {"name": "none"},
        "model": {"name": "kmeans", "n_clusters": 3, "init": "k-means++", "n_init": 2, "max_iter": 100, "tol": 1e-4,
                  "extra": {}},
        "fit": {"name": "clusterer", "device": "cpu", "assign": "fit_rows"},
    }


def classify_base() -> dict:
    return {
        "data": {"name": "synthetic", "generator": "classification", "n_rows": 400, "n_features": 6, "n_groups": 3,
                 "noise": 0.8, "seed": 5},
        "split": {"name": "kfold", "k": 4, "stratify": "truth", "salt": "t", "identity_fold": False},
        "task": {"name": "classification", "target": "truth", "classes": ["c0", "c1", "c2"],
                 "metrics": ["balanced_accuracy", "accuracy", "f1_macro"], "select": "balanced_accuracy", "groups": []},
        "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
        "features": {"name": "none"},
        "model": {"name": "logistic_regression", "c": 1.0, "l1_ratio": 0.0, "class_weight": "none", "max_iter": 200,
                  "extra": {}},
        "fit": {"name": "estimator", "device": "cpu", "control": "none", "refit": "train", "eval_train": False},
    }


def sweep(base: dict, stages: list, sweep_id: str = "t1", purpose: str = "screen", follows=(), folds=None) -> dict:
    split = base["split"]["name"]
    if folds is None:
        folds = {"subsamples": ["sub1", "sub2"], "kfold": ["fold1", "fold2"]}[split]
    return {"sweep_id": sweep_id, "description": "a test sweep", "purpose": purpose, "follows": list(follows),
            "folds": folds, "base": copy.deepcopy(base), "stages": stages}
