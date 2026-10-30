"""CPU and GPU paths agree: for every model of the template's own modules with a cuML (or XGBoost CUDA) path, the same
data through both, compared by adjusted Rand index between the two label sets, agreement of predictions, or the
correlation of scores. Fork-written modules carry their own GPU tests (test_cluster_models, test_embed_models).

These run only on a machine with an NVIDIA GPU and cuML (the rapids environment on a node); elsewhere they skip, and
with SWEEPKIT_REQUIRE naming gpu and cuml they fail instead (node/check.sh sets it). The thresholds were set before
any GPU run, from what the methods guarantee on well separated synthetic data; TESTING.md asks the node session to
record the values seen.
"""
import unittest

import numpy as np

from tests.support import load_catalog, needs_gpu, needs_module


def ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def selected(name: str, raw: dict):
    from sweepkit.registry import MODELS
    from sweepkit.settings import Selected, validate

    return Selected(name, validate(MODELS.get(name).settings_class, raw, "model"))


def host(values) -> np.ndarray:
    from sweepkit.models.base import _host

    return np.asarray(_host(values))


def blobs(n_rows: int, n_features: int, n_groups: int, noise: float, seed: int):
    from sweepkit.data.synthetic import make

    made = make("blobs", n_rows, n_features, n_groups, noise, seed)
    x = made["x"].astype(np.float64)
    return (x - x.mean(axis=0)) / x.std(axis=0), made["columns"]["truth"]


@needs_module("cuml")
@needs_gpu()
class TestClusterersAndEmbedders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.x, cls.truth = blobs(3000, 8, 4, 0.8, 7)

    def both(self, kind: str, name: str, raw: dict):
        from sweepkit.models.base import build_clusterer, build_embedder

        build = build_clusterer if kind == "clusterer" else build_embedder
        out = []
        for device in ("cpu", "cuda"):
            model = build(selected(name, raw), 1, device)
            if kind == "clusterer":
                model.fit(self.x)
                out.append(host(model.labels()))
            else:
                out.append(host(model.fit_transform(self.x)))
            self.assertEqual(model.fallbacks(), [], f"{name} on {device} fell back")
        return out

    def test_kmeans(self):
        cpu, gpu = self.both("clusterer", "kmeans", {"n_clusters": 4, "init": "k-means++", "n_init": 4,
                                                     "max_iter": 300, "tol": 1e-4, "extra": {}})
        self.assertGreaterEqual(ari(cpu, gpu), 0.99)
        self.assertGreaterEqual(ari(self.truth, gpu), 0.99)

    def test_dbscan(self):
        cpu, gpu = self.both("clusterer", "dbscan", {"eps": 0.5, "min_samples": 10, "metric": "euclidean",
                                                     "extra": {}})
        self.assertGreaterEqual(ari(cpu, gpu), 0.99)            # DBSCAN is deterministic up to border points

    def test_hdbscan_both_implementations(self):
        raw = {"min_cluster_size": 50, "min_samples": 10, "cluster_selection_epsilon": 0.0,
               "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "euclidean", "extra": {}}
        for name in ("hdbscan_sklearn",) + (("hdbscan",) if _has("hdbscan") else ()):
            cpu, gpu = self.both("clusterer", name, raw)
            self.assertGreaterEqual(ari(cpu, gpu), 0.95, name)
            self.assertGreaterEqual(ari(self.truth, gpu), 0.95, name)

    def test_pca_and_truncated_svd_span_the_same_subspace(self):
        for name, raw in (("pca", {"n_components": 3, "solver": "full", "whiten": False, "batch_size": None,
                                   "extra": {}}),
                          ("truncated_svd", {"n_components": 3, "n_iter": 7, "extra": {}})):
            cpu, gpu = self.both("embedder", name, raw)
            # the same subspace up to sign and rotation: every canonical correlation near 1
            q1, _ = np.linalg.qr(cpu - cpu.mean(axis=0))
            q2, _ = np.linalg.qr(gpu - gpu.mean(axis=0))
            self.assertGreaterEqual(float(np.linalg.svd(q1.T @ q2, compute_uv=False).min()), 0.999, name)


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


@needs_module("cuml")
@needs_gpu()
class TestDetector(unittest.TestCase):
    def test_isolation_forest_ranks_the_same_outliers(self):
        load_catalog()
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score

        from sweepkit.data.synthetic import make
        from sweepkit.models.base import build_detector

        made = make("outliers", 2000, 5, 3, 1.0, 3)
        x, flag = made["x"].astype(np.float64), made["columns"]["is_outlier"].astype(int)
        raw = {"n_estimators": 300, "max_samples": 256, "max_features": 1.0, "contamination": None, "extra": {}}
        scores = []
        for device in ("cpu", "cuda"):
            model = build_detector(selected("isolation_forest", raw), 1, device)
            model.fit(x)
            scores.append(host(model.score(x)))
            self.assertGreaterEqual(roc_auc_score(flag, scores[-1]), 0.95, device)
        self.assertGreaterEqual(float(spearmanr(scores[0], scores[1])[0]), 0.8)


def estimator_data() -> dict:
    """A classification table (three classes) and a linear regression table, 3000 rows each; the first 2000 rows fit,
    the last 1000 are predicted."""
    from sweepkit.data.synthetic import make

    made = make("classification", 3000, 8, 3, 1.0, 5)
    x = made["x"].astype(np.float32)
    reg = make("regression_linear", 3000, 8, 1, 0.5, 5)
    return {"classification": ((x - x.mean(axis=0)) / x.std(axis=0),
                               np.array([int(t[1:]) for t in made["columns"]["truth"]])),
            "regression": (reg["x"].astype(np.float32), reg["columns"]["target"].astype(np.float64))}


def cpu_and_gpu_predictions(data: dict, name: str, raw: dict, task: str):
    """([CPU predictions, GPU predictions], the true values) on the 1000 rows not fitted."""
    from sweepkit.registry import MODELS

    x, y = data[task]
    out = []
    for device in ("cpu", "cuda"):
        model = MODELS.get(name).target.make(selected(name, raw).settings, task, 1, device)
        model.fit(x[:2000], y[:2000])
        out.append(host(model.predict(x[2000:])))
    return out, y[2000:]


@needs_module("cuml")
@needs_gpu()
class TestEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.data = estimator_data()

    def predictions(self, name: str, raw: dict, task: str):
        return cpu_and_gpu_predictions(self.data, name, raw, task)

    def test_classifiers_agree(self):
        for name, raw in (
                ("logistic_regression", {"c": 1.0, "l1_ratio": 0.0, "class_weight": "none", "max_iter": 1000,
                                         "extra": {}}),
                ("linear_svm", {"c": 1.0, "class_weight": "none", "max_iter": 5000, "extra": {}}),
                ("svm", {"kernel": "rbf", "c": 1.0, "gamma": None, "degree": 3, "class_weight": "none",
                         "probability": False, "extra": {}}),
                ("knn", {"n_neighbors": 15, "weights": "uniform", "metric": "euclidean", "extra": {}}),
                ("random_forest", {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1, "max_features": 0.5,
                                   "bootstrap": True, "class_weight": "none", "n_jobs": 1, "extra": {}})):
            (cpu, gpu), y = self.predictions(name, raw, "classification")
            self.assertGreaterEqual(float(np.mean(cpu.astype(int) == gpu.astype(int))), 0.95, name)
            self.assertGreaterEqual(float(np.mean(gpu.astype(int) == y)), 0.8, name)

    def test_linear_regression_agrees(self):
        for penalty, alpha, ratio in (("none", 0.0, None), ("ridge", 1.0, None), ("lasso", 0.05, None),
                                      ("elastic_net", 0.05, 0.5)):
            raw = {"penalty": penalty, "alpha": alpha, "l1_ratio": ratio, "fit_intercept": True, "max_iter": 5000,
                   "extra": {}}
            (cpu, gpu), _ = self.predictions("linear", raw, "regression")
            self.assertGreaterEqual(float(np.corrcoef(cpu, gpu)[0, 1]), 0.999, penalty)



@needs_module("xgboost")
@needs_gpu()
class TestXGBoostOnCUDA(unittest.TestCase):
    """XGBoost's CUDA path needs no cuML, so this runs in any environment with xgboost and a GPU (main on a node)."""

    def test_xgboost_cuda_agrees(self):
        load_catalog()
        raw = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "num_leaves": None, "subsample": 1.0,
               "colsample_bytree": 1.0, "reg_lambda": 1.0, "n_jobs": 1, "extra": {}}
        (cpu, gpu), y = cpu_and_gpu_predictions(estimator_data(), "xgboost", raw, "classification")
        self.assertGreaterEqual(float(np.mean(cpu == gpu)), 0.97)
        self.assertGreaterEqual(float(np.mean(gpu.astype(int) == y)), 0.8)


if __name__ == "__main__":
    unittest.main()
