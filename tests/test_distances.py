"""Distances between rows (sweepkit.models.distances) and precomputed distance matrices.

Every distance agrees with SciPy's own computation (binary distances on the rows' on/off pattern, Mahalanobis with
the fit rows' covariance), learns from the fit rows only, and gives each method the same result whether the method
computes the distance itself or reads it precomputed: "kmedoids", "agglomerative", "pcoa", "mds", "dbscan",
"hdbscan_sklearn" and "optics". Then a precomputed matrix through the clusterer runner: each subsample's method gets
its own rows' block, silhouette uses the distances, criteria that need coordinates are undefined, and preparation
steps or a matrix whose columns are not the row IDs are refused.
"""
import json
import unittest

import numpy as np

from tests.support import TempRepo, load_catalog

METRICS = ("euclidean", "manhattan", "chebyshev", "cosine", "correlation", "canberra", "braycurtis", "mahalanobis",
           "jaccard", "hamming", "gower")


def model(name: str, raw: dict, kind: str = "clusterer", seed: int = 1):
    from sweepkit.models.base import build_clusterer, build_embedder
    from sweepkit.registry import MODELS
    from sweepkit.settings import Selected, validate

    selected = Selected(name, validate(MODELS.get(name).settings_class, raw, "model"))
    built = (build_clusterer if kind == "clusterer" else build_embedder)(selected, seed, "cpu")
    built.set_columns(tuple(f"c{j}" for j in range(8)), {})
    return built


def counts(n_rows: int, seed: int):
    from sweepkit.data.synthetic import make

    made = make("counts", n_rows, 30, 3, 0.5, seed)
    return made["x"].astype(np.float64), made["columns"]["truth"]


class TestDistances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        rng = np.random.default_rng(0)
        cls.fit = rng.normal(size=(40, 4))
        cls.new = rng.normal(size=(7, 4))

    def test_each_distance_agrees_with_scipy(self):
        from scipy.spatial.distance import cdist

        from sweepkit.models.distances import Distances

        for metric in METRICS:
            if metric == "gower":
                continue
            d = Distances(metric, {}).fit(self.fit).between(self.new, self.fit)
            if metric in ("jaccard", "hamming"):
                expected = cdist(self.new != 0, self.fit != 0, metric=metric)
            elif metric == "mahalanobis":
                expected = cdist(self.new, self.fit, metric="mahalanobis",
                                 VI=np.linalg.inv(np.cov(self.fit, rowvar=False)))
            else:
                expected = cdist(self.new, self.fit, metric="cityblock" if metric == "manhattan" else metric)
            self.assertTrue(np.allclose(d, expected), metric)

    def test_binary_distances_read_on_and_off(self):
        from sweepkit.models.distances import Distances

        a = np.array([[1.0, 0.0, 2.0, 0.0]])
        b = np.array([[3.0, 0.0, 0.0, 0.0]])
        # on: columns {0, 2} against {0}: jaccard 1 disagreement of 2 columns on in either; hamming 1 of 4 columns
        self.assertAlmostEqual(float(Distances("jaccard", {}).fit(a).between(a, b)[0, 0]), 0.5)
        self.assertAlmostEqual(float(Distances("hamming", {}).fit(a).between(a, b)[0, 0]), 0.25)

    def test_gower_between_agrees_with_the_fit_rows_matrix(self):
        from sweepkit.models.cluster.mixed import gower_matrix
        from sweepkit.models.distances import Distances

        x = self.fit.copy()
        x[:, 3] = np.round(np.abs(x[:, 3]) * 2)             # a categorical column of codes
        categorical = {3: ["a", "b", "c", "d", "e", "f", "g", "h"]}
        d = Distances("gower", categorical).fit(x)
        self.assertTrue(np.allclose(d.between(x, x), gower_matrix(x, categorical), atol=1e-6))

    def test_learned_from_the_fit_rows_only(self):
        from sweepkit.models.distances import Distances

        for metric in ("mahalanobis", "gower"):
            d = Distances(metric, {}).fit(self.fit)
            before = d.between(self.new, self.fit[:5])
            changed = self.new * 100                          # other rows never change what was learned
            d.between(changed, self.fit[:5])
            self.assertTrue(np.array_equal(before, d.between(self.new, self.fit[:5])), metric)

    def test_precomputed_blocks_are_checked(self):
        from sweepkit.models.distances import check_square

        with self.assertRaisesRegex(ValueError, "square"):
            check_square(np.zeros((3, 4)))
        with self.assertRaisesRegex(ValueError, "at least 0"):
            check_square(-np.ones((2, 2)))


class TestPrecomputedEqualsComputed(unittest.TestCase):
    """The same method on Bray-Curtis distances, computed by the method or read precomputed."""

    @classmethod
    def setUpClass(cls):
        load_catalog()
        from scipy.spatial.distance import cdist

        cls.x, cls.truth = counts(150, 2)
        cls.d = cdist(cls.x, cls.x, metric="braycurtis")

    def test_kmedoids(self):
        raw = {"n_clusters": 3, "method": "alternate", "max_iter": 100, "n_init": 4}
        computed = model("kmedoids", {**raw, "metric": "braycurtis"})
        computed.fit(self.x)
        given = model("kmedoids", {**raw, "metric": "precomputed"})
        given.fit(self.d)
        self.assertEqual(computed.labels().tolist(), given.labels().tolist())
        self.assertEqual(given.assign(self.d[:10]).tolist(), given.labels()[:10].tolist())
        from sklearn.metrics import adjusted_rand_score

        self.assertGreaterEqual(adjusted_rand_score(self.truth, given.labels()), 0.9)

    def test_agglomerative(self):
        raw = {"linkage": "average", "cut": "n_clusters", "n_clusters": 3, "distance": None, "extra": {}}
        computed = model("agglomerative", {**raw, "metric": "braycurtis"})
        computed.fit(self.x)
        given = model("agglomerative", {**raw, "metric": "precomputed"})
        given.fit(self.d)
        self.assertEqual(computed.labels().tolist(), given.labels().tolist())
        self.assertEqual(given.assign(self.d[:10]).tolist(), given.labels()[:10].tolist())
        for metric in ("mahalanobis", "jaccard", "gower"):
            other = model("agglomerative", {**raw, "metric": metric})
            other.fit(self.x)
            self.assertEqual(other.assign(self.x[:5]).tolist(), other.labels()[:5].tolist(), metric)

    def test_pcoa_and_mds(self):
        computed = model("pcoa", {"n_components": 2, "metric": "braycurtis"}, "embedder")
        given = model("pcoa", {"n_components": 2, "metric": "precomputed"}, "embedder")
        a, b = computed.fit_transform(self.x), given.fit_transform(self.d)
        self.assertTrue(np.allclose(np.abs(a), np.abs(b), atol=1e-4))
        self.assertTrue(np.allclose(given.transform(self.d[:5]), b[:5], atol=1e-4))
        raw = {"n_components": 2, "metric_mds": True, "init": "random", "n_init": 1, "max_iter": 50, "eps": 1e-3,
               "normalized_stress": False, "extra": {}}
        m1 = model("mds", {**raw, "metric": "braycurtis"}, "embedder").fit_transform(self.x)
        m2 = model("mds", {**raw, "metric": "precomputed"}, "embedder").fit_transform(self.d)
        self.assertTrue(np.allclose(m1, m2, atol=1e-5))

    def test_density_methods(self):
        from scipy.spatial.distance import cdist

        x = self.x / self.x.sum(axis=1, keepdims=True)
        d = cdist(x, x)
        for name, raw in (("dbscan", {"eps": 0.05, "min_samples": 5, "extra": {}}),
                          ("hdbscan_sklearn", {"min_cluster_size": 10, "min_samples": 5,
                                               "cluster_selection_epsilon": 0.0, "cluster_selection_method": "eom",
                                               "allow_single_cluster": False, "extra": {}}),
                          ("optics", {"min_samples": 10, "max_eps": None, "cluster_method": "xi", "xi": 0.05,
                                      "eps": None, "extra": {}})):
            computed = model(name, {**raw, "metric": "euclidean"})
            computed.fit(x)
            given = model(name, {**raw, "metric": "precomputed"})
            given.fit(d)
            self.assertEqual(computed.labels().tolist(), given.labels().tolist(), name)


class TestPrecomputedThroughTheRunner(unittest.TestCase):
    def sweep(self, prep: list, path: str) -> dict:
        base = {"data": {"name": "table", "path": path, "format": "csv", "sheet_or_key": None, "row_id": "row_id",
                         "features": ["*"], "exclude": [], "keep": ["truth"], "categorical": "error",
                         "filter_column": None, "filter_values": []},
                "split": {"name": "subsamples", "k": 2, "fraction": 0.7, "salt": "t"},
                "task": {"name": "clustering", "criteria": ["silhouette", "calinski_harabasz", "n_clusters"],
                         "select": "silhouette", "sample_rows": None, "sample_salt": "t", "reference": "truth",
                         "reveal_metrics": ["ari"], "criteria_options": {}},
                "prep": prep, "features": {"name": "none"},
                "model": {"name": "kmedoids", "n_clusters": 3, "metric": "precomputed", "method": "alternate",
                          "max_iter": 100, "n_init": 4},
                "fit": {"name": "clusterer", "device": "cpu", "assign": "fit_rows"}}
        return {"sweep_id": "pre1", "description": "t", "purpose": "learning", "follows": [], "folds": ["sub1", "sub2"],
                "base": base, "stages": [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]}]}

    def write_matrix(self, root, name: str, ids: list, columns: list, d: np.ndarray, truth) -> None:
        (root / "data").mkdir(exist_ok=True)
        lines = [",".join(["row_id"] + columns + ["truth"])]
        lines += [",".join([ids[i]] + [f"{v:.10g}" for v in d[i]] + [truth[i]]) for i in range(len(ids))]
        (root / "data" / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_each_subsample_gets_its_block_and_criteria_follow(self):
        load_catalog()
        from scipy.spatial.distance import cdist

        from sweepkit.sweep.design import expand
        from sweepkit.sweep.runner import execute

        x, truth = counts(90, 3)
        d = cdist(x, x, metric="braycurtis")
        ids = [f"s{i:02d}" for i in range(90)]
        order = list(np.random.default_rng(0).permutation(90))          # columns in another order than the rows
        with TempRepo() as root:
            self.write_matrix(root, "d.csv", ids, [ids[j] for j in order], d[:, order], truth)
            jobs, _ = expand(self.sweep([], "d.csv"))
            for job in jobs:
                execute(job, "cpu", lambda m: None, runs_root=root / "runs")
                folder = root / "runs" / job["run_id"]
                criteria = json.loads((folder / "criteria.json").read_text(encoding="utf-8"))
                self.assertIsNotNone(criteria["criteria"]["silhouette"])
                self.assertIsNone(criteria["criteria"]["calinski_harabasz"])
                self.assertIn("precomputed", json.dumps(criteria))
                reveal = json.loads((folder / "reveal.json").read_text(encoding="utf-8"))
                self.assertGreaterEqual(reveal["metrics"]["ari"], 0.9)
            # the matrix through a preparation step, and a matrix whose columns are not the row IDs, are refused
            scale = {"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}
            job = expand(self.sweep([scale], "d.csv"))[0][0]
            with self.assertRaisesRegex(ValueError, "no preparation steps"):
                execute(job, "cpu", lambda m: None, runs_root=root / "runs_prep")
            self.write_matrix(root, "bad.csv", ids, [f"x{j}" for j in range(90)], d, truth)
            job = expand(self.sweep([], "bad.csv"))[0][0]
            with self.assertRaisesRegex(ValueError, "columns must be its row IDs"):
                execute(job, "cpu", lambda m: None, runs_root=root / "runs_bad")


if __name__ == "__main__":
    unittest.main()
