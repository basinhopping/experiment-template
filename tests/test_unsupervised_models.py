"""The first clustering, embedding and outlier methods of the template, on data with known structure.

Centroid and mixture models, density methods and k-prototypes are scored against planted groups by the adjusted Rand
index (ARI). The linear embedders are checked on data whose structure is known: a low-rank table, independent
sources mixed together, a non-negative product. The detectors are scored against planted outliers by the area under
the ROC curve (AUROC). Each method is built the way its family's runner builds it. The last classes run the methods
through their fit procedures ("clusterer", "embedder", "detector") and a whole sweep of the task "outlier" through
run --local.

Thresholds were set from runs in the container (Python 3.13, scikit-learn 1.9.1, SciPy 1.18.1) over ten data seeds
or more, a margin below the worst value seen; that value is in a comment beside each. Data seeds were chosen where the
planted groups are apart (centres more than 8 units apart): blobs 5-D seeds 1 to 3, blobs 2-D seeds 2 to 4. Methods
from packages this machine lacks (hdbscan, fast_hdbscan, kmedoids) skip, and fail instead where SWEEPKIT_REQUIRE names
the package (the node); their thresholds are first proposals, to be confirmed by that run.
"""
import json
import unittest
import warnings

from tests.support import TempRepo, load_catalog, needs_module

SEPARATED_5D = (1, 2, 3)
SEPARATED_2D = (2, 3, 4)


def planted(generator: str, n_rows: int, n_features: int, n_groups: int, noise: float, seed: int, scale: bool = True):
    """(x, columns, made) from a synthetic generator; x standardized when scale is true."""
    import numpy as np

    from sweepkit.data.synthetic import make

    made = make(generator, n_rows, n_features, n_groups, noise, seed)
    x = np.asarray(made["x"], dtype=np.float64)
    if scale:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, made["columns"], made


def build(raw: dict, n_columns: int, seed: int = 7, categorical=None):
    """The model of a model section, built as its family's runner builds it, with its columns set."""
    from sweepkit.models.base import build_clusterer, build_detector, build_embedder
    from sweepkit.registry import MODELS
    from sweepkit.settings import Selected

    entry, settings = MODELS.validate_section(raw, "model")
    builder = {"clusterer": build_clusterer, "embedder": build_embedder, "detector": build_detector}
    model = builder[entry.meta["family"]](Selected(entry.name, settings), seed, "cpu")
    model.set_columns(tuple(f"f{j}" for j in range(n_columns)), categorical or {})
    return model


def fitted(raw: dict, x, seed: int = 7, categorical=None):
    model = build(raw, x.shape[1], seed, categorical)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
    return model


def ari(truth, labels) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(truth, labels))


def auroc(columns, scores) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(columns["is_outlier"]).astype(int), scores))


def job_with(model: dict, device: str = "cpu"):
    """A validated job around a model section (the clustering fixture's data, split and task)."""
    from sweepkit.runs.spec import validate_job
    from tests.sweep_fixtures import cluster_base

    base = cluster_base()
    base["model"] = model
    base["fit"]["device"] = device
    return validate_job(base)


GMM = {"name": "gmm", "n_components": 3, "covariance_type": "full", "n_init": 2, "max_iter": 200, "reg_covar": 1e-6,
       "extra": {}}
BAYESIAN_GMM = {"name": "bayesian_gmm", "n_components": 8, "covariance_type": "full",
                "weight_concentration_prior_type": "dirichlet_process", "weight_concentration_prior": 0.01,
                "max_iter": 500, "extra": {}}
KMEANS = {"name": "kmeans", "n_clusters": 3, "init": "k-means++", "n_init": 4, "max_iter": 300, "tol": 1e-4,
          "extra": {}}
MINIBATCH_KMEANS = {"name": "minibatch_kmeans", "n_clusters": 3, "batch_size": 128, "n_init": 3, "max_iter": 100,
                    "extra": {}}
BISECTING = {"name": "bisecting_kmeans", "n_clusters": 3, "bisecting_strategy": "biggest_inertia", "n_init": 2,
             "extra": {}}
FUZZY = {"name": "fuzzy_cmeans", "n_clusters": 3, "m": 2.0, "max_iter": 300, "tol": 1e-6}
KMEDOIDS = {"name": "kmedoids", "n_clusters": 3, "metric": "euclidean", "method": "alternate", "max_iter": 100,
            "n_init": 3}
KPROTOTYPES = {"name": "kprototypes", "n_clusters": 3, "gamma": None, "n_init": 4, "max_iter": 100}
HDBSCAN = {"min_cluster_size": 15, "min_samples": None, "cluster_selection_epsilon": 0.0,
           "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "euclidean", "extra": {}}
OPTICS_XI = {"name": "optics", "min_samples": 20, "max_eps": None, "cluster_method": "xi", "xi": 0.1, "eps": None,
             "metric": "euclidean", "extra": {}}
OPTICS_DBSCAN = dict(OPTICS_XI, min_samples=10, cluster_method="dbscan", xi=None, eps=0.5)
MEAN_SHIFT = {"name": "mean_shift", "bandwidth": None, "bin_seeding": True, "cluster_all": True, "max_iter": 300,
              "extra": {}}
TRUNCATED_SVD = {"name": "truncated_svd", "n_components": 2, "n_iter": 7, "extra": {}}
KERNEL_PCA = {"name": "kernel_pca", "n_components": 2, "kernel": "linear", "gamma": None, "degree": 3,
              "fit_inverse_transform": True, "extra": {}}
ICA = {"name": "ica", "n_components": 3, "algorithm": "parallel", "fun": "logcosh", "max_iter": 1000, "extra": {}}
FACTOR_ANALYSIS = {"name": "factor_analysis", "n_components": 2, "rotation": "none", "max_iter": 1000, "extra": {}}
NMF = {"name": "nmf", "n_components": 3, "init": "nndsvda", "beta_loss": "frobenius", "max_iter": 1000,
       "alpha_w": 0.0, "l1_ratio": 0.0, "extra": {}}
ISOLATION_FOREST = {"name": "isolation_forest", "n_estimators": 100, "max_samples": None, "max_features": 1.0,
                    "contamination": None, "extra": {}}
LOF = {"name": "lof", "n_neighbors": 20, "metric": "euclidean", "contamination": None, "extra": {}}
ONE_CLASS_SVM = {"name": "one_class_svm", "kernel": "rbf", "nu": 0.05, "gamma": None, "extra": {}}
ELLIPTIC_ENVELOPE = {"name": "elliptic_envelope", "contamination": 0.05, "support_fraction": None, "extra": {}}


class TestCentroidAndMixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.blobs = {seed: planted("blobs", 600, 5, 3, 1.0, seed) for seed in SEPARATED_5D}

    def test_gmm_finds_the_groups_and_bic_is_lowest_at_three(self):
        import numpy as np

        for seed, (x, columns, _) in self.blobs.items():
            model = fitted(GMM, x, seed)
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95, seed)      # 1.000 on 9 seeds
            self.assertEqual(model.assign(x).tolist(), model.labels().tolist())
            info = model.info()
            self.assertTrue(info["converged"])
            self.assertGreater(info["bic"], info["aic"])          # BIC charges more per parameter than AIC here
            probability = model.soft()["probability"]
            self.assertTrue(((probability >= 1 / 3 - 1e-9) & (probability <= 1)).all())
        x = self.blobs[1][0]
        bic = [fitted(dict(GMM, n_components=k), x).info()["bic"] for k in (2, 3, 4, 5)]
        self.assertEqual(2 + int(np.argmin(bic)), 3)                                           # 3 on 3 seeds

    def test_bayesian_gmm_switches_off_the_components_it_does_not_need(self):
        for seed, (x, columns, _) in self.blobs.items():
            model = fitted(BAYESIAN_GMM, x, seed)
            self.assertEqual(model.info()["effective_components"], 3, seed)                 # 3 on 9 seeds
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95, seed)      # 1.000 on 9 seeds

    def test_kmeans_and_minibatch_kmeans_find_the_groups(self):
        for raw in (KMEANS, MINIBATCH_KMEANS):
            for seed, (x, columns, _) in self.blobs.items():
                model = fitted(raw, x, seed)
                self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95, f"{raw['name']} {seed}")
                # 1.000 on seeds 1 to 3, 0.970 at worst over data seeds 1 to 10
                self.assertEqual(model.assign(x).tolist(), model.labels().tolist())
                self.assertGreater(model.info()["inertia"], 0)

    def test_bisecting_kmeans(self):
        for seed, (x, columns, _) in self.blobs.items():
            model = fitted(BISECTING, x, seed)
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95, seed)      # 1.000 on 9 seeds
            self.assertGreater(model.info()["inertia"], 0)
            self.assertEqual(model.params()["n_clusters"], 3)

    def test_fuzzy_cmeans_memberships_sum_to_one(self):
        import numpy as np

        for seed, (x, columns, _) in self.blobs.items():
            model = fitted(FUZZY, x, seed)
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95, seed)      # 1.000 on 9 seeds
            self.assertTrue(np.allclose(model.u.sum(axis=1), 1.0, atol=1e-9))
            self.assertTrue(np.array_equal(model.soft()["probability"], model.u.max(axis=1)))
            self.assertEqual(model.extra_files()["memberships.npy"].shape, (600, 3))
            # the largest membership is the nearest centre, so fit rows keep their labels when assigned again
            self.assertEqual(model.assign(x).tolist(), model.labels().tolist())
        with self.assertRaisesRegex(ValueError, "m must be above 1"):
            fitted(dict(FUZZY, m=1.0), self.blobs[1][0])

    def test_kmedoids_medoids_are_rows_at_a_fixed_point(self):
        import numpy as np
        from scipy.spatial.distance import cdist

        for seed, (x, columns, _) in self.blobs.items():
            model = fitted(KMEDOIDS, x, seed)
            labels = model.labels()
            self.assertGreaterEqual(ari(columns["truth"], labels), 0.95, seed)              # 1.000 on 9 seeds
            self.assertTrue(np.array_equal(model.medoids, x[model.medoid_index]))
            d = cdist(x, x)
            self.assertTrue(np.array_equal(labels, np.argmin(d[:, model.medoid_index], axis=1)))
            self.assertAlmostEqual(model.info()["total_distance"],
                                   float(d[np.arange(600), model.medoid_index[labels]].sum()), places=6)
            for c, medoid in enumerate(model.medoid_index):
                members = np.flatnonzero(labels == c)
                within = d[np.ix_(members, members)].sum(axis=0)
                self.assertAlmostEqual(float(d[medoid, members].sum()), float(within.min()), places=6)
            self.assertEqual(model.assign(x).tolist(), labels.tolist())
            self.assertEqual(model.params()["medoid_rows"], [int(i) for i in model.medoid_index])
        with self.assertRaisesRegex(ValueError, "must be below the rows"):
            fitted(dict(KMEDOIDS, n_clusters=10), self.blobs[1][0][:10])

    def test_kmedoids_with_gower_on_mixed_columns(self):
        for seed in (1, 2, 3):
            x, columns, made = planted("mixed", 300, 6, 3, 0.8, seed, scale=False)
            model = fitted(dict(KMEDOIDS, metric="gower"), x, seed, made["categorical"])
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.9, seed)        # 0.951 on 5 seeds
            # new rows are placed with the fit rows' ranges; the fit rows themselves land on their own clusters
            self.assertTrue(model.can_assign())
            self.assertEqual(model.assign(x).tolist(), model.labels().tolist())

    @needs_module("kmedoids")
    def test_kmedoids_fasterpam(self):
        x, columns, _ = self.blobs[1]
        model = fitted(dict(KMEDOIDS, method="fasterpam"), x)
        self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95)

    def test_kprototypes_on_mixed_columns(self):
        import numpy as np

        for seed in (1, 2, 3):
            x, columns, made = planted("mixed", 400, 6, 3, 0.8, seed, scale=False)
            model = fitted(KPROTOTYPES, x, seed, made["categorical"])
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.85, seed)       # 0.941 on 10 seeds
            numeric = [j for j in range(x.shape[1]) if j not in made["categorical"]]
            self.assertAlmostEqual(model.gamma_used, 0.5 * float(x[:, numeric].std(axis=0).mean()), places=12)
            self.assertEqual(model.assign(x).tolist(), model.labels().tolist())
            self.assertEqual(model.params()["categorical_columns"], ["f3", "f4", "f5"])
            self.assertTrue(np.isin(model.centers_c, [0, 1, 2, 3]).all())
        with self.assertRaisesRegex(ValueError, "no categorical column"):
            fitted(KPROTOTYPES, self.blobs[1][0])


class TestDensity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.blobs = {seed: planted("blobs", 600, 2, 3, 0.6, seed) for seed in SEPARATED_2D}

    def test_hdbscan_sklearn(self):
        import numpy as np

        for seed, (x, columns, _) in self.blobs.items():
            model = fitted({"name": "hdbscan_sklearn", **HDBSCAN}, x, seed)
            labels = model.labels()
            self.assertGreaterEqual(ari(columns["truth"], labels), 0.95, seed)              # 1.000 on 7 seeds
            probability = model.soft()["probability"]
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            self.assertTrue((probability[labels < 0] == 0).all())
            self.assertFalse(model.can_assign())             # on the CPU; cuML's has approximate_predict
            self.assertGreater(np.unique(labels[labels >= 0]).size, 0)

    def test_cuml_hdbscan_takes_euclidean_distances_only(self):
        from sweepkit import devices
        from sweepkit.runs.spec import check_job

        for name in ("hdbscan_sklearn", "hdbscan"):
            manhattan = {"name": name, **dict(HDBSCAN, metric="manhattan")}
            with self.assertRaisesRegex(ValueError, "Euclidean distances only", msg=name):
                check_job(job_with(manhattan, "cuda"))
            check_job(job_with(manhattan, "cpu"))
            check_job(job_with(manhattan, "auto"))          # auto runs it on the CPU instead
        reason = "cuML's HDBSCAN takes Euclidean distances only, not manhattan"
        self.assertEqual(devices.resolve("auto", "cuda", "cuml", reason), "cpu")
        with self.assertRaisesRegex(ValueError, "Euclidean distances only"):
            devices.resolve("cuda", "cuda", "cuml", reason)
        self.assertEqual(devices.auto_note("auto", "cuda", "cpu", reason), [f"fit.device auto ran on the CPU: {reason}"])
        self.assertEqual(devices.auto_note("auto", "cpu", "cpu", reason), [])

    def test_auto_records_why_a_setting_ran_on_the_cpu(self):
        import json
        import tempfile
        from pathlib import Path

        from sweepkit.sweep.design import expand
        from sweepkit.sweep.runner import execute
        from tests.support import TempRepo
        from tests.sweep_fixtures import cluster_base, sweep

        base = cluster_base()
        base["model"] = {"name": "hdbscan_sklearn", **dict(HDBSCAN, metric="manhattan")}
        base["fit"]["device"] = "auto"
        jobs, _ = expand(sweep(base, [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]}]))
        with TempRepo(), tempfile.TemporaryDirectory() as d:
            execute(jobs[0], "cuda", lambda m: None, runs_root=Path(d))            # a worker that offers the GPU
            provenance = json.loads((Path(d) / jobs[0]["run_id"] / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual((provenance["device_requested"], provenance["device_used"]), ("auto", "cpu"))
        self.assertEqual(provenance["fallbacks"],
                         ["fit.device auto ran on the CPU: cuML's HDBSCAN takes Euclidean distances only, not "
                          "manhattan"])

    @needs_module("hdbscan")
    def test_hdbscan_package_gives_outlier_scores_and_assigns_new_rows(self):
        import numpy as np

        x, columns, _ = self.blobs[2]
        model = fitted({"name": "hdbscan", **HDBSCAN}, x)
        labels = model.labels()
        self.assertGreaterEqual(ari(columns["truth"], labels), 0.95)
        soft = model.soft()
        self.assertEqual(sorted(soft), ["outlier_score", "probability"])
        self.assertTrue(np.isfinite(soft["outlier_score"]).all())
        self.assertTrue(model.can_assign())
        self.assertGreaterEqual(float((model.assign(x) == labels).mean()), 0.95)

    @needs_module("fast_hdbscan")
    def test_fast_hdbscan(self):
        x, columns, _ = self.blobs[2]
        model = fitted({"name": "fast_hdbscan", "min_cluster_size": 15, "min_samples": None,
                        "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.0,
                        "allow_single_cluster": False, "extra": {}}, x)
        self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95)
        self.assertFalse(model.can_assign())

    @needs_module("fast_hdbscan")
    def test_plscan_picks_its_own_resolution(self):
        x, columns, _ = self.blobs[2]
        model = fitted({"name": "plscan", "base_min_cluster_size": 10, "min_samples": 5, "max_layers": 4,
                        "layer_similarity_threshold": 0.2, "extra": {}}, x)
        self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.9)

    @needs_module("fast_hdbscan")
    def test_flasc_branches_split_a_y_into_its_arms(self):
        import numpy as np

        x, columns, _ = planted("y_shape", 600, 2, 3, 0.15, 1, scale=False)
        # A Y of 600 rows, 200 to an arm (arms 5 long), and a round blob of 300 rows about 10 away from it. HDBSCAN
        # must keep the Y whole, so that only FLASC's branches can divide it. On the node, min_cluster_size 30 found 5
        # clusters (2026-10-27: the arms came from HDBSCAN), and the Y alone with min_cluster_size 250 and
        # allow_single_cluster left every row as noise (2026-10-28: the Y never splits into two parts of 250 rows, and
        # fast_hdbscan 0.3.2 then finds no cluster, with or without allow_single_cluster). With the blob, the rows
        # split once, into the Y and the blob, each a cluster; no part of the Y can be one of its own (an arm has 200
        # rows), so the Y is one of two clusters, labeled whole.
        blob = np.random.default_rng(3).normal([15.0, 0.0], 0.3, size=(300, 2))
        x = np.vstack([x, blob])
        arms = np.concatenate([np.asarray(columns["branch"]), np.array(["blob"] * 300)])
        model = fitted({"name": "flasc_branches", "min_cluster_size": 250, "min_samples": 30,
                        "allow_single_cluster": False, "min_branch_size": 20, "label_sides_as_branches": False,
                        "extra": {}}, x)
        labels = model.labels()
        table = {int(label): {str(a): int(c) for a, c in zip(*np.unique(arms[labels == label], return_counts=True))}
                 for label in np.unique(labels)}
        self.assertEqual(model.info()["hdbscan_clusters"], 2, f"points of each arm by label: {table}")
        on_the_y = np.unique(labels[(labels >= 0) & (arms != "blob")])
        self.assertGreaterEqual(on_the_y.size, 3, f"points of each arm by label: {table}")
        # no label holds rows of both the Y and the blob: branches divide clusters, never join them
        for label in np.unique(labels[labels >= 0]):
            blob_rows = arms[labels == label] == "blob"
            self.assertTrue(blob_rows.all() or not blob_rows.any(), f"points of each arm by label: {table}")
        # Each arm should get a branch of its own: a label whose points come at least 90% from one arm. FLASC can
        # also label where the arms meet, so agreement over every point is not the measure (ARI 0.128 on the node,
        # 2026-10-27, against a bar of 0.5 set before any run); the table in the message shows each label's arms.
        pure = set()
        for label in on_the_y:
            members = arms[labels == label]
            values, counts = np.unique(members, return_counts=True)
            if members.size >= 20 and counts.max() >= 0.9 * members.size:
                pure.add(str(values[counts.argmax()]))
        self.assertEqual(pure, {"arm0", "arm1", "arm2"}, f"points of each arm by label: {table}")

    @needs_module("fast_hdbscan")
    def test_flasc_branches_says_why_when_hdbscan_finds_no_cluster(self):
        import numpy as np

        # One round blob of 400 rows with min_cluster_size 300: the rows never split into two parts of 300, so
        # fast_hdbscan 0.3.2 finds no cluster, with or without allow_single_cluster (its EOM selection returns before
        # it looks at allow_single_cluster when nothing split), and there is nothing to branch.
        x = np.random.default_rng(4).normal(0.0, 1.0, size=(400, 2))
        for single in (False, True):
            with self.subTest(allow_single_cluster=single), \
                    self.assertRaisesRegex(ValueError, "HDBSCAN left every row as noise"):
                fitted({"name": "flasc_branches", "min_cluster_size": 300, "min_samples": 30,
                        "allow_single_cluster": single, "min_branch_size": 20, "label_sides_as_branches": False,
                        "extra": {}}, x)

    def test_optics_by_xi_and_by_a_dbscan_cut(self):
        for seed in SEPARATED_2D[:2]:
            x, columns, _ = self.blobs[seed]
            for raw in (OPTICS_XI, OPTICS_DBSCAN):
                model = fitted(raw, x, seed)
                self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95,
                                        f"{raw['cluster_method']} {seed}")                  # 1.000 on 7 seeds
                self.assertFalse(model.can_assign())

    def test_optics_settings_belong_to_their_extraction_method(self):
        from sweepkit.runs.spec import check_job

        for bad in (dict(OPTICS_XI, xi=None), dict(OPTICS_XI, eps=0.5), dict(OPTICS_DBSCAN, eps=None),
                    dict(OPTICS_DBSCAN, xi=0.1)):
            with self.assertRaisesRegex(ValueError, "xi with cluster_method xi", msg=str(bad)):
                check_job(job_with(bad))
        check_job(job_with(OPTICS_XI))
        check_job(job_with(OPTICS_DBSCAN))

    def test_mean_shift_estimates_its_bandwidth(self):
        for seed, (x, columns, _) in self.blobs.items():
            model = fitted(MEAN_SHIFT, x, seed)
            self.assertGreaterEqual(ari(columns["truth"], model.labels()), 0.95, seed)      # 1.000 on 7 seeds
            self.assertTrue(model.can_assign())
            self.assertEqual(model.assign(x).tolist(), model.labels().tolist())
        self.assertFalse(fitted(dict(MEAN_SHIFT, cluster_all=False), self.blobs[2][0]).can_assign())


class TestLinearEmbedders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.low_rank = {seed: planted("low_rank", 400, 10, 2, 0.05, seed, scale=False)[0] for seed in (1, 2, 3)}

    def embed(self, raw, x, seed=7):
        model = build(raw, x.shape[1], seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model, model.fit_transform(x)

    def test_truncated_svd_keeps_a_low_rank_table(self):
        import numpy as np

        for seed, x in self.low_rank.items():
            model, e = self.embed(TRUNCATED_SVD, x, seed)
            self.assertEqual(e.shape, (400, 2))
            self.assertGreaterEqual(model.info()["explained_variance"], 0.99, seed)           # 0.9995 on 10 seeds
            rebuilt = model.inverse_transform(e)
            self.assertLessEqual(float(((rebuilt - x) ** 2).mean() / (x ** 2).mean()), 0.01)  # 0.0005 on 10 seeds
            # new rows go through the same projection (fit_transform and transform round differently)
            self.assertTrue(np.allclose(model.transform(x), e, rtol=1e-4, atol=1e-4 * float(abs(e).max())))

    def test_kernel_pca_with_a_linear_kernel_is_pca(self):
        import numpy as np

        for seed, x in self.low_rank.items():
            model, e = self.embed(KERNEL_PCA, x, seed)
            centred = x - x.mean(axis=0)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            scores = centred @ vt[:2].T
            for j in range(2):
                self.assertGreaterEqual(abs(np.corrcoef(e[:, j], scores[:, j])[0, 1]), 0.999)  # 1.000 on 10 seeds
            self.assertTrue(model.can_inverse())
            rebuilt = model.inverse_transform(e)
            self.assertLessEqual(float(((rebuilt - x) ** 2).mean() / (centred ** 2).mean()), 0.05)  # 0.013 at worst
        model, _ = self.embed(dict(KERNEL_PCA, fit_inverse_transform=False), self.low_rank[1])
        self.assertFalse(model.can_inverse())

    def test_ica_recovers_independent_sources_up_to_sign_and_order(self):
        import numpy as np

        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            sources = np.stack([rng.laplace(size=600), rng.uniform(-1, 1, size=600), rng.standard_t(5, size=600)],
                               axis=1)
            x = sources @ rng.uniform(0.5, 2, size=(3, 6))
            _, e = self.embed(ICA, x, seed)
            r = np.abs(np.corrcoef(np.hstack([sources, e]).T)[:3, 3:])
            self.assertGreaterEqual(float(r.max(axis=1).min()), 0.95, seed)                  # 0.989 on 10 seeds
            self.assertEqual(sorted(np.argmax(r, axis=1).tolist()), [0, 1, 2])               # one component each

    def test_factor_analysis_finds_the_low_rank_subspace(self):
        import numpy as np

        def smallest_cosine(a, b):
            qa, _ = np.linalg.qr(a.T)
            qb, _ = np.linalg.qr(b.T)
            return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())

        for seed, x in self.low_rank.items():
            _, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
            for rotation in ("none", "varimax"):
                model, e = self.embed(dict(FACTOR_ANALYSIS, rotation=rotation), x, seed)
                self.assertEqual(e.shape, (400, 2))
                self.assertGreaterEqual(smallest_cosine(model.model.components_, vt[:2]), 0.99)  # 1.000 on 10 seeds
                self.assertFalse(model.can_inverse())

    def test_nmf_on_a_non_negative_product(self):
        import numpy as np

        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            w = rng.uniform(0, 1, size=(300, 3))
            h = rng.uniform(0, 1, size=(3, 12)) * (rng.uniform(size=(3, 12)) < 0.6)
            x = w @ h
            for beta_loss in ("frobenius", "kullback-leibler"):
                model, e = self.embed(dict(NMF, beta_loss=beta_loss), x, seed)
                self.assertTrue((e >= 0).all())
                error = np.linalg.norm(model.inverse_transform(e) - x) / np.linalg.norm(x)
                self.assertLessEqual(float(error), 0.02, f"{beta_loss} {seed}")              # 0.005 at worst
            self.assertEqual(model.params()["solver"], "mu")      # kullback-leibler needs multiplicative updates


class TestDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.outliers = {seed: planted("outliers", 600, 4, 3, 1.0, seed) for seed in (1, 2, 3)}

    def check(self, raw, threshold, data=None):
        import numpy as np

        for seed, (x, columns, _) in (data or self.outliers).items():
            model = fitted(raw, x, seed)
            scores = model.score(x)
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreaterEqual(auroc(columns, scores), threshold, f"{raw['name']} {seed}")
            yield model, x, columns

    def test_isolation_forest(self):
        for model, x, _ in self.check(ISOLATION_FOREST, 0.98):                                # 0.997 on 10 seeds
            self.assertEqual(model.params()["n_estimators"], 100)
        for model, x, _ in self.check(dict(ISOLATION_FOREST, contamination=0.05), 0.98):
            self.assertAlmostEqual(float(model.flag(x).mean()), 0.05, places=9)             # exactly 30 of 600

    def test_local_outlier_factor_scores_new_rows(self):
        for model, x, _ in self.check(LOF, 0.98):                                              # 0.9995 on 10 seeds
            self.assertTrue(model.params()["novelty"])

    def test_one_class_svm_flags_about_nu(self):
        for model, x, _ in self.check(ONE_CLASS_SVM, 0.93):                                    # 0.956 on 10 seeds
            self.assertTrue(0.02 <= float(model.flag(x).mean()) <= 0.1)                      # 0.047 to 0.057

    def test_elliptic_envelope_on_one_gaussian_cloud(self):
        one_cloud = {seed: planted("outliers", 400, 4, 1, 1.0, seed) for seed in (1, 2, 3)}
        for model, x, _ in self.check(ELLIPTIC_ENVELOPE, 0.97, one_cloud):                    # 0.995 on 10 seeds
            self.assertAlmostEqual(float(model.flag(x).mean()), 0.05, places=9)

    @needs_module("hdbscan")
    def test_hdbscan_outliers_glosh_scores(self):
        for model, x, _ in self.check({"name": "hdbscan_outliers", "min_cluster_size": 15, "min_samples": None,
                                       "flag_quantile": 0.95, "extra": {}}, 0.9):
            self.assertTrue(0.03 <= float(model.flag(x).mean()) <= 0.07)
            self.assertIs(model.score(x), model.fit_scores)


def run_jobs(base: dict, variants: dict, folds: list) -> dict:
    """label -> the files of the first job of each configuration, run through its fit procedure (nothing written)."""
    from sweepkit.registry import FITS
    from sweepkit.runs.spec import validate_job
    from sweepkit.sweep.design import expand
    from tests.sweep_fixtures import sweep

    stages = [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]}]
    if variants:
        stages.append({"name": "methods", "kind": "variants", "priority": 4, "seeds": [1], "variants": variants})
    jobs, _ = expand(sweep(base, stages, sweep_id="fits", folds=folds))
    out = {}
    for spec in jobs:
        if spec["label"] in out:
            continue
        job = validate_job(spec["settings"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[spec["label"]] = FITS.get(job.fit_name).target.run(job, spec, "cpu", lambda message: None)
    return out


class TestThroughTheFitProcedures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_every_clusterer_through_the_clusterer_fit(self):
        from tests.sweep_fixtures import cluster_base

        base = cluster_base()
        base["split"] = {"name": "all"}
        base["fit"]["name"] = "clusterer"
        base["task"].update({"criteria": ["silhouette", "n_clusters", "noise_fraction"], "reveal_metrics": ["ari"]})
        base["model"] = GMM
        variants = {raw["name"] + ("_dbscan" if raw.get("cluster_method") == "dbscan" else ""): {"model": raw}
                    for raw in (BAYESIAN_GMM, BISECTING, FUZZY, KMEDOIDS, {"name": "hdbscan_sklearn", **HDBSCAN},
                                OPTICS_XI, OPTICS_DBSCAN, MEAN_SHIFT)}
        files = run_jobs(base, variants, ["all"])
        self.assertEqual(sorted(files), sorted(["base"] + list(variants)))
        for label, f in files.items():
            self.assertGreaterEqual(f["reveal.json"]["metrics"]["ari"], 0.95, label)          # 0.980 at worst
            self.assertEqual(f["criteria.json"]["criteria"]["n_clusters"], 3.0, label)
            rows = f["assignments.csv"].strip().splitlines()
            self.assertEqual(len(rows), 301, label)
            self.assertEqual(f["model_params.json"]["model"], label.replace("_dbscan", "") if label != "base"
                             else "gmm")
        self.assertIn("memberships.npy", files["fuzzy_cmeans"])
        self.assertTrue(rows[0].startswith("row_id,cluster,status"))
        header = files["base"]["assignments.csv"].splitlines()[0]
        self.assertEqual(header, "row_id,cluster,status,probability")

    def test_kprototypes_receives_the_categorical_columns(self):
        from tests.sweep_fixtures import cluster_base

        base = cluster_base()
        base["data"] = {"name": "synthetic", "generator": "mixed", "n_rows": 400, "n_features": 6, "n_groups": 3,
                        "noise": 0.8, "seed": 1}
        base["split"] = {"name": "all"}
        base["task"].update({"criteria": ["n_clusters"], "select": "stability", "reveal_metrics": ["ari"]})
        base["model"] = KPROTOTYPES
        files = run_jobs(base, {"gower_kmedoids": {"model": dict(KMEDOIDS, metric="gower")}}, ["all"])
        for label, f in files.items():
            self.assertGreaterEqual(f["reveal.json"]["metrics"]["ari"], 0.9, label)            # 1.000 and 0.985
        self.assertEqual(files["base"]["model_params.json"]["params"]["categorical_columns"],
                         ["cat_0", "cat_1", "cat_2"])
        self.assertEqual(files["base"]["prep_record.json"]["columns"],
                         ["num_0", "num_1", "num_2", "cat_0", "cat_1", "cat_2"])

    def test_every_linear_embedder_through_the_embedder_fit(self):
        import numpy as np

        base = {"data": {"name": "synthetic", "generator": "low_rank", "n_rows": 300, "n_features": 10,
                         "n_groups": 2, "noise": 0.05, "seed": 1},
                "split": {"name": "all"},
                "task": {"name": "embedding", "criteria": ["trustworthiness", "explained_variance",
                                                           "reconstruction_error"],
                         "select": "trustworthiness", "sample_rows": None, "sample_salt": "s", "n_neighbors": 10,
                         "reference": None, "reveal_metrics": [], "criteria_options": {}},
                "prep": [], "features": {"name": "none"}, "model": TRUNCATED_SVD,
                "fit": {"name": "embedder", "device": "cpu", "assign": "fit_rows"}}
        variants = {"kernel_pca": {"model": KERNEL_PCA}, "ica": {"model": dict(ICA, n_components=2)},
                    "factor_analysis": {"model": FACTOR_ANALYSIS},
                    "nmf": {"model": NMF, "prep": [{"name": "scale", "method": "minmax", "columns": ["*"],
                                                    "per_group": None}]}}
        files = run_jobs(base, variants, ["all"])
        for label, f in files.items():
            criteria = f["criteria.json"]
            self.assertGreaterEqual(criteria["criteria"]["trustworthiness"], 0.95, label)     # 0.982 at worst
            self.assertEqual(np.asarray(f["embedding.npy"]).shape[0], 300, label)
        svd = files["base"]["criteria.json"]["criteria"]
        self.assertGreaterEqual(svd["explained_variance"], 0.99)
        self.assertLess(svd["reconstruction_error"], 0.01)
        fa = files["factor_analysis"]["criteria.json"]
        self.assertIsNone(fa["criteria"]["reconstruction_error"])
        self.assertIn("does not report reconstruction_error", fa["undefined"]["reconstruction_error"])
        self.assertIn("does not report explained_variance", fa["undefined"]["explained_variance"])


class TestOutlierSweep(unittest.TestCase):
    """A sweep of the task "outlier" through the fit procedure "detector", run by run --local into a temporary
    repository: every detector writes its scores and flags, a defined excess-mass criterion, and agreement with the
    planted outliers where analyze never looks."""

    def test_detector_sweep_end_to_end(self):
        import csv
        import io

        load_catalog()
        from sweepkit import paths
        from sweepkit.sweep.local import run_local
        from tests.sweep_fixtures import sweep

        base = {"data": {"name": "synthetic", "generator": "outliers", "n_rows": 300, "n_features": 3, "n_groups": 2,
                         "noise": 1.0, "seed": 1},
                "split": {"name": "all"},
                "task": {"name": "outlier", "criteria": ["em_auc", "mv_auc", "flagged_fraction"], "select": "em_auc",
                         "sample_rows": None, "sample_salt": "s", "reference": "is_outlier",
                         "reveal_metrics": ["auroc", "outlier_average_precision", "precision_at_n"],
                         "criteria_options": {"em_mv_uniform_points": 2000, "em_mv_max_features": 5,
                                              "em_mv_feature_draws": 2, "em_t_max": 0.9, "em_t_points": 200,
                                              "mv_alpha_min": 0.9, "mv_alpha_max": 0.999, "mv_alpha_points": 50}},
                "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
                "features": {"name": "none"},
                "model": dict(ISOLATION_FOREST, n_estimators=50),
                "fit": {"name": "detector", "device": "cpu", "assign": "fit_rows"}}
        s = sweep(base, [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]},
                         {"name": "methods", "kind": "variants", "priority": 4, "seeds": [1],
                          "variants": {"lof": {"model": LOF}, "one_class_svm": {"model": ONE_CLASS_SVM},
                                       "elliptic_envelope": {"model": ELLIPTIC_ENVELOPE}}}],
                  sweep_id="out1", purpose="learning", folds=["all"])
        with TempRepo() as root:
            (root / "out1.json").write_text(json.dumps(s), encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_local(root / "out1.json", "cpu", 1, lambda message: None)
            self.assertEqual(result["failed"], 0)
            runs = sorted(paths.runs_dir("out1").iterdir())
            self.assertEqual(len(runs), 4)
            seen = set()
            for run in runs:
                spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
                seen.add(spec["settings"]["model"]["name"])
                rows = list(csv.DictReader(io.StringIO((run / "scores.csv").read_text(encoding="utf-8"))))
                self.assertEqual(len(rows), 300)
                self.assertTrue({r["flagged"] for r in rows} <= {"0", "1"})
                criteria = json.loads((run / "criteria.json").read_text(encoding="utf-8"))
                self.assertIsNotNone(criteria["criteria"]["em_auc"], run.name)
                self.assertIsNotNone(criteria["criteria"]["mv_auc"], run.name)
                self.assertEqual(criteria["select"]["metric"], "em_auc")
                reveal = json.loads((run / "reveal.json").read_text(encoding="utf-8"))
                self.assertGreaterEqual(reveal["metrics"]["auroc"], 0.9, spec["label"])         # 0.962 at worst
                self.assertEqual(reveal["n_with_reference"], 300)
            self.assertEqual(seen, {"isolation_forest", "lof", "one_class_svm", "elliptic_envelope"})


if __name__ == "__main__":
    unittest.main()
