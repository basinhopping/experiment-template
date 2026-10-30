"""The unsupervised criteria and the reveal metrics, called the way the families' runners call them: one
UnsupervisedContext per run and the registered function on it.

Each metric is checked on tiny inputs worked out by hand, against scikit-learn where it computes the same quantity,
for the direction its registration declares (a perfect clustering beats a shuffled one under every reveal metric,
planted blobs beat a random labelling under every label-free criterion), for noise labels (-1), for results that have
no value (Undefined), and for the criteria options a task must give.
"""
import unittest

import numpy as np

from tests.support import load_catalog
from tests.sweep_fixtures import cluster_base

GOIX = {"em_mv_uniform_points": 100000, "em_mv_max_features": 5, "em_mv_feature_draws": 50, "em_t_max": 0.9,
        "em_t_points": 10000, "mv_alpha_min": 0.9, "mv_alpha_max": 0.999, "mv_alpha_points": 991}


def context(**kw):
    """An UnsupervisedContext as a runner builds it, with every field not given left empty."""
    from sweepkit.metrics.unsupervised_context import UnsupervisedContext

    for key in ("x", "labels", "scores", "embedding"):
        if kw.get(key) is not None:
            n = np.asarray(kw[key]).shape[0]
            break
    base = dict(x=None, labels=None, embedding=None, scores=None, model_info={}, sample=np.arange(n), options={},
                refit=None, score_new=None, reference=None, n_neighbors=None, seed=1)
    base.update(kw)
    for key in ("x", "embedding", "scores"):
        if base[key] is not None:
            base[key] = np.asarray(base[key], dtype=np.float64)
    if base["labels"] is not None:
        base["labels"] = np.asarray(base["labels"], dtype=np.int64)
    if base["reference"] is not None:
        base["reference"] = np.asarray(base["reference"]).astype(str)
    return UnsupervisedContext(**base)


def value(name: str, ctx) -> float:
    from sweepkit.registry import METRICS

    return METRICS.get(name)["fn"](ctx)


def better(name: str, a: float, b: float) -> bool:
    """Whether a is better than b by the direction the metric's registration declares."""
    from sweepkit.registry import METRICS

    return a > b if METRICS.get(name)["direction"] == "max" else a < b


def blobs():
    """Three planted, well separated groups: the rows and their group numbers."""
    from sweepkit.data.synthetic import make

    made = make("blobs", 300, 4, 3, 0.6, 3)
    truth = np.array([int(t[1:]) for t in made["columns"]["truth"]])
    return made["x"].astype(np.float64), truth


def checked(settings: dict):
    """A job dict validated and checked as a sweep's expansion does it."""
    from sweepkit.runs.spec import check_job, validate_job

    job = validate_job(settings)
    check_job(job)
    return job


def outlier_job(criteria: list, select: str, options: dict) -> dict:
    base = cluster_base()
    base["data"]["generator"] = "outliers"
    base["task"] = {"name": "outlier", "criteria": criteria, "select": select, "sample_rows": 200, "sample_salt": "s",
                    "reference": "is_outlier", "reveal_metrics": ["auroc"], "criteria_options": options}
    base["model"] = {"name": "isolation_forest", "n_estimators": 50, "max_samples": None, "max_features": 1.0,
                     "contamination": None, "extra": {}}
    base["fit"] = {"name": "detector", "device": "cpu", "assign": "fit_rows"}
    return base


class TestRegistrations(unittest.TestCase):
    """The direction, label use and role each criterion declares: analyze ranks by these."""

    EXPECTED = {
        # name: (task, direction, label_free, descriptive)
        "inertia": ("clustering", "min", True, False), "calinski_harabasz": ("clustering", "max", True, False),
        "davies_bouldin": ("clustering", "min", True, False), "gap": ("clustering", "max", True, False),
        "aic": ("clustering", "min", True, False), "bic": ("clustering", "min", True, False),
        "noise_fraction": ("clustering", "min", True, True),
        "largest_cluster_fraction": ("clustering", "min", True, True),
        "ami": ("clustering", "max", False, False), "nmi": ("clustering", "max", False, False),
        "homogeneity": ("clustering", "max", False, False), "completeness": ("clustering", "max", False, False),
        "v_measure": ("clustering", "max", False, False), "fowlkes_mallows": ("clustering", "max", False, False),
        "purity": ("clustering", "max", False, False),
        "em_auc": ("outlier", "max", True, False), "mv_auc": ("outlier", "min", True, False),
        "flagged_fraction": ("outlier", "min", True, True), "auroc": ("outlier", "max", False, False),
        "outlier_average_precision": ("outlier", "max", False, False),
        "precision_at_n": ("outlier", "max", False, False),
        "reconstruction_error": ("embedding", "min", True, False),
        "reference_silhouette": ("embedding", "max", False, False),
    }

    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_each_criterion_declares_what_analyze_needs(self):
        from sweepkit.registry import METRICS

        for name, (task, direction, label_free, descriptive) in self.EXPECTED.items():
            entry = METRICS.get(name)
            self.assertEqual(entry["tasks"], (task,), name)
            self.assertEqual(entry["direction"], direction, name)
            self.assertEqual(entry["label_free"], label_free, name)
            self.assertEqual(entry["descriptive"], descriptive, name)
            # none of these decomposes over rows, so none is bootstrapped over rows
            self.assertFalse(entry["weighted"], name)
            self.assertIsNone(entry["batched"], name)


class TestClusteringCriteria(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    # two clusters of two points: (0, 0) and (2, 0) around (1, 0); (10, 0) and (10, 4) around (10, 2)
    FOUR = [[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [10.0, 4.0]]

    def test_inertia_calinski_harabasz_and_davies_bouldin_by_hand(self):
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

        ctx = context(x=self.FOUR, labels=[0, 0, 1, 1])
        # within: 1 + 1 around (1, 0) and 4 + 4 around (10, 2)
        self.assertAlmostEqual(value("inertia", ctx), 10.0, places=12)
        # between: the overall mean is (5.5, 1); 2 x (4.5^2 + 1^2) for each cluster, so 85; CH = (85 / 1) / (10 / 2)
        self.assertAlmostEqual(value("calinski_harabasz", ctx), 17.0, places=12)
        # mean distance to the centre is 1 and 2; the centres are sqrt(85) apart: (1 + 2) / sqrt(85) for both
        self.assertAlmostEqual(value("davies_bouldin", ctx), 3 / np.sqrt(85), places=12)
        x, labels = np.array(self.FOUR), np.array([0, 0, 1, 1])
        self.assertAlmostEqual(value("calinski_harabasz", ctx), calinski_harabasz_score(x, labels), places=10)
        self.assertAlmostEqual(value("davies_bouldin", ctx), davies_bouldin_score(x, labels), places=10)

    def test_noise_rows_are_left_out(self):
        clean = context(x=self.FOUR, labels=[0, 0, 1, 1])
        noisy = context(x=[[50.0, 50.0]] + self.FOUR + [[-30.0, 7.0]], labels=[-1, 0, 0, 1, 1, -1])
        for name in ("inertia", "calinski_harabasz", "davies_bouldin", "silhouette"):
            self.assertAlmostEqual(value(name, noisy), value(name, clean), places=12, msg=name)

    def test_inertia_is_the_sum_kmeans_minimizes(self):
        from sklearn.cluster import KMeans

        x, _ = blobs()
        km = KMeans(n_clusters=3, n_init=4, tol=0.0, random_state=0).fit(x)
        self.assertAlmostEqual(value("inertia", context(x=x, labels=km.labels_)), km.inertia_,
                               delta=1e-9 * km.inertia_)

    def test_descriptive_counts(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        ctx = context(x=np.zeros((5, 1)), labels=[0, 0, 0, 1, -1])
        self.assertAlmostEqual(value("noise_fraction", ctx), 1 / 5)
        self.assertAlmostEqual(value("largest_cluster_fraction", ctx), 3 / 5)    # over every row, noise included
        self.assertEqual(value("n_clusters", ctx), 2.0)
        self.assertEqual(value("noise_fraction", context(x=np.zeros((3, 1)), labels=[0, 1, 1])), 0.0)
        everything_noise = context(x=np.zeros((3, 1)), labels=[-1, -1, -1])
        self.assertEqual(value("noise_fraction", everything_noise), 1.0)
        with self.assertRaises(Undefined):
            value("largest_cluster_fraction", everything_noise)

    def test_criteria_that_need_two_clusters(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        one = context(x=self.FOUR, labels=[0, 0, 0, -1])
        for name in ("calinski_harabasz", "davies_bouldin", "silhouette"):
            with self.assertRaises(Undefined, msg=name):
                value(name, one)
        # inertia stays defined with one cluster: (0, 0), (2, 0) and (10, 0) around (4, 0) give 16 + 4 + 36
        self.assertAlmostEqual(value("inertia", one), 56.0, places=12)

    def test_aic_and_bic_come_from_the_mixture_model(self):
        from sklearn.mixture import GaussianMixture

        from sweepkit.metrics.unsupervised_context import Undefined
        from sweepkit.models.cluster.centroid import BayesianGMMSettings, GMMSettings
        from sweepkit.registry import MODELS

        x, _ = blobs()
        found = {}
        for k in (1, 3):
            model = MODELS.get("gmm").target(GMMSettings(k, "full", 1, 200, 1e-6, {}), 5, "cpu")
            model.fit(x)
            ctx = context(x=x, labels=model.labels(), model_info=model.info())
            found[k] = (value("bic", ctx), value("aic", ctx))
            reference = GaussianMixture(n_components=k, covariance_type="full", n_init=1, max_iter=200,
                                        reg_covar=1e-6, random_state=5).fit(x)
            self.assertAlmostEqual(found[k][0], reference.bic(x), places=6)
            self.assertAlmostEqual(found[k][1], reference.aic(x), places=6)
        # three planted groups: three components fit better by both criteria, in their declared direction
        self.assertTrue(better("bic", found[3][0], found[1][0]))
        self.assertTrue(better("aic", found[3][1], found[1][1]))
        model = MODELS.get("bayesian_gmm").target(BayesianGMMSettings(3, "full", "dirichlet_process", None, 200, {}),
                                                  5, "cpu")
        model.fit(x)
        for name in ("bic", "aic"):
            with self.assertRaises(Undefined, msg=name):
                value(name, context(x=x, labels=model.labels(), model_info=model.info()))

    def test_gap_statistic(self):
        from sweepkit.metrics.unsupervised_context import Undefined
        from sweepkit.streams import generator

        x = np.array(self.FOUR + [[1.0, 1.0], [9.0, 2.0]])
        labels = np.array([0, 0, 1, 1, 0, 1])

        def halves(points):
            return (points[:, 0] > np.median(points[:, 0])).astype(np.int64)

        ctx = context(x=x, labels=labels, refit=halves, options={"gap_references": 4}, seed=7)
        # Tibshirani, Walther and Hastie (2001): the mean log within-cluster sum of squares of the refitted
        # reference sets, drawn uniformly in the rows' bounding box from the run's named stream, minus the observed
        def log_w(points, groups):
            return np.log(sum(((points[groups == g] - points[groups == g].mean(axis=0)) ** 2).sum()
                              for g in np.unique(groups)))

        rng = generator(7, "criterion gap")
        references = []
        for _ in range(4):
            fake = rng.uniform(x.min(axis=0), x.max(axis=0), size=x.shape)
            references.append(log_w(fake, halves(fake.astype(np.float32))))
        self.assertAlmostEqual(value("gap", ctx), float(np.mean(references) - log_w(x, labels)), places=10)
        with self.assertRaises(Undefined):
            value("gap", context(x=x, labels=labels, refit=None, options={"gap_references": 4}))
        with self.assertRaises(ValueError):
            value("gap", context(x=x, labels=labels, refit=halves, options={"gap_references": 0}))
        same = context(x=[[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0]], labels=[0, 0, 1, 1], refit=halves,
                       options={"gap_references": 2})
        with self.assertRaises(Undefined):                 # no spread within the clusters: log 0
            value("gap", same)

    def test_planted_blobs_beat_a_random_labelling(self):
        from sklearn.cluster import KMeans

        x, truth = blobs()
        shuffled = np.random.default_rng(4).integers(0, 3, size=truth.size)

        def refit(points):
            return KMeans(n_clusters=3, n_init=2, random_state=0).fit_predict(points)

        planted = context(x=x, labels=truth, refit=refit, options={"gap_references": 3})
        random = context(x=x, labels=shuffled, refit=refit, options={"gap_references": 3})
        for name in ("inertia", "calinski_harabasz", "davies_bouldin", "gap", "silhouette", "dbcv"):
            self.assertTrue(better(name, value(name, planted), value(name, random)), name)

    def test_the_task_needs_the_options_its_criteria_read(self):
        base = cluster_base()
        base["task"]["criteria"] = ["silhouette", "gap"]
        base["task"]["criteria_options"] = {}
        with self.assertRaisesRegex(ValueError, r"criteria_options missing \['gap_references'\]"):
            checked(base)
        base["task"]["criteria_options"] = {"gap_references": 2.5}
        with self.assertRaisesRegex(ValueError, "gap_references .*must be a int"):
            checked(base)
        base["task"]["criteria_options"] = {"gap_references": 3, "spare": 1}
        with self.assertRaisesRegex(ValueError, r"unknown \['spare'\]"):
            checked(base)
        base["task"]["criteria_options"] = {"gap_references": 3}
        checked(base)

    def test_descriptive_criteria_cannot_choose(self):
        for name in ("noise_fraction", "largest_cluster_fraction"):
            base = cluster_base()
            base["task"]["criteria"] = ["silhouette", name]
            base["task"]["select"] = name
            with self.assertRaisesRegex(ValueError, "describes a result but cannot choose one", msg=name):
                checked(base)


class TestRevealMetrics(unittest.TestCase):
    REVEAL = ("ari", "ami", "nmi", "homogeneity", "completeness", "v_measure", "fowlkes_mallows", "purity")

    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_purity_of_a_two_by_two_table(self):
        # cluster 0 holds a, a; cluster 1 holds a, b, b, b: 2 + 3 of 6 rows sit with their cluster's main label
        ctx = context(labels=[0, 0, 1, 1, 1, 1], reference=["a", "a", "a", "b", "b", "b"])
        self.assertAlmostEqual(value("purity", ctx), 5 / 6)

    def test_fowlkes_mallows_by_counting_pairs(self):
        # same cluster: 1 + 6 = 7 pairs; same label: 3 + 3 = 6 pairs; both: 1 (a, a in 0) + 3 (b, b, b in 1) = 4
        ctx = context(labels=[0, 0, 1, 1, 1, 1], reference=["a", "a", "a", "b", "b", "b"])
        self.assertAlmostEqual(value("fowlkes_mallows", ctx), 4 / np.sqrt(7 * 6))

    def test_homogeneity_completeness_v_measure_and_nmi_by_hand(self):
        reference = ["a", "a", "b", "b"]
        lumped = context(labels=[0, 0, 0, 0], reference=reference)
        # one cluster holds both labels (not homogeneous) but keeps each label together (complete)
        self.assertAlmostEqual(value("homogeneity", lumped), 0.0)
        self.assertAlmostEqual(value("completeness", lumped), 1.0)
        self.assertAlmostEqual(value("v_measure", lumped), 0.0)
        split = context(labels=[0, 1, 2, 3], reference=reference)
        # H(K) = log 4, H(K | C) = log 2: completeness 1 - 1/2; V = 2 x 1 x 0.5 / 1.5; NMI with the arithmetic mean
        # of the entropies is the same number: log 2 / ((log 2 + log 4) / 2)
        self.assertAlmostEqual(value("homogeneity", split), 1.0)
        self.assertAlmostEqual(value("completeness", split), 0.5)
        self.assertAlmostEqual(value("v_measure", split), 2 / 3)
        self.assertAlmostEqual(value("nmi", split), 2 / 3)

    def test_purity_rewards_many_small_clusters(self):
        # every row its own cluster: purity 1, which is why completeness is read beside it
        ctx = context(labels=[0, 1, 2, 3, 4, 5], reference=["a", "a", "a", "b", "b", "b"])
        self.assertEqual(value("purity", ctx), 1.0)
        self.assertLess(value("completeness", ctx), 0.5)

    def test_against_scikit_learn_with_noise_as_one_more_cluster(self):
        from sklearn import metrics as sk

        rng = np.random.default_rng(9)
        labels = rng.integers(-1, 4, size=80)
        reference = rng.choice(np.array(["x", "y", "z"]), size=80)
        ctx = context(labels=labels, reference=reference)
        expected = {"ari": sk.adjusted_rand_score, "ami": sk.adjusted_mutual_info_score,
                    "nmi": sk.normalized_mutual_info_score, "homogeneity": sk.homogeneity_score,
                    "completeness": sk.completeness_score, "v_measure": sk.v_measure_score,
                    "fowlkes_mallows": sk.fowlkes_mallows_score}
        for name, fn in expected.items():
            self.assertAlmostEqual(value(name, ctx), float(fn(reference, labels)), places=12, msg=name)
        table = sk.cluster.contingency_matrix(reference, labels)
        self.assertAlmostEqual(value("purity", ctx), table.max(axis=0).sum() / 80, places=12)

    def test_noise_counts_as_a_cluster(self):
        ctx = context(labels=[0, 0, 1, 1, -1, -1], reference=["a", "a", "b", "b", "c", "c"])
        for name in self.REVEAL:
            self.assertAlmostEqual(value(name, ctx), 1.0, places=12, msg=name)

    def test_rows_without_a_reference_label_are_left_out(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        ctx = context(labels=[0, 0, 1, 1, 0], reference=["a", "a", "b", "b", ""])
        for name in self.REVEAL:
            self.assertAlmostEqual(value(name, ctx), 1.0, places=12, msg=name)
        for name in self.REVEAL:
            with self.assertRaises(Undefined, msg=name):
                value(name, context(labels=[0, 1], reference=["", ""]))
            with self.assertRaises(ValueError, msg=name):
                value(name, context(labels=[0, 1], reference=None))

    def test_label_names_do_not_matter(self):
        reference = ["a", "a", "b", "b", "b", "c"]
        first = context(labels=[0, 0, 1, 1, 2, 2], reference=reference)
        renamed = context(labels=[7, 7, 3, 3, 5, 5], reference=reference)
        for name in self.REVEAL:
            self.assertAlmostEqual(value(name, first), value(name, renamed), places=12, msg=name)

    def test_a_perfect_clustering_beats_a_shuffled_one(self):
        _, truth = blobs()
        reference = np.array(["p", "q", "r"])[truth]
        perfect = context(labels=(truth + 1) % 3, reference=reference)
        shuffled = context(labels=np.random.default_rng(2).permutation(truth), reference=reference)
        for name in self.REVEAL:
            self.assertAlmostEqual(value(name, perfect), 1.0, places=12, msg=name)
            self.assertTrue(better(name, value(name, perfect), value(name, shuffled)), name)


def triangular(n: int) -> np.ndarray:
    """n points at the quantiles of the density 1 - |x| on [-1, 1], so the sample follows it closely."""
    q = (np.arange(n) + 0.5) / n
    return np.where(q < 0.5, np.sqrt(2 * q) - 1, 1 - np.sqrt(2 * (1 - q)))


class TestOutlierCriteria(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_ranking_metrics_by_hand(self):
        from sklearn.metrics import average_precision_score, roc_auc_score

        reference = ["0", "0", "0", "1", "1"]
        perfect = context(scores=[0.1, 0.2, 0.3, 0.9, 0.8], reference=reference)
        for name in ("auroc", "outlier_average_precision", "precision_at_n"):
            self.assertEqual(value(name, perfect), 1.0, name)
        # ranked: 0.9 inlier, 0.8 outlier, 0.3 inlier, 0.2 outlier, 0.1 inlier
        mixed = context(scores=[0.1, 0.9, 0.3, 0.8, 0.2], reference=reference)
        # the outlier at 0.8 beats 2 of 3 inliers, the one at 0.2 beats 1 of 3: 3 of 6 pairs
        self.assertAlmostEqual(value("auroc", mixed), 0.5)
        # precision 1/2 at the first outlier (rank 2) and 2/4 at the second (rank 4)
        self.assertAlmostEqual(value("outlier_average_precision", mixed), 0.5)
        # the two highest scores (n = 2 known outliers) hold one outlier
        self.assertAlmostEqual(value("precision_at_n", mixed), 0.5)
        truth, scores = np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        self.assertAlmostEqual(value("auroc", mixed), roc_auc_score(truth, scores), places=12)
        self.assertAlmostEqual(value("outlier_average_precision", mixed), average_precision_score(truth, scores),
                               places=12)

    def test_reference_values_other_than_zero_and_one_are_left_out(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        ctx = context(scores=[0.1, 0.9, 0.5, 0.2, 0.8], reference=["0.0", "1.0", "", "0", "nan"])
        self.assertEqual(value("auroc", ctx), 1.0)                  # rows 0, 1 and 3 count
        self.assertEqual(value("precision_at_n", ctx), 1.0)
        for name in ("auroc", "outlier_average_precision", "precision_at_n"):
            with self.assertRaises(Undefined, msg=name):            # only inliers
                value(name, context(scores=[0.1, 0.2], reference=["0", "0"]))
            with self.assertRaises(Undefined, msg=name):            # no 0/1 value at all
                value(name, context(scores=[0.1, 0.2], reference=["", "x"]))
            with self.assertRaises(ValueError, msg=name):
                value(name, context(scores=[0.1, 0.2], reference=None))

    def test_perfect_scores_beat_reversed_ones(self):
        rng = np.random.default_rng(5)
        truth = (rng.random(60) < 0.15).astype(int)
        truth[:2] = (0, 1)
        scores = truth + rng.random(60) * 0.5
        reference = truth.astype(str)
        for name in ("auroc", "outlier_average_precision", "precision_at_n"):
            good, bad = value(name, context(scores=scores, reference=reference)), value(
                name, context(scores=-scores, reference=reference))
            self.assertEqual(good, 1.0, name)
            self.assertTrue(better(name, good, bad), name)

    def test_flagged_fraction(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        flags = np.array([True, False, False, True, False])
        # the detector runner reports the share of flagged evaluated rows as model_info["flagged"]
        ctx = context(scores=np.zeros(5), model_info={"flagged": float(flags.mean())})
        self.assertAlmostEqual(value("flagged_fraction", ctx), 0.4)
        with self.assertRaises(Undefined):
            value("flagged_fraction", context(scores=np.zeros(5)))

    def test_mass_volume_known_answers(self):
        from sweepkit.metrics.outlier import mass_volume

        alphas = np.linspace(0.9, 0.999, 100)
        grid = np.linspace(-1, 1, 40001)         # the uniform points of the box [-1, 1], volume 2
        x = triangular(4000)
        # density 1 - |x|: the set |x| <= a holds mass 2a - a^2, so mass alpha needs volume 2 (1 - sqrt(1 - alpha));
        # its integral over [0.9, 0.999] is 2 x 0.099 - (4/3)(0.1^1.5 - 0.001^1.5)
        good = mass_volume(alphas, 2.0, -np.abs(grid), -np.abs(x))
        self.assertAlmostEqual(good, 2 * 0.099 - (4 / 3) * (0.1 ** 1.5 - 0.001 ** 1.5), delta=1e-3)
        # calling the tails normal: the set |x| >= b holds mass (1 - b)^2, so mass alpha needs volume 2 sqrt(alpha)
        bad = mass_volume(alphas, 2.0, np.abs(grid), np.abs(x))
        self.assertAlmostEqual(bad, (4 / 3) * (0.999 ** 1.5 - 0.9 ** 1.5), delta=1e-3)
        self.assertTrue(better("mv_auc", good, bad))
        # on uniform data every score is as good as any other: mass alpha always needs volume 2 alpha
        uniform = np.linspace(-1, 1, 4001)
        for sign in (-1, 1):
            self.assertAlmostEqual(mass_volume(alphas, 2.0, sign * np.abs(grid), sign * np.abs(uniform)),
                                   0.999 ** 2 - 0.9 ** 2, delta=1e-3)

    def test_excess_mass_known_answers(self):
        from sweepkit.metrics.outlier import excess_mass
        from sweepkit.metrics.unsupervised_context import Undefined

        grid = np.linspace(-1, 1, 40001)
        x = triangular(4000)
        t = np.linspace(0.0, 100.0 / 2.0, 10000, endpoint=False)          # as em_auc builds it for volume 2
        # density 1 - |x|: EM(t) = integral of (f - t)+ = (1 - t)^2, which falls to 0.9 at t = 1 - sqrt(0.9),
        # 0.0513; the area is taken up to the first grid point at or past it, t = 0.055 (Goix's code does the same)
        good = excess_mass(t, 0.9, 2.0, -np.abs(grid), -np.abs(x))
        self.assertAlmostEqual(good, (1 - (1 - 0.055) ** 3) / 3, delta=1e-5)
        bad = excess_mass(t, 0.9, 2.0, np.abs(grid), np.abs(x))   # tails called normal: EM(t) = 1 - 2t
        self.assertAlmostEqual(bad, 0.05 - 0.05 ** 2, delta=1e-3)
        self.assertTrue(better("em_auc", good, bad))
        with self.assertRaises(Undefined):                         # the grid ends before EM falls to 0.9
            excess_mass(t[:5], 0.9, 2.0, -np.abs(grid), -np.abs(x))

    def gaussian_context(self, sign: float, x=None, **options):
        """Two-dimensional Gaussian rows scored by the squared distance to their centre (sign 1: right) or its
        negative (sign -1: the centre called unusual), through score_new as the detector runner gives it."""
        if x is None:
            x = np.random.default_rng(8).normal(size=(300, 2))
        calls = []

        def score_new(features, part, x_new):
            calls.append((None if features is None else [int(f) for f in features], part.shape, x_new.shape))
            centre = np.zeros(part.shape[1])
            return sign * ((np.asarray(x_new) - centre) ** 2).sum(axis=1)

        opts = {"em_mv_uniform_points": 20000, "em_mv_max_features": 5, "em_mv_feature_draws": 3, "em_t_max": 0.9,
                "em_t_points": 1000, "mv_alpha_min": 0.9, "mv_alpha_max": 0.999, "mv_alpha_points": 100}
        opts.update(options)
        return context(x=x, scores=np.zeros(x.shape[0]), score_new=score_new, options=opts), calls

    def test_em_and_mv_through_the_context(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        good, calls = self.gaussian_context(1.0)
        bad, _ = self.gaussian_context(-1.0)
        for name in ("em_auc", "mv_auc"):
            self.assertTrue(better(name, value(name, good), value(name, bad)), name)
        # at most em_mv_max_features columns: one call for the rows and one for the uniform points, all features
        self.assertEqual([c[0] for c in calls], [None, None, None, None])
        self.assertEqual(calls[1][2], (20000, 2))
        no_model = context(x=np.ones((4, 2)), scores=np.zeros(4), score_new=None, options=GOIX)
        for name in ("em_auc", "mv_auc"):
            with self.assertRaises(Undefined, msg=name):
                value(name, no_model)
        flat = np.random.default_rng(1).normal(size=(50, 2))
        flat[:, 1] = 3.0                                           # a column that does not vary: no box volume
        constant, _ = self.gaussian_context(1.0, x=flat)
        for name in ("em_auc", "mv_auc"):
            with self.assertRaisesRegex(Undefined, "does not vary", msg=name):
                value(name, constant)
        short, _ = self.gaussian_context(1.0, em_t_points=1)
        with self.assertRaisesRegex(Undefined, "em_t_max"):
            value("em_auc", short)

    def test_wide_data_are_scored_on_feature_subsets(self):
        x = np.random.default_rng(3).normal(size=(200, 4))
        ctx, calls = self.gaussian_context(1.0, x=x, em_mv_max_features=2, em_mv_feature_draws=3)
        value("mv_auc", ctx)
        self.assertEqual(len(calls), 2 * 3)                       # the rows and the uniform points, per subset
        for features, part, new in calls:
            self.assertEqual(len(features), 2)
            self.assertEqual(features, sorted(set(features)))
            self.assertTrue(all(0 <= f < 4 for f in features))
            self.assertEqual(part, (200, 2))
        self.assertEqual(calls[0][0], calls[1][0])                # the same subset for the rows and the box
        self.assertEqual(value("mv_auc", ctx), value("mv_auc", ctx))   # the subsets come from the run's stream

    def test_the_task_needs_every_em_and_mv_option(self):
        with self.assertRaises(ValueError) as caught:
            checked(outlier_job(["em_auc", "flagged_fraction"], "em_auc", {}))
        for option in GOIX:
            self.assertIn(option, str(caught.exception))
        checked(outlier_job(["em_auc", "mv_auc", "flagged_fraction"], "mv_auc", dict(GOIX)))
        with self.assertRaisesRegex(ValueError, "em_mv_uniform_points .*must be a int"):
            checked(outlier_job(["em_auc"], "em_auc", dict(GOIX, em_mv_uniform_points=1e5)))
        checked(outlier_job(["em_auc"], "em_auc", dict(GOIX, em_t_max=1)))     # a whole number serves as a float
        with self.assertRaisesRegex(ValueError, "describes a result but cannot choose one"):
            checked(outlier_job(["em_auc", "flagged_fraction"], "flagged_fraction", dict(GOIX)))


class TestEmbeddingCriteria(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_reconstruction_error_as_the_embedder_runner_computes_it(self):
        from sweepkit.metrics.unsupervised_context import Undefined
        from sweepkit.models.embed.linear import PCASettings
        from sweepkit.registry import MODELS

        # variance along x is 2 and along y 0.5: one component keeps x, so the two points off the axis each lose
        # a squared error of 1 in one of their 2 columns: 2 / (4 rows x 2 columns)
        x = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        found = {}
        for k in (1, 2):
            model = MODELS.get("pca").target(PCASettings(k, "full", False, None, {}), 1, "cpu")
            embedding = model.fit_transform(x)
            info = dict(model.info())
            info["reconstruction_error"] = float(np.mean((model.inverse_transform(embedding) - x) ** 2))
            found[k] = value("reconstruction_error", context(x=x, embedding=embedding, model_info=info))
        self.assertAlmostEqual(found[1], 0.25, places=12)
        self.assertAlmostEqual(found[2], 0.0, places=12)
        self.assertTrue(better("reconstruction_error", found[2], found[1]))
        with self.assertRaises(Undefined):
            value("reconstruction_error", context(x=x, embedding=x[:, :1], model_info={}))

    def test_reference_silhouette_by_hand(self):
        from sklearn.metrics import silhouette_score

        e = np.array([[0.0], [1.0], [10.0], [11.0]])
        ctx = context(x=e, embedding=e, reference=["a", "a", "b", "b"])
        # each point is 1 from its partner; the other class is 10.5 or 9.5 away on average
        expected = np.mean([(10.5 - 1) / 10.5, (9.5 - 1) / 9.5, (9.5 - 1) / 9.5, (10.5 - 1) / 10.5])
        self.assertAlmostEqual(value("reference_silhouette", ctx), expected, places=12)
        self.assertAlmostEqual(value("reference_silhouette", ctx), silhouette_score(e, ["a", "a", "b", "b"]),
                               places=12)

    def test_reference_silhouette_reads_the_sample_rows_that_have_a_label(self):
        from sweepkit.metrics.unsupervised_context import Undefined

        e = np.array([[0.0], [1.0], [10.0], [11.0], [5.0], [100.0]])
        reference = ["a", "a", "b", "b", "", "a"]
        sampled = context(x=e, embedding=e, reference=reference, sample=np.arange(5))
        four = context(x=e[:4], embedding=e[:4], reference=reference[:4])
        self.assertAlmostEqual(value("reference_silhouette", sampled), value("reference_silhouette", four), places=12)
        with self.assertRaises(Undefined):                       # fewer than 3 labelled rows
            value("reference_silhouette", context(x=e, embedding=e, reference=["a", "b", "", "", "", ""]))
        with self.assertRaises(Undefined):                       # one class
            value("reference_silhouette", context(x=e[:4], embedding=e[:4], reference=["a"] * 4))
        with self.assertRaises(ValueError):
            value("reference_silhouette", context(x=e, embedding=e, reference=None))

    def test_separated_classes_beat_mixed_ones(self):
        _, truth = blobs()
        reference = np.array(["p", "q", "r"])[truth]
        rng = np.random.default_rng(6)
        separated = (truth[:, None] * 10.0 + rng.normal(size=(truth.size, 2)))
        mixed = rng.normal(size=(truth.size, 2))
        a = value("reference_silhouette", context(x=separated, embedding=separated, reference=reference))
        b = value("reference_silhouette", context(x=mixed, embedding=mixed, reference=reference))
        self.assertTrue(better("reference_silhouette", a, b))


if __name__ == "__main__":
    unittest.main()
