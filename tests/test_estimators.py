"""Every supervised estimator through the fit procedure "estimator", on synthetic data with known answers.

Classification: three classes planted apart (generator classification, noise 0.5, data seed 1), where every model's
validation balanced accuracy was 0.97 or more on every fold here; the tests ask 0.85. Regression: a linear target with
half the columns informative (generator regression_linear), where the linear models' R2 was 0.88 or more over five
data seeds; lasso sets the uninformative half's coefficients to exactly zero. Multi-label: four 0/1 labels per row,
each a noisy linear rule (generator multilabel). The tasks are "classification", "regression" and "multilabel".

Also checked: each model's own setting checks refuse bad combinations before anything runs; the shuffled-label
control sits near chance on average; refit train_and_val predicts the test rows from the refitted model while
validation still comes from the model fitted on the training rows; and models without probabilities give decision
values as scores (one linear SVM per label in multi-label too). A single control run can stray far from chance when
the classes are well apart: a permuted label vector still agrees with the truth by chance in each class's region,
and the model's argmax turns a small excess into a confident prediction, so the mean over seeds and folds is the
number to read. Here it was 0.331 over 3 seeds and 4 folds, with single runs from 0.13 to 0.78; on the noisier
fixture (noise 0.8) the mean ranged from 0.28 to 0.36 over eight data seeds.

XGBoost, LightGBM and TPOT skip where they are not installed, and fail instead where SWEEPKIT_REQUIRE names them (the
node); their thresholds are first proposals. Runs are limited to one thread (threadpoolctl): histogram boosting under
OpenMP on a busy two-core machine took from 0.5 to 11 seconds a fit, against 0.05 seconds on one thread.
"""
import csv
import io
import unittest
import warnings

from tests.support import load_catalog, needs_module
from tests.sweep_fixtures import classify_base, sweep

LOGISTIC = {"name": "logistic_regression", "c": 1.0, "l1_ratio": 0.0, "class_weight": "none", "max_iter": 500,
            "extra": {}}
LINEAR_SVM = {"name": "linear_svm", "c": 1.0, "class_weight": "none", "max_iter": 5000, "extra": {}}
SVM = {"name": "svm", "kernel": "rbf", "c": 1.0, "gamma": None, "degree": 3, "class_weight": "none",
       "probability": False, "extra": {}}
KNN = {"name": "knn", "n_neighbors": 10, "weights": "uniform", "metric": "euclidean", "extra": {}}
NAIVE_BAYES = {"name": "naive_bayes", "variant": "gaussian", "smoothing": 1e-9, "extra": {}}
DECISION_TREE = {"name": "decision_tree", "max_depth": 5, "min_samples_leaf": 5, "max_features": None,
                 "class_weight": "none", "extra": {}}
RANDOM_FOREST = {"name": "random_forest", "n_estimators": 50, "max_depth": None, "min_samples_leaf": 1,
                 "max_features": 0.5, "bootstrap": True, "class_weight": "none", "n_jobs": 1, "extra": {}}
EXTRA_TREES = dict(RANDOM_FOREST, name="extra_trees", bootstrap=False)
HIST_GB = {"name": "hist_gradient_boosting", "learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 15,
           "max_depth": None, "min_samples_leaf": 10, "l2_regularization": 0.0, "early_stopping": False,
           "class_weight": "none", "extra": {}}
XGBOOST = {"name": "xgboost", "n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "num_leaves": None,
           "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_jobs": 1, "extra": {}}
LIGHTGBM = dict(XGBOOST, name="lightgbm", max_depth=None, num_leaves=15)
LINEAR = {"name": "linear", "penalty": "none", "alpha": 0.0, "l1_ratio": None, "fit_intercept": True,
          "max_iter": 5000, "extra": {}}
TPOT = {"name": "tpot", "generations": 1, "population_size": 4, "max_time_mins": 0.5, "max_eval_time_mins": 0.2,
        "n_jobs": 1, "extra": {}}
CLASSIFIERS = (LINEAR_SVM, SVM, KNN, NAIVE_BAYES, DECISION_TREE, RANDOM_FOREST, EXTRA_TREES, HIST_GB)
LABELS = ["label_0", "label_1", "label_2", "label_3"]


def classification_base(noise: float = 0.5, seed: int = 1) -> dict:
    base = classify_base()
    base["data"].update({"noise": noise, "seed": seed})
    base["task"]["metrics"] = ["balanced_accuracy", "accuracy", "f1_macro", "roc_auc"]
    base["model"] = dict(LOGISTIC)
    return base


def regression_base() -> dict:
    base = classify_base()
    base["data"] = {"name": "synthetic", "generator": "regression_linear", "n_rows": 400, "n_features": 8,
                    "n_groups": 1, "noise": 1.0, "seed": 1}
    base["split"] = {"name": "kfold", "k": 4, "stratify": None, "salt": "t", "identity_fold": False}
    base["task"] = {"name": "regression", "target": "target", "metrics": ["r2", "rmse", "mae"], "select": "rmse",
                    "groups": []}
    base["model"] = dict(LINEAR)
    return base


def multilabel_base() -> dict:
    base = classify_base()
    base["data"] = {"name": "synthetic", "generator": "multilabel", "n_rows": 600, "n_features": 8, "n_groups": 4,
                    "noise": 0.5, "seed": 1}
    base["split"] = {"name": "kfold", "k": 4, "stratify": None, "salt": "t", "identity_fold": False}
    base["task"] = {"name": "multilabel", "targets": LABELS,
                    "metrics": ["label_f1_macro", "label_f1_micro", "sample_f1", "subset_accuracy", "hamming_loss",
                                "label_roc_auc", "label_average_precision"],
                    "select": "label_f1_macro", "groups": []}
    base["model"] = dict(LOGISTIC)
    return base


def run_jobs(base: dict, variants: dict, folds: list, seeds=(1,)) -> dict:
    """label -> [(fold, seed, files)] for every job of the base and the variants, run through the fit procedure
    (nothing is written)."""
    from threadpoolctl import threadpool_limits

    from sweepkit.registry import FITS
    from sweepkit.runs.spec import validate_job
    from sweepkit.sweep.design import expand

    stages = [{"name": "base", "kind": "base", "priority": 5, "seeds": list(seeds)}]
    if variants:
        stages.append({"name": "methods", "kind": "variants", "priority": 4, "seeds": list(seeds),
                       "variants": variants})
    jobs, _ = expand(sweep(base, stages, sweep_id="estimators", folds=folds))
    out = {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in jobs:
            job = validate_job(spec["settings"])
            files = FITS.get(job.fit_name).target.run(job, spec, "cpu", lambda message: None)
            out.setdefault(spec["label"], []).append((spec["fold"], spec["seed"], files))
    return out


def rows(text: str) -> list:
    return list(csv.DictReader(io.StringIO(text)))


def job_of(base: dict, model: dict, task=None, device: str = "cpu"):
    from sweepkit.runs.spec import validate_job

    settings = dict(base, model=model, fit=dict(base["fit"], device=device))
    if task is not None:
        settings["task"] = task
    return validate_job(settings)


class TestClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        variants = {raw["name"]: {"model": raw} for raw in CLASSIFIERS}
        cls.runs = run_jobs(classification_base(), variants, ["fold1", "fold2"])

    def test_every_classifier_learns_the_planted_classes(self):
        names = {"base"} | {raw["name"] for raw in CLASSIFIERS}
        self.assertEqual(set(self.runs), names)
        for label, runs in self.runs.items():
            for fold, _, files in runs:
                metrics = files["metrics_val.json"]["metrics"]
                self.assertGreaterEqual(metrics["balanced_accuracy"], 0.85, f"{label} {fold}")   # 0.97 or more
                self.assertGreaterEqual(metrics["roc_auc"], 0.9, f"{label} {fold}")
                self.assertEqual(files["metrics_val.json"]["select"]["metric"], "balanced_accuracy")
                self.assertEqual(files["model_params.json"]["model"], label if label != "base" else
                                 "logistic_regression")

    def test_scores_are_probabilities_or_decision_values(self):
        kinds = {label: {r["scores_kind"] for r in rows(runs[0][2]["predictions_val.csv"])}
                 for label, runs in self.runs.items()}
        for label in ("linear_svm", "svm"):
            self.assertEqual(kinds[label], {"decision"}, label)
        for label in ("base", "knn", "naive_bayes", "decision_tree", "random_forest", "extra_trees",
                      "hist_gradient_boosting"):
            self.assertEqual(kinds[label], {"probability"}, label)
        first = rows(self.runs["base"][0][2]["predictions_val.csv"])[0]
        total = sum(float(first[f"score_{c}"]) for c in ("c0", "c1", "c2"))
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_log_loss_needs_probabilities(self):
        base = classification_base()
        base["task"]["metrics"] = ["balanced_accuracy", "log_loss"]
        base["model"] = dict(SVM)
        with self.assertRaisesRegex(ValueError, "needs probabilities"):
            run_jobs(base, {}, ["fold1"])
        base["model"] = dict(SVM, probability=True)
        files = run_jobs(base, {}, ["fold1"])["base"][0][2]
        self.assertGreater(files["metrics_val.json"]["metrics"]["log_loss"], 0)
        self.assertEqual({r["scores_kind"] for r in rows(files["predictions_val.csv"])}, {"probability"})

    def test_svm_without_probabilities_raises_no_deprecation_warning(self):
        import numpy as np

        from sweepkit.registry import MODELS
        from sweepkit.settings import validate

        entry = MODELS.get("svm")
        settings = validate(entry.settings_class, {k: v for k, v in SVM.items() if k != "name"}, "svm")
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(60, 3)), np.arange(60) % 2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry.target.make(settings, "classification", 1, "cpu").fit(x, y)
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, FutureWarning)], [])

    def test_training_rows_are_scored_when_asked(self):
        base = classification_base()
        base["fit"]["eval_train"] = True
        files = run_jobs(base, {}, ["fold1"])["base"][0][2]
        self.assertIn("metrics_train.json", files)
        self.assertEqual(files["metrics_train.json"]["role"], "train")
        self.assertNotIn("metrics_train.json", self.runs["base"][0][2])


class TestControlAndRefit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_shuffled_labels_sit_near_chance_on_average(self):
        import numpy as np

        runs = run_jobs(classification_base(), {"shuffled": {"fit.control": "shuffle_train_labels"}},
                        ["fold1", "fold2", "fold3", "fold4"], seeds=(1, 2, 3))
        value = {label: float(np.mean([f["metrics_val.json"]["metrics"]["balanced_accuracy"] for _, _, f in r]))
                 for label, r in runs.items()}
        self.assertEqual(len(runs["shuffled"]), 12)
        self.assertLess(abs(value["shuffled"] - 1 / 3), 0.12)          # 0.331 here; 0.28 to 0.36 at noise 0.8
        self.assertGreaterEqual(value["base"], 0.9)
        params = runs["shuffled"][0][2]["model_params.json"]
        self.assertEqual(params["model"], "logistic_regression")

    def test_refit_on_train_and_val_predicts_the_test_rows(self):
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier

        from sweepkit.families.common import folds_of, load_data, prepare
        from sweepkit.families.supervised_io import task_targets
        from sweepkit.runs.spec import validate_job

        base = classify_base()                         # noise 0.8: classes overlap, so the extra rows matter
        base["model"] = dict(KNN, n_neighbors=1)
        base["fit"]["refit"] = "train_and_val"
        _, seed, files = run_jobs(base, {}, ["fold1"])["base"][0]
        self.assertEqual(files["metrics_test.json"]["refit"], "train_and_val")
        self.assertIn("final_params", files["model_params.json"])
        job = validate_job(base)
        data = load_data(job, lambda message: None)
        fold = folds_of(job, data)[0]["fold1"]
        x, positions, _, _, _ = prepare(job, data, {k: v.copy() for k, v in fold.roles.items()}, "train", seed,
                                        "cpu", lambda message: None)
        y = task_targets(job, data.table)
        alone = KNeighborsClassifier(n_neighbors=1).fit(x["train"], y[positions["train"]])
        refit = KNeighborsClassifier(n_neighbors=1).fit(np.concatenate([x["train"], x["val"]]),
                                                        np.concatenate([y[positions["train"]], y[positions["val"]]]))
        classes = list(base["task"]["classes"])
        test_pred = np.array([classes.index(r["pred"]) for r in rows(files["predictions_test.csv"])])
        val_pred = np.array([classes.index(r["pred"]) for r in rows(files["predictions_val.csv"])])
        self.assertTrue(np.array_equal(test_pred, refit.predict(x["test"])))
        self.assertTrue(np.array_equal(val_pred, alone.predict(x["val"])))      # validation never sees the refit
        self.assertTrue((alone.predict(x["test"]) != test_pred).any())          # the refit changed some answers


class TestRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        svr = dict(SVM, kernel="linear")
        variants = {"ridge": {"model": dict(LINEAR, penalty="ridge", alpha=1.0)},
                    "lasso": {"model": dict(LINEAR, penalty="lasso", alpha=0.1)},
                    "elastic_net": {"model": dict(LINEAR, penalty="elastic_net", alpha=0.1, l1_ratio=0.5)},
                    "svr": {"model": svr}, "knn": {"model": KNN}, "decision_tree": {"model": DECISION_TREE},
                    "random_forest": {"model": RANDOM_FOREST}, "extra_trees": {"model": EXTRA_TREES},
                    "hist_gradient_boosting": {"model": HIST_GB}}
        cls.runs = run_jobs(regression_base(), variants, ["fold1", "fold2"])

    def r2(self, label: str, k: int) -> float:
        return self.runs[label][k][2]["metrics_val.json"]["metrics"]["r2"]

    def test_linear_models_explain_a_linear_target_best(self):
        linear = ("base", "ridge", "lasso", "elastic_net", "svr")
        others = ("knn", "decision_tree", "random_forest", "extra_trees", "hist_gradient_boosting")
        for k in range(2):
            for label in linear:
                self.assertGreaterEqual(self.r2(label, k), 0.8, f"{label} fold {k + 1}")       # 0.883 or more
            for label in others:
                self.assertGreater(self.r2(label, k), 0.3, f"{label} fold {k + 1}")            # 0.429 or more
                self.assertLess(self.r2(label, k), self.r2("base", k), f"{label} fold {k + 1}")

    def test_regression_files_hold_numbers_not_class_scores(self):
        files = self.runs["base"][0][2]
        header = files["predictions_val.csv"].splitlines()[0]
        self.assertEqual(header, "row_id,target,pred")
        self.assertEqual(files["metrics_val.json"]["select"], {"metric": "rmse", "direction": "min",
                                                               "value": files["metrics_val.json"]["metrics"]["rmse"]})

    def test_lasso_zeroes_the_uninformative_coefficients(self):
        import numpy as np

        from sweepkit.data.synthetic import make
        from sweepkit.registry import MODELS
        from sweepkit.settings import validate

        entry = MODELS.get("linear")
        settings = validate(entry.settings_class, {"penalty": "lasso", "alpha": 0.3, "l1_ratio": None,
                                                   "fit_intercept": True, "max_iter": 5000, "extra": {}}, "linear")
        for seed in (1, 2, 3):
            made = make("regression_linear", 400, 8, 1, 1.0, seed)
            truth = np.array([float(v) for v in made["columns"]["coefficients"][0].split(",")])
            model = entry.target.make(settings, "regression", 1, "cpu").fit(made["x"], made["columns"]["target"])
            informative = truth != 0
            self.assertEqual(int(informative.sum()), 4)
            self.assertTrue((model.coef_[~informative] == 0).all(), seed)       # 4 of 4 on 10 seeds
            self.assertTrue((np.abs(model.coef_[informative]) > 0.5).all(), seed)   # 0.718 at the smallest


class TestMultilabel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        variants = {raw["name"]: {"model": raw} for raw in (LINEAR_SVM, SVM, KNN, NAIVE_BAYES, DECISION_TREE,
                                                             RANDOM_FOREST, EXTRA_TREES)}
        variants["shuffled"] = {"fit.control": "shuffle_train_labels"}
        cls.runs = run_jobs(multilabel_base(), variants, ["fold1", "fold2"])

    def metric(self, label: str, name: str) -> list:
        return [f["metrics_val.json"]["metrics"][name] for _, _, f in self.runs[label]]

    def test_every_multilabel_model_learns_the_labels(self):
        for label in self.runs:
            if label == "shuffled":
                continue
            self.assertGreaterEqual(min(self.metric(label, "label_f1_macro")), 0.55, label)   # 0.646 at worst
            self.assertGreaterEqual(min(self.metric(label, "label_roc_auc")), 0.65, label)    # 0.706 at worst
        for label in ("base", "linear_svm"):
            self.assertGreaterEqual(min(self.metric(label, "label_f1_macro")), 0.7, label)    # 0.802 at worst
            self.assertGreaterEqual(min(self.metric(label, "label_roc_auc")), 0.85, label)    # 0.891 at worst

    def test_one_linear_svm_per_label_gives_decision_values(self):
        import numpy as np

        files = self.runs["linear_svm"][0][2]
        table = rows(files["predictions_val.csv"])
        scores = np.array([[float(r[f"score_{name}"]) for name in LABELS] for r in table])
        predicted = np.array([[int(r[f"pred_{name}"]) for name in LABELS] for r in table])
        clear = np.abs(scores) > 1e-5                                     # values are written with 6 decimals
        self.assertTrue(np.array_equal((scores > 0)[clear], (predicted == 1)[clear]))   # the sign decides
        self.assertTrue((scores < 0).any())                               # decision values, not probabilities

    def test_shuffled_labels_lose_the_ranking(self):
        import numpy as np

        self.assertLess(abs(float(np.mean(self.metric("shuffled", "label_roc_auc"))) - 0.5), 0.15)
        self.assertEqual(set(self.runs["base"][0][2]["metrics_val.json"]["detail"]["positives"]), set(LABELS))


class TestSettingChecks(unittest.TestCase):
    """Each model's own checks refuse a bad combination when the sweep is expanded, before anything runs."""

    @classmethod
    def setUpClass(cls):
        load_catalog()

    def refused(self, job_maker, message: str):
        from sweepkit.runs.spec import check_job

        with self.assertRaisesRegex(ValueError, message):
            check_job(job_maker())

    def test_classification_settings(self):
        base = classification_base()
        self.refused(lambda: job_of(base, dict(LOGISTIC, l1_ratio=1.5)), r"l1_ratio in \[0, 1\]")
        self.refused(lambda: job_of(base, dict(LOGISTIC, c=0.0)), "c must be positive")
        self.refused(lambda: job_of(base, dict(XGBOOST, num_leaves=31)), "num_leaves is LightGBM's")
        self.refused(lambda: job_of(base, dict(LIGHTGBM, num_leaves=None)), "give num_leaves")
        self.refused(lambda: job_of(base, NAIVE_BAYES, device="cuda"), "has no GPU path")
        self.refused(lambda: job_of(base, dict(RANDOM_FOREST, max_depth=None), device="cuda"), "needs a max_depth")
        from sweepkit.runs.spec import check_job

        check_job(job_of(base, dict(RANDOM_FOREST, max_depth=8), device="cuda"))
        check_job(job_of(base, dict(LOGISTIC, c=None)))                    # no penalty

    def test_regression_settings(self):
        base = regression_base()
        self.refused(lambda: job_of(base, dict(LINEAR, alpha=1.0)), "alpha is 0 with penalty none")
        self.refused(lambda: job_of(base, dict(LINEAR, penalty="ridge", alpha=1.0, l1_ratio=0.5)),
                     "l1_ratio is set with penalty elastic_net only")
        self.refused(lambda: job_of(base, dict(LINEAR, penalty="elastic_net", alpha=1.0)),
                     "l1_ratio is set with penalty elastic_net only")
        for model in (dict(SVM, class_weight="balanced"), dict(RANDOM_FOREST, class_weight="balanced"),
                      dict(EXTRA_TREES, class_weight="balanced")):
            self.refused(lambda: job_of(base, model), "class_weight is none for regression")
        self.refused(lambda: job_of(base, NAIVE_BAYES), "not task 'regression'")
        self.refused(lambda: job_of(base, LOGISTIC), "not task 'regression'")

    def test_tasks_a_model_does_not_do(self):
        base = multilabel_base()
        for model in (HIST_GB, XGBOOST, LIGHTGBM, TPOT):
            self.refused(lambda: job_of(base, model), "not task 'multilabel'")
        self.refused(lambda: job_of(classification_base(), LINEAR), "not task 'classification'")


class TestSVMWithoutGPUProbabilities(unittest.TestCase):
    """cuML 26.10's SVC gives no probabilities (first node run, 2026-10-27): with probability true, fit.device cuda is
    refused and auto runs on the CPU, recording why."""

    @classmethod
    def setUpClass(cls):
        load_catalog()

    SVM_PROBABILITIES = {"name": "svm", "kernel": "rbf", "c": 1.0, "gamma": None, "degree": 3, "class_weight": "none",
                         "probability": True, "extra": {}}

    def test_cuda_is_refused_and_auto_runs_on_the_cpu(self):
        import json
        import tempfile
        from pathlib import Path

        from sweepkit.runs.spec import check_job
        from sweepkit.sweep.design import expand
        from sweepkit.sweep.runner import execute
        from tests.support import TempRepo

        base = classification_base()
        with self.assertRaisesRegex(ValueError, "cuML's SVC gives no probabilities"):
            check_job(job_of(base, self.SVM_PROBABILITIES, device="cuda"))
        check_job(job_of(base, dict(self.SVM_PROBABILITIES, probability=False), device="cuda"))
        settings = dict(base, model=self.SVM_PROBABILITIES, fit=dict(base["fit"], device="auto"))
        jobs, _ = expand(sweep(settings, [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]}]))
        with TempRepo(), tempfile.TemporaryDirectory() as d:
            execute(jobs[0], "cuda", lambda m: None, runs_root=Path(d))            # a worker that offers the GPU
            provenance = json.loads((Path(d) / jobs[0]["run_id"] / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual((provenance["device_requested"], provenance["device_used"]), ("auto", "cpu"))
        self.assertEqual(provenance["fallbacks"],
                         ["fit.device auto ran on the CPU: cuML's SVC gives no probabilities (probability true)"])


class TestOptionalLibraries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    @needs_module("xgboost")
    def test_xgboost(self):
        base = classification_base()
        base["model"] = XGBOOST
        metrics = run_jobs(base, {}, ["fold1"])["base"][0][2]["metrics_val.json"]["metrics"]
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.85)
        reg = regression_base()
        reg["model"] = XGBOOST
        self.assertGreater(run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.json"]["metrics"]["r2"], 0.5)

    @needs_module("lightgbm")
    def test_lightgbm(self):
        base = classification_base()
        base["model"] = LIGHTGBM
        metrics = run_jobs(base, {}, ["fold1"])["base"][0][2]["metrics_val.json"]["metrics"]
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.85)
        reg = regression_base()
        reg["model"] = LIGHTGBM
        self.assertGreater(run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.json"]["metrics"]["r2"], 0.5)

    @needs_module("tpot")
    def test_tpot_with_a_short_budget(self):
        base = classification_base()
        base["model"] = TPOT
        metrics = run_jobs(base, {}, ["fold1"])["base"][0][2]["metrics_val.json"]["metrics"]
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
