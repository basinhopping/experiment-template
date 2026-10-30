"""Every preparation step on small arrays with known answers, run the way the runners run them: run_prep fits each
step on the fit (training) rows and applies it to every role.

The rule every step keeps: what it learns comes from the fit rows alone. Changing the values of the other rows changes
neither the fit rows' output nor the step's record, and the other rows go through the statistics learned on the fit
rows. A step that must see other rows would be declared transductive; no step of the template is.

The steps: "impute", "scale", "drop_duplicates", "drop_constant", "outliers", "subsample", "onehot",
"select_columns", "select_features", "transform", "reduce", "batch_correct"."""
import dataclasses
import types
import unittest
from unittest import mock

import numpy as np

from tests.support import load_catalog, needs_module
from tests.sweep_fixtures import cluster_base

NAN = np.nan
TEMPLATE_STEPS = ("impute", "scale", "drop_duplicates", "drop_constant", "outliers", "subsample", "onehot",
                  "select_columns", "select_features", "transform", "reduce", "batch_correct")
REMOVES_ROWS = ("impute", "drop_duplicates", "outliers", "subsample")


def settings(name: str, **fields):
    """A step's settings, validated the way a sweep's prep list is validated."""
    from sweepkit.registry import PREP

    _, validated = PREP.validate_section({"name": name, **fields}, "prep[0]")
    return validated


def table(n_rows: int, **columns):
    from sweepkit.data.base import RowTable

    return RowTable([f"r{i:03d}" for i in range(n_rows)], {k: np.asarray(v) for k, v in columns.items()})


@dataclasses.dataclass
class Prepared:
    x: dict
    positions: dict
    names: tuple
    categorical: dict
    records: list


def run(steps, x, roles: dict, rows=None, names=None, categorical=None, fit_role: str = "train", seed: int = 1):
    """run_prep on (name, fields) steps: every role's rows of x, the step fitted on fit_role."""
    from sweepkit.prep.base import PrepContext, run_prep

    x = np.asarray(x, dtype=np.float64)
    rows = rows if rows is not None else table(x.shape[0])
    names = tuple(names) if names is not None else tuple(f"f{j}" for j in range(x.shape[1]))
    context = PrepContext(table=rows, feature_names=names, categorical=dict(categorical or {}), seed=seed,
                          device="cpu")
    steps = tuple((name, settings(name, **fields)) for name, fields in steps)
    x_by_role = {role: x[np.asarray(p, dtype=np.int64)] for role, p in roles.items()}
    positions = {role: np.asarray(p, dtype=np.int64) for role, p in roles.items()}
    return Prepared(*run_prep(steps, x_by_role, positions, fit_role, context, lambda message: None))


def refused(prep: list) -> None:
    """Validate and check a whole job with this prep list (the checks a sweep runs when it is expanded)."""
    from sweepkit.runs.spec import check_job, validate_job

    job = cluster_base()
    job["prep"] = prep
    check_job(validate_job(job))


IMPUTE = {"fill_value": None, "n_neighbors": None, "max_iter": None}
SCALE = {"columns": ["*"], "per_group": None}
TRANSFORM = {"columns": ["*"], "offset": None, "cofactor": None, "pseudocount": None, "reference": None,
             "target_sum": None, "n_quantiles": None, "output": None}
OUTLIERS = {"low_quantile": None, "high_quantile": None, "n_sd": None, "detector": None}
FEATURES = {"n": None, "threshold": None, "n_bins": None}
BATCH = {"tolerance": None, "max_iter": None, "theta": None}
PCA_SETTINGS = {"name": "pca", "n_components": 2, "solver": "full", "whiten": False, "batch_size": None,
                "extra": {}}


class PrepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def assert_fit_rows_only(self, steps, x, roles, perturb, fit_role="train", **kwargs) -> tuple:
        """Changing the rows outside the fit role changes nothing the steps learned (the fit rows' output and the
        records stay the same), and each other row goes through the learned statistics on its own: its output does
        not depend on which other rows are transformed with it."""
        first = run(steps, x, roles, fit_role=fit_role, **kwargs)
        changed = np.array(x, dtype=np.float64, copy=True)
        for role, p in roles.items():
            if role != fit_role:
                changed[np.asarray(p)] = perturb(changed[np.asarray(p)])
        second = run(steps, changed, roles, fit_role=fit_role, **kwargs)
        np.testing.assert_array_equal(first.x[fit_role], second.x[fit_role])
        np.testing.assert_array_equal(first.positions[fit_role], second.positions[fit_role])
        self.assertEqual(first.records, second.records)
        halves = {role: (p if role == fit_role else np.asarray(p)[:max(1, len(p) // 2)]) for role, p in roles.items()}
        third = run(steps, x, halves, fit_role=fit_role, **kwargs)
        for role, p in halves.items():
            if role != fit_role:
                np.testing.assert_array_equal(third.x[role], first.x[role][:len(p)])
        return first, second


class TestImpute(PrepTest):
    x = np.array([[1, 10], [NAN, 20], [3, NAN], [5, 40], [11, 90],      # training rows 0 to 4
                  [NAN, NAN], [100, NAN], [NAN, 1000]])                 # validation rows 5 to 7
    roles = {"train": [0, 1, 2, 3, 4], "val": [5, 6, 7]}

    def test_impute_mean_median_and_constant_use_the_fit_rows(self):
        cases = {"mean": (5.0, 40.0), "median": (4.0, 30.0), "constant": (0.5, 0.5)}
        for method, (first, second) in cases.items():
            with self.subTest(method=method):
                fields = dict(IMPUTE, method=method, fill_value=0.5 if method == "constant" else None)
                out = run([("impute", fields)], self.x, self.roles)
                np.testing.assert_array_equal(out.x["val"], [[first, second], [100, second], [first, 1000]])
                np.testing.assert_array_equal(out.x["train"][1:3], [[first, 20], [3, second]])
                self.assertEqual(out.records[0]["missing_per_column"], {"f0": 1, "f1": 1})
                self.assert_fit_rows_only([("impute", fields)], self.x, self.roles, lambda v: v * 7 + 3)

    def test_impute_drop_rows_takes_rows_out_of_the_fit_only(self):
        out = run([("impute", dict(IMPUTE, method="drop_rows"))], self.x, self.roles)
        record = out.records[0]
        self.assertEqual((record["rows_in"], record["rows_removed"]), (5, 2))
        self.assertEqual(record["removed_row_ids"], ["r001", "r002"])
        np.testing.assert_array_equal(out.positions["train"], [0, 3, 4])
        np.testing.assert_array_equal(out.x["train"], [[1, 10], [5, 40], [11, 90]])
        np.testing.assert_array_equal(out.positions["val"], [5, 6, 7])
        self.assertTrue(np.isnan(out.x["val"]).any(), "rows outside the fit are never removed")

    def test_impute_knn_and_iterative_follow_scikit_learn_fitted_on_the_fit_rows(self):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer, KNNImputer

        from sweepkit.streams import int_seed

        fit, val = self.x[self.roles["train"]], self.x[self.roles["val"]]
        expected = {"knn": KNNImputer(n_neighbors=2).fit(fit),
                    "iterative": IterativeImputer(max_iter=5, random_state=int_seed(1, "prep impute")).fit(fit)}
        for method, model in expected.items():
            with self.subTest(method=method):
                fields = dict(IMPUTE, method=method, n_neighbors=2 if method == "knn" else None,
                              max_iter=5 if method == "iterative" else None)
                out = run([("impute", fields)], self.x, self.roles)
                np.testing.assert_allclose(out.x["val"], model.transform(val), rtol=1e-12)
                np.testing.assert_allclose(out.x["train"], model.transform(fit), rtol=1e-12)
                self.assert_fit_rows_only([("impute", fields)], self.x, self.roles, lambda v: v * 7 + 3)

    def test_impute_iterative_gives_a_row_the_same_values_whatever_rows_come_with_it(self):
        fields = dict(IMPUTE, method="iterative", max_iter=5)
        together = run([("impute", fields)], self.x, self.roles)
        alone = run([("impute", fields)], self.x, {"train": self.roles["train"], "val": [5]})   # every value missing
        np.testing.assert_array_equal(alone.x["val"][0], together.x["val"][0])

    def test_impute_refuses_a_column_without_a_fit_value(self):
        x = np.array([[1, NAN], [2, NAN], [3, 4]])
        for method, option in (("mean", {}), ("knn", {"n_neighbors": 1}), ("iterative", {"max_iter": 3})):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "column f1 has no value in the fit rows"):
                    run([("impute", dict(IMPUTE, method=method, **option))], x, {"train": [0, 1], "val": [2]})

    def test_impute_fills_categorical_codes_with_the_most_frequent_fit_code(self):
        x = np.array([[1, 0], [2, 2], [3, 2], [5, NAN], [7, 1],      # training rows: code 2 is the most frequent
                      [NAN, NAN], [9, NAN], [9, NAN]])                # validation rows: their own codes do not count
        out = run([("impute", dict(IMPUTE, method="mean"))], x, self.roles,
                  categorical={1: ("red", "green", "blue")})
        np.testing.assert_array_equal(out.x["train"][3], [5, 2])
        np.testing.assert_array_equal(out.x["val"], [[3.6, 2], [9, 2], [9, 2]])

    def test_impute_refuses_settings_of_another_method(self):
        for fields in (dict(IMPUTE, method="constant"), dict(IMPUTE, method="mean", n_neighbors=3),
                       dict(IMPUTE, method="knn", max_iter=4)):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "prep impute"):
                    refused([{"name": "impute", **fields}])
        refused([{"name": "impute", **dict(IMPUTE, method="knn", n_neighbors=3)}])

    def test_a_step_that_removes_every_fit_row_is_refused(self):
        x = np.array([[NAN, 1], [2, NAN], [3, 4]])
        with self.assertRaisesRegex(ValueError, "removed every fit row"):
            run([("impute", dict(IMPUTE, method="drop_rows"))], x, {"train": [0, 1], "val": [2]})


class TestScale(PrepTest):
    x = np.array([[1, 5], [2, 5], [3, 5], [4, 5],      # training rows: the second column is constant
                  [6, 7], [0, 5]])                     # validation rows
    roles = {"train": [0, 1, 2, 3], "val": [4, 5]}

    def test_scale_zscore_gives_mean_zero_and_sd_one_on_the_fit_rows(self):
        fields = dict(SCALE, method="zscore")
        out = run([("scale", fields)], self.x, self.roles)
        np.testing.assert_allclose(out.x["train"][:, 0].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(out.x["train"][:, 0].std(), 1, rtol=1e-12)
        np.testing.assert_array_equal(out.x["train"][:, 1], 0, "a constant column is centred only")
        sd = np.std([1, 2, 3, 4])
        np.testing.assert_allclose(out.x["val"], [[(6 - 2.5) / sd, 2], [(0 - 2.5) / sd, 0]], rtol=1e-12)
        self.assertEqual(out.records[0]["columns"], ["f0", "f1"])
        self.assert_fit_rows_only([("scale", fields)], self.x, self.roles, lambda v: v * 9 - 4)

    def test_scale_minmax_robust_and_maxabs(self):
        expected = {"minmax": lambda v: (v - 1) / 3, "robust": lambda v: (v - 2.5) / 1.5, "maxabs": lambda v: v / 4}
        for method, formula in expected.items():
            with self.subTest(method=method):
                out = run([("scale", dict(SCALE, method=method))], self.x, self.roles)
                np.testing.assert_allclose(out.x["train"][:, 0], formula(np.array([1, 2, 3, 4.0])), rtol=1e-12)
                np.testing.assert_allclose(out.x["val"][:, 0], formula(np.array([6, 0.0])), rtol=1e-12)
                self.assert_fit_rows_only([("scale", dict(SCALE, method=method))], self.x, self.roles,
                                          lambda v: v * 9 - 4)

    def test_scale_rows_by_their_l1_or_l2_norm(self):
        x = np.array([[3, -4], [0, 0], [1, 1]])
        l1 = run([("scale", dict(SCALE, method="l1_rows"))], x, {"train": [0, 1], "val": [2]})
        np.testing.assert_allclose(l1.x["train"], [[3 / 7, -4 / 7], [0, 0]])
        np.testing.assert_allclose(l1.x["val"], [[0.5, 0.5]])
        l2 = run([("scale", dict(SCALE, method="l2_rows"))], x, {"train": [0, 1], "val": [2]})
        np.testing.assert_allclose(l2.x["train"], [[0.6, -0.8], [0, 0]])

    def test_scale_named_columns_only_and_never_categorical_codes(self):
        x = np.array([[1, 10, 2], [3, 30, 0], [5, 50, 1]])
        out = run([("scale", dict(SCALE, method="minmax", columns=["re:^a"]))], x, {"train": [0, 1], "val": [2]},
                  names=("a1", "a2", "b1"))
        np.testing.assert_allclose(out.x["val"], [[2, 2, 1]])
        self.assertEqual(out.records[0]["columns"], ["a1", "a2"])
        coded = run([("scale", dict(SCALE, method="zscore"))], x, {"train": [0, 1], "val": [2]},
                    categorical={2: ("x", "y", "z")})
        np.testing.assert_array_equal(coded.x["val"][:, 2], [1], "categorical codes are left alone")
        with self.assertRaisesRegex(ValueError, "no numeric column selected"):
            run([("scale", dict(SCALE, method="zscore", columns=["c"]))], x, {"train": [0, 1], "val": [2]},
                names=("a", "b", "c"), categorical={2: ("x", "y", "z")})

    def test_scale_per_group_uses_each_groups_own_fit_statistics(self):
        x = np.array([[1.0], [3.0], [10.0], [30.0], [40.0], [5.0]])
        rows = table(6, plate=["p1", "p1", "p2", "p2", "p2", "p3"])
        out = run([("scale", dict(SCALE, method="zscore", per_group="plate"))], x,
                  {"train": [0, 1, 2, 3], "val": [4]}, rows=rows)
        np.testing.assert_allclose(out.x["train"][:, 0], [-1, 1, -1, 1])
        np.testing.assert_allclose(out.x["val"][:, 0], [2.0])
        with self.assertRaisesRegex(ValueError, r"groups \['p3'\] have no fit rows"):
            run([("scale", dict(SCALE, method="zscore", per_group="plate"))], x,
                {"train": [0, 1, 2, 3], "val": [5]}, rows=rows)

    def test_scale_refuses_per_group_with_per_row_norms(self):
        with self.assertRaisesRegex(ValueError, "per-row scaling takes per_group None"):
            refused([{"name": "scale", "method": "l2_rows", "columns": ["*"], "per_group": "group"}])


class TestDropDuplicates(PrepTest):
    x = np.array([[1, 2], [3, 4], [1, 2], [3.0000001, 4], [5, 6],     # training rows: row 2 repeats row 0
                  [1, 2]])                                             # a validation row repeating row 0
    roles = {"train": [0, 1, 2, 3, 4], "val": [5]}

    def test_drop_duplicates_keeps_the_first_of_repeated_fit_rows(self):
        out = run([("drop_duplicates", {"decimals": None})], self.x, self.roles)
        record = out.records[0]
        self.assertEqual((record["rows_removed"], record["removed_row_ids"]), (1, ["r002"]))
        np.testing.assert_array_equal(out.positions["train"], [0, 1, 3, 4])
        np.testing.assert_array_equal(out.positions["val"], [5])
        rounded = run([("drop_duplicates", {"decimals": 3})], self.x, self.roles)
        self.assertEqual(rounded.records[0]["removed_row_ids"], ["r002", "r003"])
        self.assertEqual(rounded.records[0]["decimals"], 3)
        self.assert_fit_rows_only([("drop_duplicates", {"decimals": None})], self.x, self.roles, lambda v: v + 1)


class TestDropConstant(PrepTest):
    # training rows 0 to 5; the validation row varies the column that is constant on the fit rows
    x = np.array([[1, 7, 0, 1.0], [2, 7, 0, 1.1], [3, 7, 0, 1.2], [4, 7, 0, 1.0], [5, 7, 0, 1.1], [6, 7, 1, 1.2],
                  [9, 100, 1, 1.0]])
    roles = {"train": [0, 1, 2, 3, 4, 5], "val": [6]}
    fields = {"min_variance": 0.0, "max_frequency_ratio": None, "min_unique_fraction": None}

    def test_drop_constant_removes_the_column_constant_on_the_fit_rows(self):
        out = run([("drop_constant", self.fields)], self.x, self.roles, categorical={3: ("a", "b")})
        self.assertEqual(out.names, ("f0", "f2", "f3"))
        self.assertEqual(out.records[0]["dropped_columns"], ["f1"])
        np.testing.assert_array_equal(out.x["val"], [[9, 1, 1.0]])
        self.assertEqual(out.categorical, {2: ("a", "b")}, "categorical indices follow the kept columns")
        self.assert_fit_rows_only([("drop_constant", self.fields)], self.x, self.roles, lambda v: v * 3)

    def test_drop_constant_variance_floor_and_near_constant_rule(self):
        floor = run([("drop_constant", dict(self.fields, min_variance=0.05))], self.x, self.roles)
        self.assertEqual(floor.records[0]["dropped_columns"], ["f1", "f3"])
        near = run([("drop_constant", dict(self.fields, max_frequency_ratio=4.0, min_unique_fraction=0.5))],
                   self.x, self.roles)
        self.assertEqual(near.records[0]["dropped_columns"], ["f1", "f2"])

    def test_drop_constant_refusals(self):
        with self.assertRaisesRegex(ValueError, "every column would be dropped"):
            run([("drop_constant", self.fields)], np.ones((4, 2)), {"train": [0, 1, 2], "val": [3]})
        with self.assertRaisesRegex(ValueError, "set max_frequency_ratio and min_unique_fraction together"):
            refused([{"name": "drop_constant", "min_variance": 0.0, "max_frequency_ratio": 5.0,
                      "min_unique_fraction": None}])


class TestOutliers(PrepTest):
    def test_outliers_clip_quantiles_learned_on_the_fit_rows(self):
        x = np.column_stack([np.arange(101.0), np.arange(101.0)[::-1], np.full(101, 2.0)])
        x = np.vstack([x, [[-50, 1000, 2], [500, -7, 1]]])
        roles = {"train": np.arange(101), "val": [101, 102]}
        fields = dict(OUTLIERS, method="clip_quantiles", low_quantile=0.1, high_quantile=0.9)
        out = run([("outliers", fields)], x, roles, categorical={2: ("a", "b", "c")})
        self.assertEqual((out.records[0]["low"], out.records[0]["high"]), ([10.0, 10.0], [90.0, 90.0]))
        np.testing.assert_array_equal(out.x["val"], [[10, 90, 2], [90, 10, 1]])
        self.assertEqual((out.x["train"].min(), out.x["train"][:, :2].max()), (2.0, 90.0))
        self.assert_fit_rows_only([("outliers", fields)], x, roles, lambda v: v * 50, categorical={2: ("a", "b", "c")})

    def test_outliers_clip_sd(self):
        x = np.array([[0.0], [0.0], [0.0], [0.0], [10.0], [20.0], [-100.0]])
        fields = dict(OUTLIERS, method="clip_sd", n_sd=1.5)
        out = run([("outliers", fields)], x, {"train": [0, 1, 2, 3, 4], "val": [5, 6]})
        self.assertEqual((out.records[0]["low"], out.records[0]["high"]), ([-4.0], [8.0]))      # mean 2, SD 4
        np.testing.assert_array_equal(out.x["train"][:, 0], [0, 0, 0, 0, 8])
        np.testing.assert_array_equal(out.x["val"][:, 0], [8, -4])

    def test_outliers_remove_by_detector_takes_flagged_fit_rows_out(self):
        rng = np.random.default_rng(0)
        inliers = rng.normal(size=(38, 2))
        x = np.vstack([inliers[:19], [[25.0, 25.0]], inliers[19:], [[-30.0, 20.0]],     # far rows 19 and 39
                       [[40.0, -40.0]], rng.normal(size=(4, 2))])                        # validation rows 40 to 44
        roles = {"train": np.arange(40), "val": np.arange(40, 45)}
        detector = {"name": "elliptic_envelope", "contamination": 0.05, "support_fraction": None, "extra": {}}
        fields = dict(OUTLIERS, method="remove_by_detector", detector=detector)
        out = run([("outliers", fields)], x, roles)
        record = out.records[0]
        self.assertEqual((record["detector"], record["removed_row_ids"]), ("elliptic_envelope", ["r019", "r039"]))
        np.testing.assert_array_equal(out.positions["val"], np.arange(40, 45), "a far validation row stays")
        np.testing.assert_array_equal(out.x["train"], x[np.setdiff1d(np.arange(40), [19, 39])])
        self.assert_fit_rows_only([("outliers", fields)], x, roles, lambda v: v * 10)

    def test_outliers_refuse_settings_of_another_method(self):
        for fields in (dict(OUTLIERS, method="clip_quantiles", low_quantile=0.1, high_quantile=0.9, n_sd=2.0),
                       dict(OUTLIERS, method="clip_quantiles", low_quantile=0.9, high_quantile=0.1),
                       dict(OUTLIERS, method="clip_sd"), dict(OUTLIERS, method="remove_by_detector")):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "prep outliers"):
                    refused([{"name": "outliers", **fields}])


class TestSubsample(PrepTest):
    rows = table(40, kind=["a"] * 30 + ["b"] * 10)
    x = np.arange(80.0).reshape(40, 2)
    fields = {"method": "random", "fraction": 0.5, "n_rows": None, "stratify": None, "salt": "s"}

    def test_subsample_keeps_a_fixed_share_of_the_fit_rows(self):
        roles = {"train": np.arange(40), "val": [0]}
        out = run([("subsample", self.fields)], self.x, roles, rows=self.rows)
        self.assertEqual((out.records[0]["kept"], out.records[0]["rows_removed"]), (20, 20))
        again = run([("subsample", self.fields)], self.x, roles, rows=self.rows)
        np.testing.assert_array_equal(out.positions["train"], again.positions["train"])
        other = run([("subsample", dict(self.fields, salt="other"))], self.x, roles, rows=self.rows)
        self.assertNotEqual(out.positions["train"].tolist(), other.positions["train"].tolist())
        counted = run([("subsample", dict(self.fields, fraction=None, n_rows=7))], self.x, roles, rows=self.rows)
        self.assertEqual(counted.positions["train"].size, 7)
        np.testing.assert_array_equal(out.positions["val"], [0], "rows outside the fit are never removed")

    def test_subsample_depends_on_the_row_ids_only(self):
        roles = {"train": np.arange(30), "val": np.arange(30, 40)}
        first = run([("subsample", self.fields)], self.x, roles, rows=self.rows)
        shuffled = run([("subsample", self.fields)], self.x[::-1].copy(), roles, rows=self.rows)
        np.testing.assert_array_equal(first.positions["train"], shuffled.positions["train"])
        self.assertEqual(first.positions["train"].size, 15)
        self.assert_fit_rows_only([("subsample", self.fields)], self.x, roles, lambda v: -v, rows=self.rows)

    def test_subsample_stratified_keeps_each_strata_share(self):
        fields = dict(self.fields, method="stratified", stratify="kind")
        out = run([("subsample", fields)], self.x, {"train": np.arange(40), "val": [0]}, rows=self.rows)
        kept = self.rows.column("kind")[out.positions["train"]]
        self.assertEqual(((kept == "a").sum(), (kept == "b").sum()), (15, 5))

    def test_subsample_refusals(self):
        bad = [dict(self.fields, n_rows=5), dict(self.fields, method="stratified"),
               dict(self.fields, stratify="group")]
        for fields in bad:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "prep subsample"):
                    refused([{"name": "subsample", **fields}])


class TestOneHot(PrepTest):
    categories = {1: ("red", "green", "blue")}
    x = np.array([[1.5, 0, 9], [2.5, 1, 8], [3.5, 0, 7],      # training rows hold no blue
                  [4.5, 2, 6]])                               # the validation row is blue
    roles = {"train": [0, 1, 2], "val": [3]}

    def test_onehot_one_column_per_category_of_the_dataset(self):
        out = run([("onehot", {"drop_first": False})], self.x, self.roles, categorical=self.categories)
        self.assertEqual(out.names, ("f0", "f1=red", "f1=green", "f1=blue", "f2"))
        np.testing.assert_array_equal(out.x["train"], [[1.5, 1, 0, 0, 9], [2.5, 0, 1, 0, 8], [3.5, 1, 0, 0, 7]])
        np.testing.assert_array_equal(out.x["val"], [[4.5, 0, 0, 1, 6]])
        self.assertEqual(out.categorical, {})
        self.assertEqual((out.records[0]["expanded"], out.records[0]["drop_first"]), (["f1"], False))
        first_dropped = run([("onehot", {"drop_first": True})], self.x, self.roles, categorical=self.categories)
        self.assertEqual(first_dropped.names, ("f0", "f1=green", "f1=blue", "f2"))
        self.assert_fit_rows_only([("onehot", {"drop_first": False})], self.x, self.roles,
                                  lambda v: np.column_stack([v[:, 0] * 2, v[:, 1], v[:, 2] - 1]),
                                  categorical=self.categories)

    def test_onehot_refuses_unknown_codes_and_data_without_categories(self):
        with self.assertRaisesRegex(ValueError, "code outside its categories"):
            run([("onehot", {"drop_first": False})], np.array([[1, 0], [2, 3]]), {"train": [0], "val": [1]},
                categorical={1: ("a", "b")})
        with self.assertRaisesRegex(ValueError, "no categorical column"):
            run([("onehot", {"drop_first": False})], self.x, self.roles)


class TestSelectColumns(PrepTest):
    def test_select_columns_by_name_and_pattern(self):
        x = np.arange(12.0).reshape(3, 4)
        names = ("age", "height", "weight", "site_code")
        out = run([("select_columns", {"keep": ["weight", "re:^a"]})], x, {"train": [0, 1], "val": [2]},
                  names=names, categorical={3: ("s1", "s2")})
        self.assertEqual(out.names, ("age", "weight"))
        np.testing.assert_array_equal(out.x["val"], [[8, 10]])
        self.assertEqual(out.records[0]["kept"], ["age", "weight"])
        self.assertEqual(out.categorical, {})
        coded = run([("select_columns", {"keep": ["re:code$", "age"]})], x, {"train": [0, 1], "val": [2]},
                    names=names, categorical={3: ("s1", "s2")})
        self.assertEqual((coded.names, coded.categorical), (("age", "site_code"), {1: ("s1", "s2")}))
        every = run([("select_columns", {"keep": ["*"]})], x, {"train": [0, 1], "val": [2]}, names=names)
        self.assertEqual(every.names, names)
        with self.assertRaisesRegex(ValueError, "'bmi' matches no column"):
            run([("select_columns", {"keep": ["bmi"]})], x, {"train": [0, 1], "val": [2]}, names=names)


class TestSelectFeatures(PrepTest):
    rng = np.random.default_rng(5)
    x = np.column_stack([rng.normal(0, 1, 40), rng.normal(0, 10, 40), rng.normal(0, 5, 40), rng.normal(0, 0.1, 40)])
    roles = {"train": np.arange(30), "val": np.arange(30, 40)}

    def test_select_features_by_fit_row_variance(self):
        fields = dict(FEATURES, method="variance_top", n=2)
        out = run([("select_features", fields)], self.x, self.roles)
        self.assertEqual(out.names, ("f1", "f2"))
        self.assertEqual(out.records[0]["kept"], ["f1", "f2"])
        # validation rows a thousand times wider in the first column change nothing chosen
        self.assert_fit_rows_only([("select_features", fields)], self.x, self.roles,
                                  lambda v: np.column_stack([v[:, 0] * 1000, v[:, 1:]]))
        threshold = run([("select_features", dict(FEATURES, method="variance_threshold", threshold=5.0))],
                        self.x, self.roles)
        self.assertEqual(threshold.names, ("f1", "f2"))

    def test_select_features_drops_a_column_correlated_with_an_earlier_one(self):
        x = np.column_stack([self.x[:, 0], 2 * self.x[:, 0] + 1, self.x[:, 2]])
        out = run([("select_features", dict(FEATURES, method="correlation", threshold=0.9))], x, self.roles)
        self.assertEqual(out.names, ("f0", "f2"))

    def test_select_features_highly_variable_finds_the_overdispersed_counts(self):
        steady = np.tile(np.array([[4, 5, 6, 5, 4, 6]] * 4).T, (2, 1))           # four genes, variance near mean
        bursty = np.tile(np.array([[0, 0, 0, 0, 0, 30], [0, 0, 60, 0, 0, 0]]).T, (2, 1))
        counts = np.column_stack([steady[:, :2], bursty[:, 0], steady[:, 2:], bursty[:, 1]]).astype(float)
        out = run([("select_features", dict(FEATURES, method="highly_variable", n=2, n_bins=1))],
                  np.log1p(counts), {"train": np.arange(12), "val": [0]})
        self.assertEqual(out.names, ("f2", "f5"))
        self.assertEqual(sorted(out.records[0]["kept"]), ["f2", "f5"])

    def test_select_features_keeps_categorical_columns_and_checks_its_settings(self):
        x = np.column_stack([self.x[:, :2], np.tile([0.0, 1.0], 20)])
        out = run([("select_features", dict(FEATURES, method="variance_top", n=1))], x, self.roles,
                  categorical={2: ("a", "b")})
        self.assertEqual((out.names, out.categorical), (("f1", "f2"), {1: ("a", "b")}))
        for fields in (dict(FEATURES, method="variance_top"), dict(FEATURES, method="correlation", n=2),
                       dict(FEATURES, method="highly_variable", n=2)):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "prep select_features"):
                    refused([{"name": "select_features", **fields}])


class TestTransform(PrepTest):
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0], [7.0, 16.0]])
    roles = {"train": [0, 1, 2], "val": [3]}

    def test_transform_fixed_formulas(self):
        cases = {"log": ({"offset": 1.0}, lambda v: np.log(v + 1)), "log1p": ({}, np.log1p),
                 "arcsinh": ({"cofactor": 5.0}, lambda v: np.arcsinh(v / 5)), "sqrt": ({}, np.sqrt)}
        for method, (options, formula) in cases.items():
            with self.subTest(method=method):
                out = run([("transform", dict(TRANSFORM, method=method, **options))], self.x, self.roles)
                np.testing.assert_allclose(out.x["train"], formula(self.x[:3]), rtol=1e-12)
                np.testing.assert_allclose(out.x["val"], formula(self.x[3:]), rtol=1e-12)

    def test_transform_power_lambdas_learned_on_the_fit_rows(self):
        from sklearn.preprocessing import PowerTransformer

        for method, library in (("boxcox", "box-cox"), ("yeojohnson", "yeo-johnson")):
            with self.subTest(method=method):
                fields = dict(TRANSFORM, method=method)
                out = run([("transform", fields)], self.x, self.roles)
                model = PowerTransformer(method=library, standardize=False).fit(self.x[:3])
                np.testing.assert_allclose(out.records[0]["lambdas"], model.lambdas_, rtol=1e-12)
                np.testing.assert_allclose(out.x["val"], model.transform(self.x[3:]), rtol=1e-12)
                self.assert_fit_rows_only([("transform", fields)], self.x, self.roles, lambda v: v * 4 + 1)

    def test_transform_quantile_and_rank_against_the_fit_rows(self):
        from sklearn.preprocessing import QuantileTransformer

        fields = dict(TRANSFORM, method="quantile", n_quantiles=10, output="uniform")
        out = run([("transform", fields)], self.x, self.roles)
        model = QuantileTransformer(n_quantiles=3, output_distribution="uniform", subsample=3).fit(self.x[:3])
        np.testing.assert_allclose(out.x["val"], model.transform(self.x[3:]))
        np.testing.assert_allclose(out.x["train"], [[0, 0], [0.5, 0.5], [1, 1]], atol=1e-7)
        x = np.array([[1.0], [2.0], [3.0], [4.0], [2.5], [10.0], [0.0], [3.0]])
        ranked = run([("transform", dict(TRANSFORM, method="rank"))], x, {"train": [0, 1, 2, 3], "val": [4, 5, 6, 7]})
        np.testing.assert_allclose(ranked.x["val"][:, 0], [0.5, 1.0, 0.0, 0.75])
        np.testing.assert_allclose(ranked.x["train"][:, 0], [0.25, 0.5, 0.75, 1.0])
        self.assert_fit_rows_only([("transform", fields)], self.x, self.roles, lambda v: v * 4 + 1)

    def test_transform_compositions_and_total_counts(self):
        x = np.array([[1.0, 3.0, 7.0], [0.0, 1.0, 3.0], [2.0, 2.0, 2.0]])
        clr = run([("transform", dict(TRANSFORM, method="clr", pseudocount=1.0))], x, {"train": [0, 1], "val": [2]})
        np.testing.assert_allclose(clr.x["train"].sum(axis=1), 0, atol=1e-12)
        logs = np.log(x[0] + 1)
        np.testing.assert_allclose(clr.x["train"][0], logs - logs.mean(), rtol=1e-12)
        alr = run([("transform", dict(TRANSFORM, method="alr", pseudocount=1.0, reference="f1"))], x,
                  {"train": [0, 1], "val": [2]}, categorical={})
        self.assertEqual(alr.names, ("f0", "f2"))
        np.testing.assert_allclose(alr.x["train"][0], [np.log(2 / 4), np.log(8 / 4)], rtol=1e-12)
        total = run([("transform", dict(TRANSFORM, method="total_count", target_sum=10.0))], x,
                    {"train": [0, 1], "val": [2]})
        np.testing.assert_allclose(total.x["train"].sum(axis=1), 10)
        np.testing.assert_allclose(total.x["val"], [[10 / 3, 10 / 3, 10 / 3]])

    def test_transform_refuses_values_and_settings_it_cannot_use(self):
        with self.assertRaisesRegex(ValueError, "at or below -offset"):
            run([("transform", dict(TRANSFORM, method="log", offset=1.0))], np.array([[0.0], [-1.0]]),
                {"train": [0], "val": [1]})
        with self.assertRaisesRegex(ValueError, "a value is negative"):
            run([("transform", dict(TRANSFORM, method="sqrt"))], np.array([[1.0], [-2.0]]), {"train": [0], "val": [1]})
        with self.assertRaisesRegex(ValueError, "rows have no counts"):
            run([("transform", dict(TRANSFORM, method="total_count", target_sum=10.0))],
                np.array([[1.0, 1.0], [0.0, 0.0]]), {"train": [0], "val": [1]})
        for fields in (dict(TRANSFORM, method="log"), dict(TRANSFORM, method="arcsinh", cofactor=0.0),
                       dict(TRANSFORM, method="sqrt", offset=1.0), dict(TRANSFORM, method="alr", pseudocount=1.0)):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "prep transform"):
                    refused([{"name": "transform", **fields}])


class TestReduce(PrepTest):
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(40, 2))
    x = latent @ rng.normal(size=(2, 5)) + 0.01 * rng.normal(size=(40, 5))
    roles = {"train": np.arange(30), "val": np.arange(30, 40)}

    def test_reduce_with_pca_fitted_on_the_fit_rows(self):
        from sklearn.decomposition import PCA

        out = run([("reduce", {"method": PCA_SETTINGS})], self.x, self.roles)
        self.assertEqual(out.names, ("pca_1", "pca_2"))
        model = PCA(n_components=2, svd_solver="full").fit(self.x[:30].astype(np.float32))
        np.testing.assert_allclose(np.abs(out.x["val"]), np.abs(model.transform(self.x[30:].astype(np.float32))),
                                   rtol=1e-4, atol=1e-4)
        record = out.records[0]
        self.assertEqual((record["method"], record["components"], record["columns_out"]), ("pca", 2, 2))
        self.assertGreater(record["explained_variance"], 0.99)
        self.assert_fit_rows_only([("reduce", {"method": PCA_SETTINGS})], self.x, self.roles, lambda v: v * 5)

    def test_reduce_without_a_transform_needs_the_evaluated_rows_to_be_the_fit_rows(self):
        mds = {"name": "mds", "n_components": 2, "metric_mds": True, "metric": "euclidean", "init": "random",
               "n_init": 1, "max_iter": 50, "eps": 1e-3, "normalized_stress": False, "extra": {}}
        with self.assertRaisesRegex(ValueError, "mds cannot embed rows it was not fitted on"):
            run([("reduce", {"method": mds})], self.x, self.roles)
        every = np.arange(40)
        out = run([("reduce", {"method": mds})], self.x, {"fit": every, "eval": every.copy()}, fit_role="fit")
        np.testing.assert_array_equal(out.x["fit"], out.x["eval"])
        self.assertEqual(out.names, ("mds_1", "mds_2"))
        pcoa = run([("reduce", {"method": {"name": "pcoa", "n_components": 2, "metric": "euclidean"}})], self.x,
                   self.roles)
        self.assertEqual(pcoa.x["val"].shape, (10, 2), "PCoA places new rows by Gower's add-a-point formula")

    @needs_module("umap")
    def test_reduce_with_umap(self):
        umap = {"name": "umap", "n_components": 2, "n_neighbors": 10, "min_dist": 0.0, "spread": 1.0,
                "metric": "euclidean", "n_epochs": 50, "learning_rate": 1.0, "init": "spectral",
                "set_op_mix_ratio": 1.0, "local_connectivity": 1.0, "repulsion_strength": 1.0,
                "negative_sample_rate": 5, "densmap": False, "gpu_build_algo": "brute_force_knn", "extra": {}}
        out = run([("reduce", {"method": umap})], self.x, self.roles)
        self.assertEqual(out.names, ("umap_1", "umap_2"))
        self.assertEqual((out.x["train"].shape, out.x["val"].shape), ((30, 2), (10, 2)))



class TestBatchCorrect(PrepTest):
    def test_batch_correct_center_moves_each_batch_mean_to_the_fit_mean(self):
        x = np.array([[1.0, 0], [2, 0], [3, 1], [11, 1], [12, 0], [13, 1], [10, 0], [0, 1], [5, 0]])
        rows = table(9, batch=["A", "A", "A", "B", "B", "B", "A", "B", "C"])
        fields = dict(BATCH, method="center", batch="batch")
        roles = {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7]}
        out = run([("batch_correct", fields)], x, roles, rows=rows, categorical={1: ("no", "yes")})
        np.testing.assert_allclose(out.x["train"][:, 0], [6, 7, 8, 6, 7, 8])
        np.testing.assert_allclose(out.x["val"][:, 0], [15, -5])
        np.testing.assert_array_equal(out.x["val"][:, 1], [0, 1], "categorical codes are left alone")
        self.assert_fit_rows_only([("batch_correct", fields)], x, roles, lambda v: v * 3, rows=rows,
                                  categorical={1: ("no", "yes")})
        with self.assertRaisesRegex(ValueError, r"batches \['C'\] have no fit rows"):
            run([("batch_correct", fields)], x, {"train": [0, 1, 2, 3, 4, 5], "val": [8]}, rows=rows)

    def test_batch_correct_combat_removes_additive_and_multiplicative_effects(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 6)) * np.array([1, 2, 0.5, 1.5, 1, 3]) + np.array([0, 5, -2, 1, 10, 0])
        x[20:] = x[20:] * 2.0 + 5.0
        batch = np.array(["A"] * 20 + ["B"] * 20)
        rows = table(40, batch=batch)
        roles = {"train": np.r_[0:16, 20:36], "val": np.r_[16:20, 36:40]}
        fields = dict(BATCH, method="combat", batch="batch", tolerance=1e-4, max_iter=100)
        out = run([("batch_correct", fields)], x, roles, rows=rows)
        fit, b = x[roles["train"]], batch[roles["train"]]
        before = np.abs(fit[b == "A"].mean(axis=0) - fit[b == "B"].mean(axis=0))
        after = np.abs(out.x["train"][b == "A"].mean(axis=0) - out.x["train"][b == "B"].mean(axis=0))
        self.assertTrue(np.all(after < 0.1 * before), f"before {before}, after {after}")
        ratio = out.x["train"][b == "B"].std(axis=0) / out.x["train"][b == "A"].std(axis=0)
        self.assertTrue(np.all((ratio > 0.75) & (ratio < 1.35)), ratio)
        record = out.records[0]
        self.assertEqual(record["batches"], ["A", "B"])
        self.assertTrue(all(1 <= n <= 100 for n in record["iterations"]))
        self.assert_fit_rows_only([("batch_correct", fields)], x, roles, lambda v: v * 3 - 1, rows=rows)
        with self.assertRaisesRegex(ValueError, "fewer than 2 fit rows"):
            run([("batch_correct", fields)], x, {"train": np.r_[0:16, 20], "val": [21]}, rows=rows)

    @needs_module("harmonypy")
    def test_batch_correct_harmony_corrects_the_fit_rows_themselves(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(60, 4))
        x[30:] += 3.0
        rows = table(60, batch=["A"] * 30 + ["B"] * 30)
        fields = dict(BATCH, method="harmony", batch="batch", max_iter=10, theta=2.0)
        every = np.arange(60)
        out = run([("batch_correct", fields)], x, {"fit": every, "eval": every.copy()}, rows=rows, fit_role="fit")
        self.assertEqual(out.x["eval"].shape, (60, 4))
        with self.assertRaisesRegex(ValueError, "corrects the fit rows themselves"):
            run([("batch_correct", fields)], x, {"train": np.arange(50), "val": np.arange(50, 60)}, rows=rows)

    def test_batch_correct_refuses_settings_of_another_method(self):
        for fields in (dict(BATCH, method="center", batch="b", tolerance=1e-4),
                       dict(BATCH, method="combat", batch="b", tolerance=1e-4),
                       dict(BATCH, method="harmony", batch="b", max_iter=10)):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "prep batch_correct"):
                    refused([{"name": "batch_correct", **fields}])


class TestEveryStep(PrepTest):
    """Every step of the template, in one loop: declared as the runners expect, and learning from the fit rows
    only."""

    def cases(self):
        rng = np.random.default_rng(7)
        positive = rng.uniform(1, 5, size=(24, 4))
        rows = table(24, batch=["A", "B"] * 12)
        roles = {"train": np.arange(16), "val": np.arange(16, 24)}
        coded = np.column_stack([positive[:, :3], np.tile([0.0, 1.0, 2.0], 8)])
        double = lambda v: v * 2 + 1       # noqa: E731
        return {
            "impute": ([("impute", dict(IMPUTE, method="median"))], positive, {}, double),
            "scale": ([("scale", dict(SCALE, method="robust"))], positive, {}, double),
            "drop_duplicates": ([("drop_duplicates", {"decimals": 2})], np.vstack([positive[:8]] * 3), {}, double),
            "drop_constant": ([("drop_constant", {"min_variance": 0.5, "max_frequency_ratio": None,
                                                  "min_unique_fraction": None})], positive, {}, double),
            "outliers": ([("outliers", dict(OUTLIERS, method="clip_quantiles", low_quantile=0.2,
                                            high_quantile=0.8))], positive, {}, double),
            "subsample": ([("subsample", {"method": "random", "fraction": 0.5, "n_rows": None, "stratify": None,
                                          "salt": "s"})], positive, {}, double),
            "onehot": ([("onehot", {"drop_first": True})], coded, {"categorical": {3: ("a", "b", "c")}},
                       lambda v: np.column_stack([v[:, :3] * 2 + 1, v[:, 3]])),
            "select_columns": ([("select_columns", {"keep": ["f0", "f2"]})], positive, {}, double),
            "select_features": ([("select_features", dict(FEATURES, method="variance_top", n=2))], positive, {},
                                double),
            "transform": ([("transform", dict(TRANSFORM, method="yeojohnson"))], positive, {}, double),
            "reduce": ([("reduce", {"method": PCA_SETTINGS})], positive, {}, double),
            "batch_correct": ([("batch_correct", dict(BATCH, method="combat", batch="batch", tolerance=1e-4,
                                                      max_iter=50))], positive, {"rows": rows}, double),
        }, roles

    def test_every_template_step_learns_from_the_fit_rows_only(self):
        from sweepkit.registry import PREP

        cases, roles = self.cases()
        self.assertEqual(sorted(cases), sorted(TEMPLATE_STEPS))
        for name in TEMPLATE_STEPS:
            steps, x, kwargs, perturb = cases[name]
            if PREP.get(name).meta.get("transductive", False):
                continue                     # a transductive step may learn from other rows, and reports say so
            with self.subTest(step=name):
                first, _ = self.assert_fit_rows_only(steps, x, roles, perturb, **kwargs)
                record = first.records[0]
                self.assertEqual((record["step"], record["name"], record["rows_in"]), (0, name, 16))
                self.assertEqual(record["columns_out"], len(first.names))

    def test_declarations_match_what_the_steps_do(self):
        from sweepkit.registry import PREP

        for name in TEMPLATE_STEPS:
            meta = PREP.get(name).meta
            with self.subTest(step=name):
                self.assertEqual(meta.get("removes_rows", False), name in REMOVES_ROWS)
                self.assertIn(meta.get("transductive", False), (False, True))
                self.assertEqual(meta["kinds"], ("tabular",))

    def test_a_step_that_removes_rows_must_say_so(self):
        from sweepkit.prep.base import PrepContext, PrepStep, run_prep
        from sweepkit.registry import PREP, Entry

        class Remover(PrepStep):
            def fit(self, x, positions):
                self.flags = np.arange(x.shape[0]) == 0

            def removed(self):
                return self.flags

            def transform(self, x, positions):
                return x

        undeclared = Entry("prep step", "remover", object, Remover, {"kinds": ("tabular",)})
        context = PrepContext(table=table(4), feature_names=("f0",), categorical={}, seed=1, device="cpu")
        with mock.patch.object(PREP, "get", lambda name: undeclared):
            with self.assertRaisesRegex(RuntimeError, "not declared removes_rows"):
                run_prep((("remover", None),), {"train": np.ones((3, 1)), "val": np.ones((1, 1))},
                         {"train": np.arange(3), "val": np.array([3])}, "train", context, lambda m: None)

    def test_the_runners_prepare_gives_float32_and_the_same_values(self):
        from sweepkit.data.base import Data
        from sweepkit.families.common import prepare

        x = np.array([[1.0, 5.0], [2.0, 6.0], [4.0, 9.0], [8.0, 1.0]])
        data = Data("tabular", table(4), x=x, feature_names=("a", "b"))
        steps = (("scale", settings("scale", **dict(SCALE, method="zscore"))),)
        job = types.SimpleNamespace(prep=steps)
        out, positions, names, categorical, records = prepare(job, data, {"train": np.arange(3), "val": np.array([3])},
                                                              "train", 1, "cpu", lambda m: None)
        direct = run([("scale", dict(SCALE, method="zscore"))], x, {"train": [0, 1, 2], "val": [3]}, names=("a", "b"))
        self.assertEqual((out["val"].dtype, names), (np.float32, ("a", "b")))
        np.testing.assert_allclose(out["val"], direct.x["val"], rtol=1e-6)
        self.assertEqual(records[0]["method"], "zscore")


if __name__ == "__main__":
    unittest.main()
