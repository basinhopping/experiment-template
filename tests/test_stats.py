"""Statistics against known answers, including a pairing check that a wrong bootstrap would fail."""
import math
import unittest

from tests.support import load_catalog


class TestTInterval(unittest.TestCase):
    def test_five_seeds(self):
        from sweepkit.eval.stats import t_interval

        r = t_interval([1, 2, 3, 4, 5], 0.95)
        t975_df4 = 2.7764451051977987
        self.assertAlmostEqual(r["t_critical"], t975_df4, places=9)
        self.assertAlmostEqual(r["sd"], math.sqrt(2.5), places=12)
        self.assertAlmostEqual(r["half_width"], t975_df4 * math.sqrt(2.5) / math.sqrt(5), places=9)
        self.assertAlmostEqual(r["half_width"] / r["sd"], 1.2416, places=3)    # 1.24 x SD for 5 seeds

    def test_needs_two_values(self):
        from sweepkit.eval.stats import t_interval

        with self.assertRaises(ValueError):
            t_interval([1.0], 0.95)


class TestHolmPooAndRounding(unittest.TestCase):
    def test_holm_known_answer(self):
        from sweepkit.eval.stats import holm

        # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max
        for got, want in zip(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])
        with self.assertRaises(ValueError):
            holm([1.2])

    def test_probability_of_outperforming_both_directions(self):
        from sweepkit.eval.stats import probability_of_outperforming

        self.assertAlmostEqual(probability_of_outperforming([1, 2, 3], [2, 2, 2], "max"), 0.5)
        self.assertAlmostEqual(probability_of_outperforming([3, 4], [1, 2], "max"), 1.0)
        self.assertAlmostEqual(probability_of_outperforming([3, 4], [1, 2], "min"), 0.0)
        self.assertAlmostEqual(probability_of_outperforming([1], [1], "max"), 0.5)

    def test_probability_of_outperforming_by_fold(self):
        import numpy as np

        from sweepkit.eval.stats import probability_of_outperforming_by_fold

        a = np.array([[0.5], [0.6], [0.4]])                  # one seed per fold
        b = np.array([[0.4], [0.7], [0.3]])
        r = probability_of_outperforming_by_fold(a, b, 1000, 0.95, np.random.default_rng(0), 0.75, "max")
        self.assertAlmostEqual(r["p_a_beats_b"], 2 / 3)       # better in two of three folds
        self.assertTrue(0.0 <= r["low"] <= 2 / 3 <= r["high"] <= 1.0)
        flipped = probability_of_outperforming_by_fold(a, b, 1000, 0.95, np.random.default_rng(0), 0.75, "min")
        self.assertAlmostEqual(flipped["p_a_beats_b"], 1 / 3)
        # two seeds against one in each fold: all pairs within the fold, ties counted half
        two = probability_of_outperforming_by_fold(np.array([[0.5, 0.3], [0.2, 0.2]]), np.array([[0.4], [0.2]]),
                                                   500, 0.95, np.random.default_rng(1), 0.75, "max")
        self.assertAlmostEqual(two["p_a_beats_b"], (0.5 + 0.5) / 2)
        with self.assertRaises(ValueError):
            probability_of_outperforming_by_fold(a, b[:2], 100, 0.95, np.random.default_rng(0), 0.75, "max")

    def test_verdict_follows_the_direction(self):
        from sweepkit.eval.stats import verdict

        self.assertEqual(verdict(0.1, 0.01, 0.05, "max"), "better")
        self.assertEqual(verdict(0.1, 0.01, 0.05, "min"), "worse")
        self.assertEqual(verdict(-0.1, 0.01, 0.05, "min"), "better")
        self.assertEqual(verdict(0.1, 0.05, 0.05, "max"), "no detectable difference")

    def test_round_half_up_from_the_exact_value(self):
        from sweepkit.eval.stats import round_half_up

        self.assertEqual(round_half_up(0.6005, 3), "0.601")
        self.assertEqual(round_half_up(0.12345, 4), "0.1235")
        self.assertEqual(round_half_up(2.5, 0), "3")


class TestSeedAndUnitBootstraps(unittest.TestCase):
    def test_a_bootstrap_p_is_never_zero(self):
        import numpy as np

        from sweepkit.eval.stats import holm, seed_bootstrap

        a = {1: 0.90, 2: 0.91, 3: 0.92}
        b = {1: 0.10, 2: 0.11, 3: 0.12}                      # every draw puts A ahead
        r = seed_bootstrap(a, b, 2000, 0.95, np.random.default_rng(5), "max")
        self.assertAlmostEqual(r["p_two_sided"], 2 / 2001)
        self.assertAlmostEqual(r["p_floor"], 2 / 2001)
        self.assertGreater(holm([r["p_two_sided"]])[0], 0.0)

    def test_seed_bootstrap_observed_difference(self):
        import numpy as np

        from sweepkit.eval.stats import seed_bootstrap

        r = seed_bootstrap({1: 0.5, 2: 0.6, 3: 0.7}, {1: 0.4, 2: 0.4, 3: 0.4}, 2000, 0.95,
                           np.random.default_rng(0), "max")
        self.assertAlmostEqual(r["difference"], 0.2, places=12)
        self.assertGreaterEqual(r["low"], 0.1 - 1e-12)
        self.assertLessEqual(r["high"], 0.3 + 1e-12)

    def test_units_are_paired(self):
        import numpy as np

        from sweepkit.eval.stats import paired_units_bootstrap

        # B is A minus 0.05 in every fold: paired, the difference never varies
        a = {"f1": 0.9, "f2": 0.5, "f3": 0.7, "f4": 0.3}
        b = {k: v - 0.05 for k, v in a.items()}
        r = paired_units_bootstrap(a, b, 1000, 0.95, np.random.default_rng(1), "max", "fold")
        self.assertAlmostEqual(r["low"], 0.05, places=12)
        self.assertAlmostEqual(r["high"], 0.05, places=12)
        self.assertAlmostEqual(r["p_bootstrap"], 2 / 1001)       # every draw on one side: the floor, never 0
        # four folds that agree: the sign-flip test cannot go below 2 / 2^4
        self.assertAlmostEqual(r["p_two_sided"], 0.125)
        with self.assertRaises(ValueError):
            paired_units_bootstrap({"f1": 1.0}, {"f1": 0.0}, 10, 0.95, np.random.default_rng(1), "max", "fold")


class TestFoldsAndSeedsBootstrap(unittest.TestCase):
    def test_observed_difference_and_fold_spread(self):
        import numpy as np

        from sweepkit.eval.stats import folds_and_seeds_bootstrap

        a = np.array([[0.50, 0.52, 0.51], [0.60, 0.61, 0.62], [0.40, 0.41, 0.39]])     # [folds, seeds]
        b = a - np.array([[0.02], [0.03], [0.01]])
        r = folds_and_seeds_bootstrap(a, b, 2000, 0.95, np.random.default_rng(1), "max")
        self.assertAlmostEqual(r["difference"], 0.02, places=12)
        # folds give differences 0.01 to 0.03; independent seed draws for A and B widen that a little
        self.assertTrue(0.0 < r["low"] < 0.02 < r["high"] < 0.04)
        self.assertEqual(r["resampled"], "folds (paired) and seeds")

    def test_a_deterministic_method_still_gets_a_spread_from_the_folds(self):
        import numpy as np

        from sweepkit.eval.stats import folds_and_seeds_bootstrap

        a = np.array([[0.30] * 3, [0.36] * 3, [0.33] * 3])          # the same value for every seed
        b = np.array([[0.31] * 3, [0.32] * 3, [0.30] * 3])
        r = folds_and_seeds_bootstrap(a, b, 2000, 0.95, np.random.default_rng(2), "max")
        self.assertLess(r["low"], r["high"])
        self.assertAlmostEqual(r["difference"], 0.02, places=12)

    def test_no_spread_is_refused_and_identical_values_are_no_difference(self):
        import numpy as np

        from sweepkit.eval.stats import NoVariation, folds_and_seeds_bootstrap, verdict

        a = np.full((1, 3), 0.30)
        with self.assertRaises(NoVariation):
            folds_and_seeds_bootstrap(a, a - 0.01, 500, 0.95, np.random.default_rng(3), "max")
        same = folds_and_seeds_bootstrap(a, a.copy(), 500, 0.95, np.random.default_rng(3), "max")
        self.assertEqual((same["difference"], same["p_two_sided"]), (0.0, 1.0))
        self.assertEqual(verdict(same["difference"], same["p_two_sided"], 0.05, "max"), "no detectable difference")

    def test_few_folds_that_agree_cannot_give_a_small_p(self):
        import numpy as np

        from sweepkit.eval.stats import folds_and_seeds_bootstrap, sign_flip_p

        self.assertAlmostEqual(sign_flip_p(np.array([0.01, 0.02, 0.03]), np.random.default_rng(0)), 0.25)
        self.assertAlmostEqual(sign_flip_p(np.array([0.01, -0.01]), np.random.default_rng(0)), 1.0)
        self.assertAlmostEqual(sign_flip_p(np.full(8, 0.01), np.random.default_rng(0)), 2 / 256)
        a = np.array([[0.41] * 3, [0.40] * 3, [0.42] * 3])                 # deterministic, 3 folds, all ahead
        r = folds_and_seeds_bootstrap(a, a - 0.003, 2000, 0.95, np.random.default_rng(4), "max")
        self.assertAlmostEqual(r["p_bootstrap"], 2 / 2001)       # (k + 1) / (n + 1) in each tail, doubled
        self.assertAlmostEqual(r["p_two_sided"], 0.25)
        ten = np.repeat(a, 4, axis=0)[:10]
        r = folds_and_seeds_bootstrap(ten, ten - 0.003, 2000, 0.95, np.random.default_rng(4), "max")
        self.assertAlmostEqual(r["p_two_sided"], 2 / 1024)


class TestPairedBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def make(self):
        import numpy as np

        rng = np.random.default_rng(0)
        folds_a, folds_b = [], []
        for fold in range(2):
            y = np.repeat(np.arange(4), 100)
            right = rng.random(400) < 0.6                     # B right on about 60%
            wrong_class0 = np.flatnonzero((y == 0) & ~right)[:20]
            better = right.copy()
            better[wrong_class0] = True                       # A also right on 20 more class-0 rows
            ids = [f"f{fold}_{i}" for i in range(400)]

            def preds(correct):
                return np.tile(np.where(correct, y, (y + 1) % 4), (3, 1))

            common = {"row_ids": ids, "y_true": y, "scores": None, "scores_kind": "none",
                      "names": ("a", "b", "c", "d")}
            folds_b.append(dict(common, y_pred=preds(right)))
            folds_a.append(dict(common, y_pred=preds(better)))
        return folds_a, folds_b

    def test_observed_difference_by_hand(self):
        import numpy as np

        from sweepkit.eval.stats import paired_bootstrap
        from sweepkit.registry import METRICS

        a, b = self.make()
        r = paired_bootstrap(a, b, METRICS.get("balanced_accuracy"), 2000, 0.95, np.random.default_rng(1), 500)
        # 20 more right of 100 class-0 rows: recall +0.2, balanced accuracy +0.05 in each fold
        self.assertAlmostEqual(r["difference"], 0.05, places=12)
        self.assertTrue(r["low"] < 0.05 < r["high"])
        self.assertGreater(r["low"], 0.0)
        self.assertTrue(r["batched"])

    def test_pairing_narrows_the_interval(self):
        import numpy as np

        from sweepkit.eval.stats import paired_bootstrap
        from sweepkit.registry import METRICS

        metric = METRICS.get("balanced_accuracy")
        a, b = self.make()
        paired = paired_bootstrap(a, b, metric, 2000, 0.95, np.random.default_rng(2), 500)
        # An unpaired bootstrap: separate row resamples for each model.
        rng = np.random.default_rng(3)
        diffs = []
        for _ in range(2000):
            value = [0.0, 0.0]
            for fa, fb in zip(a, b):
                for k, f in enumerate((fa, fb)):
                    w = rng.multinomial(400, np.full(400, 1 / 400), size=1).astype(float)
                    value[k] += metric["batched"](f, w)[0].mean() / 2
            diffs.append(value[0] - value[1])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        self.assertLess(paired["high"] - paired["low"], 0.6 * (hi - lo))

    def test_slow_path_agrees_with_the_batched_one(self):
        import numpy as np

        from sweepkit.eval.stats import paired_bootstrap
        from sweepkit.registry import METRICS

        a, b = self.make()
        entry = dict(METRICS.get("f1_macro"))
        fast = paired_bootstrap(a, b, entry, 200, 0.95, np.random.default_rng(4), 50)
        slow = paired_bootstrap(a, b, dict(entry, batched=None), 200, 0.95, np.random.default_rng(4), 50)
        for key in ("difference", "low", "high", "p_two_sided"):
            self.assertAlmostEqual(fast[key], slow[key], places=10)
        self.assertFalse(slow["batched"])

    def test_unmatched_rows_refused(self):
        import numpy as np

        from sweepkit.eval.stats import paired_bootstrap
        from sweepkit.registry import METRICS

        a, b = self.make()
        b[1] = dict(b[1], row_ids=list(reversed(b[1]["row_ids"])))
        with self.assertRaisesRegex(ValueError, "cannot be paired"):
            paired_bootstrap(a, b, METRICS.get("balanced_accuracy"), 10, 0.95, np.random.default_rng(0), 10)

    def test_a_class_left_out_of_a_draw_stops_the_comparison(self):
        import numpy as np

        from sweepkit.eval.stats import UndefinedInDraw, paired_bootstrap
        from sweepkit.registry import METRICS

        y = np.array([0] * 30 + [1])                         # class 1 has one row: some draws leave it out
        fold = {"row_ids": [f"r{i}" for i in range(31)], "y_true": y, "y_pred": np.tile(y, (2, 1)), "scores": None,
                "scores_kind": "none", "names": ("a", "b")}
        for entry in (METRICS.get("balanced_accuracy"), dict(METRICS.get("balanced_accuracy"), batched=None)):
            with self.assertRaisesRegex(UndefinedInDraw, "undefined in a bootstrap draw"):
                paired_bootstrap([fold], [dict(fold)], entry, 200, 0.95, np.random.default_rng(0), 50)

    def test_metric_without_weights_refused(self):
        import numpy as np

        from sweepkit.eval.stats import paired_bootstrap
        from sweepkit.registry import METRICS

        a, b = self.make()
        with self.assertRaisesRegex(ValueError, "row weights"):
            paired_bootstrap(a, b, METRICS.get("silhouette"), 10, 0.95, np.random.default_rng(0), 10)


if __name__ == "__main__":
    unittest.main()
