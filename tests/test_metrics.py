"""Supervised metrics against hand-computed answers, and every batched form (the paired bootstrap's fast path)
against its metric under random row weights."""
import unittest

from tests.support import load_catalog


def outputs(y_true, y_pred, scores=None, kind="none", names=("a", "b", "c")):
    import numpy as np

    from sweepkit.tasks.supervised import Outputs

    return Outputs(np.asarray(y_true), np.asarray(y_pred), None if scores is None else np.asarray(scores), kind, names)


class TestClassificationByHand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_counts(self):
        from sweepkit.registry import METRICS

        # truth a a a b b c, predictions a a b b c c
        o = outputs([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2])
        get = lambda name, w=None: METRICS.get(name)["fn"](o, w)
        self.assertAlmostEqual(get("accuracy"), 4 / 6)
        self.assertAlmostEqual(get("balanced_accuracy"), (2 / 3 + 1 / 2 + 1) / 3)
        # precision a 1, b 1/2, c 1/2; recall a 2/3, b 1/2, c 1; F1 a 0.8, b 0.5, c 2/3
        self.assertAlmostEqual(get("precision_macro"), (1 + 0.5 + 0.5) / 3)
        self.assertAlmostEqual(get("f1_macro"), (0.8 + 0.5 + 2 / 3) / 3)
        self.assertAlmostEqual(get("f1_weighted"), (0.8 * 3 + 0.5 * 2 + 2 / 3) / 6)
        # weights count rows: doubling the first row equals adding it once more
        o2 = outputs([0, 0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(get("accuracy", [2, 1, 1, 1, 1, 1]), METRICS.get("accuracy")["fn"](o2, None))

    def test_undefined_recall_raises(self):
        from sweepkit.registry import METRICS

        with self.assertRaises(ValueError):
            METRICS.get("balanced_accuracy")["fn"](outputs([0, 0, 1], [0, 1, 1]), None)

    def test_regression_by_hand(self):
        from sweepkit.registry import METRICS

        o = outputs([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 5.0], names=())
        get = lambda name: METRICS.get(name)["fn"](o, None)
        self.assertAlmostEqual(get("mse"), 0.5)
        self.assertAlmostEqual(get("mae"), 0.5)
        self.assertAlmostEqual(get("r2"), 1 - 2.0 / 5.0)
        self.assertAlmostEqual(get("max_error"), 1.0)


class TestBatchedForms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def fold(self, task: str):
        import numpy as np

        rng = np.random.default_rng(7)
        n, seeds = 60, 3
        if task == "classification":
            y = np.arange(n) % 3
            pred = np.where(rng.random((seeds, n)) < 0.6, y[None, :], rng.integers(0, 3, (seeds, n)))
            scores = rng.dirichlet(np.ones(3), size=(seeds, n))
            return {"y_true": y, "y_pred": pred, "scores": scores, "scores_kind": "probability",
                    "names": ("a", "b", "c")}
        y = rng.normal(size=n)
        return {"y_true": y, "y_pred": y[None, :] + rng.normal(scale=0.5, size=(seeds, n)), "scores": None,
                "scores_kind": "none", "names": ()}

    def test_every_batched_form_matches_its_metric(self):
        import numpy as np

        from sweepkit.registry import METRICS
        from sweepkit.tasks.supervised import Outputs

        rng = np.random.default_rng(11)
        checked = []
        for name in METRICS.names():
            entry = METRICS.get(name)
            if entry["batched"] is None:
                continue
            task = entry["tasks"][0]
            fold = self.fold(task)
            n = fold["y_true"].shape[0]
            weights = rng.multinomial(n, np.full(n, 1 / n), size=4).astype(float)
            fast = entry["batched"](fold, weights)
            self.assertEqual(fast.shape, (4, fold["y_pred"].shape[0]), name)
            for b in range(4):
                for s in range(fold["y_pred"].shape[0]):
                    o = Outputs(fold["y_true"], fold["y_pred"][s],
                                None if fold["scores"] is None else fold["scores"][s], fold["scores_kind"],
                                fold["names"])
                    self.assertAlmostEqual(float(fast[b, s]), entry["fn"](o, weights[b]), places=10, msg=name)
            checked.append(name)
        for name in ("accuracy", "balanced_accuracy", "f1_macro", "mcc", "cohen_kappa", "log_loss", "brier", "mse",
                     "rmse", "mae", "r2", "pearson", "ccc"):
            self.assertIn(name, checked)


if __name__ == "__main__":
    unittest.main()
