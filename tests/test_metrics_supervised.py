"""The supervised metrics tests/test_metrics.py leaves out: answers worked out by hand, agreement with scikit-learn
and scipy, row weights, decision values against probabilities, results that have no value, and every batched form.

Row weights are a bootstrap draw's counts, so for every weighted metric a weight of 2 must give the same number as
the row written twice, and equal weights the same number as no weights.
"""
import unittest

import numpy as np

from tests.support import load_catalog

CLASSIFICATION = ("f1_micro", "recall_macro", "roc_auc", "average_precision")
REGRESSION = ("median_ae", "spearman", "max_error")
MULTILABEL = ("hamming_loss", "subset_accuracy", "sample_f1", "label_f1_macro", "label_f1_micro", "label_roc_auc",
              "label_average_precision")


def outputs(y_true, y_pred, scores=None, kind="none", names=("a", "b", "c")):
    from sweepkit.tasks.supervised import Outputs

    return Outputs(np.asarray(y_true), np.asarray(y_pred), None if scores is None else np.asarray(scores, dtype=float),
                   kind, tuple(names))


def metric(name: str, o, w=None) -> float:
    from sweepkit.registry import METRICS

    return METRICS.get(name)["fn"](o, w)


def repeated(o, counts):
    """The outputs with row i written counts[i] times."""
    from sweepkit.tasks.supervised import Outputs

    index = np.repeat(np.arange(len(counts)), np.asarray(counts, dtype=int))
    return Outputs(o.y_true[index], o.y_pred[index], None if o.scores is None else o.scores[index], o.scores_kind,
                   o.names)


def random_classification(rng, n: int, k: int):
    y = np.arange(n) % k
    pred = np.where(rng.random(n) < 0.6, y, rng.integers(0, k, n))
    scores = rng.dirichlet(np.ones(k), size=n)
    scores[np.arange(n), y] += 0.3 * rng.random(n)
    scores /= scores.sum(axis=1, keepdims=True)
    return outputs(y, pred, scores, "probability", tuple("abcdefg"[:k]))


def random_regression(rng, n: int):
    y = np.round(rng.normal(size=n), 1)                     # rounded, so some values tie
    return outputs(y, np.round(y + rng.normal(scale=0.7, size=n), 1), names=())


def random_multilabel(rng, n: int, labels: int):
    logits = rng.normal(size=(n, labels))
    y = (logits + rng.normal(scale=0.8, size=(n, labels)) > 0).astype(int)
    y[0], y[1] = 1, 0                                       # every label both on and off somewhere
    return outputs(y, (logits > 0).astype(int), 1 / (1 + np.exp(-logits)), "probability",
                   tuple(f"l{j}" for j in range(labels)))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()


class TestClassification(Base):
    def test_f1_micro_and_recall_macro_by_hand(self):
        from sklearn.metrics import f1_score, recall_score

        # truth a a a b b c, predictions a a b b c c: recall a 2/3, b 1/2, c 1
        y, p = [0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]
        o = outputs(y, p)
        self.assertAlmostEqual(metric("f1_micro", o), 4 / 6)            # single label: equals accuracy
        self.assertAlmostEqual(metric("f1_micro", o), metric("accuracy", o))
        self.assertAlmostEqual(metric("recall_macro", o), (2 / 3 + 1 / 2 + 1) / 3)
        self.assertAlmostEqual(metric("recall_macro", o), metric("balanced_accuracy", o))
        self.assertAlmostEqual(metric("f1_micro", o), f1_score(y, p, average="micro"), places=12)
        self.assertAlmostEqual(metric("recall_macro", o), recall_score(y, p, average="macro"), places=12)

    def test_constant_predictions_and_a_missing_class(self):
        o = outputs([0, 0, 1, 2], [0, 0, 0, 0])
        self.assertAlmostEqual(metric("f1_micro", o), 2 / 4)
        self.assertAlmostEqual(metric("recall_macro", o), (1 + 0 + 0) / 3)
        missing = outputs([0, 0, 1, 1], [0, 1, 1, 1])                   # class c has no row
        self.assertAlmostEqual(metric("f1_micro", missing), 3 / 4)
        with self.assertRaises(ValueError):                              # recall of c is undefined
            metric("recall_macro", missing)

    def test_roc_auc_and_average_precision_binary_by_hand(self):
        # positives at 0.35 and 0.8, negatives at 0.1 and 0.4: 3 of the 4 (positive, negative) pairs ordered right
        y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        o = outputs(y, [0, 1, 0, 1], np.stack([1 - np.array(s), s], axis=1), "probability", ("no", "yes"))
        self.assertAlmostEqual(metric("roc_auc", o), 0.75)
        # ranked 0.8 (yes), 0.4, 0.35 (yes), 0.1: precision 1 at the first positive, 2/3 at the second
        self.assertAlmostEqual(metric("average_precision", o), (1 + 2 / 3) / 2)

    def test_multiclass_one_against_the_rest(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.preprocessing import label_binarize

        o = random_classification(np.random.default_rng(1), 60, 3)
        self.assertAlmostEqual(metric("roc_auc", o), roc_auc_score(o.y_true, o.scores, multi_class="ovr",
                                                                   average="macro"), places=12)
        onehot = label_binarize(o.y_true, classes=[0, 1, 2])
        self.assertAlmostEqual(metric("average_precision", o),
                               average_precision_score(onehot, o.scores, average="macro"), places=12)

    def test_decision_values_rank_like_probabilities(self):
        rng = np.random.default_rng(2)
        y = np.arange(40) % 2
        d = y + rng.normal(size=40)
        p = 1 / (1 + np.exp(-d))
        probability = outputs(y, (d > 0).astype(int), np.stack([1 - p, p], axis=1), "probability", ("no", "yes"))
        # the estimator family's decision values for two classes: the negated and the plain decision function
        decision = outputs(y, (d > 0).astype(int), np.stack([-d, d], axis=1), "decision", ("no", "yes"))
        for name in ("roc_auc", "average_precision"):
            self.assertAlmostEqual(metric(name, probability), metric(name, decision), places=12, msg=name)
        with self.assertRaises(ValueError):                              # probabilities only
            metric("log_loss", decision)

    def test_undefined_without_both_sides(self):
        three = outputs([0, 0, 1, 1], [0, 0, 1, 1], np.full((4, 3), 1 / 3), "probability")
        for name in ("roc_auc", "average_precision"):                     # class c has no row
            with self.assertRaisesRegex(ValueError, "class c has no rows", msg=name):
                metric(name, three)
        s = [0.2, 0.7, 0.4]
        only_yes = outputs([1, 1, 1], [1, 1, 1], np.stack([1 - np.array(s), s], axis=1), "probability", ("no", "yes"))
        only_no = outputs([0, 0, 0], [0, 0, 0], np.stack([1 - np.array(s), s], axis=1), "probability", ("no", "yes"))
        with self.assertRaises(ValueError):
            metric("roc_auc", only_yes)
        with self.assertRaises(ValueError):
            metric("roc_auc", only_no)
        with self.assertRaisesRegex(ValueError, "class yes has no rows"):
            metric("average_precision", only_no)
        self.assertEqual(metric("average_precision", only_yes), 1.0)    # every row positive: precision 1 throughout
        both = outputs([0, 1, 1], [0, 1, 1], np.stack([1 - np.array(s), s], axis=1), "probability", ("no", "yes"))
        with self.assertRaises(ValueError):                              # a weight of 0 is no row
            metric("roc_auc", both, [0.0, 1.0, 1.0])
        for name in ("roc_auc", "average_precision"):
            with self.assertRaisesRegex(ValueError, "needs class scores", msg=name):
                metric(name, outputs([0, 1, 2], [0, 1, 2]))

    def test_row_weights_count_rows(self):
        rng = np.random.default_rng(3)
        o = random_classification(rng, 30, 3)
        counts = rng.integers(1, 4, size=30)
        counts[:3] = (2, 0, 3)
        for name in CLASSIFICATION:
            self.assertAlmostEqual(metric(name, o, np.ones(30)), metric(name, o), places=12, msg=name)
            self.assertAlmostEqual(metric(name, o, counts.astype(float)), metric(name, repeated(o, counts)),
                                   places=12, msg=name)


class TestRegression(Base):
    def test_median_absolute_error_by_hand(self):
        from sklearn.metrics import median_absolute_error

        even = outputs([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 5.0], names=())
        self.assertAlmostEqual(metric("median_ae", even), 0.5)           # errors 0, 0, 1, 1: the two middle ones
        self.assertAlmostEqual(metric("median_ae", even), median_absolute_error(even.y_true, even.y_pred))
        odd = outputs([0.0, 0.0, 0.0], [0.0, 1.0, -3.0], names=())
        self.assertAlmostEqual(metric("median_ae", odd), 1.0)            # errors 0, 1, 3
        # the first row twice: errors 0, 0, 0, 1, 1 have median 0
        self.assertAlmostEqual(metric("median_ae", even, [2.0, 1.0, 1.0, 1.0]), 0.0)
        self.assertAlmostEqual(metric("median_ae", even, [0.0, 1.0, 1.0, 1.0]), 1.0)

    def test_median_absolute_error_with_weights_against_scikit_learn(self):
        from sklearn.metrics import median_absolute_error

        rng = np.random.default_rng(4)
        for _ in range(20):
            o = random_regression(rng, int(rng.integers(2, 25)))
            w = rng.integers(0, 4, size=o.y_true.size).astype(float)
            w[0] = 1.0
            self.assertAlmostEqual(metric("median_ae", o, w),
                                   median_absolute_error(o.y_true, o.y_pred, sample_weight=w), places=12)
            self.assertAlmostEqual(metric("median_ae", o), float(np.median(np.abs(o.y_true - o.y_pred))), places=12)

    def test_spearman_by_hand_and_against_scipy(self):
        from scipy.stats import spearmanr

        o = outputs([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0], names=())
        # one swap: rank differences 0, 1, 1, 0, so 1 - 6 x 2 / (4 x 15)
        self.assertAlmostEqual(metric("spearman", o), 0.8)
        rng = np.random.default_rng(5)
        o = random_regression(rng, 40)
        self.assertAlmostEqual(metric("spearman", o), spearmanr(o.y_true, o.y_pred).statistic, places=12)
        # ranks only: any increasing change of the predictions leaves it alone
        stretched = outputs(o.y_true, np.exp(o.y_pred), names=())
        self.assertAlmostEqual(metric("spearman", stretched), metric("spearman", o), places=12)
        with self.assertRaisesRegex(ValueError, "does not vary"):
            metric("spearman", outputs(o.y_true, np.ones(40), names=()))

    def test_max_error(self):
        o = outputs([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.5, 7.0], names=())
        self.assertEqual(metric("max_error", o), 3.0)
        self.assertEqual(metric("max_error", o, [1.0, 1.0, 1.0, 0.0]), 0.5)   # a row of weight 0 is no row

    def test_row_weights_count_rows(self):
        rng = np.random.default_rng(6)
        for _ in range(10):
            o = random_regression(rng, 25)
            counts = rng.integers(0, 4, size=25)
            counts[:2] = (1, 2)
            for name in REGRESSION:
                self.assertAlmostEqual(metric(name, o, np.ones(25)), metric(name, o), places=12, msg=name)
                self.assertAlmostEqual(metric(name, o, counts.astype(float)), metric(name, repeated(o, counts)),
                                       places=12, msg=name)


class TestMultilabel(Base):
    # rows: labels on in truth and in prediction
    Y = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]]
    P = [[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 0]]

    def test_counts_by_hand(self):
        o = outputs(self.Y, self.P, names=("x", "y", "z"))
        self.assertAlmostEqual(metric("subset_accuracy", o), 2 / 4)      # rows 1 and 3 fully right
        self.assertAlmostEqual(metric("hamming_loss", o), 3 / 12)        # one wrong in row 0, two in row 2
        # x: 2 right, nothing wrong (F1 1); y: 1 right, 1 missed (2/3); z: 1 missed, 1 extra (0)
        self.assertAlmostEqual(metric("label_f1_macro", o), (1 + 2 / 3 + 0) / 3)
        # every decision: 3 right, 1 extra, 2 missed, so 2 x 3 / (2 x 3 + 1 + 2)
        self.assertAlmostEqual(metric("label_f1_micro", o), 6 / 9)
        # rows: 2 x 1 / (2 + 1), 1, 2 x 1 / (2 + 2), and 1 for a row with no label on either side
        self.assertAlmostEqual(metric("sample_f1", o), (2 / 3 + 1 + 1 / 2 + 1) / 4)

    def test_counts_against_scikit_learn(self):
        from sklearn.metrics import accuracy_score, f1_score, hamming_loss

        o = random_multilabel(np.random.default_rng(7), 50, 4)
        self.assertAlmostEqual(metric("subset_accuracy", o), accuracy_score(o.y_true, o.y_pred), places=12)
        self.assertAlmostEqual(metric("hamming_loss", o), hamming_loss(o.y_true, o.y_pred), places=12)
        self.assertAlmostEqual(metric("label_f1_micro", o), f1_score(o.y_true, o.y_pred, average="micro"), places=12)
        self.assertAlmostEqual(metric("label_f1_macro", o), f1_score(o.y_true, o.y_pred, average="macro",
                                                                     zero_division=0), places=12)
        self.assertAlmostEqual(metric("sample_f1", o), f1_score(o.y_true, o.y_pred, average="samples",
                                                                zero_division=1), places=12)

    def test_constant_predictions(self):
        o = outputs(self.Y, np.zeros((4, 3), dtype=int), names=("x", "y", "z"))
        self.assertAlmostEqual(metric("subset_accuracy", o), 1 / 4)      # only the row with no label is right
        self.assertAlmostEqual(metric("hamming_loss", o), 5 / 12)        # every true label missed
        self.assertEqual(metric("label_f1_macro", o), 0.0)
        self.assertEqual(metric("label_f1_micro", o), 0.0)
        self.assertAlmostEqual(metric("sample_f1", o), 1 / 4)            # 1 for the empty row, 0 for the others

    def test_ranking_metrics_by_hand_and_against_scikit_learn(self):
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = [[1, 0], [0, 1], [1, 1], [0, 0]]
        s = [[0.9, 0.2], [0.3, 0.6], [0.4, 0.7], [0.5, 0.1]]
        o = outputs(y, (np.array(s) > 0.5).astype(int), s, "probability", ("x", "y"))
        # x: positives 0.9 and 0.4 against negatives 0.3 and 0.5: 3 of 4 pairs; y: positives above both negatives
        self.assertAlmostEqual(metric("label_roc_auc", o), (3 / 4 + 1) / 2)
        # x ranked 0.9 (on), 0.5, 0.4 (on), 0.3: (1 + 2/3) / 2; y: both positives first, 1
        self.assertAlmostEqual(metric("label_average_precision", o), ((1 + 2 / 3) / 2 + 1) / 2)
        o = random_multilabel(np.random.default_rng(8), 60, 3)
        self.assertAlmostEqual(metric("label_roc_auc", o), roc_auc_score(o.y_true, o.scores, average="macro"),
                               places=12)
        self.assertAlmostEqual(metric("label_average_precision", o),
                               average_precision_score(o.y_true, o.scores, average="macro"), places=12)

    def test_decision_values_rank_like_probabilities(self):
        rng = np.random.default_rng(9)
        logits = rng.normal(size=(40, 3))
        y = (logits + rng.normal(size=(40, 3)) > 0).astype(int)
        y[0], y[1] = 1, 0
        probability = outputs(y, (logits > 0).astype(int), 1 / (1 + np.exp(-logits)), "probability", "xyz")
        decision = outputs(y, (logits > 0).astype(int), logits, "decision", "xyz")
        for name in ("label_roc_auc", "label_average_precision"):
            self.assertAlmostEqual(metric(name, probability), metric(name, decision), places=12, msg=name)

    def test_undefined_without_both_sides(self):
        s = [[0.9, 0.2], [0.3, 0.6], [0.4, 0.7]]
        always = outputs([[1, 0], [1, 1], [1, 0]], [[1, 0], [1, 1], [1, 0]], s, "probability", ("x", "y"))
        with self.assertRaisesRegex(ValueError, "label x is always or never on"):
            metric("label_roc_auc", always)
        self.assertGreater(metric("label_average_precision", always), 0)   # always on: precision is still defined
        never = outputs([[0, 1], [0, 1], [0, 0]], [[0, 1], [0, 1], [0, 0]], s, "probability", ("x", "y"))
        with self.assertRaisesRegex(ValueError, "label x is always or never on"):
            metric("label_roc_auc", never)
        with self.assertRaisesRegex(ValueError, "label x is never on"):
            metric("label_average_precision", never)
        mixed = outputs([[1, 1], [0, 0], [1, 0]], [[1, 1], [0, 0], [1, 0]], s, "probability", ("x", "y"))
        with self.assertRaisesRegex(ValueError, "label y"):           # its only positive row weighs 0
            metric("label_average_precision", mixed, [0.0, 1.0, 1.0])
        for name in ("label_roc_auc", "label_average_precision"):
            with self.assertRaisesRegex(ValueError, "needs label scores", msg=name):
                metric(name, outputs(self.Y, self.P, names=("x", "y", "z")))

    def test_row_weights_count_rows(self):
        rng = np.random.default_rng(10)
        o = random_multilabel(rng, 30, 3)
        counts = rng.integers(0, 4, size=30)
        counts[:2] = (2, 1)
        for name in MULTILABEL:
            self.assertAlmostEqual(metric(name, o, np.ones(30)), metric(name, o), places=12, msg=name)
            self.assertAlmostEqual(metric(name, o, counts.astype(float)), metric(name, repeated(o, counts)),
                                   places=12, msg=name)


class TestBatchedForms(Base):
    """Every metric of this file that has a batched form gives, for each draw and seed, what the metric gives."""

    def fold(self, task: str, rng) -> dict:
        n, seeds = 45, 3
        if task == "classification":
            made = [random_classification(rng, n, 3) for _ in range(seeds)]
        elif task == "regression":
            first = random_regression(rng, n)
            made = [outputs(first.y_true, first.y_true + rng.normal(scale=0.5, size=n), names=())
                    for _ in range(seeds)]
        else:
            made = [random_multilabel(rng, n, 3) for _ in range(seeds)]
        y = made[0].y_true
        return {"y_true": y, "y_pred": np.stack([m.y_pred for m in made]),
                "scores": None if made[0].scores is None else np.stack([m.scores for m in made]),
                "scores_kind": made[0].scores_kind, "names": made[0].names}

    def test_batched_forms_match_their_metrics(self):
        from sweepkit.registry import METRICS
        from sweepkit.tasks.supervised import Outputs

        rng = np.random.default_rng(11)
        checked = []
        for name in CLASSIFICATION + REGRESSION + MULTILABEL:
            entry = METRICS.get(name)
            if entry["batched"] is None:
                continue
            fold = self.fold(entry["tasks"][0], rng)
            n = fold["y_true"].shape[0]
            weights = rng.multinomial(n, np.full(n, 1 / n), size=5).astype(float)
            fast = entry["batched"](fold, weights)
            self.assertEqual(fast.shape, (5, 3), name)
            for b in range(5):
                for s in range(3):
                    o = Outputs(fold["y_true"], fold["y_pred"][s],
                                None if fold["scores"] is None else fold["scores"][s], fold["scores_kind"],
                                fold["names"])
                    self.assertAlmostEqual(float(fast[b, s]), entry["fn"](o, weights[b]), places=10, msg=name)
            checked.append(name)
        for name in ("f1_micro", "recall_macro"):
            self.assertIn(name, checked)

    def test_a_draw_without_a_class_is_undefined_both_ways(self):
        from sweepkit.registry import METRICS
        from sweepkit.tasks.supervised import Outputs

        fold = {"y_true": np.array([0, 0, 1, 1, 2, 2]), "y_pred": np.array([[0, 1, 1, 1, 2, 0]]), "scores": None,
                "scores_kind": "none", "names": ("a", "b", "c")}
        weights = np.array([[1.0, 2.0, 0.0, 0.0, 1.0, 2.0]])            # no weight on class b
        entry = METRICS.get("recall_macro")
        with self.assertRaises(ValueError):
            entry["batched"](fold, weights)
        with self.assertRaises(ValueError):
            entry["fn"](Outputs(fold["y_true"], fold["y_pred"][0], None, "none", fold["names"]), weights[0])
        # f1_micro stays defined: rows 0 and 4 are right with weight 1 each, of a total weight of 6
        fast = METRICS.get("f1_micro")["batched"](fold, weights)
        self.assertAlmostEqual(float(fast[0, 0]), 2 / 6)


if __name__ == "__main__":
    unittest.main()
