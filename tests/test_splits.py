"""Every split scheme on small synthetic tables: which rows play which role, the same folds on every call, and the
refusals of settings that cannot work.

Supervised schemes ("kfold", "group_kfold", "leave_one_group_out", "fixed", "holdout") give each fold the roles
train, val and test; unsupervised schemes ("all", "subsamples", "fit_holdout") give fit and eval. The identity check
fold (CHECK_FOLD) is the one fold whose validation rows are its test rows."""
import collections
import unittest

import numpy as np

from tests.support import load_catalog
from tests.sweep_fixtures import classify_base, cluster_base

SUPERVISED = ("kfold", "group_kfold", "leave_one_group_out", "fixed", "holdout")
UNSUPERVISED = ("all", "subsamples", "fit_holdout")


def synthetic_table(n_rows: int = 120, n_classes: int = 3, seed: int = 1):
    """The columns of the synthetic classification generator: truth (c0, c1, ...), group (g0 to g4) and
    published_split (train, val or test)."""
    from sweepkit.data.base import RowTable
    from sweepkit.data.synthetic import make

    made = make("classification", n_rows, 3, n_classes, 1.0, seed)
    return RowTable([f"r{i:04d}" for i in range(n_rows)], made["columns"])


def hand_table(**columns):
    from sweepkit.data.base import RowTable

    n = len(next(iter(columns.values())))
    return RowTable([f"r{i:04d}" for i in range(n)], {k: np.asarray(v) for k, v in columns.items()})


def make_folds(name: str, fields: dict, table) -> tuple:
    """(settings, scheme, folds) for a split section, validated the way a sweep validates it."""
    from sweepkit.registry import SPLITS

    entry, settings = SPLITS.validate_section({"name": name, **fields}, "split")
    return settings, entry.target, entry.target.make(settings, table)


def roles_of(folds) -> dict:
    return {f.name: {r: set(p.tolist()) for r, p in f.roles.items()} for f in folds}


def checked_job(settings: dict) -> None:
    from sweepkit.runs.spec import check_job, validate_job

    check_job(validate_job(settings))


class SchemeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def assert_supervised_fold(self, fold, n_rows: int, every_row: bool = True):
        """Roles train, val and test, pairwise disjoint, and (for these schemes) every row in one of them."""
        self.assertEqual(set(fold.roles), {"train", "val", "test"})
        train, val, test = (set(fold.roles[r].tolist()) for r in ("train", "val", "test"))
        self.assertFalse(train & val or train & test or val & test, fold.name)
        if every_row:
            self.assertEqual(train | val | test, set(range(n_rows)), fold.name)
        for positions in fold.roles.values():
            self.assertTrue(np.all(np.diff(positions) > 0), "roles hold sorted positions, each once")

    def assert_same_folds(self, first, second):
        self.assertEqual([f.name for f in first], [f.name for f in second])
        for a, b in zip(first, second):
            self.assertEqual(a.selection_on_test, b.selection_on_test)
            for role in a.roles:
                np.testing.assert_array_equal(a.roles[role], b.roles[role])


class TestRegistered(SchemeTest):
    def test_every_scheme_is_registered_with_its_roles(self):
        from sweepkit.registry import SPLITS
        from sweepkit.splits.base import SUPERVISED_ROLES, UNSUPERVISED_ROLES

        for name in SUPERVISED:
            self.assertEqual(SPLITS.get(name).meta["roles"], SUPERVISED_ROLES, name)
        for name in UNSUPERVISED:
            self.assertEqual(SPLITS.get(name).meta["roles"], UNSUPERVISED_ROLES, name)

    def test_declared_fold_names_match_the_folds_made(self):
        table = synthetic_table()
        cases = [("kfold", {"k": 4, "stratify": "truth", "salt": "s", "identity_fold": True}),
                 ("group_kfold", {"group": "group", "k": 3, "salt": "s", "identity_fold": True}),
                 ("leave_one_group_out", {"group": "group", "order": ["g0", "g1", "g2", "g3", "g4"],
                                          "identity_fold": False}),
                 ("fixed", {"column": "published_split", "identity_fold": True}),
                 ("holdout", {"val_fraction": 0.2, "test_fraction": 0.2, "stratify": None, "salt": "s",
                              "identity_fold": False}),
                 ("all", {}), ("subsamples", {"k": 3, "fraction": 0.5, "salt": "s"}),
                 ("fit_holdout", {"fit_fraction": 0.75, "salt": "s"})]
        for name, fields in cases:
            with self.subTest(split=name):
                settings, scheme, folds = make_folds(name, fields, table)
                self.assertEqual(tuple(f.name for f in folds), tuple(scheme.fold_names(settings)))

    def test_a_supervised_task_refuses_an_unsupervised_split_and_the_reverse(self):
        job = classify_base()
        job["split"] = {"name": "all"}
        with self.assertRaisesRegex(ValueError, "needs folds with roles"):
            checked_job(job)
        job = cluster_base()
        job["split"] = {"name": "kfold", "k": 3, "stratify": None, "salt": "s", "identity_fold": False}
        with self.assertRaisesRegex(ValueError, "needs folds with roles"):
            checked_job(job)


class TestKFold(SchemeTest):
    fields = {"k": 5, "stratify": "truth", "salt": "example", "identity_fold": False}

    def test_kfold_roles_partition_and_the_next_group_validates(self):
        table = synthetic_table()
        _, _, folds = make_folds("kfold", self.fields, table)
        self.assertEqual([f.name for f in folds], ["fold1", "fold2", "fold3", "fold4", "fold5"])
        for fold in folds:
            self.assert_supervised_fold(fold, len(table))
        tested = np.concatenate([f.roles["test"] for f in folds])
        self.assertEqual(sorted(tested.tolist()), list(range(len(table))), "every row is tested exactly once")
        for i, fold in enumerate(folds):
            following = folds[(i + 1) % len(folds)]
            np.testing.assert_array_equal(fold.roles["val"], following.roles["test"])
        sizes = [f.roles["test"].size for f in folds]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_kfold_stratification_keeps_class_shares_within_one_row(self):
        table = synthetic_table(n_rows=121, n_classes=3)
        _, _, folds = make_folds("kfold", self.fields, table)
        truth = table.column("truth")
        for cls in np.unique(truth):
            counts = [int((truth[f.roles["test"]] == cls).sum()) for f in folds]
            self.assertLessEqual(max(counts) - min(counts), 1, f"class {cls}: {counts}")

    def test_kfold_is_the_same_on_every_call_and_the_salt_changes_it(self):
        table = synthetic_table()
        _, _, first = make_folds("kfold", self.fields, table)
        _, _, second = make_folds("kfold", self.fields, table)
        self.assert_same_folds(first, second)
        _, _, other = make_folds("kfold", dict(self.fields, salt="another"), table)
        self.assertNotEqual(roles_of(first), roles_of(other))
        from sweepkit.splits.base import folds_hash

        self.assertEqual(folds_hash(table.ids, list(first)), folds_hash(table.ids, list(second)))
        self.assertNotEqual(folds_hash(table.ids, list(first)), folds_hash(table.ids, list(other)))

    def test_kfold_without_stratification_still_deals_every_row(self):
        table = synthetic_table(n_rows=50)
        _, _, folds = make_folds("kfold", dict(self.fields, stratify=None, k=3), table)
        sizes = sorted(f.roles["test"].size for f in folds)
        self.assertEqual(sizes, [16, 17, 17])
        for fold in folds:
            self.assert_supervised_fold(fold, len(table))

    def test_kfold_refuses_k_below_three(self):
        from sweepkit.registry import SPLITS

        table = synthetic_table()
        entry, two = SPLITS.validate_section({"name": "kfold", **dict(self.fields, k=2)}, "split")
        with self.assertRaisesRegex(ValueError, "at least 3"):
            entry.target.make(two, table)
        job = classify_base()
        job["split"] = {"name": "kfold", **dict(self.fields, k=2)}
        with self.assertRaisesRegex(ValueError, "k must be at least 3"):
            checked_job(job)


class TestIdentityFold(SchemeTest):
    def test_the_identity_fold_validates_on_the_first_folds_test_rows(self):
        from sweepkit.splits.base import CHECK_FOLD

        table = synthetic_table()
        _, _, folds = make_folds("kfold", {"k": 4, "stratify": "truth", "salt": "s", "identity_fold": True}, table)
        self.assertEqual([f.name for f in folds], ["fold1", "fold2", "fold3", "fold4", CHECK_FOLD])
        identity, first = folds[-1], folds[0]
        self.assertTrue(identity.selection_on_test)
        self.assertFalse(any(f.selection_on_test for f in folds[:-1]))
        np.testing.assert_array_equal(identity.roles["val"], identity.roles["test"])
        np.testing.assert_array_equal(identity.roles["test"], first.roles["test"])
        np.testing.assert_array_equal(identity.roles["train"], first.roles["train"])
        self.assert_supervised_fold(first, len(table))

    def test_train_final_never_holds_test_rows(self):
        table = synthetic_table()
        _, _, folds = make_folds("kfold", {"k": 4, "stratify": None, "salt": "s", "identity_fold": True}, table)
        normal, identity = folds[0], folds[-1]
        np.testing.assert_array_equal(normal.train_final(), np.union1d(normal.roles["train"], normal.roles["val"]))
        np.testing.assert_array_equal(identity.train_final(), identity.roles["train"])
        for fold in folds:
            self.assertFalse(set(fold.train_final().tolist()) & set(fold.roles["test"].tolist()), fold.name)

    def test_every_supervised_scheme_can_add_the_identity_fold(self):
        from sweepkit.splits.base import CHECK_FOLD

        table = synthetic_table()
        cases = [("kfold", {"k": 3, "stratify": None, "salt": "s"}),
                 ("group_kfold", {"group": "group", "k": 3, "salt": "s"}),
                 ("leave_one_group_out", {"group": "group", "order": ["g0", "g1", "g2", "g3", "g4"]}),
                 ("fixed", {"column": "published_split"}),
                 ("holdout", {"val_fraction": 0.2, "test_fraction": 0.2, "stratify": "truth", "salt": "s"})]
        for name, fields in cases:
            with self.subTest(split=name):
                _, scheme, folds = make_folds(name, dict(fields, identity_fold=True), table)
                self.assertEqual(folds[-1].name, CHECK_FOLD)
                self.assertEqual(sum(f.selection_on_test for f in folds), 1)
                np.testing.assert_array_equal(folds[-1].roles["test"], folds[0].roles["test"])

    def test_fold_refuses_overlaps_empty_roles_and_a_misnamed_check_fold(self):
        from sweepkit.splits.base import CHECK_FOLD, Fold

        def roles(train, val, test):
            return {"train": np.array(train), "val": np.array(val), "test": np.array(test)}

        Fold("ok", roles([0, 1], [2], [3]), "", False)
        Fold(CHECK_FOLD, roles([0, 1], [2, 3], [2, 3]), "", True)
        for bad, message in [(lambda: Fold("f", roles([0, 1], [1], [3]), "", False), "both train and val"),
                             (lambda: Fold("f", roles([0, 1], [2], [2]), "", False), "both val and test"),
                             (lambda: Fold("f", roles([0, 3], [2], [3]), "", False), "both train and test"),
                             (lambda: Fold("f", roles([], [2], [3]), "", False), "no rows in role train"),
                             (lambda: Fold("f", roles([0, 0], [2], [3]), "", False), "lists a row twice"),
                             (lambda: Fold("f", {"fit": np.array([0]), "test": np.array([1])}, "", False),
                              "roles must be"),
                             (lambda: Fold("f", {"fit": np.array([[0]]), "eval": np.array([1])}, "", False),
                              "one-dimensional"),
                             (lambda: Fold("check", roles([0, 1], [2], [2]), "", True), "named 'identity'"),
                             (lambda: Fold(CHECK_FOLD, roles([0, 1], [2], [3]), "", False), "named 'identity'"),
                             (lambda: Fold(CHECK_FOLD, roles([0, 1], [2], [3]), "", True),
                              "validation rows to be the test rows")]:
            with self.subTest(expected=message):
                with self.assertRaisesRegex(ValueError, message):
                    bad()


class TestGroupKFold(SchemeTest):
    def test_group_kfold_never_puts_a_group_in_two_roles(self):
        table = synthetic_table()
        groups = table.column("group")
        for k in (3, 5):
            with self.subTest(k=k):
                _, _, folds = make_folds("group_kfold", {"group": "group", "k": k, "salt": "s",
                                                         "identity_fold": False}, table)
                tested = collections.Counter()
                for fold in folds:
                    self.assert_supervised_fold(fold, len(table))
                    sets = {r: set(groups[p].tolist()) for r, p in fold.roles.items()}
                    self.assertFalse(sets["train"] & sets["val"] or sets["train"] & sets["test"]
                                     or sets["val"] & sets["test"], fold.name)
                    tested.update(sets["test"])
                self.assertEqual(tested, collections.Counter({f"g{g}": 1 for g in range(5)}),
                                 "every group is tested in exactly one fold")

    def test_group_kfold_split_report_shows_no_group_overlap(self):
        from sweepkit.splits.base import split_report

        table = synthetic_table()
        _, _, folds = make_folds("group_kfold", {"group": "group", "k": 3, "salt": "s", "identity_fold": False},
                                 table)
        report = split_report(list(folds), table, "truth", ("group",))
        for name, entry in report.items():
            self.assertEqual(set(entry["group_overlap"]["group"].values()), {0}, name)
            self.assertEqual(sum(entry["roles"][r]["rows"] for r in ("train", "val", "test")), len(table))
        _, _, plain = make_folds("kfold", {"k": 3, "stratify": None, "salt": "s", "identity_fold": False}, table)
        overlap = split_report(list(plain), table, None, ("group",))["fold1"]["group_overlap"]["group"]
        self.assertGreater(sum(overlap.values()), 0, "row-level folds share groups across roles")

    def test_group_kfold_is_stable_and_refuses_too_few_groups(self):
        table = synthetic_table()
        fields = {"group": "group", "k": 3, "salt": "s", "identity_fold": False}
        self.assert_same_folds(make_folds("group_kfold", fields, table)[2], make_folds("group_kfold", fields, table)[2])
        with self.assertRaisesRegex(ValueError, "5 groups in 'group', fewer than k = 6"):
            make_folds("group_kfold", dict(fields, k=6), table)
        job = classify_base()
        job["split"] = {"name": "group_kfold", **dict(fields, k=2)}
        with self.assertRaisesRegex(ValueError, "k must be at least 3"):
            checked_job(job)


class TestLeaveOneGroupOut(SchemeTest):
    order = ["g2", "g0", "g4", "g1", "g3"]

    def test_leave_one_group_out_tests_each_group_and_validates_on_the_next(self):
        table = synthetic_table()
        groups = table.column("group")
        _, _, folds = make_folds("leave_one_group_out", {"group": "group", "order": self.order,
                                                         "identity_fold": False}, table)
        self.assertEqual([f.name for f in folds], [f"test_{g}" for g in self.order])
        for i, fold in enumerate(folds):
            self.assert_supervised_fold(fold, len(table))
            test_group, val_group = self.order[i], self.order[(i + 1) % len(self.order)]
            self.assertEqual(set(groups[fold.roles["test"]].tolist()), {test_group})
            self.assertEqual(set(groups[fold.roles["val"]].tolist()), {val_group})
            self.assertEqual(set(groups[fold.roles["train"]].tolist()),
                             set(self.order) - {test_group, val_group})
            np.testing.assert_array_equal(fold.roles["test"], np.flatnonzero(groups == test_group))

    def test_leave_one_group_out_needs_every_value_listed_once(self):
        table = synthetic_table()
        with self.assertRaisesRegex(ValueError, "list every value once"):
            make_folds("leave_one_group_out", {"group": "group", "order": ["g0", "g1", "g2", "g3"],
                                               "identity_fold": False}, table)
        for order in (["g0", "g1"], ["g0", "g1", "g1", "g2", "g3"]):
            job = classify_base()
            job["split"] = {"name": "leave_one_group_out", "group": "group", "order": order, "identity_fold": False}
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, "at least 3 distinct values"):
                    checked_job(job)

    def test_fold_names_are_cleaned_and_must_stay_distinct(self):
        table = hand_table(site=["site A", "site/B", "site-C"] * 4)
        _, _, folds = make_folds("leave_one_group_out", {"group": "site", "order": ["site A", "site/B", "site-C"],
                                                         "identity_fold": False}, table)
        self.assertEqual([f.name for f in folds], ["test_site_A", "test_site_B", "test_site-C"])
        from sweepkit.registry import SPLITS

        _, clash = SPLITS.validate_section({"name": "leave_one_group_out", "group": "site",
                                            "order": ["a b", "a_b", "c"], "identity_fold": False}, "split")
        with self.assertRaisesRegex(ValueError, "repeated fold names"):
            SPLITS.get("leave_one_group_out").target.fold_names(clash)


class TestFixedAndHoldout(SchemeTest):
    def test_fixed_reads_the_published_split_column(self):
        table = synthetic_table()
        column = table.column("published_split")
        _, _, folds = make_folds("fixed", {"column": "published_split", "identity_fold": False}, table)
        self.assertEqual([f.name for f in folds], ["fixed"])
        fold = folds[0]
        self.assert_supervised_fold(fold, len(table))
        for role in ("train", "val", "test"):
            np.testing.assert_array_equal(fold.roles[role], np.flatnonzero(column == role))
        self.assertEqual({r: fold.roles[r].size for r in fold.roles}, {"train": 72, "val": 24, "test": 24})

    def test_fixed_refuses_other_values(self):
        table = hand_table(role=["train", "val", "test", "holdout"])
        with self.assertRaisesRegex(ValueError, r"holds \['holdout'\]; only train, val and test"):
            make_folds("fixed", {"column": "role", "identity_fold": False}, table)

    def test_holdout_fractions_within_each_stratum(self):
        table = synthetic_table()            # 40 rows of each of three classes
        truth = table.column("truth")
        _, _, folds = make_folds("holdout", {"val_fraction": 0.2, "test_fraction": 0.25, "stratify": "truth",
                                             "salt": "s", "identity_fold": False}, table)
        fold = folds[0]
        self.assert_supervised_fold(fold, len(table))
        for cls in np.unique(truth):
            counts = {r: int((truth[p] == cls).sum()) for r, p in fold.roles.items()}
            self.assertEqual(counts, {"train": 22, "val": 8, "test": 10}, cls)
        _, _, plain = make_folds("holdout", {"val_fraction": 0.2, "test_fraction": 0.25, "stratify": None,
                                             "salt": "s", "identity_fold": False}, table)
        self.assertEqual({r: p.size for r, p in plain[0].roles.items()}, {"train": 66, "val": 24, "test": 30})
        self.assert_same_folds(plain, make_folds("holdout", {"val_fraction": 0.2, "test_fraction": 0.25,
                                                             "stratify": None, "salt": "s",
                                                             "identity_fold": False}, table)[2])

    def test_holdout_refuses_fractions_that_leave_no_training_rows(self):
        for val, test in ((0.5, 0.5), (0.0, 0.2), (0.2, 1.0)):
            job = classify_base()
            job["split"] = {"name": "holdout", "val_fraction": val, "test_fraction": test, "stratify": None,
                            "salt": "s", "identity_fold": False}
            with self.subTest(val=val, test=test):
                with self.assertRaisesRegex(ValueError, "fractions above 0, adding to less than 1"):
                    checked_job(job)


class TestUnsupervised(SchemeTest):
    def test_all_fits_and_evaluates_every_row(self):
        table = synthetic_table(n_rows=30)
        _, _, folds = make_folds("all", {}, table)
        self.assertEqual(len(folds), 1)
        fold = folds[0]
        self.assertEqual((fold.name, set(fold.roles)), ("all", {"fit", "eval"}))
        np.testing.assert_array_equal(fold.roles["fit"], np.arange(30))
        np.testing.assert_array_equal(fold.roles["eval"], np.arange(30))

    def test_subsamples_are_fixed_distinct_and_of_the_given_size(self):
        table = synthetic_table()
        fields = {"k": 3, "fraction": 0.5, "salt": "s"}
        _, _, folds = make_folds("subsamples", fields, table)
        self.assertEqual([f.name for f in folds], ["sub1", "sub2", "sub3"])
        chosen = []
        for fold in folds:
            np.testing.assert_array_equal(fold.roles["fit"], fold.roles["eval"])
            self.assertEqual(fold.roles["fit"].size, 60)
            chosen.append(set(fold.roles["fit"].tolist()))
        self.assertTrue(chosen[0] != chosen[1] and chosen[1] != chosen[2] and chosen[0] != chosen[2])
        self.assertTrue(chosen[0] & chosen[1], "subsamples overlap, which stability across folds compares")
        self.assert_same_folds(folds, make_folds("subsamples", fields, table)[2])
        whole = make_folds("subsamples", {"k": 2, "fraction": 1.0, "salt": "s"}, table)[2]
        np.testing.assert_array_equal(whole[0].roles["fit"], np.arange(len(table)))

    def test_subsamples_refusals(self):
        table = synthetic_table(n_rows=8)
        with self.assertRaisesRegex(ValueError, "leaves fewer than 2"):
            make_folds("subsamples", {"k": 2, "fraction": 0.1, "salt": "s"}, table)
        for k, fraction in ((0, 0.5), (2, 0.0), (2, 1.5)):
            job = cluster_base()
            job["split"] = {"name": "subsamples", "k": k, "fraction": fraction, "salt": "s"}
            with self.subTest(k=k, fraction=fraction):
                with self.assertRaisesRegex(ValueError, "k at least 1, fraction in"):
                    checked_job(job)

    def test_fit_holdout_partitions_the_rows(self):
        table = synthetic_table()
        _, _, folds = make_folds("fit_holdout", {"fit_fraction": 0.7, "salt": "s"}, table)
        fold = folds[0]
        self.assertEqual(fold.name, "holdout")
        fit, evaluated = set(fold.roles["fit"].tolist()), set(fold.roles["eval"].tolist())
        self.assertEqual((len(fit), len(evaluated)), (84, 36))
        self.assertFalse(fit & evaluated)
        self.assertEqual(fit | evaluated, set(range(len(table))))
        self.assert_same_folds(folds, make_folds("fit_holdout", {"fit_fraction": 0.7, "salt": "s"}, table)[2])

    def test_fit_holdout_needs_assign_predict(self):
        job = cluster_base()
        job["split"] = {"name": "fit_holdout", "fit_fraction": 0.7, "salt": "s"}
        with self.assertRaisesRegex(ValueError, "needs assign predict"):
            checked_job(job)
        job["fit"]["assign"] = "predict"
        checked_job(job)
        job["split"]["fit_fraction"] = 1.0
        with self.assertRaisesRegex(ValueError, "fit_fraction in"):
            checked_job(job)


class TestOrderHelpers(SchemeTest):
    def test_stable_order_is_a_fixed_shuffle(self):
        from sweepkit.splits.base import stable_order

        items = [f"u{i}" for i in range(20)]
        first = stable_order(items, "salt")
        self.assertEqual(first, stable_order(list(reversed(items)), "salt"))
        self.assertEqual(sorted(first), sorted(items))
        self.assertNotEqual(first, items)
        self.assertNotEqual(first, stable_order(items, "other"))

    def test_deal_balances_groups_and_strata(self):
        from sweepkit.splits.base import deal, deal_rows

        units = [f"u{i}" for i in range(23)]
        stratum = {u: ("a" if i % 3 else "b") for i, u in enumerate(units)}
        groups = deal(units, stratum.get, 4, "s")
        sizes = collections.Counter(groups.values())
        self.assertLessEqual(max(sizes.values()) - min(sizes.values()), 1)
        for s in ("a", "b"):
            counts = collections.Counter(g for u, g in groups.items() if stratum[u] == s)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1, s)
        rows = deal_rows(np.array(units), np.array([stratum[u] for u in units]), 4, "s")
        self.assertEqual(sorted(collections.Counter(rows.tolist()).values()), [5, 6, 6, 6])
        with self.assertRaisesRegex(ValueError, "at least 2 groups"):
            deal(units, stratum.get, 1, "s")
        with self.assertRaisesRegex(ValueError, "units repeat"):
            deal(["u1", "u1"], stratum.get, 2, "s")


if __name__ == "__main__":
    unittest.main()
