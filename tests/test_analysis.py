"""The choose-then-judge flow on real runs of small synthetic sweeps: status, analyze, select, rule-fix,
test-report, reveal-report, test-compare, compare-runs and smoke.

The runs are made by the real runners (run --local), so the files analyzed are the files a job writes. Every test that
chooses records each file it opens and fails if a test or reveal file is among them.
"""
import builtins
import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from tests.support import TempRepo, load_catalog
from tests.sweep_fixtures import classify_base, cluster_base, sweep

JUDGING = ("metrics_test.json", "predictions_test.csv", "reveal.json", "reveal_reference.csv")


@contextlib.contextmanager
def opened_files():
    """Every path opened for reading through open() or io.open() inside the block (writes are not counted: a run
    writes its own test and reveal files)."""
    seen = []
    real = builtins.open

    def recording(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if not any(c in mode for c in "wxa+"):
            seen.append(str(file))
        return real(file, *args, **kwargs)

    with mock.patch.object(builtins, "open", recording), mock.patch.object(io, "open", recording):
        yield seen


def cli(*argv) -> int:
    from sweepkit import cli as module

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return module.main(list(argv))


def run_sweep(root: Path, s: dict) -> None:
    path = root / f"{s['sweep_id']}.json"
    path.write_text(json.dumps(s), encoding="utf-8")
    assert cli("run", "--sweep-file", str(path), "--device", "cpu", "--max-consecutive-failures", "2") == 0


def analyze_saved(sweep_id: str) -> dict:
    from sweepkit import paths

    assert cli("analyze", "--sweep", sweep_id, "--confidence", "0.95", "--n-boot", "400", "--seed", "1",
               "--meaningful", "0.75", "--top", "10") == 0
    folder = sorted(p for p in paths.analysis_dir().iterdir() if p.name.startswith(sweep_id + "_"))[-1]
    return json.loads((folder / "report.json").read_text(encoding="utf-8"))


def set_discipline(root: Path, value: str) -> None:
    (root / "project.json").write_text(json.dumps({"project": "testproject", "discipline": value}), encoding="utf-8")


LATE = "a test that fixes its RULE after the runs it covers"


def fix_rule(root: Path, name: str, sweeps: list, families, late_reason: str = LATE) -> str:
    from sweepkit import paths

    (root / "RULE.txt").write_text("which results are read, fixed before they exist\n", encoding="utf-8")
    fam = "none"
    if families is not None:
        (root / f"{name}_families.json").write_text(json.dumps({"families": families}), encoding="utf-8")
        fam = str(root / f"{name}_families.json")
    assert cli("rule-fix", "--name", name, "--rule-file", str(root / "RULE.txt"), "--sweeps", *sweeps,
               "--families", fam, "--late-reason", late_reason) == 0
    return [p.name for p in paths.rules_dir().iterdir() if p.name.startswith(name + "_")][0]


class Repo(unittest.TestCase):
    """A temporary repository shared by the tests of one class."""

    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.repo = TempRepo()
        cls.root = cls.repo.__enter__()
        cls.make()

    @classmethod
    def tearDownClass(cls):
        cls.repo.__exit__(None, None, None)

    @classmethod
    def make(cls):
        pass


class TestClusteringFlow(Repo):
    @classmethod
    def make(cls):
        base = cluster_base()
        # eight subsamples: a label-free comparison's p cannot go below 2 / 2^folds (the sign-flip bound), so two or
        # three subsamples can show stability but never a difference
        base["split"] = {"name": "subsamples", "k": 8, "fraction": 0.8, "salt": "t"}
        cls.sweep = sweep(base, [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2, 3]},
            {"name": "k", "kind": "one_factor_at_a_time", "priority": 4, "seeds": [1, 2, 3],
             "factors": {"model.n_clusters": [2, 5]}},
            {"name": "same", "kind": "variants", "priority": 3, "seeds": [1, 2, 3],
             "variants": {"copy_x_off": {"model.extra": {"copy_x": False}}}},
        ], sweep_id="clu1", folds=[f"sub{i}" for i in range(1, 9)])
        run_sweep(cls.root, cls.sweep)
        with opened_files() as seen:
            cls.report = analyze_saved("clu1")
            assert cli("status", "--sweep", "clu1", "--top", "5", "--since", "all") == 0
            assert cli("select", "--sweep", "clu1", "--config", cls.report["ranked"][0]["config_hash"][:10], "--name",
                       "while_watched", "--reason", "the highest silhouette") == 0
        cls.opened_while_choosing = list(seen)

    def config(self, label: str) -> dict:
        return next(s for s in self.report["ranked"] if s["label"] == label)

    def test_choosing_never_opens_a_judging_file(self):
        self.assertTrue(any(p.endswith("criteria.json") for p in self.opened_while_choosing))
        self.assertTrue(any(p.endswith("assignments.csv") for p in self.opened_while_choosing))
        self.assertEqual([p for p in self.opened_while_choosing if Path(p).name in JUDGING], [])

    def test_ranking_and_verdicts(self):
        report = self.report
        self.assertEqual(report["select"], {"metric": "silhouette", "direction": "max"})
        self.assertEqual(report["n_runs"], report["n_expected_runs"])
        top = report["ranked"][0]
        self.assertIn(top["label"], ("base", "copy_x_off"))
        for label in ("model.n_clusters=2", "model.n_clusters=5"):
            s = self.config(label)
            self.assertEqual(s["versus_baseline"]["verdict"], "worse", label)
            self.assertLess(s["value"], self.config("base")["value"])
        self.assertEqual(report["n_compared"], 3)

    def test_identical_outputs_are_grouped(self):
        base = self.config("base")["config_hash"]
        same = self.config("copy_x_off")["config_hash"]
        self.assertIn(sorted([base, same]), self.report["identical_outputs"])
        self.assertEqual(self.config("copy_x_off")["versus_baseline"]["verdict"], "no detectable difference")

    def test_stability_of_well_separated_blobs(self):
        stab = self.config("base")["stability"]
        self.assertEqual(stab["measure"], "adjusted Rand index")
        self.assertEqual(stab["across_seeds"]["n_pairs"], 8 * 3)      # 8 folds x 3 seed pairs
        self.assertEqual(stab["across_folds"]["n_pairs"], 3 * 28)    # 3 seeds x 28 fold pairs
        self.assertGreater(stab["value"], 0.9)

    def test_reveal_waits_for_a_rule_fixed_before_the_selection(self):
        from sweepkit.analysis.rules import DisciplineError
        from sweepkit.analysis.selection import reveal_report

        best = self.config("base")["config_hash"]
        self.assertEqual(cli("select", "--sweep", "clu1", "--config", best[:10], "--name", "early",
                             "--reason", "highest silhouette"), 0)
        with self.assertRaises(DisciplineError):
            reveal_report("early", 0.95)
        self.assertEqual(cli("reveal-report", "--selection", "early", "--confidence", "0.95"), 1)
        set_discipline(self.root, "advisory")
        try:
            report = reveal_report("early", 0.95)
            self.assertTrue(report["warnings"][0].startswith("WARNING (advisory discipline)"))
        finally:
            set_discipline(self.root, "enforced")
        fix_rule(self.root, "clurule", ["clu1"], None)
        self.assertEqual(cli("select", "--sweep", "clu1", "--config", best[:10], "--name", "late",
                             "--reason", "highest silhouette"), 0)
        report = reveal_report("late", 0.95)
        # the RULE came after the sweep's runs: allowed with a reason, and every report that uses it says so
        self.assertEqual(report["warnings"], [f"NOTE RULE clurule was fixed after 96 runs of clu1 existed: {LATE}"])
        self.assertAlmostEqual(report["metrics"]["ari"]["mean"], 1.0, places=9)

    def test_select_refuses_bad_requests(self):
        from sweepkit.analysis.selection import selection_record

        best = self.config("base")["config_hash"]
        with self.assertRaisesRegex(ValueError, "8 characters"):
            selection_record("clu1", best[:6], "x", "reason")
        with self.assertRaisesRegex(ValueError, "reason"):
            selection_record("clu1", best[:10], "x", "  ")
        with self.assertRaisesRegex(ValueError, "matches 0"):
            selection_record("clu1", "ffffffffff", "x", "reason")
        self.assertEqual(cli("select", "--sweep", "clu1", "--config", best[:10], "--name", "twice",
                             "--reason", "first"), 0)
        # a selection is written once: the same name again is refused, with a one-line error
        self.assertEqual(cli("select", "--sweep", "clu1", "--config", best[:10], "--name", "twice", "--reason",
                             "again"), 1)

    def test_atlas_lists_every_configuration(self):
        from sweepkit import paths

        self.assertEqual(cli("atlas", "--format", "csv"), 0)
        folder = sorted(paths.atlas_dir().iterdir())[-1]
        rows = (folder / "configs.csv").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows) - 1, 4)
        values = (folder / "values.csv").read_text(encoding="utf-8")
        self.assertIn("model.n_clusters,2,1,1,clu1", values)


class TestAnExportIsAReading(Repo):
    """test-compare refuses a RULE fixed after a configuration's reveal files were read, whichever command read them:
    here export --reveal yes, which records its reading beside the export."""

    @classmethod
    def make(cls):
        cls.sweep = sweep(cluster_base(), [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]},
            {"name": "k", "kind": "one_factor_at_a_time", "priority": 4, "seeds": [1, 2],
             "factors": {"model.n_clusters": [2]}},
        ], sweep_id="clx")
        run_sweep(cls.root, cls.sweep)
        cls.report = analyze_saved("clx")

    def test_a_rule_fixed_after_an_export_with_reveal_cannot_compare_its_configuration(self):
        base = next(s for s in self.report["ranked"] if s["label"] == "base")["config_hash"]
        two = next(s for s in self.report["ranked"] if s["label"] == "model.n_clusters=2")["config_hash"]
        fix_rule(self.root, "before_export", ["clx"], None)
        self.assertEqual(cli("select", "--sweep", "clx", "--config", base[:10], "--name", "exported", "--reason",
                             "the highest silhouette"), 0)
        self.assertEqual(cli("export", "--selection", "exported", "--runs", "all", "--reveal", "yes", "--plots", "no",
                             "--top-features", "3", "--formats", "csv"), 0)
        families = [{"name": "k", "files": "reveal", "metric": "ari",
                     "comparisons": [{"a": {"sweep": "clx", "config": base[:8]},
                                      "b": {"sweep": "clx", "config": two[:8]}}]}]
        after = fix_rule(self.root, "after_export", ["clx"], families)
        self.assertEqual(cli("test-compare", "--rule", after, "--confidence", "0.95", "--n-boot", "200", "--seed", "1",
                             "--meaningful", "0.75"), 1)


class TestSupervisedFlow(Repo):
    @classmethod
    def make(cls):
        base = classify_base()
        base["task"]["groups"] = ["group"]
        cls.sweep = sweep(base, [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2, 3]},
            {"name": "vars", "kind": "variants", "priority": 4, "seeds": [1, 2, 3],
             "variants": {"tiny_c": {"model.c": 0.001}}},
            {"name": "ctrl", "kind": "variants", "priority": 3, "seeds": [1, 2],
             "variants": {"shuffled": {"fit.control": "shuffle_train_labels"}}},
        ], sweep_id="sup1")
        run_sweep(cls.root, cls.sweep)
        with opened_files() as seen:
            cls.report = analyze_saved("sup1")
        cls.opened_while_choosing = list(seen)

    def test_choosing_never_opens_a_judging_file(self):
        self.assertTrue(any(p.endswith("predictions_val.csv") for p in self.opened_while_choosing))
        self.assertEqual([p for p in self.opened_while_choosing if Path(p).name in JUDGING], [])

    def test_controls_apart_and_groups_reported(self):
        self.assertEqual([s["label"] for s in self.report["controls"]], ["shuffled"])
        self.assertNotIn("shuffled", [s["label"] for s in self.report["ranked"]])
        base = next(s for s in self.report["ranked"] if s["label"] == "base")
        self.assertEqual(sorted(base["by_group"]["group"]), ["g0", "g1", "g2", "g3", "g4"])
        tiny = next(s for s in self.report["ranked"] if s["label"] == "tiny_c")
        self.assertTrue(tiny["versus_baseline"]["bootstrap"]["batched"])
        self.assertEqual(tiny["versus_baseline"]["bootstrap"]["resampled"], "seeds and rows")

    def test_test_report_and_rule_families(self):
        import numpy as np

        from sweepkit import paths
        from sweepkit.analysis.records import complete_runs

        ranked = self.report["ranked"]
        best, other = ranked[0]["config_hash"], ranked[1]["config_hash"]
        families = [{"name": "main", "files": "test", "metric": "balanced_accuracy",
                     "comparisons": [{"a": {"selection": "chosen"}, "b": {"sweep": "sup1", "config": other[:8]}}]},
                    {"name": "extra", "files": "test", "metric": "f1_macro",
                     "comparisons": [{"a": {"selection": "chosen"}, "b": {"sweep": "sup1", "config": other[:8]}},
                                     {"a": {"sweep": "sup1", "config": other[:8]},
                                      "b": {"sweep": "sup1", "config": best[:8]}}]}]
        rule = fix_rule(self.root, "suprule", ["sup1"], families)
        self.assertEqual(cli("select", "--sweep", "sup1", "--config", best[:10], "--name", "chosen",
                             "--reason", "highest balanced accuracy"), 0)
        self.assertEqual(cli("test-report", "--selection", "chosen", "--confidence", "0.95"), 0)
        report_folder = sorted(paths.test_reports_dir().iterdir())[-1]
        report = json.loads((report_folder / "report.json").read_text(encoding="utf-8"))
        # the mean the report gives equals the mean of the runs' own test files
        values = []
        for run in complete_runs("sup1"):
            if run["spec"]["config_hash"] == best:
                values.append(json.loads((run["folder"] / "metrics_test.json").read_text())["metrics"]["balanced_accuracy"])
        self.assertAlmostEqual(report["metrics"]["balanced_accuracy"]["mean"], float(np.mean(values)), places=12)
        reading = json.loads((report_folder / "reading.json").read_text(encoding="utf-8"))
        self.assertEqual(reading["configs"], [{"sweep_id": "sup1", "config_hash": best}])
        self.assertEqual(cli("test-compare", "--rule", rule, "--confidence", "0.95", "--n-boot", "400", "--seed", "2",
                             "--meaningful", "0.75"), 0)
        compared = json.loads((sorted(paths.comparisons_dir().iterdir())[-1] / "report.json").read_text())
        extra = next(f for f in compared["families"] if f["name"] == "extra")
        p = [c["bootstrap"]["p_two_sided"] for c in extra["comparisons"]]
        holm = [c["holm_p"] for c in extra["comparisons"]]
        self.assertAlmostEqual(min(holm), min(1.0, 2 * min(p)), places=12)
        # a RULE fixed after these readings may not license comparisons of the same configurations
        late = fix_rule(self.root, "late", ["sup1"], families)
        self.assertEqual(cli("test-compare", "--rule", late, "--confidence", "0.95", "--n-boot", "400", "--seed", "2",
                             "--meaningful", "0.75"), 1)

    def test_families_file_is_checked_when_the_rule_is_fixed(self):
        from sweepkit.analysis.rules import check_families

        good = {"name": "f", "files": "test", "metric": "accuracy",
                "comparisons": [{"a": {"selection": "x"}, "b": {"sweep": "s", "config": "0123456789"}}]}
        check_families({"families": [good]})
        bad = [dict(good, files="val"), dict(good, metric="silhouette", files="reveal"),
               dict(good, comparisons=[{"a": {"selection": "x"}, "b": {"selection": "x"}}]),
               dict(good, comparisons=[{"a": {"sweep": "s", "config": "0123"}, "b": {"selection": "x"}}])]
        for family in bad:
            with self.assertRaises(ValueError):
                check_families({"families": [family]})
        with self.assertRaises(KeyError):
            check_families({"families": [dict(good, metric="no_such_metric")]})


class TestIdentityCheckFold(Repo):
    """The identity fold validates on test rows: no step that chooses reads its results, and test-report checks that
    its validation and test numbers agree."""

    @classmethod
    def make(cls):
        base = classify_base()
        base["split"]["identity_fold"] = True
        cls.sweep = sweep(base, [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]},
            {"name": "vars", "kind": "variants", "priority": 4, "seeds": [1, 2],
             "variants": {"tiny_c": {"model.c": 0.001}}},
        ], sweep_id="idf1", folds=["fold1", "fold2", "fold3", "fold4", "identity"])
        run_sweep(cls.root, cls.sweep)
        with opened_files() as seen:
            cls.report = analyze_saved("idf1")
            assert cli("status", "--sweep", "idf1", "--top", "5", "--since", "all") == 0
            assert cli("atlas", "--format", "csv") == 0
        cls.opened_while_choosing = list(seen)

    def test_choosing_reads_no_result_of_the_identity_fold(self):
        from sweepkit.analysis.records import CHECK_RUN_READABLE

        identity = {Path(p).name for p in self.opened_while_choosing if "_identity_seed" in p}
        self.assertIn("spec.json", identity)                 # read, to learn the run's fold
        self.assertLessEqual(identity, set(CHECK_RUN_READABLE))
        self.assertEqual(self.report["folds"], ["fold1", "fold2", "fold3", "fold4"])
        self.assertEqual(self.report["check_folds"], ["identity"])
        self.assertEqual(self.report["n_runs"], 2 * 2 * 5)

    def test_status_counts_the_identity_fold_runs_without_their_numbers(self):
        from sweepkit.sweep.status import render, summarise

        s = summarise("idf1", None)
        self.assertEqual((s["runs_complete"], s["runs_identity"], s["runs_expected"]), (20, 4, 20))
        self.assertIn("runs: 20 complete of 20 (4 on the identity check fold, whose numbers are not shown), 0 failed",
                      render(s, 5))
        self.assertEqual(len(s["ranked"]), 4)            # 2 configurations x 2 seeds, each over the 4 folds

    def test_records_refuse_identity_results_to_a_choosing_step(self):
        from sweepkit.analysis.records import JudgingFileRefused, complete_runs, read_json, read_rows

        run = next(r for r in complete_runs("idf1") if r["spec"]["fold"] == "identity")
        for reader, name in ((read_json, "metrics_val.json"), (read_rows, "predictions_val.csv"),
                             (read_json, "model_params.json")):
            with self.assertRaises(JudgingFileRefused):
                reader(run["folder"], name, "choose")
        self.assertIn("metrics", read_json(run["folder"], "metrics_val.json", "judge"))

    def test_a_fold_whose_validation_rows_are_test_rows_must_be_the_identity_fold(self):
        import numpy as np

        from sweepkit.splits.base import Fold

        rows = {"train": np.arange(5), "val": np.arange(5, 8), "test": np.arange(5, 8)}
        Fold("identity", dict(rows), "", True)
        with self.assertRaisesRegex(ValueError, "named 'identity'"):
            Fold("check", dict(rows), "", True)
        with self.assertRaisesRegex(ValueError, "named 'identity'"):
            Fold("identity", {"train": np.arange(5), "val": np.arange(5, 8), "test": np.arange(8, 10)}, "", False)

    def test_test_report_checks_the_identity_fold(self):
        from sweepkit.analysis.selection import render, test_report

        best = self.report["ranked"][0]["config_hash"]
        fix_rule(self.root, "idrule", ["idf1"], None)
        self.assertEqual(cli("select", "--sweep", "idf1", "--config", best[:10], "--name", "idchosen",
                             "--reason", "highest balanced accuracy"), 0)
        report = test_report("idchosen", 0.95)
        self.assertEqual(report["identity_check"], {"runs": 2, "compared": True, "agree": True, "differing": []})
        for by_fold in report["select_by_fold"].values():
            self.assertEqual(sorted(by_fold), ["fold1", "fold2", "fold3", "fold4"])
        self.assertIn("identity check fold: validation and test numbers agree in all 2 runs",
                      render(report, "test"))


class TestStabilityChooses(Repo):
    @classmethod
    def make(cls):
        base = cluster_base()
        base["split"] = {"name": "subsamples", "k": 3, "fraction": 0.7, "salt": "t"}
        base["task"]["select"] = "stability"
        cls.sweep = sweep(base, [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]},
            {"name": "k", "kind": "one_factor_at_a_time", "priority": 4, "seeds": [1, 2],
             "factors": {"model.n_clusters": [8]}},
        ], sweep_id="stab1", folds=["sub1", "sub2", "sub3"])
        run_sweep(cls.root, cls.sweep)
        cls.report = analyze_saved("stab1")

    def test_ranked_by_stability_and_compared_by_folds(self):
        ranked = self.report["ranked"]
        self.assertEqual(self.report["select"], {"metric": "stability", "direction": "max"})
        self.assertEqual(ranked[0]["label"], "base")
        self.assertEqual([s["value"] for s in ranked], sorted((s["value"] for s in ranked), reverse=True))
        k8 = ranked[1]
        self.assertEqual(k8["versus_baseline"]["units"], "folds")
        self.assertEqual(k8["versus_baseline"]["bootstrap"]["n_folds"], 3)


class TestPrintedStability(Repo):
    """analyze's text names both parts of stability, across seeds and across folds (the dry run, 2026-10-30: t-SNE
    was 1.000 across seeds and 0.114 across subsamples, and only report.json said so), and shows a value smaller than
    its last decimal with significant digits (an excess-mass criterion of about 1e-4 printed as 0.0000)."""

    @classmethod
    def make(cls):
        cls.sweep = sweep(cluster_base(), [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]},
            {"name": "k", "kind": "one_factor_at_a_time", "priority": 4, "seeds": [1, 2],
             "factors": {"model.n_clusters": [4]}},
        ], sweep_id="print1")
        run_sweep(cls.root, cls.sweep)
        cls.report = analyze_saved("print1")

    def test_both_parts_of_stability_and_small_values(self):
        from sweepkit.analysis.validation import render

        text = render(self.report, 5)
        lines = [line for line in text.splitlines() if "stability (adjusted Rand index)" in line]
        self.assertEqual(len(lines), 2, text)
        for line in lines:
            self.assertIn("across seeds", line)
            self.assertIn("across folds", line)
        tiny = json.loads(json.dumps(self.report))
        tiny["ranked"][0]["value"] = 0.000144
        self.assertIn("  0.000144", render(tiny, 5))
        self.assertNotIn("  0.0001 ", render(tiny, 5))

class TestLateRule(Repo):
    """A RULE is fixed before the results it governs: rule-fix refuses one fixed after runs of its sweeps exist,
    unless a reason is given, which is recorded and printed by every report that uses the RULE."""

    @classmethod
    def make(cls):
        cls.sweep = sweep(cluster_base(), [{"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]}],
                          sweep_id="late1")
        run_sweep(cls.root, cls.sweep)
        (cls.root / "RULE.txt").write_text("which results are read, fixed before they exist\n", encoding="utf-8")

    def rule_fix(self, name: str, reason: str) -> int:
        return cli("rule-fix", "--name", name, "--rule-file", str(self.root / "RULE.txt"), "--sweeps", "late1",
                   "--families", "none", "--late-reason", reason)

    def test_refused_without_a_reason_recorded_with_one(self):
        from sweepkit import paths
        from sweepkit.analysis.rules import DisciplineError, rule_fix

        with self.assertRaisesRegex(DisciplineError, "would be fixed after 4 runs of late1 exist"):
            rule_fix("nope", self.root / "RULE.txt", ["late1"], None, None)
        self.assertEqual(self.rule_fix("nope", "none"), 1)
        self.assertFalse(paths.rules_dir().is_dir() and any(f.name.startswith("nope_")
                                                           for f in paths.rules_dir().iterdir()))
        self.assertEqual(self.rule_fix("reasoned", "the screen's RULE was forgotten; nothing judging was read"), 0)
        folder = next(f for f in paths.rules_dir().iterdir() if f.name.startswith("reasoned_"))
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        self.assertEqual((record["runs_at_fix"], record["late_reason"], record["warning"]),
                         ({"late1": 4}, "the screen's RULE was forgotten; nothing judging was read", None))
        base = analyze_saved("late1")["ranked"][0]["config_hash"]
        self.assertEqual(cli("select", "--sweep", "late1", "--config", base[:10], "--name", "late_choice",
                             "--reason", "the baseline"), 0)
        from sweepkit.analysis.selection import reveal_report

        self.assertEqual(reveal_report("late_choice", 0.95)["warnings"],
                         ["NOTE RULE reasoned was fixed after 4 runs of late1 existed: the screen's RULE was "
                          "forgotten; nothing judging was read"])

    def test_advisory_discipline_records_a_warning_instead(self):
        from sweepkit import paths

        set_discipline(self.root, "advisory")
        try:
            self.assertEqual(self.rule_fix("advised", "none"), 0)
        finally:
            set_discipline(self.root, "enforced")
        folder = next(f for f in paths.rules_dir().iterdir() if f.name.startswith("advised_"))
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        self.assertTrue(record["warning"].startswith("WARNING (advisory discipline): RULE advised would be fixed"))
        self.assertIsNone(record["late_reason"])

    def test_a_rule_before_any_run_is_not_late(self):
        from sweepkit import paths

        self.assertEqual(cli("rule-fix", "--name", "early", "--rule-file", str(self.root / "RULE.txt"), "--sweeps",
                             "not_run_yet", "--families", "none", "--late-reason", "none"), 0)
        folder = next(f for f in paths.rules_dir().iterdir() if f.name.startswith("early_"))
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        self.assertEqual((record["runs_at_fix"], record["late_reason"]), ({"not_run_yet": 0}, None))


class TestOneSeedOverSubsamples(Repo):
    """A deterministic method (the same result for every seed) is compared on one seed over its subsamples; one
    subsample with one seed is not compared."""

    @classmethod
    def make(cls):
        base = cluster_base()
        base["split"] = {"name": "subsamples", "k": 8, "fraction": 0.8, "salt": "t"}
        base["model"] = {"name": "agglomerative", "linkage": "ward", "metric": "euclidean", "cut": "n_clusters",
                         "n_clusters": 3, "distance": None, "extra": {}}
        stages = [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]},
                  {"name": "k", "kind": "one_factor_at_a_time", "priority": 4, "seeds": [1],
                   "factors": {"model.n_clusters": [2]}}]
        run_sweep(cls.root, sweep(base, stages, sweep_id="det8", folds=[f"sub{i}" for i in range(1, 9)]))
        run_sweep(cls.root, sweep(base, stages, sweep_id="det1", folds=["sub1"]))
        cls.eight, cls.one = analyze_saved("det8"), analyze_saved("det1")

    def test_one_seed_over_eight_subsamples_is_compared(self):
        two = next(s for s in self.eight["ranked"] if s["label"] == "model.n_clusters=2")
        v = two["versus_baseline"]
        self.assertEqual((v["units"], v["seeds"], v["seeds_baseline"]), ("folds and seeds", [1], [1]))
        self.assertEqual(sorted(v["difference_by_fold"]), [f"sub{i}" for i in range(1, 9)])
        self.assertTrue(all(d < 0 for d in v["difference_by_fold"].values()))        # two clusters for three blobs
        self.assertEqual(v["p_better"]["p_a_beats_b"], 0.0)                         # worse in every subsample
        self.assertAlmostEqual(v["bootstrap"]["p_two_sided"], 2 / 2 ** 8)         # the sign-flip bound over 8 folds
        self.assertEqual(v["verdict"], "worse")

    def test_one_subsample_and_one_seed_is_not_compared(self):
        two = next(s for s in self.one["ranked"] if s["label"] == "model.n_clusters=2")
        self.assertNotIn("versus_baseline", two)
        self.assertIn("needs two or more folds (subsamples), or two seeds of one fold", two["not_compared"])


class TestCompareRuns(Repo):
    @classmethod
    def make(cls):
        cls.sweep = sweep(cluster_base(), [{"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]}],
                          sweep_id="orig", purpose="learning")
        run_sweep(cls.root, cls.sweep)
        # another seed, so other run IDs: no other sweep holds a copy of its runs
        run_sweep(cls.root, sweep(cluster_base(), [{"name": "base", "kind": "base", "priority": 5, "seeds": [7]}],
                                  sweep_id="lonely", purpose="learning"))

    def copy_as(self, sweep_id: str, code: str):
        """A second sweep holding the same run IDs, as if made by other code."""
        from sweepkit import paths

        shutil.copytree(paths.sweeps_dir() / "orig", paths.sweeps_dir() / sweep_id)
        shutil.copytree(paths.runs_dir("orig"), paths.runs_dir(sweep_id))
        for folder in paths.runs_dir(sweep_id).iterdir():
            provenance = json.loads((folder / "provenance.json").read_text())
            provenance["code_sha256"] = code
            (folder / "provenance.json").write_text(json.dumps(provenance))

    def test_a_sweep_without_copies_has_nothing_to_compare(self):
        from sweepkit.analysis.compare import compare_sweep, render

        report = compare_sweep("lonely", 3)
        self.assertEqual((report["verdict"], report["n_copies"]), ("NO COPIES", 0))
        self.assertIn("NO COPIES: none of the 2 runs has a copy", render(report))
        self.assertNotIn("DIFFERENT", render(report))
        self.assertEqual(cli("compare-runs", "--sweep", "lonely", "--differences-shown", "2"), 0)

    def test_same_then_incomplete_then_different(self):
        from sweepkit import paths
        from sweepkit.analysis.compare import compare_sweep

        self.copy_as("again", "0" * 64)
        self.assertEqual(compare_sweep("again", 3)["verdict"], "SAME")
        self.assertEqual(cli("compare-runs", "--sweep", "again", "--differences-shown", "2"), 0)
        runs = sorted(paths.runs_dir("again").iterdir())
        shutil.rmtree(runs[-1])
        self.assertEqual(compare_sweep("again", 3)["verdict"], "INCOMPLETE")
        self.assertEqual(cli("compare-runs", "--sweep", "again", "--differences-shown", "2"), 3)
        text = (runs[0] / "assignments.csv").read_text().splitlines()
        text[1] = text[1] + "0"
        (runs[0] / "assignments.csv").write_text("\n".join(text) + "\n")
        report = compare_sweep("again", 3)
        self.assertEqual(report["verdict"], "DIFFERENT")
        self.assertEqual(report["differing"], [runs[0].name])
        self.assertEqual(cli("compare-runs", "--sweep", "again", "--differences-shown", "2"), 1)


class TestStatusAndSmoke(Repo):
    @classmethod
    def make(cls):
        cls.sweep = sweep(cluster_base(), [
            {"name": "base", "kind": "base", "priority": 5, "seeds": [1]},
            {"name": "big_k", "kind": "variants", "priority": 4, "seeds": [1],
             "variants": {"k_too_big": {"model.n_clusters": 250}}},
        ], sweep_id="st1", purpose="learning")
        (cls.root / "st1.json").write_text(json.dumps(cls.sweep))

    def test_smoke_reports_the_failing_configuration(self):
        from sweepkit import paths

        with opened_files() as seen:
            self.assertEqual(cli("smoke", "--sweep-file", str(self.root / "st1.json"), "--device", "cpu", "--rows",
                                 "20"), 1)
        self.assertEqual([p for p in seen if Path(p).name in JUDGING], [])
        folder = sorted(paths.smoke_dir().iterdir())[-1]
        table = (folder / "table.csv").read_text(encoding="utf-8")
        self.assertIn(",ok,", table)
        self.assertIn(",FAILED,", table)
        self.assertFalse((paths.sweeps_dir() / "st1").exists())     # smoke records nothing as a sweep

    def test_status_since_and_swap_failed(self):
        from sweepkit import paths
        from sweepkit.sweep.status import render, summarise

        self.assertEqual(cli("run", "--sweep-file", str(self.root / "st1.json"), "--device", "cpu",
                             "--max-consecutive-failures", "5"), 1)        # k_too_big fails
        s = summarise("st1", None)
        self.assertEqual(s["runs_complete"], 2)
        self.assertEqual(len(s["runs_failed"]), 2)
        self.assertEqual(summarise("st1", "2999-01-01T00:00:00")["runs_complete"], 0)
        (paths.queue_dir() / "SWAP_FAILED").write_text("digest differs", encoding="utf-8")
        self.assertTrue(render(summarise("st1", None), 3).startswith("ALERT SWAP_FAILED"))


class TestCommandLine(unittest.TestCase):
    def test_every_argument_is_required(self):
        from sweepkit.cli import parser

        p = parser()
        sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
        self.assertGreater(len(sub.choices), 15)
        for name, command in sub.choices.items():
            for action in command._actions:
                if action.option_strings and action.dest != "help":
                    self.assertTrue(action.required, f"{name} {action.option_strings}")

    def test_missing_argument_stops(self):
        from sweepkit.cli import parser

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            parser().parse_args(["analyze", "--sweep", "x"])
        self.assertEqual(stop.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
