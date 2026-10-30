"""export, cluster-report, describe-model and propose on real runs of a small clustering sweep."""
import contextlib
import io
import json
import sys
import types
import unittest

from tests.support import TempRepo, load_catalog, needs_module
from tests.sweep_fixtures import cluster_base, sweep


def cli(*argv) -> int:
    from sweepkit import cli as module

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return module.main(list(argv))


class Flow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.repo = TempRepo()
        cls.root = cls.repo.__enter__()
        base = cluster_base()
        s = sweep(base, [{"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]},
                         {"name": "grid", "kind": "grid", "priority": 4, "seeds": [1, 2],
                          "factors": {"model.n_clusters": [2, 3, 4], "model.n_init": [1, 2]}}],
                  sweep_id="ex1", purpose="learning")
        (cls.root / "ex1.json").write_text(json.dumps(s), encoding="utf-8")
        assert cli("run", "--sweep-file", str(cls.root / "ex1.json"), "--device", "cpu",
                   "--max-consecutive-failures", "1") == 0
        assert cli("analyze", "--sweep", "ex1", "--confidence", "0.95", "--n-boot", "200", "--seed", "1",
                   "--meaningful", "0.75", "--top", "3") == 0
        from sweepkit import paths

        report = json.loads(sorted(paths.analysis_dir().iterdir())[-1].joinpath("report.json").read_text())
        cls.best = next(r for r in report["ranked"] if r["label"] == "base")["config_hash"]
        assert cli("select", "--sweep", "ex1", "--config", cls.best[:10], "--name", "early", "--reason", "base") == 0
        (cls.root / "RULE.txt").write_text("rule\n", encoding="utf-8")
        assert cli("rule-fix", "--name", "ex", "--rule-file", str(cls.root / "RULE.txt"), "--sweeps", "ex1",
                   "--families", "none", "--late-reason", "a test that fixes its RULE after its runs") == 0
        assert cli("select", "--sweep", "ex1", "--config", cls.best[:10], "--name", "chosen", "--reason", "base") == 0
        r = sweep(base, [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]},
                         {"name": "random", "kind": "random", "priority": 4, "seeds": [1], "n_configs": 3,
                          "sampling_seed": 7, "skip_draws": 0,
                          "space": {"model.n_clusters": [2, 4, 5], "model.tol": [0.001, 0.01, 0.1]}}],
                  sweep_id="ex3", purpose="learning")
        (cls.root / "ex3.json").write_text(json.dumps(r), encoding="utf-8")
        assert cli("run", "--sweep-file", str(cls.root / "ex3.json"), "--device", "cpu",
                   "--max-consecutive-failures", "1") == 0

    @classmethod
    def tearDownClass(cls):
        cls.repo.__exit__(None, None, None)

    def latest(self, prefix):
        from sweepkit import paths

        return sorted(p for p in paths.exports_dir().iterdir()
                      if p.name.startswith(prefix) and p.name[len(prefix)].isdigit())[-1]

    def test_export_without_and_with_reveal(self):
        self.assertEqual(cli("export", "--selection", "chosen", "--runs", "all", "--reveal", "no", "--plots", "yes",
                             "--top-features", "2", "--formats", "csv"), 0)
        out = self.latest("chosen_")
        names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        for name in ("rows.csv", "sizes.csv", "cluster_summary.csv", "distinguishing.csv", "plots/clusters_pca.png",
                     "settings.json", "packages.json", "README.txt", "provenance.json"):
            self.assertIn(name, names)
        self.assertNotIn("contingency.csv", names)
        self.assertNotIn("reading.json", names)
        rows = (out / "rows.csv").read_text().strip().splitlines()
        self.assertEqual(len(rows) - 1, 4 * 240)          # 2 seeds x 2 folds x 240 evaluated rows
        readme = (out / "README.txt").read_text()
        self.assertIn(f"rows.csv: one line per evaluated row of every exported run, with the columns "
                      f"{rows[0].replace(',', ', ')}:", readme)      # the README names the file's own columns
        self.assertEqual(cli("export", "--selection", "early", "--runs", "all", "--reveal", "yes", "--plots", "no",
                             "--top-features", "2", "--formats", "csv"), 1)   # selected before the RULE: refused
        self.assertEqual(cli("export", "--selection", "chosen", "--runs", "all", "--reveal", "yes", "--plots", "no",
                             "--top-features", "2", "--formats", "csv"), 0)
        out = self.latest("chosen_")
        table = (out / "contingency.csv").read_text().strip().splitlines()
        self.assertEqual(len(table) - 1, 4 * 3)            # blobs: each cluster holds one planted group
        self.assertTrue((out / "reading.json").is_file())

    def test_cluster_report_and_describe_model(self):
        self.assertEqual(cli("cluster-report", "--selection", "chosen", "--runs", "all", "--reveal", "no",
                             "--top-features", "2"), 0)
        report = (self.latest("chosen_report_") / "report.txt").read_text()
        self.assertIn("sizes: cluster 0", report)
        self.assertEqual(cli("describe-model", "--source", "selection:chosen", "--config", "selected"), 0)
        from sweepkit import paths

        d = json.loads(sorted(paths.architectures_dir().iterdir())[-1].joinpath("description.json").read_text())
        self.assertEqual(d["library_parameters"]["n_clusters"], 3)
        from sweepkit.analysis.describe import PER_RUN

        self.assertEqual(d["library_parameters"]["random_state"], PER_RUN)      # each run's own seed, not one number
        self.assertTrue(d["runs"] and all(r["same"] for r in d["runs"]))

    def _stub_optuna(self):
        """A stand-in for Optuna with the calls propose makes, sampling at random: it tests propose's own logic
        (trials from finished runs, no repeats, a sweep file that expands) where Optuna is not installed."""
        import random

        class Trial:
            def __init__(self, params):
                self.params = params

        class Study:
            def __init__(self, seed):
                self.rng = random.Random(seed)
                self.trials = []

            def add_trial(self, trial):
                self.trials.append(trial)

            def ask(self, distributions):
                return Trial({k: self.rng.choice(d.choices) for k, d in distributions.items()})

        module = types.ModuleType("optuna")
        module.__version__ = "stub"
        module.distributions = types.SimpleNamespace(CategoricalDistribution=lambda choices: types.SimpleNamespace(
            choices=list(choices)))
        module.trial = types.SimpleNamespace(create_trial=lambda params, distributions, value: (params, value))
        module.samplers = types.SimpleNamespace(TPESampler=lambda seed, n_startup_trials: seed)
        module.create_study = lambda direction, sampler: Study(sampler)
        module.logging = types.SimpleNamespace(set_verbosity=lambda level: None, WARNING=30)
        return module

    def test_propose_writes_untried_configurations(self):
        from unittest import mock

        from sweepkit.analysis.propose import finished_trials, propose, _space

        with mock.patch.dict(sys.modules, {"optuna": self._stub_optuna()}):
            space = _space(json.loads((self.root / "ex1.json").read_text()), "grid")
            trials = finished_trials("ex1", space)
            self.assertEqual(len(trials), 6)     # the base sits on levels 3 and 2 too: the grid leaves it out
            with self.assertRaisesRegex(ValueError, "already tried"):
                propose("ex1", "grid", 1, 0, 0, "ex2", [4, 5], 3)    # all 6 cells of the grid are tried
            result = propose("ex3", "random", 2, 0, 0, "ex4", [11, 12], 3)
        tried = {json.dumps(t["params"], sort_keys=True) for t in result["record"]["trials"]}
        self.assertEqual(len(tried), 3)
        for p in result["record"]["proposals"]:
            self.assertNotIn(json.dumps(p["params"], sort_keys=True), tried)
        new = result["sweep_file"]
        self.assertEqual((new["sweep_id"], new["follows"], new["stages"][0]["seeds"]), ("ex4", ["ex3"], [11, 12]))
        from sweepkit.sweep.design import expand

        self.assertEqual(expand(new)[1]["configs"], 3)

    @needs_module("optuna")
    def test_propose_with_optuna(self):
        from sweepkit.analysis.propose import propose

        with self.assertRaisesRegex(ValueError, "already tried"):
            propose("ex1", "grid", 1, 0, 0, "ex2", [4, 5], 3)
        result = propose("ex3", "random", 2, 0, 0, "ex4", [11, 12], 3)
        self.assertEqual(len(result["record"]["proposals"]), 2)


if __name__ == "__main__":
    unittest.main()
