"""The node checks of TESTING.md stay runnable: the sweeps in sweeps/checks expand, the overnight queue check's models
all have a GPU path and ask for the GPU, the laptop-to-GPU check's stand-in script compares label files and, where
the hdbscan package is installed, clusters a small table, and the flow check's sweeps fit a node round, match its
table and families in TESTING.md, and smoke-test on the data its own make-synthetic lines write."""
import contextlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO, TempRepo, load_catalog, needs_module, required

CHECKS = REPO / "sweeps" / "checks"
STANDIN = REPO / "tests" / "laptop_standin.py"


class TestCheckSweeps(unittest.TestCase):
    def test_every_check_sweep_expands(self):
        load_catalog()
        from sweepkit.sweep.design import expand

        files = sorted(CHECKS.glob("*.json"))
        self.assertEqual([f.name for f in files], ["flow_cluster.json", "flow_embedding.json", "flow_estimators.json",
                                                   "flow_mlp.json", "flow_outliers.json", "flow_regression.json",
                                                   "gpu_queue.json", "laptop_hdbscan.json"])
        for path in files:
            jobs, _ = expand(json.loads(path.read_text(encoding="utf-8")))
            self.assertGreater(len(jobs), 0, path.name)

    def test_the_queue_check_runs_only_models_with_a_gpu_path_on_the_gpu(self):
        load_catalog()
        from sweepkit.registry import MODELS
        from sweepkit.sweep.design import expand

        jobs, _ = expand(json.loads((CHECKS / "gpu_queue.json").read_text(encoding="utf-8")))
        self.assertEqual(len(jobs), 24)
        for job in jobs:
            settings = job["settings"]
            self.assertEqual(settings["fit"]["device"], "cuda")
            self.assertEqual(MODELS.get(settings["model"]["name"]).meta["gpu"], "cuml", settings["model"]["name"])


FLOW = ["flow_estimators", "flow_mlp", "flow_regression", "flow_cluster", "flow_embedding", "flow_outliers"]


def flow_section() -> str:
    text = (REPO / "TESTING.md").read_text(encoding="utf-8")
    return text.split("## The flow check\n", 1)[1].split("\n## ", 1)[0]


class TestFlowCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        from sweepkit.sweep.design import expand

        cls.expanded = {}
        for name in FLOW:
            sweep = json.loads((CHECKS / f"{name}.json").read_text(encoding="utf-8"))
            cls.expanded[name] = (sweep, *expand(sweep))

    def test_every_job_fits_a_node_round_and_reads_the_uploaded_tables(self):
        for name, (sweep, jobs, summary) in self.expanded.items():
            self.assertEqual(sweep["sweep_id"], f"check_{name}")
            devices = {j["settings"]["fit"]["device"] for j in jobs}
            self.assertEqual(devices, {"cuda"} if name == "flow_mlp" else {"auto"}, name)
            paths = {j["settings"]["data"]["path"] for j in jobs}
            self.assertEqual(len(paths), 1, name)
            self.assertRegex(paths.pop(), r"^flow/(classify|regress|blobs|outliers)\.csv$")
        self.assertEqual({j["settings"]["fit"]["epochs"] for j in self.expanded["flow_mlp"][1]}, {1})

    def test_the_table_in_testing_md_matches_the_sweeps(self):
        rows = re.findall(r"^\| `(flow_\w+)\.json` \|.*\| (\d+) \| (\d+) \|$", flow_section(), re.M)
        self.assertEqual([r[0] for r in rows], FLOW)
        for name, configs, jobs in rows:
            _, expanded_jobs, summary = self.expanded[name]
            self.assertEqual((summary["configs"], summary["jobs"]), (int(configs), int(jobs)), name)
        total = sum(len(self.expanded[n][1]) for n in FLOW)
        self.assertIn(f"the round is {total} jobs", flow_section())

    def test_the_families_name_the_shuffled_controls(self):
        block = flow_section().split("```json\n", 1)[1].split("```", 1)[0]
        families = json.loads(block)["families"]
        from sweepkit.analysis.rules import check_families

        check_families({"families": families})
        for family in families:
            (comparison,) = family["comparisons"]
            sweep = comparison["b"]["sweep"]
            name = sweep.removeprefix("check_")
            self.assertEqual(comparison["a"]["selection"], f"{name}_choice")
            control = [j for j in self.expanded[name][1] if j["stage"] == "control"]
            self.assertTrue(control, name)
            self.assertTrue(all(j["config_hash"].startswith(comparison["b"]["config"]) for j in control), name)

    def test_the_make_synthetic_lines_write_the_tables_each_sweep_smokes_on(self):
        """The four make-synthetic lines of TESTING.md, run as written; each flow sweep whose packages are installed
        here is then smoke-tested on them, on the device its workers would offer: the CPU, or the GPU for a sweep that
        asks for one (on the node, check.sh requires every package and the GPU)."""
        from sweepkit import cli
        from sweepkit.devices import gpu_visible
        from sweepkit.scaffold.additions import missing_packages

        lines = [line.strip() for line in flow_section().splitlines() if "sweepkit.cli make-synthetic" in line]
        self.assertEqual(len(lines), 4)
        with TempRepo() as root:
            for line in lines:
                argv = shlex.split(line.split("sweepkit.cli ", 1)[1])
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(argv), 0, line)
            (root / "data" / "flow").mkdir()
            for f in sorted((root / "data" / "synthetic").iterdir()):
                shutil.move(str(f), str(root / "data" / "flow" / f.name))
            self.assertEqual(len(list((root / "data" / "flow").iterdir())), 12)
            for name in FLOW:
                with self.subTest(sweep=name):
                    jobs = self.expanded[name][1]
                    device = "cuda" if any(j["settings"]["fit"]["device"] == "cuda" for j in jobs) else "cpu"
                    missing = sorted({m for job in jobs for m in missing_packages(job, device)})
                    if device == "cuda" and not gpu_visible():
                        missing.append("gpu")
                    if missing:
                        must = sorted(set(missing) & required())
                        if must:
                            self.fail(f"{name}: {must} missing, but SWEEPKIT_REQUIRE says this machine must run it")
                        continue
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                        code = cli.main(["smoke", "--sweep-file", str(CHECKS / f"{name}.json"), "--device", device,
                                         "--rows", "100"])
                    self.assertEqual(code, 0, out.getvalue()[-2000:])
                    self.assertIn("SMOKE OK", out.getvalue())


class TestLaptopStandin(unittest.TestCase):
    def test_compare_reads_labels_and_assignments(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.csv").write_text("row_id,label\nr1,0\nr2,0\nr3,1\nr4,-1\n")
            (d / "b.csv").write_text("row_id,cluster,status\nr1,5,kept\nr2,5,kept\nr3,7,kept\nr4,-1,kept\n")
            (d / "rows.csv").write_text("row_id,f0,truth\nr1,0.1,c0\nr2,0.2,c0\nr3,5.0,c1\nr4,9.0,c1\n")
            done = subprocess.run([sys.executable, str(STANDIN), "--compare", str(d / "a.csv"), str(d / "b.csv"),
                                   str(d / "rows.csv")], capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("4 rows in both: adjusted Rand index 1.000000, identical labels False", done.stdout)
            self.assertIn("against the truth", done.stdout)

    @needs_module("hdbscan")
    def test_the_script_clusters_a_small_table(self):
        import numpy as np

        rng = np.random.default_rng(0)
        centres = np.repeat(np.arange(3) * 20.0, 600)
        x = rng.normal(size=(1800, 4)) + centres[:, None]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lines = ["row_id," + ",".join(f"f{j}" for j in range(4)) + ",truth"]
            lines += [f"r{i}," + ",".join(f"{v:.6f}" for v in row) + f",c{i // 600}" for i, row in enumerate(x)]
            (d / "rows.csv").write_text("\n".join(lines) + "\n")
            done = subprocess.run([sys.executable, str(STANDIN), str(d / "rows.csv"), str(d / "labels.csv")],
                                  capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("1800 rows, 4 columns: 3 clusters", done.stdout)


if __name__ == "__main__":
    unittest.main()
