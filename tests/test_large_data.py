"""The large-data path, end to end, at a size set by SWEEPKIT_LARGE_ROWS (skipped when it is not set):

    1. make-synthetic writes a CSV of planted blobs in chunks (10 columns, 5 groups)
    2. convert turns it into a memory-mapped float32 store, in chunks
    3. a screen on two subsamples of at most 20000 rows chooses the number of clusters by silhouette
    4. estimate measures the chosen configuration on growing parts of all the rows and predicts the full run
    5. the full run: the chosen configuration on every row, one seed
and prints, for each step, the wall time and the peak memory of the process that ran it, with the estimate's
prediction beside the full run's measurement. Every step runs as its own process, as from a terminal, in a temporary
copy of the code.

    SWEEPKIT_LARGE_ROWS=50000 python -m unittest tests.test_large_data        any machine, about a minute
    SWEEPKIT_LARGE_ROWS=1000000 python -m unittest tests.test_large_data      the brief's million rows (TESTING.md)

Checks: the screen chooses the planted 5 clusters; the full run recovers them (adjusted Rand index at least 0.95,
read from the run's reveal file: a test reads it directly, no choice depends on it); the conversion's peak memory
stays below the CSV's size (it reads in chunks); the full run's peak memory stays within twice the estimate's
prediction (at a million rows three runs measured 1.35 to 1.4 times, the reason for the estimate's margin of 1.5).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.support import REPO

ROWS = int(os.environ.get("SWEEPKIT_LARGE_ROWS", "0") or 0)
LAUNCH = ("import atexit, runpy, sys\n"
          "def peak():\n"
          "    try:\n"
          "        import resource\n"
          "    except ImportError:\n"
          "        return\n"
          "    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
          "    print('PEAK_RSS_MB', value / (1024 * 1024) if sys.platform == 'darwin' else value / 1024,"
          " file=sys.stderr)\n"
          "atexit.register(peak)\n"
          "sys.argv = ['sweepkit'] + sys.argv[1:]\n"
          "runpy.run_module('sweepkit.cli', run_name='__main__')\n")


def clustering_sweep(sweep_id: str, split: dict, folds: list, stages: list, sample_rows: int) -> dict:
    return {"sweep_id": sweep_id, "description": "the large-data test", "purpose": "screen", "follows": [],
            "folds": folds,
            "base": {"data": {"name": "store", "store": "large", "features": ["*"], "keep": ["truth"],
                              "filter_column": None, "filter_values": []},
                     "split": split,
                     "task": {"name": "clustering", "criteria": ["silhouette", "calinski_harabasz", "n_clusters"],
                              "select": "silhouette", "sample_rows": sample_rows, "sample_salt": "large",
                              "reference": "truth", "reveal_metrics": ["ari"], "criteria_options": {}},
                     "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
                     "features": {"name": "none"},
                     "model": {"name": "kmeans", "n_clusters": 5, "init": "k-means++", "n_init": 2, "max_iter": 300,
                               "tol": 1e-4, "extra": {}},
                     "fit": {"name": "clusterer", "device": "cpu", "assign": "fit_rows"}},
            "stages": stages}


@unittest.skipUnless(ROWS > 0, "set SWEEPKIT_LARGE_ROWS (for example 50000, or 1000000 on a node) to run it")
class TestLargeData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "repo"
        cls.root.mkdir()
        shutil.copytree(REPO / "src", cls.root / "src", ignore=shutil.ignore_patterns("__pycache__", "project"))
        shutil.copy(REPO / "project.json", cls.root / "project.json")
        cls.steps = []

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def step(self, label: str, *argv) -> str:
        env = dict(os.environ, PYTHONPATH=str(self.root / "src"))
        started = time.perf_counter()
        done = subprocess.run([sys.executable, "-c", LAUNCH, *argv], cwd=self.root, env=env, capture_output=True,
                              text=True, timeout=6 * 3600)
        seconds = time.perf_counter() - started
        peak = next((float(line.split()[1]) for line in done.stderr.splitlines() if line.startswith("PEAK_RSS_MB")),
                    None)
        self.steps.append((label, seconds, peak))
        if done.returncode != 0:
            raise AssertionError(f"{label} exited {done.returncode}:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
        return done.stdout

    def write(self, sweep: dict) -> str:
        path = self.root / f"{sweep['sweep_id']}.json"
        path.write_text(json.dumps(sweep), encoding="utf-8")
        return path.name

    def test_large_data_path(self):
        self.step("make-synthetic", "make-synthetic", "--generator", "blobs", "--n-rows", str(ROWS), "--n-features",
                  "10", "--n-groups", "5", "--noise", "1.0", "--seed", "11", "--name", "large", "--chunk-rows",
                  "100000", "--sample-rows", "1000")
        csv_mb = (self.root / "data" / "synthetic" / "large.csv").stat().st_size / 1e6
        self.step("convert", "convert", "--source", "synthetic/large.csv", "--name", "large", "--sep", "comma",
                  "--row-id", "row_id", "--features", "*", "--exclude", "none", "--keep", "truth", "group",
                  "published_split", "--chunk-rows", "100000", "--to", "store")
        fraction = min(1.0, 20000 / ROWS)
        screen = clustering_sweep(
            "large_screen", {"name": "subsamples", "k": 2, "fraction": fraction, "salt": "large"}, ["sub1", "sub2"],
            [{"name": "base", "kind": "base", "priority": 5, "seeds": [1, 2]},
             {"name": "k", "kind": "one_factor_at_a_time", "priority": 4, "seeds": [1, 2],
              "factors": {"model.n_clusters": [3, 4, 6]}}], 5000)
        self.step("screen on subsamples", "run", "--sweep-file", self.write(screen), "--device", "cpu",
                  "--max-consecutive-failures", "1")
        self.step("analyze", "analyze", "--sweep", "large_screen", "--confidence", "0.95", "--n-boot", "200",
                  "--seed", "1", "--meaningful", "0.75", "--top", "4")
        report = json.loads(sorted((self.root / "artifacts" / "analysis").glob("large_screen_*"))[-1]
                            .joinpath("report.json").read_text(encoding="utf-8"))
        best = report["ranked"][0]
        self.assertEqual(best["label"], "base", "the screen should choose the planted 5 clusters")
        full = clustering_sweep("large_full", {"name": "all"}, ["all"],
                                [{"name": "base", "kind": "base", "priority": 5, "seeds": [1]}], 5000)
        name = self.write(full)
        # parts of at least 10000 rows (past the 5000-row sample the criteria use), at most all the rows
        first = min(0.2, max(0.01, 10000 / ROWS))
        self.step("estimate", "estimate", "--sweep-file", name, "--config", "base", "--device", "cpu", "--fractions",
                  str(first), str(2 * first), str(5 * first), "--workers", "1")
        estimate = json.loads(sorted((self.root / "artifacts" / "estimates").glob("large_full_*"))[-1]
                              .joinpath("report.json").read_text(encoding="utf-8"))
        self.step("full run, every row", "run", "--sweep-file", name, "--device", "cpu", "--max-consecutive-failures",
                  "1")
        run = next((self.root / "artifacts" / "runs" / "large_full").iterdir())
        reveal = json.loads((run / "reveal.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(reveal["metrics"]["ari"], 0.95)
        lines = [f"large-data test, {ROWS} rows, CSV {csv_mb:.0f} MB:"]
        for label, seconds, peak in self.steps:
            lines.append(f"  {label:24s} {seconds:9.1f} s   peak memory "
                         + ("n/a" if peak is None else f"{peak:8.0f} MB"))
        predicted = estimate["per_job"]
        lines.append(f"  estimate of the full run: {predicted['seconds']:.1f} s"
                     + (f", {predicted['peak_memory_mb']:.0f} MB" if "peak_memory_mb" in predicted else ""))
        print("\n".join(lines), file=sys.stderr)
        convert_peak = next(p for label, _, p in self.steps if label == "convert")
        if convert_peak is not None and csv_mb > 200:
            self.assertLess(convert_peak, csv_mb, "convert should read in chunks, never the whole CSV")
        full_peak = next(p for label, _, p in self.steps if label == "full run, every row")
        if full_peak is not None and "peak_memory_mb" in predicted:
            self.assertLess(full_peak, 2 * predicted["peak_memory_mb"],
                            "the full run's peak memory should stay within twice the estimate")


if __name__ == "__main__":
    unittest.main()
