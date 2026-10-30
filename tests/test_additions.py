"""check-additions on the template itself, and scaffold end to end in a temporary copy of the repository.

check-additions refuses a registered component or stage kind without an example sweep that uses it or a test that
names it; the template must pass its own check. The scaffold test writes one component of every kind into a copy (the
project's own folder, src/project/, and once into the template's own folders), then runs what a user would run next:
the new tests, check-additions, make-example-data and smoke on each new example sweep. Every command runs as a
separate process, as from a terminal. The copy leaves out the project's own files (src/project/, sweeps/project/,
tests/test_project_*.py), and the new components' names start with zz_scaffold_, so a project's own additions never
collide with them.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO, load_catalog

KINDS = ("clusterer", "embedder", "detector", "estimator", "prep", "metric", "split", "dataset", "stage_kind",
         "torch_model")
PREFIX = "zz_scaffold_"


class TestCheckAdditions(unittest.TestCase):
    def test_every_component_has_an_example_that_uses_it_and_a_test_that_names_it(self):
        load_catalog()
        from sweepkit.scaffold.additions import check_additions

        result = check_additions()
        self.assertEqual(result["problems"], [], f"{len(result['problems'])} problems")
        self.assertGreater(result["checked"], 100)

    def test_a_stage_kind_without_an_example_or_a_test_is_reported(self):
        load_catalog()
        from sweepkit.scaffold.additions import check_additions
        from sweepkit.sweep.design import STAGE_KINDS, register_stage_kind

        name = "_".join(("unexampled", "kind"))          # built here, so that no test names it in quotes
        register_stage_kind(name, ())(lambda stage, base, context: [])
        try:
            problems = check_additions()["problems"]
        finally:
            del STAGE_KINDS[name]
        self.assertIn(f"stage kind {name}: no example sweep has a stage of this kind", problems)
        self.assertIn(f"stage kind {name}: no test under tests/ names it", problems)
        self.assertEqual(len(problems), 2, problems)


def run(command: list, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(cwd / "src"))
    env.pop("SWEEPKIT_REQUIRE", None)
    done = subprocess.run([sys.executable, *command], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    if check and done.returncode != 0:
        raise AssertionError(f"{command} exited {done.returncode}:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
    return done


class TestScaffold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "repo"
        cls.root.mkdir()
        for item in ("src", "tests", "sweeps", "templates", "project.json"):
            source = REPO / item
            if source.is_dir():
                shutil.copytree(source, cls.root / item,
                                ignore=shutil.ignore_patterns("__pycache__", "project", "test_project_*"))
            else:
                shutil.copy(source, cls.root / item)
        cls.written = {}
        for kind in KINDS:
            out = run(["-m", "sweepkit.cli", "scaffold", "--kind", kind, "--name", f"{PREFIX}{kind}", "--where",
                       "project"], cls.root).stdout
            cls.written[kind] = out
        run(["-m", "sweepkit.cli", "scaffold", "--kind", "clusterer", "--name", f"{PREFIX}tmpl", "--where",
             "template"], cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files_and_catalog_lines(self):
        for kind in KINDS:
            self.assertTrue((self.root / "src" / "project" / f"{PREFIX}{kind}.py").is_file(), kind)
            self.assertTrue((self.root / "tests" / f"test_project_{PREFIX}{kind}.py").is_file(), kind)
            has_example = kind != "metric"
            self.assertEqual((self.root / "sweeps" / "project" / f"{PREFIX}{kind}.json").is_file(), has_example, kind)
        catalog = (self.root / "src" / "project" / "catalog.py").read_text(encoding="utf-8")
        listed = [line.strip().strip('",') for line in catalog.splitlines() if line.strip().startswith('"project.')]
        self.assertEqual(listed, [f"project.{PREFIX}{kind}" for kind in KINDS])
        template = (self.root / "src" / "sweepkit" / "catalog.py").read_text(encoding="utf-8")
        self.assertIn(f'"sweepkit.models.cluster.{PREFIX}tmpl"', template)
        self.assertTrue((self.root / "sweeps" / "examples" / f"{PREFIX}tmpl.json").is_file())
        self.assertTrue((self.root / "tests" / f"test_clusterer_{PREFIX}tmpl.py").is_file())

    def test_refusals(self):
        again = run(["-m", "sweepkit.cli", "scaffold", "--kind", "clusterer", "--name", f"{PREFIX}clusterer",
                     "--where", "project"], self.root, check=False)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("registered already", again.stdout + again.stderr)
        for name in ("Bad", "kmeans", "x", "trailing_"):
            done = run(["-m", "sweepkit.cli", "scaffold", "--kind", "clusterer", "--name", name, "--where", "project"],
                       self.root, check=False)
            self.assertNotEqual(done.returncode, 0, name)
        done = run(["-m", "sweepkit.cli", "scaffold", "--kind", "no_such_kind", "--name", "my_net", "--where",
                    "project"], self.root, check=False)
        self.assertNotEqual(done.returncode, 0)

    def test_new_tests_pass_and_check_additions_names_no_new_component(self):
        tests = [f"tests.test_project_{PREFIX}{kind}" for kind in KINDS] + [f"tests.test_clusterer_{PREFIX}tmpl"]
        done = run(["-m", "unittest", *tests], self.root)
        self.assertIn("OK", done.stderr)
        report = run(["-m", "sweepkit.cli", "check-additions", "--show", "1000"], self.root, check=False).stdout
        new = [line for line in report.splitlines() if PREFIX in line]
        self.assertEqual(new, [])

    def test_examples_run(self):
        import importlib.util

        run(["-m", "sweepkit.cli", "make-example-data", "--confirm", "yes"], self.root)
        self.assertTrue((self.root / "data" / "synthetic" / f"{PREFIX}dataset_example.npz").is_file())
        # the torch model's example needs torch to run; its file is checked above and its test skips without torch
        skipped = {"metric"} | (set() if importlib.util.find_spec("torch") else {"torch_model"})
        examples = [self.root / "sweeps" / "project" / f"{PREFIX}{kind}.json" for kind in KINDS if kind not in skipped]
        examples.append(self.root / "sweeps" / "examples" / f"{PREFIX}tmpl.json")
        for path in examples:
            done = run(["-m", "sweepkit.cli", "smoke", "--sweep-file", str(path.relative_to(self.root)), "--device",
                        "cpu", "--rows", "30"], self.root, check=False)
            self.assertIn("SMOKE OK", done.stdout, f"{path.name}:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
            self.assertNotIn("identical models", done.stdout, path.name)
            json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
