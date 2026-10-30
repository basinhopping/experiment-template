"""The payload tool (src/sweepkit/payload.py), on a small repository in a temporary folder standing in for the person's
computer and a second folder standing in for the cloud container: a snapshot of the code, work on an unpacked copy,
and the payload that installs only onto the files it was built from. The tool runs as a script, as it does in use,
with the Python the tests run on and, for one round trip, the oldest Python 3.8 to 3.11 present.
"""
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO

TOOL = REPO / "src" / "sweepkit" / "payload.py"


def tool(*args, python=None):
    return subprocess.run([python or sys.executable, *map(str, args)], capture_output=True, text=True, timeout=300)


def contents(root: Path) -> dict:
    """Every file under root with its SHA-256, to show that a refusal changed nothing."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class TestPayload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.pc = root / "pc"
        self.box = root / "box"
        files = {
            "README.md": "# a project\n",
            "project.json": '{"project": "demo", "discipline": "enforced"}\n',
            "src/sweepkit/core.py": "VALUE = 'core'\n",
            "src/project/model.py": "def fit():\n    return 1\n",
            "docs/NOTES.md": "notes\n",
            "tests/test_core.py": "import unittest\n",
            "notebooks/look.ipynb": "{}\n",
            "data/rows.csv": "a,b\n1,2\n",
        }
        for rel, text in files.items():
            (self.pc / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.pc / rel).write_text(text, encoding="utf-8")
        (self.pc / "src" / "sweepkit" / "__pycache__").mkdir()
        (self.pc / "src" / "sweepkit" / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"\0compiled")

    def snapshot(self, *paths, python=None) -> Path:
        done = tool(TOOL, "snapshot", self.pc, self.box / "in", *paths, python=python)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        record = sorted((self.box / "in").glob("snapshot_*.json"))[-1]
        archive = record.parent / (record.stem + ".tar.gz")
        expected = (record.parent / (archive.name + ".sha256")).read_text().split()[0]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)
        unpacked = self.box / ("work_" + record.stem)
        unpacked.mkdir()
        unpack(archive, unpacked)
        checked = tool(TOOL, "check", record, unpacked, python=python)
        self.assertEqual(checked.returncode, 0, checked.stdout)
        self.assertIn("SNAPSHOT OK", checked.stdout)
        return record

    def make(self, record: Path, *paths, python=None) -> Path:
        unpacked = self.box / ("work_" + record.stem)
        done = tool(TOOL, "make", record, unpacked, self.box / "out", *paths, python=python)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        archive = sorted((self.box / "out").glob("payload_*.tar.gz"))[-1]
        expected = (archive.parent / (archive.name + ".sha256")).read_text().split()[0]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)
        landing = self.pc / "artifacts" / "payloads"
        landing.mkdir(parents=True, exist_ok=True)
        unpack(archive, landing)
        return landing / archive.name[: -len(".tar.gz")]

    def work(self, record: Path):
        unpacked = self.box / ("work_" + record.stem)
        (unpacked / "docs" / "NOTES.md").write_text("notes, revised\n", encoding="utf-8")
        (unpacked / "src" / "project" / "extra.py").write_text("EXTRA = True\n", encoding="utf-8")
        (unpacked / "tests" / "test_core.py").unlink()
        (unpacked / "src" / "sweepkit" / "__pycache__").mkdir(exist_ok=True)
        (unpacked / "src" / "sweepkit" / "__pycache__" / "x.pyc").write_bytes(b"left by a test run")
        reading = unpacked / "artifacts" / "analyses" / "r1_2026-10-27"
        reading.mkdir(parents=True)
        (reading / "README.md").write_text("what was read\n", encoding="utf-8")
        return unpacked

    def test_the_digest_and_the_code_items_are_the_template_s(self):
        from sweepkit import hashing, payload

        self.assertEqual(payload.CODE_ITEMS, hashing.CODE_ITEMS)
        self.assertEqual(payload.tree_digest(REPO), hashing.tree_digest(REPO))
        self.assertEqual(payload.tree_digest(self.pc), hashing.tree_digest(self.pc))

    def test_standard_library_only_and_python_3_8_syntax(self):
        text = TOOL.read_text(encoding="utf-8")
        tree = ast.parse(text, feature_version=(3, 8))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        self.assertEqual(modules - set(sys.stdlib_module_names), set())
        for name in ("removeprefix", "removesuffix", "is_relative_to", "tomllib"):
            self.assertNotIn(name, text, f"{name} needs a Python newer than 3.8")

    def test_round_trip_installs_the_work_and_checks_it(self):
        record = self.snapshot()
        unpacked = self.work(record)
        folder = self.make(record, "artifacts/analyses")
        before_data = (self.pc / "data" / "rows.csv").read_bytes()
        done = tool(folder / "payload.py", "install", self.pc)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("5. TREE OK", done.stdout)
        from sweepkit.hashing import tree_digest

        self.assertEqual(tree_digest(self.pc), tree_digest(unpacked))
        self.assertEqual((self.pc / "docs" / "NOTES.md").read_text(), "notes, revised\n")
        self.assertEqual((self.pc / "src" / "project" / "extra.py").read_text(), "EXTRA = True\n")
        self.assertEqual((self.pc / "artifacts" / "analyses" / "r1_2026-10-27" / "README.md").read_text(),
                         "what was read\n")
        self.assertFalse((self.pc / "tests" / "test_core.py").exists())
        self.assertTrue((folder / "removed" / "tests" / "test_core.py").is_file())
        self.assertEqual((self.pc / "data" / "rows.csv").read_bytes(), before_data)
        self.assertEqual((folder / "NEW.txt").read_text().split(),
                         ["artifacts/analyses/r1_2026-10-27/README.md", "src/project/extra.py"])
        self.assertNotIn("pyc", (folder / "MANIFEST.sha256").read_text())
        again = tool(folder / "payload.py", "install", self.pc)
        self.assertEqual(again.returncode, 0, again.stdout)
        self.assertIn("4 already in their new state", again.stdout)

    def test_refusals_change_nothing(self):
        record = self.snapshot()
        self.work(record)
        folder = self.make(record, "artifacts/analyses")
        notes = self.pc / "docs" / "NOTES.md"
        cases = {
            "a replaced file edited since": (notes, "edited on the computer\n", "step 2", "docs/NOTES.md: changed"),
            "a removed file edited since": (self.pc / "tests" / "test_core.py", "import os\n", "step 2",
                                            "tests/test_core.py: changed"),
            "a new file already there": (self.pc / "src" / "project" / "extra.py", "OTHER = 1\n", "step 2",
                                         "src/project/extra.py: exists with other content"),
            "other code changed since": (self.pc / "src" / "project" / "model.py", "def fit():\n    return 2\n",
                                         "step 3", "src/project/model.py: changed since the snapshot"),
            "other code added since": (self.pc / "docs" / "MORE.md", "more\n", "step 3",
                                       "docs/MORE.md: added since the snapshot"),
        }
        for label, (path, text, step, says) in cases.items():
            with self.subTest(label):
                old = path.read_bytes() if path.exists() else None
                path.write_text(text, encoding="utf-8")
                state = contents(self.pc)
                done = tool(folder / "payload.py", "install", self.pc)
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn(f"REFUSED: {step}", done.stdout)
                self.assertIn(says, done.stdout)
                self.assertIn("nothing was changed", done.stdout)
                self.assertEqual(contents(self.pc), state)
                if old is None:
                    path.unlink()
                else:
                    path.write_bytes(old)
        damaged = folder / "payload" / "docs" / "NOTES.md"
        damaged.write_text("damaged in transit\n", encoding="utf-8")
        state = contents(self.pc)
        done = tool(folder / "payload.py", "install", self.pc)
        self.assertIn("REFUSED: step 1, the payload is damaged", done.stdout)
        self.assertIn("differs from MANIFEST.sha256: docs/NOTES.md", done.stdout)
        self.assertEqual(contents(self.pc), state)

    def test_make_leaves_out_what_tools_leave_behind_and_says_when_nothing_changed(self):
        record = self.snapshot()
        unpacked = self.box / ("work_" + record.stem)
        nothing = tool(TOOL, "make", record, unpacked, self.box / "out")
        self.assertEqual(nothing.returncode, 0)
        self.assertIn("nothing changed", nothing.stdout)
        self.assertFalse((self.box / "out").exists())
        # opening a notebook and browsing the folder leave files that are not code: still nothing to send
        checkpoint = unpacked / "notebooks" / ".ipynb_checkpoints" / "look-checkpoint.ipynb"
        checkpoint.parent.mkdir()
        checkpoint.write_text("{}\n", encoding="utf-8")
        (unpacked / "docs" / ".DS_Store").write_bytes(b"\0")
        (unpacked / "src" / "sweepkit.egg-info").mkdir()
        (unpacked / "src" / "sweepkit.egg-info" / "PKG-INFO").write_text("x\n", encoding="utf-8")
        (unpacked / "src" / "odd.pyc").mkdir()
        (unpacked / "src" / "odd.pyc" / "inside.txt").write_text("x\n", encoding="utf-8")
        left = tool(TOOL, "make", record, unpacked, self.box / "out")
        self.assertEqual(left.returncode, 0)
        self.assertIn("nothing changed", left.stdout)
        self.assertEqual(tool(TOOL, "check", record, unpacked).returncode, 0)

    def test_check_finds_a_changed_or_extra_file(self):
        record = self.snapshot()
        unpacked = self.box / ("work_" + record.stem)
        (unpacked / "README.md").write_text("# changed\n", encoding="utf-8")
        (unpacked / "docs" / "EXTRA.md").write_text("extra\n", encoding="utf-8")
        done = tool(TOOL, "check", record, unpacked)
        self.assertEqual(done.returncode, 1)
        self.assertIn("differs: README.md", done.stdout)
        self.assertIn("not in the snapshot: docs/EXTRA.md", done.stdout)

    def test_a_snapshot_of_some_paths_has_no_tree_digest(self):
        record = self.snapshot("docs")
        self.assertFalse(json.loads(record.read_text())["covers_code_items"])
        unpacked = self.box / ("work_" + record.stem)
        (unpacked / "docs" / "NOTES.md").write_text("notes, revised\n", encoding="utf-8")
        folder = self.make(record)
        self.assertFalse((folder / "DIGEST.txt").exists())
        done = tool(folder / "payload.py", "install", self.pc)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("3. no DIGEST.txt", done.stdout)
        self.assertEqual((self.pc / "docs" / "NOTES.md").read_text(), "notes, revised\n")

    def test_a_round_trip_on_the_oldest_python_present(self):
        older = next((shutil.which(f"python3.{minor}") for minor in (8, 9, 10, 11)
                      if shutil.which(f"python3.{minor}")), None)
        if older is None:
            self.skipTest("no Python 3.8 to 3.11 here")
        record = self.snapshot(python=older)
        self.work(record)
        folder = self.make(record, "artifacts/analyses", python=older)
        done = tool(folder / "payload.py", "install", self.pc, python=older)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("5. TREE OK", done.stdout)


def unpack(archive: Path, folder: Path) -> None:
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(folder, filter="data")
        else:
            tar.extractall(folder)


if __name__ == "__main__":
    unittest.main()
