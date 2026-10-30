"""Hashes: canonical JSON, file hashes, and the code-tree digest computed the same way in Python (any machine) and
in the shell (node/tree_digest.sh), so a person can check that a copy arrived complete."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sweepkit.hashing import canonical_json, sha256_file, tree_digest, tree_files
from tests.support import REPO


class TestHashing(unittest.TestCase):
    def test_canonical_json_sorts_and_refuses_nan(self):
        self.assertEqual(canonical_json({"b": 1, "a": [1, 2]}), '{"a":[1,2],"b":1}')
        with self.assertRaises(ValueError):
            canonical_json({"a": float("nan")})

    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_bytes(b"abc")
            self.assertEqual(sha256_file(p), (3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"))

    def _tree(self, root: Path) -> None:
        (root / "src" / "pkg").mkdir(parents=True)
        (root / "src" / "pkg" / "a.py").write_text("print(1)\n")
        (root / "README.md").write_text("hello\n")
        (root / "node").mkdir()
        (root / "node" / "x.sh").write_text("echo x\n")
        for skipped in ("data", "artifacts", "queue", "logs", "reports", ".venv-main", "src/pkg/__pycache__"):
            (root / skipped).mkdir(parents=True, exist_ok=True)
            (root / skipped / "junk.txt").write_text("not code\n")
        # what tools leave inside the code items: not code, so not in the digest
        for left in ("notebooks/.ipynb_checkpoints/walk-checkpoint.ipynb", "src/pkg/.pytest_cache/v",
                     "src/sweepkit.egg-info/PKG-INFO", "docs/.DS_Store", "docs/Thumbs.db", "src/pkg/old.pyc",
                     "src/pkg/odd.pyc/inside.txt"):        # find prunes a folder named *.pyc as well
            (root / left).parent.mkdir(parents=True, exist_ok=True)
            (root / left).write_text("left by a tool\n")
        (root / "node.local.txt").write_text("someone@somewhere\n")
        (root / "STATE.md").write_text("notes stay where they were written\n")
        (root / "project.json").write_text('{"project": "p", "discipline": "enforced"}')

    def test_tree_files_hold_the_code_items_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._tree(root)
            self.assertEqual(sorted(tree_files(root)), ["README.md", "node/x.sh", "project.json", "src/pkg/a.py"])

    @unittest.skipIf(shutil.which("bash") is None or shutil.which("sha256sum") is None, "no bash or sha256sum here")
    @unittest.skipIf(not (REPO / "node" / "tree_digest.sh").is_file(), "node/ was removed from this copy")
    def test_python_and_shell_digests_agree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._tree(root)
            shell = subprocess.run(["bash", str(REPO / "node" / "tree_digest.sh"), str(root)], capture_output=True,
                                   text=True, check=True).stdout.strip()
            self.assertEqual(shell, tree_digest(root))
            (root / "README.md").write_text("changed\n")
            self.assertNotEqual(shell, tree_digest(root))

    def test_crlf_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._tree(root)
            before = tree_digest(root)
            (root / "README.md").write_bytes(b"hello\r\n")
            self.assertNotEqual(before, tree_digest(root), "a checkout that turned LF into CRLF must show")


if __name__ == "__main__":
    unittest.main()
