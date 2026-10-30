"""Shared helpers for the tests."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def required() -> set:
    return {x.strip() for x in os.environ.get("SWEEPKIT_REQUIRE", "").split(",") if x.strip()}


def _skip_or_fail(reason: str, must_run: bool):
    """Skip, or, when this machine must run the test (SWEEPKIT_REQUIRE), fail every test it covers. Works on test
    methods and on whole test classes."""
    if not must_run:
        return unittest.skip(reason)
    message = f"{reason}, but SWEEPKIT_REQUIRE says this machine must run it"

    def decorator(obj):
        if isinstance(obj, type):
            obj.setUpClass = classmethod(lambda cls: None)

            def setUp(self):
                self.fail(message)

            obj.setUp = setUp
            return obj

        def failing(self, *args, **kwargs):
            self.fail(message)

        failing.__name__ = obj.__name__
        return failing

    return decorator


def needs_module(module: str):
    """Skip when a module is missing, unless SWEEPKIT_REQUIRE names it; then fail."""
    if importlib.util.find_spec(module) is not None:
        return lambda obj: obj
    return _skip_or_fail(f"{module} is not installed", module in required())


def needs_gpu():
    """Skip without a visible NVIDIA GPU, unless SWEEPKIT_REQUIRE has gpu."""
    from sweepkit.devices import gpu_visible

    if gpu_visible():
        return lambda obj: obj
    return _skip_or_fail("no GPU is visible", "gpu" in required())


class TempRepo:
    """A context that points sweepkit's paths at a temporary repository copy (artifacts/, queue/, data/), so tests
    never write into the real tree. The code itself is not copied: paths.repo_root is patched."""

    def __enter__(self):
        from unittest import mock

        from sweepkit import paths

        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._dir.name)
        (root / "src" / "sweepkit").mkdir(parents=True)
        (root / "src" / "sweepkit" / "__init__.py").write_text("", encoding="utf-8")
        (root / "project.json").write_text('{"project": "testproject", "discipline": "enforced"}', encoding="utf-8")
        self._patch = mock.patch.object(paths, "repo_root", lambda: root)
        self._patch.start()
        self._package = mock.patch.object(paths, "package_dir", lambda: REPO / "src" / "sweepkit")
        self._package.start()
        self.root = root
        from sweepkit.families.common import clear_caches

        clear_caches()
        return root

    def __exit__(self, *exc):
        import gc

        from sweepkit.families.common import clear_caches

        self._package.stop()
        self._patch.stop()
        # cached tables and memory-mapped stores hold their files open; on a network file system an open file
        # cannot be deleted (the first node run, 2026-10-27, errored here after its tests had passed)
        clear_caches()
        gc.collect()
        self._dir.cleanup()
        return False


def load_catalog():
    from sweepkit import catalog

    catalog.load_all()


def git_ignored(relative: Path) -> bool:
    """Whether .gitignore keeps a path (relative to the repository) out of git, for the plain patterns it uses:
    anchored ones (/data/, /node.local.txt, /.venv*/) and unanchored ones (__pycache__/, *.pyc). The checks of what
    the repository publishes (tests/test_docs.py) skip these, so a person's own login file, state notes, data and
    environments never fail them."""
    import fnmatch

    path = (REPO / ".gitignore")
    if not path.is_file():
        return False
    parts = relative.parts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        anchored, folder, core = line.startswith("/"), line.endswith("/"), line.strip("/")
        if anchored:
            if folder and len(parts) > 1 and fnmatch.fnmatch(parts[0], core):
                return True
            if not folder and fnmatch.fnmatch(relative.as_posix(), core):
                return True
        elif folder:
            if any(fnmatch.fnmatch(part, core) for part in parts[:-1]):
                return True
        elif fnmatch.fnmatch(parts[-1], core):
            return True
    return False
