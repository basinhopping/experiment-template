"""Hashes for files, settings and code. Used for manifests, run IDs, provenance and the code-tree digest."""
import hashlib
import json
from pathlib import Path

# Folders of a working tree that are not code: never part of the code-tree digest (tree_digest).
NOT_CODE = ("data", "artifacts", "queue", "logs", "reports", ".git", "CODE_VERSIONS.log")
# What a push sends to a node and what the code-tree digest covers: the code, the sweeps, the scripts, the docs and
# the templates. A project's notes (STATE.md, PROJECT.md, findings.csv, ...) stay where they were written, so the
# same code gives the same digest on every machine whatever notes sit beside it. node/tree_digest.sh and the push
# scripts list the same items.
CODE_ITEMS = ("src", "tests", "sweeps", "node", "env", "docs", "templates", "notebooks", "README.md", "START_HERE.md",
              "CLAUDE.md", "TESTING.md", "LICENSE", "project.json", ".gitignore", ".gitattributes")


def sha256_file(path: Path) -> tuple:
    """Return (number of bytes, SHA-256 hex) of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    n_bytes = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            n_bytes += len(chunk)
    return n_bytes, digest.hexdigest()


def canonical_json(obj) -> str:
    """JSON with sorted keys and no whitespace, so equal content always gives equal text.

    NaN and infinity are refused: they have no stable JSON form.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(canonical_json(obj))


def code_hash(package_dir: Path) -> str:
    """SHA-256 over every .py file of a package: sorted relative paths and file contents.

    Two trees give the same hash exactly when their Python files are identical.
    """
    digest = hashlib.sha256()
    files = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    if not files:
        raise RuntimeError(f"no .py files under {package_dir}")
    for path in files:
        rel = path.relative_to(package_dir).as_posix()
        _, file_digest = sha256_file(path)
        digest.update(f"{rel}\0{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()


# What tools leave in a working copy, which is not code and .gitignore keeps out of git: opening a notebook writes
# .ipynb_checkpoints, a file browser .DS_Store or Thumbs.db. The digest leaves them out, so they do not make a copy
# look changed (node/tree_digest.sh and payload.py leave out the same names).
TOOL_LEFTOVERS = ("__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store",
                  "Thumbs.db")


def left_by_a_tool(parts) -> bool:
    return any(part in TOOL_LEFTOVERS or part.endswith((".egg-info", ".pyc")) for part in parts)


def tree_files(root: Path) -> list:
    """Every file of the code items (CODE_ITEMS) under root, without what tools leave (__pycache__, compiled files,
    notebook checkpoints, file-browser files, TOOL_LEFTOVERS). Paths relative to root, as POSIX text."""
    out = []
    root = Path(root)
    for item in CODE_ITEMS:
        top = root / item
        if top.is_file():
            out.append(item)
            continue
        if not top.is_dir():
            continue
        for path in top.rglob("*"):
            rel = path.relative_to(root)
            if left_by_a_tool(rel.parts) or not path.is_file():
                continue
            out.append(rel.as_posix())
    return out


def tree_digest(root: Path) -> str:
    """The code-tree digest: the same number as node/tree_digest.sh computes with

        find <the code items> <prune TOOL_LEFTOVERS, *.egg-info and *.pyc> -type f -print0 | LC_ALL=C sort -z
            | xargs -0 sha256sum
            | cut -c1-16,65- | sha256sum

    Each file contributes one line "<first 16 hex of its SHA-256>  ./<path>", sorted bytewise by path. Windows, a Mac
    and the node compute the same digest for the same files, so a person can check that a copy arrived complete.
    """
    root = Path(root)
    names = sorted(("./" + rel for rel in tree_files(root)), key=lambda s: s.encode("utf-8"))
    lines = []
    for name in names:
        _, digest = sha256_file(root / name[2:])
        lines.append(f"{digest[:16]}  {name}\n")
    return sha256_text("".join(lines))
