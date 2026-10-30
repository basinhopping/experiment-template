"""ingest: copy a received folder into data/received/<name>/ and record its manifest; verify-data checks a copy
against its manifest again later.

The received folder is only read. Run the same command until it reports "finalized"; each call continues where the
last stopped (--max-seconds bounds one call, for shells that allow a few minutes per command). The phases:

    1. list the received folder (every file and its size) and fix the copy plan
    2. copy every file, hashing the bytes as they are read
    3. read every copy back and check its hash
    4. list the received folder again: nothing may have changed during the copy
    5. list the copy: exactly the planned files, at the planned sizes
    6. write the manifest and a summary

Outputs:
    data/received/<name>/                 the copy
    data/manifests/<name>.csv             path, bytes, SHA-256 of every file
    data/manifests/<name>.json            what was copied, when, by which code (the source path as the copying
                                          machine saw it, which never leaves data/)
    data/_work/ingest_<name>/             resumable state and logs

Settings then point at the copy: {"data": {"name": "table", "path": "received/<name>/<file>", ...}}.
"""
import json
import re
import time
from pathlib import Path

from sweepkit import paths
from sweepkit.data.manifest import read_manifest, verify_tree, write_manifest
from sweepkit.data.transfer import CopyItem, Transfer
from sweepkit.data.walk import ResumableWalk
from sweepkit.hashing import sha256_file
from sweepkit.runs.provenance import collect, utc_now

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def ingest(name: str, source: Path, budget_s: float) -> dict:
    if not NAME_RE.match(name) or name.startswith("."):
        raise ValueError(f"name {name!r}: use letters, digits, _ . - only")
    start = time.monotonic()

    def remaining() -> float:
        return max(budget_s - (time.monotonic() - start), 0.0)

    dest = paths.received_dir() / name
    work = paths.data_work_dir() / f"ingest_{name}"
    manifest_csv = paths.manifests_dir() / f"{name}.csv"
    summary_json = paths.manifests_dir() / f"{name}.json"
    if summary_json.exists():
        return {"phase": "already_ingested", "summary": str(summary_json)}
    source = Path(source).expanduser()
    if not source.is_dir():
        raise NotADirectoryError(f"{source} is not a folder")
    if paths.data_dir().resolve() in source.resolve().parents or source.resolve() == paths.data_dir().resolve():
        raise ValueError(f"{source} is inside data/ already; ingest copies folders from outside it")
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "calls.log", "a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()}\t{source}\n")

    transfer = Transfer(work, dest, {"received": source})
    if not transfer.has_plan():
        listing = ResumableWalk(source, work / "list_source")
        if not listing.run(remaining()):
            return {"phase": "1/6 listing the received folder", **listing.progress()}
        sizes = listing.sizes()
        if not sizes:
            raise RuntimeError(f"{source} holds no files")
        transfer.write_plan([CopyItem(rel, "received", rel, size) for rel, size in sorted(sizes.items())])

    status = transfer.status()
    if status["n_verified"] < status["n_planned"]:
        status = transfer.run(remaining())
        if status["phase"] != "done":
            status["phase"] = {"copying": "2/6 copying", "verifying": "3/6 reading copies back"}[status["phase"]]
            return status

    planned = {item.dest: item.size for item in transfer.load_plan()}
    again = ResumableWalk(source, work / "list_source_again")
    if not again.run(remaining()):
        return {"phase": "4/6 listing the received folder again", **again.progress()}
    if again.sizes() != planned:
        now = again.sizes()
        changed = sorted(k for k in set(now) | set(planned) if now.get(k) != planned.get(k))
        raise RuntimeError(f"{source} changed during the ingest: {changed[:10]}")
    copy_listing = ResumableWalk(dest, work / "list_copy")
    if not copy_listing.run(remaining()):
        return {"phase": "5/6 listing the copy", **copy_listing.progress()}
    if copy_listing.sizes() != planned:
        now = copy_listing.sizes()
        differ = sorted(k for k in set(now) | set(planned) if now.get(k) != planned.get(k))
        raise RuntimeError(f"{dest} differs from the plan: {differ[:10]}")

    rows = transfer.manifest_rows()
    write_manifest(manifest_csv, rows)
    _, manifest_sha = sha256_file(manifest_csv)
    summary = {
        "name": name,
        "copy": dest.relative_to(paths.repo_root()).as_posix(),
        "manifest": manifest_csv.relative_to(paths.repo_root()).as_posix(),
        "manifest_sha256": manifest_sha,
        "n_files": len(rows),
        "bytes": sum(planned.values()),
        "source_as_seen_by_the_copying_machine": str(source),
        "checks": {
            "every_copy_read_back_and_its_hash_matched": True,
            "received_folder_unchanged_during_the_ingest": True,
            "copy_holds_exactly_the_planned_files_at_the_planned_sizes": True,
        },
        "finalized_utc": utc_now(),
        "provenance": collect("sweepkit.cli ingest"),
    }
    with open(summary_json, "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    return {"phase": "finalized", "summary": str(summary_json), "n_files": len(rows),
            "bytes": summary["bytes"], "manifest_sha256": manifest_sha}


def verify(name: str) -> dict:
    """Re-hash every file of data/received/<name>/ against its manifest."""
    manifest_csv = paths.manifests_dir() / f"{name}.csv"
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"no manifest data/manifests/{name}.csv; ingest the folder first")
    report = verify_tree(paths.received_dir() / name, read_manifest(manifest_csv), frozenset())
    _, sha = sha256_file(manifest_csv)
    return {"name": name, "manifest_sha256": sha, **report}
