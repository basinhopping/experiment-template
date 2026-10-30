"""Move work between a cloud session's container and the person's computer as one checked payload.

A cloud session with the person's folder linked (CLAUDE.md, "From a cloud container to the person's computer") keeps
the repository on the computer and edits it there. Work that needs the container starts from a snapshot of the
computer's files and comes back as a payload that installs only onto the files it was built from:

    on the computer:   python3 src/sweepkit/payload.py snapshot <repository> <folder> [<path> ...]
    in the container:  python3 <unpacked>/src/sweepkit/payload.py check <snapshot>.json <unpacked>
                       (the work, in <unpacked>)
                       python3 <unpacked>/src/sweepkit/payload.py make <snapshot>.json <unpacked> <folder> [<path> ...]
    on the computer:   python3 <payload folder>/payload.py install <repository>

snapshot  packs the code items (CODE_ITEMS: the same list as sweepkit.hashing and the push scripts), or the paths
          given, into <folder>/snapshot_<utc>.tar.gz, with snapshot_<utc>.tar.gz.sha256 and snapshot_<utc>.json
          (each file's SHA-256, the paths, and the code-tree digest when every code item was packed).
check     confirms that an unpacked snapshot has exactly the record's files, each with its SHA-256.
make      compares the worked folder with the record, under the snapshot's paths and any paths added (a reading
          folder made in the container, for example), and writes <folder>/payload_<utc>/: payload/ (each changed or
          new file at its path), MANIFEST.sha256 (their SHA-256), BEFORE.sha256 (each replaced file's SHA-256 in the
          snapshot), NEW.txt, REMOVED.sha256 (files removed in the container, with their SHA-256 in the snapshot),
          SNAPSHOT.json, DIGEST.txt (the worked folder's code-tree digest, when the snapshot covered every code item)
          and a copy of this file; then payload_<utc>.tar.gz and its .sha256 beside that folder. It refuses files a
          tool leaves behind (.ipynb_checkpoints, *.egg-info and the like) among the new files.
install   refuses, changing nothing, unless every payload file matches MANIFEST.sha256, every file it replaces or
          removes is still the snapshot's version (or already in its new state), every new file is absent or
          identical, and the code items with the payload applied give DIGEST.txt, so nothing else in the code
          changed since the snapshot. Then it writes each file over in place, moves each removed file aside into the
          payload folder's removed/ (it never deletes), checks every file against the manifest, and compares the
          repository's code-tree digest with DIGEST.txt. Running it again after it succeeded changes nothing.

Standard library only, and Python 3.8 or newer, so it runs in the computer's shell without the project's
environment. The code-tree digest is the number sweepkit.hashing.tree_digest, node/tree_digest.sh and the
tree-digest command give.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

CODE_ITEMS = ("src", "tests", "sweeps", "node", "env", "docs", "templates", "notebooks", "README.md", "START_HERE.md",
              "CLAUDE.md", "TESTING.md", "LICENSE", "project.json", ".gitignore", ".gitattributes")
# What editors, notebooks and installers leave beside the work: never part of a payload's new files.
LEFT_BEHIND = (".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store", "Thumbs.db")


def left_by_a_tool(parts) -> bool:
    """What tools leave in a working copy (caches, compiled files, notebook checkpoints, file-browser files): not
    code, so no snapshot, payload or digest counts it (the same names as sweepkit.hashing.TOOL_LEFTOVERS)."""
    return any(part == "__pycache__" or part in LEFT_BEHIND or part.endswith((".egg-info", ".pyc")) for part in parts)


class Refused(Exception):
    """A check failed; nothing was changed."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(rel: str) -> str:
    """A path inside the repository, as POSIX text: not absolute, no '..', no empty parts."""
    path = PurePosixPath(rel.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts) or ":" in rel:
        raise Refused(f"not a path inside the repository: {rel!r}")
    return path.as_posix()


def is_code(rel: str) -> bool:
    return rel.split("/", 1)[0] in CODE_ITEMS


def files_under(root: Path, paths) -> list:
    """Every file under the paths (relative to root), without __pycache__ folders and compiled files, as sorted POSIX
    paths relative to root: for the code items, the same files as sweepkit.hashing.tree_files."""
    out = set()
    root = Path(root)
    for item in paths:
        top = root / item
        if top.is_file():
            out.add(PurePosixPath(Path(item).as_posix()).as_posix())
            continue
        if not top.is_dir():
            continue
        for path in top.rglob("*"):
            rel = path.relative_to(root)
            if left_by_a_tool(rel.parts) or not path.is_file():
                continue
            out.add(rel.as_posix())
    return sorted(out)


def digest_of(hashes: dict) -> str:
    """The code-tree digest from {path: SHA-256} of the code items' files: one line '<first 16 hex>  ./<path>' per
    file, sorted bytewise by path, then the SHA-256 of the lines."""
    names = sorted(hashes, key=lambda rel: ("./" + rel).encode("utf-8"))
    text = "".join(f"{hashes[rel][:16]}  ./{rel}\n" for rel in names)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hashes(root: Path) -> dict:
    return {rel: sha256_of(Path(root) / rel) for rel in files_under(root, CODE_ITEMS)}


def tree_digest(root: Path) -> str:
    return digest_of(code_hashes(root))


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read_sums(path: Path) -> dict:
    """{path: SHA-256} from a file in sha256sum's format ('<hash>  <path>' per line)."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line[:64], line[66:]
        if len(digest) != 64 or line[64:66] not in ("  ", " *"):
            raise Refused(f"{path.name}: not a sha256sum line: {line!r}")
        out[safe_relative(rel)] = digest.lower()
    return out


def sums_text(hashes: dict, names) -> str:
    return "".join(f"{hashes[rel]}  {rel}\n" for rel in names)


def read_lines(path: Path) -> list:
    if not path.is_file():
        return []
    return [safe_relative(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def free_name(folder: Path, prefix: str, suffixes) -> str:
    """<prefix>_<utc>, waiting a second while something of that name exists, so two calls never share a name."""
    while True:
        name = f"{prefix}_{utc_stamp()}"
        if not any((folder / (name + suffix)).exists() for suffix in suffixes):
            return name
        time.sleep(1)


def write_archive_hash(archive: Path) -> str:
    digest = sha256_of(archive)
    write_text(archive.with_name(archive.name + ".sha256"), f"{digest}  {archive.name}\n")
    return digest


# ----------------------------------------------------------------------------------------------------------------
# snapshot and check


def snapshot(repository: Path, folder: Path, paths) -> Path:
    """Pack the code items (or the paths given) of the repository with a record of every file's SHA-256."""
    repository = Path(repository).resolve()
    folder = Path(folder)
    if not repository.is_dir():
        raise Refused(f"no folder {repository}")
    present = [item for item in CODE_ITEMS if (repository / item).exists()]
    paths = [safe_relative(p) for p in paths] if paths else present
    files = files_under(repository, paths)
    if not files:
        raise Refused(f"no files under {paths} in {repository}")
    folder.mkdir(parents=True, exist_ok=True)
    name = free_name(folder, "snapshot", (".tar.gz", ".json"))
    archive = folder / f"{name}.tar.gz"
    hashes = {}
    with tarfile.open(archive, "w:gz") as tar:
        for rel in files:
            hashes[rel] = sha256_of(repository / rel)
            tar.add(str(repository / rel), arcname=rel, recursive=False)
    covered = all(item in paths for item in present)
    record = {"name": name, "created_utc": utc_stamp(), "repository": repository.name, "paths": paths,
              "covers_code_items": covered,
              "tree_digest": digest_of({r: h for r, h in hashes.items() if is_code(r)}) if covered else None,
              "files": hashes}
    write_text(folder / f"{name}.json", json.dumps(record, indent=1, sort_keys=True) + "\n")
    digest = write_archive_hash(archive)
    print(f"snapshot {archive}: {len(files)} files, sha256 {digest}")
    print(f"record {folder / (name + '.json')}"
          + (f"; code-tree digest {record['tree_digest']}" if covered else "; not every code item: no tree digest"))
    print(f"stage {name}.tar.gz, {name}.tar.gz.sha256 and {name}.json; in the container, check the archive's "
          f"SHA-256, unpack it into a new empty folder and run: payload.py check {name}.json <that folder>")
    return archive


def check(record_path: Path, folder: Path) -> None:
    """An unpacked snapshot holds exactly the record's files, each with its SHA-256."""
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    expected = record["files"]
    present = set(files_under(Path(folder), record["paths"]))
    problems = [f"missing: {rel}" for rel in sorted(set(expected) - present)]
    problems += [f"not in the snapshot: {rel}" for rel in sorted(present - set(expected))]
    problems += [f"differs: {rel}" for rel in sorted(set(expected) & present)
                 if sha256_of(Path(folder) / rel) != expected[rel]]
    if problems:
        raise Refused("SNAPSHOT DIFFERS from its record:\n  " + "\n  ".join(problems))
    print(f"SNAPSHOT OK: {len(expected)} files in {folder} match {Path(record_path).name}")


# ----------------------------------------------------------------------------------------------------------------
# make


def make(record_path: Path, worked: Path, folder: Path, extra_paths) -> Path | None:
    """The payload that turns the snapshot into the worked folder, with the checks install needs."""
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    worked = Path(worked)
    folder = Path(folder)
    base = record["files"]
    paths = list(record["paths"]) + [safe_relative(p) for p in extra_paths if safe_relative(p) not in record["paths"]]
    current = {rel: sha256_of(worked / rel) for rel in files_under(worked, paths)}
    changed = sorted(rel for rel in current if rel in base and current[rel] != base[rel])
    new = sorted(rel for rel in current if rel not in base)
    removed = sorted(rel for rel in base if rel not in current)
    if not (changed or new or removed):
        print(f"nothing changed since {record['name']}: no payload written")
        return None
    folder.mkdir(parents=True, exist_ok=True)
    name = free_name(folder, "payload", ("", ".tar.gz"))
    out = folder / name
    (out / "payload").mkdir(parents=True)
    shipped = sorted(changed + new)
    for rel in shipped:
        target = out / "payload" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(worked / rel, target)
        if sha256_of(target) != current[rel]:
            raise Refused(f"{rel} changed while the payload was being made; make it again")
    write_text(out / "MANIFEST.sha256", sums_text(current, shipped))
    write_text(out / "BEFORE.sha256", sums_text(base, changed))
    write_text(out / "NEW.txt", "".join(f"{rel}\n" for rel in new))
    write_text(out / "REMOVED.sha256", sums_text(base, removed))
    write_text(out / "SNAPSHOT.json", json.dumps(record, indent=1, sort_keys=True) + "\n")
    if record.get("covers_code_items"):
        write_text(out / "DIGEST.txt", digest_of({rel: h for rel, h in current.items() if is_code(rel)}) + "\n")
    shutil.copyfile(Path(__file__).resolve(), out / "payload.py")
    archive = folder / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(str(out), arcname=name)
    digest = write_archive_hash(archive)
    print(f"payload {archive}: {len(changed)} changed, {len(new)} new, {len(removed)} removed; sha256 {digest}")
    for label, names in (("changed", changed), ("new", new), ("removed", removed)):
        for rel in names:
            print(f"  {label}: {rel}")
    if (out / "DIGEST.txt").exists():
        print(f"code-tree digest after the install: {(out / 'DIGEST.txt').read_text(encoding='utf-8').strip()}")
    print(f"send {name}.tar.gz and {name}.tar.gz.sha256 to the computer, check the SHA-256 there, unpack, and run: "
          f"python3 {name}/payload.py install <repository>")
    return archive


# ----------------------------------------------------------------------------------------------------------------
# install


def install(repository: Path, here: Path) -> None:
    """Install the payload in `here` (the folder holding this copy) into the repository, or refuse unchanged."""
    if not (here / "MANIFEST.sha256").is_file() or not (here / "payload").is_dir():
        raise Refused(f"{here} is not an unpacked payload: run the payload.py inside the payload folder")
    repository = Path(repository).resolve()
    if not repository.is_dir():
        raise Refused(f"no folder {repository}")
    manifest = read_sums(here / "MANIFEST.sha256")
    before = read_sums(here / "BEFORE.sha256")
    removed = read_sums(here / "REMOVED.sha256")
    new = read_lines(here / "NEW.txt")
    digest_file = here / "DIGEST.txt"
    expected = digest_file.read_text(encoding="utf-8").strip() if digest_file.is_file() else None
    if set(before) - set(manifest) or set(new) - set(manifest) or set(before) & set(new):
        raise Refused("BEFORE.sha256 and NEW.txt do not match MANIFEST.sha256: the payload is incomplete")

    # 1. the payload against its manifest
    shipped = {p.relative_to(here / "payload").as_posix() for p in (here / "payload").rglob("*") if p.is_file()}
    problems = [f"missing from the payload: {rel}" for rel in sorted(set(manifest) - shipped)]
    problems += [f"not in MANIFEST.sha256: {rel}" for rel in sorted(shipped - set(manifest))]
    problems += [f"differs from MANIFEST.sha256: {rel}" for rel in sorted(set(manifest) & shipped)
                 if sha256_of(here / "payload" / rel) != manifest[rel]]
    if problems:
        raise Refused("step 1, the payload is damaged; nothing was changed:\n  " + "\n  ".join(problems))
    print(f"1. payload: {len(manifest)} files match MANIFEST.sha256")

    # 2. the files it replaces, adds and removes are the versions it was built on, or already in their new state
    done = 0
    for rel, digest in sorted(before.items()):
        target = repository / rel
        if not target.is_file():
            problems.append(f"{rel}: missing (the payload replaces the version {digest[:12]})")
            continue
        now = sha256_of(target)
        if now == manifest[rel]:
            done += 1
        elif now != digest:
            problems.append(f"{rel}: changed since the snapshot (the payload replaces it)")
    for rel in new:
        target = repository / rel
        if target.exists() or target.is_symlink():
            if target.is_file() and sha256_of(target) == manifest[rel]:
                done += 1
            else:
                problems.append(f"{rel}: exists with other content (the payload adds it)")
    for rel, digest in sorted(removed.items()):
        target = repository / rel
        if not target.exists():
            done += 1
        elif not target.is_file() or sha256_of(target) != digest:
            problems.append(f"{rel}: changed since the snapshot (the payload removes it)")
    if problems:
        raise Refused("step 2, the repository does not hold the versions the payload was built on; nothing was "
                      "changed:\n  "
                      + "\n  ".join(problems))
    print(f"2. the repository holds the versions the payload was built on ({len(before)} replaced, {len(new)} new, "
          f"{len(removed)} removed; {done} already in their new state)")

    # 3. nothing else in the code changed since the snapshot
    if expected:
        now = code_hashes(repository)
        would = dict(now)
        would.update({rel: digest for rel, digest in manifest.items() if is_code(rel)})
        for rel in removed:
            would.pop(rel, None)
        if digest_of(would) != expected:
            snapshot_files = json.loads((here / "SNAPSHOT.json").read_text(encoding="utf-8"))["files"]
            then = {rel: digest for rel, digest in snapshot_files.items() if is_code(rel)}
            touched = set(manifest) | set(removed)
            others = []
            for rel in sorted(set(now) | set(then)):
                if rel in touched:
                    continue
                if rel not in then:
                    others.append(f"{rel}: added since the snapshot")
                elif rel not in now:
                    others.append(f"{rel}: removed since the snapshot")
                elif now[rel] != then[rel]:
                    others.append(f"{rel}: changed since the snapshot")
            raise Refused("step 3, other code changed since the snapshot, so the result would not be the tree the "
                          "payload was built and tested as; nothing was changed. Take a new snapshot and "
                          "rebuild on it:\n  "
                          + "\n  ".join(others or ["(no single file found: compare the snapshot by hand)"]))
        print("3. the code items with the payload applied give DIGEST.txt: nothing else changed since the snapshot")
    else:
        print("3. no DIGEST.txt (the snapshot did not cover every code item): the rest of the code was not compared")

    # 4. write over in place (never unlink: a linked folder may refuse deletes), move removed files aside
    for rel in sorted(manifest):
        target = repository / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.is_file() and sha256_of(target) == manifest[rel]):
            shutil.copyfile(here / "payload" / rel, target)
    moved, not_moved = [], []
    for rel in sorted(removed):
        target = repository / rel
        if target.exists():
            aside = here / "removed" / rel
            aside.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(target, aside)
                moved.append(rel)
            except OSError as error:
                not_moved.append(f"{rel} ({error.strerror})")

    # 5. check the result
    bad = [rel for rel in sorted(manifest) if sha256_of(repository / rel) != manifest[rel]]
    if bad:
        raise Refused("step 5, written but not matching MANIFEST.sha256 (run install again):\n  "
                      + "\n  ".join(bad))
    print(f"4. written: {len(manifest)} files match MANIFEST.sha256 in the repository"
          + (f"; moved aside into {here / 'removed'}: {', '.join(moved)}" if moved else ""))
    if not_moved:
        print("   NOT MOVED, remove these yourself or ask the person to:\n     " + "\n     ".join(not_moved))
    if expected:
        got = tree_digest(repository)
        if got != expected:
            raise Refused(f"step 5, TREE DIFFERS after writing: {got}, expected {expected}")
        print(f"5. TREE OK {got}")
    print(f"INSTALLED {here.name} into {repository}")


# ----------------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="payload.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("snapshot", help="on the computer: pack the code items (or the paths given) with a record")
    s.add_argument("repository")
    s.add_argument("folder")
    s.add_argument("paths", nargs="*")
    c = sub.add_parser("check", help="in the container: an unpacked snapshot matches its record")
    c.add_argument("record")
    c.add_argument("folder")
    m = sub.add_parser("make", help="in the container: the payload from the snapshot's record and the worked folder")
    m.add_argument("record")
    m.add_argument("worked")
    m.add_argument("folder")
    m.add_argument("paths", nargs="*", help="paths beyond the snapshot's whose new files go in the payload")
    i = sub.add_parser("install", help="on the computer: install this payload into the repository, or refuse")
    i.add_argument("repository")
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            snapshot(Path(args.repository), Path(args.folder), args.paths)
        elif args.command == "check":
            check(Path(args.record), Path(args.folder))
        elif args.command == "make":
            make(Path(args.record), Path(args.worked), Path(args.folder), args.paths)
        else:
            install(Path(args.repository), Path(__file__).resolve().parent)
    except Refused as refusal:
        print(f"REFUSED: {refusal}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
