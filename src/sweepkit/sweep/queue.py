"""A job queue made of files.

    queue/pending/          waiting jobs, one JSON file each; names sort by priority, then order, and end with the
                            device the job needs (any, cpu, cuda or mps)
    queue/running/          taken by a worker: the file is moved here with the worker's name added
    queue/done/             finished (or found already complete)
    queue/failed/           failed; the run's own folder holds the error
    queue/parked/<name>/    jobs set aside on purpose (park); no worker looks here
    queue/STOP              if this file exists, every worker finishes its current job and exits
    queue/STOP_<group>      the same for the workers of one group only (worker IDs <group>_<n>, for example gpu2_1
                            or cpu_3)

A worker takes a job by renaming its file from pending/ to running/. A rename is atomic, so two workers can never
take the same job: the second one's rename finds no file and it moves on.
"""
import json
import os
import re
from pathlib import Path


def pending_name(priority: int, seq: int, sweep_id: str, run_id: str) -> str:
    return f"{999 - priority:03d}_{seq:06d}__{sweep_id}__{run_id}.json"


def group_of(worker_id: str) -> str:
    """gpu2_1 -> gpu2; cpu_3 -> cpu."""
    return worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id


class Queue:
    STATES = ("pending", "running", "done", "failed")

    def __init__(self, root: Path):
        self.root = Path(root)
        for state in self.STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        (self.root / "parked").mkdir(parents=True, exist_ok=True)

    def folder(self, state: str) -> Path:
        return self.root / state

    def put(self, name: str, spec: dict) -> None:
        """Add a job. Refuses a name already pending (a sweep is enqueued once; see enqueue)."""
        with open(self.folder("pending") / name, "x", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=1)

    def take(self, worker_id: str, accepts: tuple):
        """The next job this worker can run: its device need (the last part of the name, see enqueue.device_need)
        must be one the worker accepts, for example ("cuda", "any") for a GPU worker."""
        for path in sorted(self.folder("pending").glob("*.json")):
            need = path.stem.rsplit("__", 1)[-1]
            if need not in accepts:
                continue
            target = self.folder("running") / f"{path.stem}.{worker_id}.json"
            try:
                os.rename(path, target)
            except FileNotFoundError:
                continue
            return target
        return None

    def finish(self, claimed: Path, state: str) -> Path:
        if state not in ("done", "failed"):
            raise ValueError(state)
        target = self.folder(state) / claimed.name
        os.rename(claimed, target)
        return target

    @staticmethod
    def original_name(path: Path) -> str:
        """The pending name of a running, done or failed job file (the worker's name removed)."""
        return path.name.rsplit(".", 2)[0] + ".json"

    def requeue_running(self) -> int:
        """Move every running job back to pending. Only when no worker is running."""
        moved = 0
        for path in sorted(self.folder("running").glob("*.json")):
            os.rename(path, self.folder("pending") / self.original_name(path))
            moved += 1
        return moved

    def requeue_failed(self, sweep_id: str) -> int:
        """Move one sweep's failed jobs back to pending, under their original names. Their failed run folders stay
        where they are, renamed .failed-<utc>, so the error is kept."""
        moved = 0
        for path in sorted(self.folder("failed").glob(f"*__{sweep_id}__*.json")):
            os.rename(path, self.folder("pending") / self.original_name(path))
            moved += 1
        return moved

    def park(self, sweep_id: str, name: str) -> int:
        """Move one sweep's pending jobs to queue/parked/<name>/, where no worker looks."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"park name {name!r}: letters, digits, _ and - only")
        target = self.root / "parked" / name
        target.mkdir(parents=True, exist_ok=True)
        moved = 0
        for path in sorted(self.folder("pending").glob(f"*__{sweep_id}__*.json")):
            os.rename(path, target / path.name)
            moved += 1
        return moved

    def unpark(self, name: str) -> int:
        source = self.root / "parked" / name
        if not source.is_dir():
            raise FileNotFoundError(f"nothing parked under {name!r}")
        moved = 0
        for path in sorted(source.glob("*.json")):
            os.rename(path, self.folder("pending") / path.name)
            moved += 1
        return moved

    def stop_requested(self, worker_id: str | None = None) -> bool:
        """queue/STOP stops every worker; queue/STOP_<group> stops the workers of that group."""
        if (self.root / "STOP").exists():
            return True
        if worker_id is None:
            return False
        return (self.root / f"STOP_{group_of(worker_id)}").exists()

    def counts(self, sweep_id: str | None) -> dict:
        out = {}
        for state in self.STATES:
            files = self.folder(state).glob("*.json")
            out[state] = sum(1 for f in files if sweep_id is None or f"__{sweep_id}__" in f.name)
        out["parked"] = sum(1 for f in (self.root / "parked").rglob("*.json")
                            if sweep_id is None or f"__{sweep_id}__" in f.name)
        return out

    def pending_priorities(self) -> dict:
        """sweep_id -> set of priorities of its pending jobs."""
        out = {}
        for path in self.folder("pending").glob("*.json"):
            parts = path.name.split("__")
            if len(parts) >= 3:
                out.setdefault(parts[1], set()).add(999 - int(parts[0][:3]))
        return out
