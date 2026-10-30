"""Enqueue a sweep: expand it, check it against the sweeps it follows, record it, and write one queue file per job.

    artifacts/sweeps/<sweep_id>/sweep_file.json   the sweep file exactly as given
    artifacts/sweeps/<sweep_id>/summary.json      stages, configuration and job counts, seeds, code hash
    artifacts/sweeps/<sweep_id>/jobs.csv          run_id, stage, label, priority, fold, seed, config hash, device
    artifacts/sweeps/<sweep_id>/provenance.json
    queue/pending/*.json                          one job each

A sweep is enqueued once: if artifacts/sweeps/<sweep_id> exists, nothing happens. A changed sweep gets a new
sweep_id, so results of different sweeps never mix.

A confirm or final sweep may not reuse a seed of a sweep it follows (a confirmation on seeds its screen already used
is not a confirmation); enqueue refuses it. It warns when a sweep shares its priority with pending work of another
sweep, and when a sweep meant to rank or report runs one seed.

Several sweep files are enqueued together or not at all (enqueue_all): each is expanded and checked first (its
sweep_id given once, not enqueued before, the sweeps it follows recorded or among those given), and only then is any
recorded or queued, so a refusal leaves the queue as it was. A round's sweeps go in that way.
"""
import csv
import io
import json
from pathlib import Path

from sweepkit import paths
from sweepkit.hashing import sha256_file
from sweepkit.runs.provenance import code_hashes, collect
from sweepkit.runs.store import AlreadyComplete, OutputFolder
from sweepkit.sweep.design import SweepError, expand, warnings_for
from sweepkit.sweep.queue import Queue, pending_name


def load_sweep_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def device_need(job: dict) -> str:
    """What a worker must offer to run the job: the fit section's device (auto runs anywhere)."""
    device = job["settings"]["fit"].get("device", "cpu")
    return "any" if device == "auto" else device


def recorded_seeds(sweep_id: str, given: dict = None) -> set:
    """The seeds of a sweep that another follows: from its record, or from the sweeps enqueued with it (given)."""
    if given and sweep_id in given:
        return given[sweep_id]
    path = paths.sweeps_dir() / sweep_id / "summary.json"
    if not path.is_file():
        raise SweepError(f"follows {sweep_id!r}: that sweep has no record in artifacts/sweeps/; enqueue or pull it "
                         "first, or list only sweeps that ran")
    with open(path, encoding="utf-8") as handle:
        return set(json.load(handle)["seeds"])


def check_follows(sweep: dict, summary: dict, given: dict = None) -> list:
    """Raises when a confirm or final sweep reuses a seed of a sweep it follows; returns notes."""
    notes = []
    for other in sweep["follows"]:
        overlap = sorted(set(summary["seeds"]) & recorded_seeds(other, given))
        if overlap and sweep["purpose"] in ("confirm", "final"):
            raise SweepError(f"{sweep['sweep_id']} is a {sweep['purpose']} sweep but reuses seeds {overlap} of "
                             f"{other}, which it follows; confirm on seeds no earlier sweep of the chain used")
        if overlap:
            notes.append(f"shares seeds {overlap} with {other}, which it follows")
    return notes


def enqueue(sweep_file: Path, log) -> dict:
    return enqueue_all([sweep_file], log)[0]


def enqueue_all(sweep_files: list, log) -> list:
    """Enqueue several sweeps together, or none: every one is expanded and checked before any is recorded or queued.
    Returns each sweep's summary, in the order given."""
    checked = []
    for sweep_file in sweep_files:
        sweep = load_sweep_file(sweep_file)
        jobs, summary = expand(sweep)
        checked.append((Path(sweep_file), sweep, jobs, summary))
    ids = [sweep["sweep_id"] for _, sweep, _, _ in checked]
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if twice:
        raise SweepError(f"sweep_id given more than once: {twice}; nothing was enqueued")
    given = {sweep["sweep_id"]: set(summary["seeds"]) for _, sweep, _, summary in checked}
    notes = {}
    for _, sweep, _, summary in checked:
        if (paths.sweeps_dir() / sweep["sweep_id"]).exists():
            raise FileExistsError(f"sweep {sweep['sweep_id']} was already enqueued (artifacts/sweeps/"
                                  f"{sweep['sweep_id']}); give a changed sweep a new sweep_id. Nothing was enqueued")
        notes[sweep["sweep_id"]] = check_follows(sweep, summary, given)
    return [_record_and_queue(f, sweep, jobs, summary, notes[sweep["sweep_id"]], log)
            for f, sweep, jobs, summary in checked]


def _record_and_queue(sweep_file: Path, sweep: dict, jobs: list, summary: dict, notes: list, log) -> dict:
    sweep_id = sweep["sweep_id"]
    for warning in warnings_for(sweep, summary):
        log(f"WARNING {warning}")
    queue = Queue(paths.queue_dir())
    pending = queue.pending_priorities()
    mine = {stage["priority"] for stage in summary["stages"]}
    for other, priorities in sorted(pending.items()):
        shared = sorted(mine & priorities)
        if other != sweep_id and shared:
            log(f"WARNING priority {shared} is shared with pending jobs of {other}: they will interleave; give the "
                "sweep you need finished first a higher priority")
    try:
        folder = OutputFolder(paths.sweeps_dir(), sweep_id).begin()
    except AlreadyComplete:
        raise FileExistsError(f"sweep {sweep_id} was already enqueued (artifacts/sweeps/{sweep_id}); give a changed "
                              "sweep a new sweep_id") from None
    try:
        _, file_sha = sha256_file(sweep_file)
        summary.update({"sweep_file": Path(sweep_file).name, "sweep_file_sha256": file_sha,
                        "code_sha256": code_hashes(), "notes": notes})
        folder.write_json("sweep_file.json", sweep)
        folder.write_json("summary.json", summary)
        folder.write_json("provenance.json", collect("sweepkit.cli enqueue"))
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["seq", "run_id", "stage", "label", "priority", "fold", "seed", "config_hash", "device"])
        for seq, job in enumerate(jobs):
            writer.writerow([seq, job["run_id"], job["stage"], job["label"], job["priority"], job["fold"],
                             job["seed"], job["config_hash"], device_need(job)])
        folder.write_text("jobs.csv", buffer.getvalue())
        folder.complete()
    except BaseException as err:
        folder.fail(err)
        raise
    for seq, job in enumerate(jobs):
        name = pending_name(job["priority"], seq, sweep_id, job["run_id"])
        queue.put(name.replace(".json", f"__{device_need(job)}.json"), job)
    return summary


def record_sweep(sweep_file: Path, sweep: dict, jobs: list, summary: dict, log) -> bool:
    """Record a sweep that runs in this process without the queue (run --local). Returns False when the same file
    was recorded before (the run resumes), and refuses a changed file under the same sweep_id."""
    record = paths.sweeps_dir() / sweep["sweep_id"]
    if record.exists():
        recorded = json.loads((record / "sweep_file.json").read_text(encoding="utf-8"))
        if recorded != sweep:
            raise FileExistsError(f"sweep {sweep['sweep_id']} was recorded with a different file; give a changed "
                                  "sweep a new sweep_id")
        log(f"sweep {sweep['sweep_id']} was recorded before; resuming (complete runs are skipped)")
        return False
    notes = check_follows(sweep, summary)
    folder = OutputFolder(paths.sweeps_dir(), sweep["sweep_id"]).begin()
    try:
        _, file_sha = sha256_file(sweep_file)
        folder.write_json("sweep_file.json", sweep)
        folder.write_json("summary.json", {**summary, "sweep_file": Path(sweep_file).name,
                                           "sweep_file_sha256": file_sha, "code_sha256": code_hashes(),
                                           "notes": notes})
        folder.write_json("provenance.json", collect("sweepkit.cli run"))
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["seq", "run_id", "stage", "label", "priority", "fold", "seed", "config_hash", "device"])
        for seq, job in enumerate(jobs):
            writer.writerow([seq, job["run_id"], job["stage"], job["label"], job["priority"], job["fold"],
                             job["seed"], job["config_hash"], device_need(job)])
        folder.write_text("jobs.csv", buffer.getvalue())
        folder.complete()
    except BaseException as err:
        folder.fail(err)
        raise
    return True
