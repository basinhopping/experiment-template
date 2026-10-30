"""Local mode: everything on one machine, with no tmux and no node.

run_local     every job of a sweep file in this process, highest priority first, into the same run folders as on a
              node. Resumable: a finished run folder is skipped, so running the same file again continues it; a
              changed file under the same sweep_id is refused.
local_workers N worker processes on this machine reading the same file queue as on a node (enqueue first), each
              with its own log under logs/local/; returns when every worker has exited (the queue holds nothing they
              can run, or queue/STOP exists). Works on Windows, a Mac and Linux.
"""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

from sweepkit import paths
from sweepkit.runs.store import AlreadyComplete
from sweepkit.sweep.design import expand
from sweepkit.sweep.enqueue import load_sweep_file, record_sweep
from sweepkit.sweep.runner import describe, execute
from sweepkit.sweep.worker import check_offer


def run_local(sweep_file: Path, offered: str, max_consecutive_failures: int, log) -> dict:
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be at least 1")
    check_offer(offered)
    sweep = load_sweep_file(sweep_file)
    jobs, summary = expand(sweep)
    record_sweep(sweep_file, sweep, jobs, summary, log)
    order = sorted(range(len(jobs)), key=lambda k: (-jobs[k]["priority"], k))
    done = skipped = failed = in_a_row = 0
    for n, k in enumerate(order, 1):
        spec = jobs[k]
        if (paths.runs_dir(spec["sweep_id"]) / spec["run_id"]).is_dir():
            skipped += 1
            continue
        started = time.perf_counter()
        try:
            chosen = execute(spec, offered, log)
        except AlreadyComplete:
            skipped += 1
            continue
        except Exception as err:
            failed += 1
            in_a_row += 1
            log(f"JOB FAILED {spec['run_id']} ({n} of {len(jobs)}): {type(err).__name__}: {err}")
            if in_a_row >= max_consecutive_failures:
                log(f"{in_a_row} jobs failed in a row; stopping")
                break
            continue
        done += 1
        in_a_row = 0
        log(f"JOB DONE {spec['run_id']} ({n} of {len(jobs)}) {describe(chosen)}, "
            f"{time.perf_counter() - started:.1f}s")
    result = {"sweep_id": sweep["sweep_id"], "jobs": len(jobs), "done": done, "skipped_complete": skipped,
              "failed": failed}
    log(str(result))
    return result


def local_workers(n_workers: int, offered: str, min_free_gb: float, max_consecutive_failures: int, log) -> int:
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    check_offer(offered)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = paths.logs_dir() / "local"
    folder.mkdir(parents=True, exist_ok=True)
    processes = []
    for k in range(1, n_workers + 1):
        worker_id = f"local_{k}"
        log_path = folder / f"{worker_id}_{stamp}.log"
        handle = open(log_path, "w", encoding="utf-8")
        command = [sys.executable, "-m", "sweepkit.cli", "worker", "--worker-id", worker_id, "--device", offered,
                   "--min-free-gb", str(min_free_gb), "--max-consecutive-failures", str(max_consecutive_failures)]
        env = dict(os.environ)
        src = str(paths.repo_root() / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        processes.append((worker_id, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                                      cwd=str(paths.repo_root()), env=env), handle, log_path))
        log(f"started {worker_id}, log {log_path}")
    codes = {}
    for worker_id, process, handle, log_path in processes:
        codes[worker_id] = process.wait()
        handle.close()
        log(f"{worker_id} exited with code {codes[worker_id]}")
    return 0 if all(c == 0 for c in codes.values()) else 1
