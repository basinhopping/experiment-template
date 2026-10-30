"""The worker: take the next job it can run from the queue, run it into its own folder, repeat.

Started by node/worker.sh (one tmux session per GPU group on a shared node) or by `local-workers` (one process per
worker on one machine). A GPU worker is pinned to one GPU with CUDA_VISIBLE_DEVICES before Python starts, so inside
the process that GPU is cuda:0.

It exits when the queue holds nothing it can run, when queue/STOP exists, when queue/STOP_<group> exists for its own
group (worker IDs <group>_<n>), when the disk has less free space than --min-free-gb, or after
--max-consecutive-failures failed jobs in a row (a problem that fails every job, a broken GPU or a bug, cannot move
the whole queue to failed/ overnight). Every job ends with one line starting "JOB DONE", "JOB FAILED" or
"JOB SKIPPED", so the logs can be counted without matching library warnings that contain the word failed.
"""
import datetime
import json
import os
import shutil
import time
import traceback

from sweepkit import devices, paths
from sweepkit.runs.store import AlreadyComplete
from sweepkit.sweep.queue import Queue
from sweepkit.sweep.runner import describe, execute

ACCEPTS = {"cpu": ("cpu", "any"), "cuda": ("cuda", "any"), "mps": ("mps", "cpu", "any")}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def check_offer(offered: str) -> None:
    if offered not in ACCEPTS:
        raise ValueError(f"--device {offered!r}: cpu, cuda or mps")
    if offered == "cuda" and not devices.gpu_visible():
        raise RuntimeError("--device cuda, but no GPU is visible to this process. On a shared node this usually means "
                           "torch or CuPy loaded before the MPS fix: start workers through node/worker.sh.")
    if offered == "mps" and not devices.mps_available():
        raise RuntimeError("--device mps, but PyTorch's MPS backend is not available here")


def run(worker_id: str, offered: str, min_free_gb: float, max_consecutive_failures: int) -> int:
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be at least 1")
    check_offer(offered)
    accepts = ACCEPTS[offered]
    queue = Queue(paths.queue_dir())
    print(f"{now()} worker {worker_id} (device {offered}, takes jobs needing {', '.join(accepts)}; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, pid {os.getpid()})", flush=True)
    done = failed = skipped = in_a_row = 0
    while True:
        if queue.stop_requested(worker_id):
            print(f"{now()} worker {worker_id}: queue/STOP or this group's STOP file exists, stopping", flush=True)
            break
        free_gb = shutil.disk_usage(paths.repo_root()).free / 1e9
        if free_gb < min_free_gb:
            print(f"{now()} worker {worker_id}: {free_gb:.1f} GB free, below {min_free_gb}; stopping", flush=True)
            break
        claimed = queue.take(worker_id, accepts)
        if claimed is None:
            print(f"{now()} worker {worker_id}: nothing left that this worker can run", flush=True)
            break
        spec = json.loads(claimed.read_text(encoding="utf-8"))
        started = time.perf_counter()
        print(f"{now()} worker {worker_id}: start {spec['run_id']} ({spec['sweep_id']}, {spec['stage']}: "
              f"{spec['label']})", flush=True)
        try:
            chosen = execute(spec, offered, lambda m: print(m, flush=True))
        except AlreadyComplete:
            queue.finish(claimed, "done")
            skipped += 1
            print(f"{now()} JOB SKIPPED {spec['run_id']}: already complete", flush=True)
            continue
        except Exception as err:
            queue.finish(claimed, "failed")
            failed += 1
            in_a_row += 1
            traceback.print_exc()
            print(f"{now()} JOB FAILED {spec['run_id']}: {type(err).__name__}: {err}", flush=True)
            del err
            _empty_gpu_cache(offered)
            if in_a_row >= max_consecutive_failures:
                print(f"{now()} worker {worker_id}: {in_a_row} jobs failed in a row, stopping so the rest of the "
                      "queue stays pending", flush=True)
                break
            continue
        queue.finish(claimed, "done")
        done += 1
        in_a_row = 0
        print(f"{now()} JOB DONE {spec['run_id']} {describe(chosen)}, {time.perf_counter() - started:.0f}s",
              flush=True)
        _empty_gpu_cache(offered)
    print(f"{now()} worker {worker_id} finished: {done} done, {failed} failed, {skipped} skipped", flush=True)
    return 0


def _empty_gpu_cache(offered: str) -> None:
    if offered != "cuda":
        return
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
