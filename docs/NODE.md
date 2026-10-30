# A shared GPU node

Everything the template needs on a shared multi-GPU Linux machine lives in `node/`. On a laptop, a home GPU or a Mac
nothing in it is needed: local mode (`run`, `local-workers`) does the same work, and the folder can be deleted
(the last section). docs/HOW_WE_WORK.md says why each rule below exists.

The person runs every command on the node; Claude writes the lines and checks each output (CLAUDE.md). Claude never
logs in to a node and never holds a login.

## Names

Everything is named after the project (`project.json`, `"project"`), so one person can run two projects from the
template on the same node without collisions, as long as neither name is the other's followed by `_` (with
`myproject` and `myproject_b`, the session `myproject_b_gpu0` would look like one of `myproject`'s, and its checks
would refuse to start):

| What | Where |
|---|---|
| the code | `~/<project>` (installed by `node/unpack.sh`, or a clone) |
| archives sent and received | `~/<project>_incoming`, `~/<project>_outgoing` |
| environments | `~/venvs/<project>-<environment>` (main, rapids, automl) |
| tmux sessions | `<project>_gpu<g>`, `<project>_cpu`, `<project>_schedule`, `<project>_<round>` |
| model downloads | `~/cache/huggingface` (HF_HOME) |
| logs | `~/<project>/logs/` (`logs/<round>/` for a round) |

Nothing uses `/scratch` or any folder other users share.

## The login file

`node.local.txt` beside README.md holds one line, the node's login (`user@host`), and nothing else. `.gitignore`
keeps it out of git; the push and pull scripts read it and send it nowhere. Each person writes their own; nobody else's
login is ever used, written down or asked for.

## The node layer

| Script | What it does |
|---|---|
| `node/pc/push.ps1`, `node/pc/push.sh` | on the computer: pack the code (and with `-Data`/`--data` a folder of `data/`) with SHA-256 and copy it to `~/<project>_incoming` in one connection, with this computer's repository folder (`pc_folder.txt`, for the pull line pack prints); prints the line to run there and the tree digest |
| `node/unpack.sh` | on the node: check the archive's SHA-256 and install it, moving the old code aside into `~/<project>_incoming/replaced/` instead of deleting it in place (copies moved aside at earlier installs are deleted then, so the code replaced last is kept); refused while any session of the project runs |
| `node/tree_digest.sh` | one SHA-256 over the code; the computer's `tree-digest` command prints the same number for the same code |
| `node/setup_env.sh` | build one environment from profiles, once, with exact pins, recording what was installed |
| `node/check.sh` | the check before a long run: every test with `SWEEPKIT_REQUIRE` (missing packages or GPU fail instead of skip), then `env-check` |
| `node/run.sh` | any sweepkit command with the project's environment, its output kept in `logs/` |
| `node/free_gpus.sh` | the GPUs no other user is on (no process of another user, at most 2000 MiB in use) |
| `node/disk_check.sh` | is there room for what a step will add |
| `node/smoke_enqueue.sh` | smoke-test every sweep file given, on the device the workers will offer (with `SMOKE_DEVICE=cuda`, on a free GPU of `SMOKE_GPUS`), then enqueue them all in one `enqueue`, which checks every one before it queues any; when any fails, none is enqueued and the script exits 1 |
| `node/worker.sh` | n worker processes on one GPU (or the CPU) in a session that ends when the queue is empty or a stop file appears |
| `node/schedule.sh` | the night's GPU schedule: from when to when, on which GPUs, ending with `queue/STOP` at most 7 days after the start; checks every step and the workers' settings before it starts anything (`--check` only checks, and wants the end at least 30 minutes ahead), refuses while `slurmd` is active and never starts a GPU another user is on; its workers run in the environment `ENVIRONMENT` names (main unless set: `ENVIRONMENT=rapids` for cuML jobs), with the `MIN_FREE_GB`, `MAX_FAILS` and `THREADS` set before it |
| `node/chain.sh` | the steps of a round's `plan.txt`, in order, in one session that ends by itself, each step's output in `logs/<round>/<n>_<script>.txt`; stops at the first failure |
| `node/pack.sh` | pack finished results with a SHA-256 for every file; `--after` packs only what earlier archives lack. It ends with the line to paste on the computer (after "paste this on your computer"), with the folder the last push recorded |
| `node/pc/pull.ps1`, `node/pc/pull.sh` | on the computer: fetch an archive, check every file, unpack it into `artifacts/`; refuses to overwrite |
| `node/swap_and_restart.sh` | install new code under workers started by hand or by a schedule, in its own session `<project>_swap` with `ENVIRONMENT` and `MIN_FREE_GB` set. Before any change it checks its settings, the archive, the tree it runs from, that the code in place can be put back as it is, the workers' schedule after it, the sweeps' devices, that no CPU workers run (it starts GPU workers only), and that nothing else uses the code (a round's chain: a round keeps the code it was prepared with; a notebook). It stops the workers, installs and checks the digest, checks the schedule again, smoke-tests and enqueues new sweeps (all or none), and replaces `<project>_schedule` with one running the workers until the end it is given. On any failure it restores the old code or leaves the queue as it was, and adds a line to `queue/SWAP_FAILED`, which `status` prints first (move it into `logs/` once read; the next swap that goes ahead does) |
| `node/rapids_run.sh` | run someone's own script unchanged on the GPU through cuML's zero-code-change mode, with the log of where every call ran |
| `node/test_all.sh` | the testing matrix of TESTING.md for the environments named, on one free GPU, within a number of hours, keeping `MIN_FREE_GB` free; in a session that ends by itself; results in `artifacts/test_all/<utc>/` |
| `node/common.sh` | the names above, the CUDA MPS fix, CUDA's GPU numbers in nvidia-smi's order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), thread caps; sourced by the others |

## The first time on a node

The exact lines, with what each should print, are in TESTING.md ("On a shared GPU node"). In order:

1. Look before touching anything, with this line, which only reads (the person logs in and pastes it):

   ```
   echo "--- $(date -u +%FT%TZ); the node's local time: $(date '+%F %T %Z, UTC%z')"; df -h ~; (timeout 20 quota -s 2>/dev/null || true); echo "--- the largest items in the home folder"; (timeout 300 du -xh --max-depth=1 ~ 2>/dev/null | sort -h | tail -12); echo "--- CPUs: $(nproc); memory: $(free -g | awk '/^Mem:/ {print $2 " GB, " $7 " GB available"}')"; echo "--- GPUs: index, ID, name, driver, memory used, memory total, use"; nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo "--- GPU processes: GPU ID, pid, memory, then user, running for, command"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | while IFS=', ' read -r u p m x; do echo "$u, $p, $m $x: $(ps -o user:32=,etime=,comm= -p "$p" 2>/dev/null)"; done; echo "--- your tmux sessions"; (tmux ls 2>/dev/null || echo "none"); echo "--- your crontab"; (crontab -l 2>/dev/null | grep -v '^#' || echo "none"); echo "--- slurmd: $(systemctl is-active slurmd 2>/dev/null)"; for v in 3.12 3.13; do echo "python$v: $(python$v --version 2>&1)"; done; for h in pypi.org files.pythonhosted.org pypi.nvidia.com github.com huggingface.co; do echo "$h: $(curl -sI -o /dev/null -w '%{http_code}' --max-time 10 https://$h)"; done
   ```

   It takes up to a few minutes (`du` walks the home folder) and prints:
   - the time in UTC and on the node's clock, with its zone: a round's `--end-time` is in the node's local time;
   - the home folder's file system, its size and free space (`df`; on a shared node a home folder often has its own
     limit, far smaller than the drive it sits on), any quota, and the largest items in the home folder;
   - the CPUs, and the memory with how much is available;
   - each GPU with its ID, driver version, memory in use and use, then every process on a GPU with that GPU's ID, the
     process's user, how long it has run and its command (an MPS server of another user's, `nvidia-cuda-mps-server`,
     does not make a GPU busy; anything else of another user's does);
   - the person's own tmux sessions and crontab: other work of the same account that may need disk or GPUs;
   - whether Slurm's `slurmd` is active (`active` stops everything here: "Adding Slurm" below);
   - the Python versions (`setup_env.sh` needs 3.12 or 3.13);
   - whether the node reaches PyPI, its file server, NVIDIA's package index (RAPIDS), GitHub and Hugging Face: any
     three-digit code means it does, `000` that it does not.

   The driver must be new enough for the CUDA builds the profiles pin (CUDA 13: driver 580.65 or newer for RAPIDS;
   docs/PROFILES.md). Write what the line shows into PROJECT.md ("Compute") with its date, then ask the person what
   the node cannot show (CLAUDE.md, "A shared node"), and write the node's rules there as its owner states them. The
   scripts count free space in whole GB, rounded down: 15.6 GB free counts as 15.
2. The code: `git clone <the project's repository> ~/<project>` when the node can reach GitHub, or `push` from the
   computer and `bash ~/<project>_incoming/unpack.sh <archive>`. Check the tree digest against the computer's.
3. The environments the project needs (docs/PROFILES.md), each once:
   `bash ~/<project>/node/setup_env.sh main core torch unsupervised`, and for RAPIDS
   `bash ~/<project>/node/setup_env.sh rapids rapids`.
4. `bash ~/<project>/node/check.sh main core torch unsupervised` (and `rapids rapids`): every test passes with
   nothing skipped for a missing package, and the GPU is seen. The tests that use a GPU run on the first free GPU
   (`CHECK_GPUS=2,3` limits the choice to the GPUs the project may use), never on another user's.
5. The testing matrix for those environments on one free GPU, in a window agreed with the person, inside a tmux
   session that ends by itself:
   `tmux new -d -s <project>_testall "MIN_FREE_GB=<keep> bash ~/<project>/node/test_all.sh <g> <hours> main rapids"`.
   TESTING.md has every line, what it should print, and how much room each environment takes.

## A night

A round's preparation folder (`round-new`, docs/HOW_WE_WORK.md) holds the night's lines in `lines/`, one file per
step, each naming its shell, what it should print, and when to stop and paste the output. The round's plan is in
`sweeps/rounds/<round>_prep_<date>/plan.txt`: it reaches the node with the code, and the tree digest lines 2 and 3
check covers it.

1. push (on the computer)
2. install and test: unpack, the tree digest, `check.sh` in the round's environment with its profiles (on the node),
   refused while the project's sessions run
3. start: check the schedule (its GPUs and its end) without starting anything, smoke-test the round's sweeps on one
   of its GPUs and enqueue them, then workers on the round's GPUs that no one else is using, until the queue is
   empty (at the latest the stop time), all inside one session running `node/chain.sh` on the round's plan, in the
   round's environment. It does not start when none of the round's GPUs is free or less than the round's floor of
   free disk (`round-new --min-free-gb`) is left; a schedule that would be refused, or a sweep that fails its smoke
   test, stops the chain before any worker starts, with none of the round's sweeps enqueued; the workers stop when no
   job is left for them, at the end time or at the floor. Once no job waits or runs and the workers have finished,
   the schedule ends (`schedule.sh --end-when-empty`) and the session with it: `chain.txt` ends with `CHAIN DONE`,
   and the next round's line 2 is not refused for a session that only waits
4. status, as often as wanted: `bash node/run.sh status --sweep <id> --top 10 --since <utc>`
5. pack, once the sweeps are finished: `status` first, because pack refuses only while a folder under the paths it
   is given is unfinished (`*.tmp`, a run being written) and knows nothing of jobs still in the queue
6. pull (on the computer), which checks every file

Smoke-testing before a round: a sweep that fails its smoke test in line 3 stops the whole round with nothing
enqueued. When a round is to show which methods run at all (one sweep per method, a failure being a finding), test
each sweep first, on one of the round's GPUs, in a session that ends by itself, and leave a failing one out of the
round (its error stays in `artifacts/smoke/`):

```
cd ~/<project> && tmux new -d -s <project>_smoke "for f in sweeps/project/<prefix>*.json; do echo \"=== \$f\"; ENVIRONMENT=<environment> CUDA_VISIBLE_DEVICES=<g> bash node/run.sh smoke --sweep-file \$f --device cuda --rows 200; done > logs/presmoke_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1"
```

`<environment>` is the round's (main, or rapids for cuML's GPU paths): the session takes tmux's environment, not
the shell's. When the session is gone: `grep -h -E '^=== |FAILED|SMOKE ' $(ls -t logs/presmoke_*.log | head -1)`.

Stopping early: `touch ~/<project>/queue/STOP`; each worker finishes its current job and exits. Remove it on purpose
before starting again: a worker exits at once while it exists.

Jobs left in `queue/running` by a worker that died: once every worker has stopped,
`bash node/run.sh requeue-running --all-workers-stopped yes`. Failed jobs, after fixing what failed them:
`requeue-failed`. Work to keep for later: `park` and `unpark`, never deleting queue files.

## Moving a laptop's script to the node's GPU

The common case: a Python script that runs a scikit-learn, UMAP or HDBSCAN analysis on one large CSV, too slowly on a
laptop's CPU.

1. The data's size against the node's free disk, counting the results: `node/disk_check.sh <GB>`. A home folder on
   a shared node often has its own limit, far smaller than the drive it sits on (`df -h ~` shows it), and your runs
   stop when it fills.
2. Send the data: `push --data <folder>` (the folder under `data/`), checked by SHA-256, unpacked into
   `~/<project>/data/<folder>/`.
3. Run the script unchanged first, on a free GPU, in the rapids environment, through cuML's zero-code-change mode,
   inside a tmux session that ends when the script does (so a dropped connection does not stop it):
   `tmux new-session -d -s <project>_rapids "CUDA_VISIBLE_DEVICES=<g> bash ~/<project>/node/rapids_run.sh cuml+cudf
   <script.py>"` (`cuml` alone leaves pandas on the CPU). `tmux ls` shows whether it still runs; its output is in
   `logs/rapids_<script>_<utc>.log`. Its `summary.txt` counts the calls that ran on the GPU and those that fell back
   to the CPU, with every reason (for HDBSCAN, any metric other than Euclidean falls back, for example). A fallback is
   not an error, but it is where the script did not get the GPU.
4. Bring the outputs home with `pack` and `pull`.
5. Then the template's version of the same analysis as a sweep, checked against the script's output on a sample
   (CLAUDE.md, "Existing code"), so the settings can be varied, seeded, compared and recorded.

## The CUDA MPS fix

Another user's CUDA MPS daemon can make CUDA unavailable to this user's processes unless two variables are empty
before Python starts. `node/common.sh` empties `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` and sets
`SWEEPKIT_CLEAR_MPS=1`, and the package clears them again when it is imported (`src/sweepkit/__init__.py`), before
anything imports torch or CuPy. Without it, a run can quietly fall back to the CPU; every run records the device that
actually ran.

## Adding Slurm

The template does not use Slurm: on the node it was built for, Slurm was configured but not running, so a Slurm path
could not be tested. `node/schedule.sh` and `node/chain.sh` refuse to start while `slurmd` is active, because Slurm
could give the same GPUs to batch jobs. To run workers under Slurm instead:

1. Write a batch script (for example slurm_worker.sbatch in `node/`) that asks for one GPU (`#SBATCH --gres=gpu:1`), a
   time limit and the CPUs and memory of n workers, sources `node/common.sh`, and runs
   `bash node/worker.sh gpu "$CUDA_VISIBLE_DEVICES" <n>`. Slurm sets `CUDA_VISIBLE_DEVICES` to the GPU it allocated,
   and `worker.sh` sets that variable to its second argument, so passing Slurm's value through keeps the allocated GPU
   whether Slurm numbers it 0 inside the job (devices confined by cgroups) or by its index on the node.
2. Submit one per GPU wanted (`sbatch --array=1-4 ...`). Workers on any number of allocations share the same file
   queue safely: a worker takes a job by renaming its file, and a rename is atomic.
3. Stopping is the same: `queue/STOP`, and the time limit ends the allocation.
4. Leave the `slurmd` refusal in the schedule; the Slurm path replaces the schedule rather than running beside it.

Test it as the node layer is tested (`tests/test_node_layer.py` runs the scripts with stand-ins for tmux and
nvidia-smi) before the first real night.

## Without a node

Delete `node/`. Local mode, the tests and everything else work without it:
`tests/test_node_layer.py` (TestWithoutNode) copies the repository without `node/`, runs the core tests and a local
sweep. On one machine:

- `python -m sweepkit.cli run --sweep-file <file> --device cpu --max-consecutive-failures 3` runs a sweep in one
  process (resumable: run it again to continue);
- `enqueue`, then `local-workers --n 4 --device cuda ...` runs the same file queue with N workers and no tmux, on
  Windows, a Mac or Linux.
