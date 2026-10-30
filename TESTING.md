# Testing

How to check a copy of the template: a fresh clone, or a project after a change. First on your own computer, then on
a shared GPU node. Every step gives the line to run and what it should print. The last section is the testing
matrix: every component with its tests, the bounds they assert, its smoke entry and its CPU and GPU agreement test.

A Claude session that tests a fresh clone follows this file word for word and writes down every step that is unclear,
wrong, or needs something this file does not say. Commands on a shared node are always the person's to run: Claude
gives one line, waits for the pasted output, compares it with what the line should print, then gives the next.

## The checks

| Check | What it shows | Where it runs |
|---|---|---|
| The unit tests | every component recovers known structure in synthetic data to a stated bound; the discipline refuses what it must; the generated docs are current; no file holds a name, address or home path | everywhere; a test whose package or GPU is missing skips |
| `env-check` | the installed versions match the pins of the profiles, and which devices the process sees | every environment |
| `check-additions` | every component and stage kind has an example sweep that uses it, and every one, metrics included, a test that names it (a unit test runs it too) | everywhere |
| `smoke-all` | every component runs once through the real runner on 100 rows, and the device it ran on is recorded | every environment, on the CPU and on a GPU |
| The agreement tests | a method's CPU and GPU versions agree on the same data | an NVIDIA GPU: cuML and cuGraph in the rapids environment, XGBoost in main |
| The large-data test | a CSV of `SWEEPKIT_LARGE_ROWS` rows generated and converted in chunks, settings chosen on subsamples, an estimate, then the full run, with each step's time and peak memory | anywhere; a million rows on a node |
| The flow check (below) | the whole flow end to end: data sent with push, a round with one sweep per kind of method run through its own lines on the node, results pulled and read in the fixed order | the computer and a node |

Three environment variables change what the tests do:

| Variable | Effect |
|---|---|
| `SWEEPKIT_REQUIRE` | a comma-separated list of import names (for example `torch,cuml`) and `gpu`: the tests that need them fail instead of skipping. `python -m sweepkit.cli required-modules --profiles <profiles>` prints the list for an environment. |
| `SWEEPKIT_NETWORK=1` | lets the tests that download a small pretrained model run (about 100 MB, into the Hugging Face cache); without it they skip |
| `SWEEPKIT_LARGE_ROWS` | the size of the large-data test (for example `50000` at home, `1000000` on a node); without it the test skips |

If Claude's shell cannot install packages (pip prints "No matching distribution found", or a proxy refuses the
connection), the person runs the lines below in their own terminal and pastes the output, one line at a time, as on
a node. Note it in the findings.

## On your own computer

Use one shell and keep to it: **Command Prompt** on Windows, **Terminal** on a Mac or Linux. Start in an empty folder.

### 1. Get a copy

Command Prompt, or Terminal:

```
git clone -c core.autocrlf=false https://github.com/<owner>/experiment-template.git && cd experiment-template
```

Should print `Cloning into 'experiment-template'...` and nothing that says error or fatal. No login is needed for a
public repository; if git asks for one, stop. `core.autocrlf=false` keeps the line endings the scripts need.

### 2. Check Python

- Windows: `py -3.12 --version` (or `py -3.13 --version`)
- Mac or Linux: `python3.12 --version` (or `python3.13 --version`)

Should print `Python 3.12.x` or `Python 3.13.x`. Without either, START_HERE.md says how to install one.

### 3. Build the main environment

The default profiles, `core torch unsupervised`. It downloads the packages, which takes several minutes.

- Windows (Command Prompt; PyPI's Windows build of torch is CPU only):

  ```
  powershell -ExecutionPolicy Bypass -File env\setup.ps1 -Environment main -Profiles "core,torch,unsupervised" -TorchIndex cpu
  ```

  With Python 3.13 instead of 3.12, add `-Python 3.13`.

- Mac or Linux:

  ```
  bash env/setup.sh main core torch unsupervised
  ```

  On Linux without an NVIDIA GPU, `TORCH_INDEX=cpu bash env/setup.sh main core torch unsupervised` downloads the much
  smaller CPU build of torch.

Near the end it should print `exact versions written to ...artifacts...env...main_<utc>.txt` and a `next:` line
(then a line on activating the environment). It refuses if `.venv-main` exists already.

To test more of the template, add on-request profiles to the same line: `boosting`, `search`, `singlecell`,
`pretrained`, `graph`, and `extras` once you have read its licences in docs/PROFILES.md. The rapids profiles need Linux
with an NVIDIA GPU (the node).

### 4. Check the environment

- Windows: `.venv-main\Scripts\python -m sweepkit.cli env-check --profiles core torch unsupervised`
- Mac or Linux: `.venv-main/bin/python -m sweepkit.cli env-check --profiles core torch unsupervised`

Should print `env-check of profiles core, torch, unsupervised:`, then `<n> of <n> pinned packages as pinned`, the
devices the process sees (`GPU visible: ...; MPS available: ...`, and with an NVIDIA GPU a `CUDA devices` line),
`ENV OK` and a `saved` line. A package whose installed version is not its pin gets a `DIFFERENT <package>: pinned
..., installed ...` line, a package that does not import a `cannot import` line, and the verdict is `ENV DIFFERS`:
write those lines down.

### 5. Run the unit tests and check-additions

- Windows: `.venv-main\Scripts\python -m unittest discover -s tests -t .`
- Mac or Linux: `.venv-main/bin/python -m unittest discover -s tests -t .`

Takes a few minutes. Should end with `Ran <n> tests in <s>s` and `OK (skipped=<m>)`. Skips are expected: tests for
profiles you did not install, for a GPU this computer does not have, and the large-data test. Nothing should fail. For
a failure, keep the whole block from `FAIL:` or `ERROR:` to the line of dashes after it. Then:

- Windows: `.venv-main\Scripts\python -m sweepkit.cli check-additions --show 50`
- Mac or Linux: `.venv-main/bin/python -m sweepkit.cli check-additions --show 50`

Should end with `<n> components and <m> example sweeps checked: ADDITIONS OK`. Otherwise it prints a `PROBLEM`
line for each of up to 50 problems (an example sweep that does not expand, a component or stage kind that no example
sweep uses, or one that no test names) and ends with `<k> PROBLEMS`.

### 6. Smoke-test every component on the CPU

- Windows: `.venv-main\Scripts\python -m sweepkit.cli smoke-all --profile all --device cpu --rows 100`
- Mac or Linux: `.venv-main/bin/python -m sweepkit.cli smoke-all --profile all --device cpu --rows 100`

It writes the small example data files first, then prints one line per component (`ok`, `FAILED` or `skipped`, with
the seconds and the device) and ends with `smoke-all (all, cpu): <a> ok, 0 failed, <b> skipped; saved <folder>` and
`SMOKE-ALL OK`. The table of every component is `<folder>/table.csv`: kind, name, profile, example, status, seconds,
device, fallbacks, error. A component is skipped when a package its example needs is not installed here.

### 7. On a GPU, if this computer has one

- **An NVIDIA GPU on Windows**: build a second environment with the CUDA build of torch, then smoke-test on the GPU:

  ```
  powershell -ExecutionPolicy Bypass -File env\setup.ps1 -Environment gpu -Profiles "core,torch,unsupervised" -TorchIndex cu130
  ```
  ```
  .venv-gpu\Scripts\python -m sweepkit.cli smoke-all --profile all --device cuda --rows 100
  ```
  ```
  set SWEEPKIT_REQUIRE=torch,gpu&& .venv-gpu\Scripts\python -m unittest tests.test_torch_family tests.test_torch_models tests.test_deep_models tests.test_graph_models
  ```

  (`-TorchIndex cu126` for a GPU older than Turing.)

- **An NVIDIA GPU on Linux**: the main environment already has PyPI's CUDA build of torch:

  ```
  .venv-main/bin/python -m sweepkit.cli smoke-all --profile all --device cuda --rows 100
  ```
  ```
  SWEEPKIT_REQUIRE=torch,gpu .venv-main/bin/python -m unittest tests.test_torch_family tests.test_torch_models tests.test_deep_models tests.test_graph_models
  ```

- **A Mac with Apple silicon**: the neural networks can use the GPU through PyTorch's MPS backend:

  ```
  .venv-main/bin/python -m sweepkit.cli smoke-all --profile torch --device mps --rows 100
  ```

The `device` column of the smoke table says where each component ran: `cuda` (or `mps`) for the neural networks, `cpu`
for the methods whose GPU path needs RAPIDS, which runs only on Linux (the node). The tests should end with `OK`; the
smoke-all with `SMOKE-ALL OK`.

### 8. The large-data test (optional at home)

- Windows: `set SWEEPKIT_LARGE_ROWS=50000&& .venv-main\Scripts\python -m unittest -v tests.test_large_data`
- Mac or Linux: `SWEEPKIT_LARGE_ROWS=50000 .venv-main/bin/python -m unittest -v tests.test_large_data`

About half a minute at 50,000 rows. Should print each step (make-synthetic, convert, screen on subsamples, analyze,
estimate, full run, every row) with its seconds and peak memory, the estimate beside the full run, and `OK`.

### 9. What to report

The last two lines of the unit tests, every failure in full, the last line of `check-additions` (and its problems,
if any), the smoke table (`table.csv`) of each smoke-all, the `env-check` verdict with any `DIFFERENT` or `cannot
import` lines, and the large-data lines.

## On a shared GPU node

The person logs in; Claude never sees the login. One line at a time, each compared with what it should print before
the next. Only GPUs no other user is on; the free disk measured before anything is downloaded; work in tmux sessions
that end by themselves; never another user's process or session; nothing left running at the end (CLAUDE.md, rule 8;
docs/NODE.md).

The lines below name the test copy `tmpltest`: its folder is `~/tmpltest`, its environments `~/venvs/tmpltest-<name>`
and its tmux sessions `tmpltest_<name>`. Replace `<owner>` with the repository's GitHub owner, and `<keep>`,
`<hours>` and `<g>` with what steps 3 and 6 settle.

### 1. Look before touching anything

The read-only first line of docs/NODE.md ("The first time on a node", step 1), which says what each part prints:

```
echo "--- $(date -u +%FT%TZ)"; df -h ~; (timeout 20 quota -s 2>/dev/null || true); echo "--- the largest items in the home folder"; (timeout 300 du -xh --max-depth=1 ~ 2>/dev/null | sort -h | tail -12); echo "--- CPUs: $(nproc); memory: $(free -g | awk '/^Mem:/ {print $2 " GB, " $7 " GB available"}')"; echo "--- GPUs: index, ID, name, driver, memory used, memory total, use"; nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo "--- GPU processes: GPU ID, pid, memory, then user, running for, command"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | while IFS=', ' read -r u p m x; do echo "$u, $p, $m $x: $(ps -o user:32=,etime=,comm= -p "$p" 2>/dev/null)"; done; echo "--- your tmux sessions"; (tmux ls 2>/dev/null || echo "none"); echo "--- your crontab"; (crontab -l 2>/dev/null | grep -v '^#' || echo "none"); echo "--- slurmd: $(systemctl is-active slurmd 2>/dev/null)"; for v in 3.12 3.13; do echo "python$v: $(python$v --version 2>&1)"; done; for h in pypi.org files.pythonhosted.org pypi.nvidia.com github.com huggingface.co; do echo "$h: $(curl -sI -o /dev/null -w '%{http_code}' --max-time 10 https://$h)"; done
```

Stop and paste if Slurm says `active` (docs/NODE.md, "Adding Slurm"), if there is no Python 3.12 or 3.13, if the
driver is older than the CUDA builds need, or if PyPI answers `000` (then the code still goes over with
`node/pc/push.ps1` or `push.sh`, but the environments cannot be built).

### 2. Get the code

```
cd ~ && git clone -c core.autocrlf=false https://github.com/<owner>/experiment-template.git tmpltest && cd ~/tmpltest && sed -i 's/"project": "myproject"/"project": "tmpltest"/' project.json && cat project.json
```

Should print `{"project": "tmpltest", "discipline": "enforced"}`. Stop if `~/tmpltest` exists already.

Before the repository is on GitHub, or when the node cannot reach it, the code goes over from the computer instead.
In a copy of the repository whose `project.json` names `tmpltest`, the person writes their login into
`node.local.txt` (one line, `user@host`), then, from that copy, on Windows:

```
powershell -ExecutionPolicy Bypass -File node\pc\push.ps1
```

(`bash node/pc/push.sh` on a Mac or Linux.) It prints the archive's name and SHA-256, asks for the node's password
once (scp), then the line to run on the node, `bash ~/tmpltest_incoming/unpack.sh tmpltest_code_<utc>.tar.gz`, and
the tree digest of the code sent. That line, on the node, ends with `tree digest <64 hex>`, which must be the same
number.

### 3. Agree on the room, the window and what to test

Three things to settle with the person before anything is built, and to write down in the findings:

- **What stays free.** `<keep>`: the gigabytes that must stay free in the home folder for everything else that runs
  from it. Another project whose workers stop below a floor needs that floor and a margin (a floor of 10 GB: keep
  15). With nothing else there, 5.
- **The window.** A time when GPU `<g>` is free and may be used: a weekend, or one GPU on an evening the person
  chooses, ending before any cutoff the node or the person keeps. `<hours>` is its length less half an hour for
  checking and packing.
- **What to test.** One environment at a time: build it, test it, and remove it when the next one would not fit.
  Each needs its size free on top of `<keep>`, and 2 GB more for the test's own files.

| Environment | Profiles | Size | What its test covers |
|---|---|---|---|
| main, every profile | `core torch unsupervised boosting search singlecell pretrained graph extras` | 7.1 GB, measured | every component except RAPIDS and TPOT, the neural networks on the GPU |
| main, no neural networks | `core unsupervised` | about 1.5 GB, estimated | the clustering, reduction and outlier methods on the CPU |
| rapids | `rapids rapids-singlecell` | 5.7 GB, measured | cuML, cuGraph and rapids-singlecell on the GPU, and their agreement with the CPU versions |
| automl | `core automl` | 1.4 GB, measured | TPOT |

The whole matrix is main with every profile, then rapids, then automl. A project that needs only some profiles tests
those: for HDBSCAN on one CSV, main without neural networks and then rapids. One environment at a time, the most the
test needs free at once is the largest environment, 2 GB and `<keep>`: about 25 GB for the whole matrix with `<keep>` 15.

Three sizes were measured on the node of the node test (2026-10-28; `test_all.sh` records each environment's size in
its summary). The size of main without neural networks is an estimate: from the packages' download sizes on PyPI,
taking an installed environment as about twice its download. Another node (another Python, another CUDA) can differ:
`setup_env.sh` prints each environment's real size at its end.

### 4. Room for the environment

```
bash ~/tmpltest/node/disk_check.sh <size + 2 + keep>
```

With the number worked out from the table: for main with every profile and `<keep>` 15, `disk_check.sh 25`. Should
print `free: <n> GB in <the home file system>; needed: <the number> GB` and not `NOT ENOUGH DISK`. If it says that,
stop: remove an environment already tested (step 9), or test a smaller set.

### 5. Build it

The environment's line from the table, in a tmux session that ends by itself:

```
tmux new -d -s tmpltest_setup "bash ~/tmpltest/node/setup_env.sh main core torch unsupervised boosting search singlecell pretrained graph extras"
```

The others: `setup_env.sh rapids rapids rapids-singlecell`, `setup_env.sh automl core automl`, and for main without
neural networks `setup_env.sh main core unsupervised`. Prints nothing. `setup_env.sh` uses `python3.12`, or
`python3.13` when there is no 3.12. To check on it (every few minutes):

```
tmux ls 2>/dev/null | grep tmpltest_ ; tail -n 4 $(ls -t ~/tmpltest/logs/setup_env_*.log | head -1)
```

While it runs, `tmpltest_setup` is listed. When it is gone, the log ends with the environment's size (`du -sh`), the
free space after it and `setup_env: done. Next: ...`. Write the size down. A log that stops at a pip error: paste it
and stop.

### 6. Choose a free GPU

```
bash ~/tmpltest/node/free_gpus.sh
```

Prints the index of every GPU no other user is on (no process of another user, at most 2000 MiB in use), one per
line, and the reason for each busy one. Choose as `<g>` one that is free and inside the agreed window. If none is,
stop: try later.

### 7. Test it, within the window

```
tmux new -d -s tmpltest_testall "SWEEPKIT_NETWORK=1 MIN_FREE_GB=<keep> bash ~/tmpltest/node/test_all.sh <g> <hours> main"
```

With the environment's name last (`main`, `rapids` or `automl`). Prints nothing. `node/test_all.sh` refuses to start
if GPU `<g>` has become busy, if Slurm is active, or if less than `<keep>` plus 2 GB is free. It runs `env-check`,
every unit test with `SWEEPKIT_REQUIRE` naming the environment's packages and the GPU, the CPU and GPU agreement tests
one by one, and `smoke-all` on the GPU and on the CPU; in main, then the large-data test at a million rows. Each step
runs with the time left, and a step that cannot start in time, or would start below `<keep>` free, is recorded as
not run. Leave out `SWEEPKIT_NETWORK=1` if the node must not download models. It has not been timed on a node yet:
three hours is a cautious first window for main with every profile, and `summary.csv` records each step's seconds
for next time.

### 8. Check on it, then read the results

```
tmux ls 2>/dev/null | grep tmpltest_ ; tail -n 4 $(ls -t ~/tmpltest/logs/test_all_*.log | head -1)
```

The log starts with the time the run ends by. While it runs, `tmpltest_testall` is listed and the log shows the step
it is on (`step <n>: <name> (<environment>), at most <s> s`) and how each finished step ended. It ends with
`TEST-ALL DONE: <n> steps: <a> ok, <b> failed, <c> timed out, <d> not run; <free> GB free` and the folder of its
results. Then:

```
cat $(ls -d ~/tmpltest/artifacts/test_all/*/ | tail -1)summary.txt
```

The environment's size, then one line per step with how it ended and the last line of its log. Every step's whole log
is in the same folder (`<nn>_<step>.log`), with each smoke-all table beside its log
(`<nn>_smoke_<device>_<environment>_table.csv`). For a step that failed: `grep -A 30 -E "^(FAIL|ERROR):" <the log>`.

### 9. Make room for the next environment, if needed

When step 4 for the next environment would say `NOT ENOUGH DISK`, remove the one just tested. It is the test copy's
own environment, which nothing else uses:

```
rm -rf ~/venvs/tmpltest-main && df -h ~
```

With that environment's name in place of `main`. Then steps 4 to 9 for the next environment.

### 10. Bring the results home

```
bash ~/tmpltest/node/pack.sh test_all
```

Once, after the last environment's test: it packs every finished run of `test_all.sh` and refuses while one is still
running. It prints the archive's name and, after "paste this on your computer", the line that pulls it there
(`node/pc/pull.ps1` or `pull.sh`, in the copy the code was pushed from: its `project.json` names `tmpltest` and it has
the person's own `node.local.txt`). The pull checks every file's SHA-256. To bring results home between environments
instead, a later pack takes `--after <the earlier archive's name>`, so only the new runs travel (the pull refuses to
overwrite a file).

### 11. Leave nothing running, then nothing behind

```
tmux ls 2>/dev/null | grep tmpltest_ || echo "no tmpltest sessions" ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Should print `no tmpltest sessions`, and no process of yours on any GPU. Once the results are on the computer (step
10), remove everything the test made, so the next test, or a project that takes the name, starts clean (step 2
stops when `~/tmpltest` exists):

```
rm -rf ~/tmpltest ~/tmpltest_incoming ~/tmpltest_outgoing ~/venvs/tmpltest-* && ls -d ~/tmpltest* ~/venvs/tmpltest-* 2>&1 | head -3; df -h ~
```

Should print `ls: cannot access` lines and the free space.

## The overnight queue check

A night at small scale: smoke-test and enqueue a sweep, start workers on one GPU through the schedule, let them take
every job, stop at the schedule's end, and read the status. `sweeps/checks/gpu_queue.json` runs k-means and HDBSCAN
through cuML with `fit.device` `cuda` (4 configurations, 24 jobs on 50,000 synthetic rows), so it needs the rapids
environment (node steps 4 and 5). On the node, with `<g>` a free GPU and `<keep>` from step 3:

```
cd ~/tmpltest && ENVIRONMENT=rapids SMOKE_DEVICE=cuda SMOKE_GPUS=<g> SMOKE_ROWS=2000 bash node/smoke_enqueue.sh sweeps/checks/gpu_queue.json
```

Should print `smoke tests on GPU <g>`, the smoke test's lines ending with `SMOKE OK`, then the sweep's ID with 4
configurations and 24 jobs, and `ALL DONE <utc>, pending 24`.

```
S="$(date +%s):<g> $(date -d '+20 min' +%s):end"; tmux new -d -s tmpltest_schedule "ENVIRONMENT=rapids MIN_FREE_GB=<keep> bash ~/tmpltest/node/schedule.sh 2 $S"
```

Two workers on GPU `<g>` in the session `tmpltest_gpu<g>`, which ends when the queue is empty; `queue/STOP` after 20
minutes ends the schedule. When neither session is listed (`tmux ls`):

```
cd ~/tmpltest && ENVIRONMENT=rapids bash node/run.sh status --sweep check_gpu_queue --top 5 --since all
```

Should report 24 runs complete and none failed. `logs/worker_gpu<g>_*` hold a `JOB DONE` line per job. Remove
`queue/STOP` before workers are started again (`rm ~/tmpltest/queue/STOP`).

## The laptop-to-GPU check

The case the template was first built for: someone runs HDBSCAN on one large CSV on a laptop's CPU, too slowly, and
wants it on the node's GPU with the results back on the laptop. CLAUDE.md ("Existing code", "From a laptop's CPU to a
node's GPU") and docs/NODE.md ("Moving a laptop's script to the node's GPU") are the steps; this check walks them with
a stand-in.

1. A stand-in dataset with known clusters, on the computer:

   ```
   .venv-main\Scripts\python -m sweepkit.cli make-synthetic --generator blobs --n-rows 1000000 --n-features 50 --n-groups 8 --noise 1.0 --seed 3 --name standin --chunk-rows 100000 --sample-rows 20000
   ```

   (`.venv-main/bin/python` on a Mac or Linux.) It writes `data/synthetic/standin.csv` (a million rows: `row_id`,
   `f0` to `f49`, the known cluster in `truth`, and `group` and `published_split`), `standin_sample.csv` (20,000 of
   those rows) and `standin.json` (their record). `push --data` sends one folder of `data/`, so move the three into
   a folder of their own:

   ```
   mkdir data\standin && move data\synthetic\standin* data\standin\
   ```

   (`mkdir data/standin && mv data/synthetic/standin* data/standin/` on a Mac or Linux.) The paths below are under
   `data/standin/`.
2. A stand-in for the person's script: `tests/laptop_standin.py` reads a CSV with pandas, takes the `f` columns,
   runs the hdbscan package's `HDBSCAN` with its settings written out, and writes `row_id,label`. Run it unchanged on
   the sample on the computer (the unsupervised profile has the hdbscan package) and keep its output:
   `python tests/laptop_standin.py data/standin/standin_sample.csv standin_cpu.csv`.
3. The template's version, as CLAUDE.md says: the same settings as a sweep, `sweeps/checks/laptop_hdbscan.json` (the
   `table` dataset on `standin/standin_sample.csv`, model `hdbscan`, split `all`, task `clustering` with reference
   `truth`; the `store` dataset after `convert` for the whole file). Run it with `run --sweep-file`; its labels
   (`assignments.csv` in the run folder) and the script's should give an adjusted Rand index of 1.0:
   `python tests/laptop_standin.py --compare standin_cpu.csv <the run folder>/assignments.csv
   data/standin/standin_sample.csv` (the last file adds each one's agreement with the planted clusters).
4. On the node: the disk first, `bash ~/tmpltest/node/disk_check.sh <keep + 2>` (the million-row CSV is about
   0.5 GB and arrives as an archive of the same size before it is unpacked; its store and the runs take about 1 GB
   more). Then the folder over with its SHA-256: on the computer
   `powershell -ExecutionPolicy Bypass -File node\pc\push.ps1 -Data standin`, or `bash node/pc/push.sh --data standin`
   on a Mac or Linux, then the `unpack.sh` line it prints, on the node (docs/NODE.md); it puts the folder in
   `~/tmpltest/data/standin/`. Then the script unchanged, in the rapids environment (node steps 4 and 5) on a free GPU,
   inside a tmux session that ends by itself:

   ```
   tmux new -d -s tmpltest_rapids "CUDA_VISIBLE_DEVICES=<g> bash ~/tmpltest/node/rapids_run.sh cuml+cudf <the script> <its arguments>"
   ```

   When the session is gone, `cat $(ls -d ~/tmpltest/artifacts/rapids/*/ | tail -1)summary.txt` counts the calls
   that ran on the GPU and those that fell back to the CPU, with each reason. HDBSCAN with the Euclidean metric
   should have run on the GPU; cuML's log lines say `ran on GPU` for it.
5. The template's version at full size on the GPU (the rapids environment, `fit.device` `cuda`), then its labels
   home with `export` and `pack` and `pull`.
6. Compare: the script's CPU labels and the GPU labels on the sample rows (adjusted Rand index, near 1 but not
   necessarily exactly 1: cuML's HDBSCAN is a different implementation), and the GPU run with the known clusters
   (`reveal-report`, after the choice is recorded).

## The flow check

The whole flow as a person meets it, start to finish, at small scale: four tables made by code and sent to the node
with push, one sweep for each kind of method, a round prepared and its RULE fixed, the round's own lines on the node
(install and test, start, status, pack, pull), and the reading on the computer in its fixed order. It shows that the
flow works and that each kind of configuration runs through it, not how good the methods are. It needs the main
environment with the profiles core, torch, unsupervised and boosting, on the computer and on the node, and one free
GPU for about half an hour; the round is 216 jobs of a few seconds each.

| Sweep (`sweeps/checks/`) | What it runs | Data | Configurations | Jobs |
|---|---|---|---|---|
| `flow_estimators.json` | classification: logistic regression (the base), random forest, histogram boosting, XGBoost (on the GPU), k nearest neighbours, a shuffled-labels control; 3 folds and the identity check fold | `classify.csv` | 6 | 72 |
| `flow_mlp.json` | classification: a PyTorch network on the GPU, one epoch, three shapes | `classify.csv` | 3 | 27 |
| `flow_regression.json` | regression: linear (the base), ridge, random forest, histogram boosting, a shuffled-target control | `regress.csv` | 5 | 45 |
| `flow_cluster.json` | clustering on 3 subsamples: k-means (the base) and a Gaussian mixture on 3 seeds, Ward's agglomerative clustering and HDBSCAN on 1, k from 3 to 5 | `blobs.csv` | 6 | 42 |
| `flow_embedding.json` | two-dimensional embeddings fitted on 70% of the rows: PCA (the base), UMAP, truncated SVD | `blobs.csv` | 3 | 9 |
| `flow_outliers.json` | outlier detection on 3 subsamples: isolation forest (the base), elliptic envelope, local outlier factor | `outliers.csv` | 3 | 21 |

Every sweep uses `fit.device` `auto` (the PyTorch one `cuda`), because a round's workers run on GPUs and take only
those jobs (`round-new` refuses a sweep with jobs on `cpu`).

1. The data, on the computer (Command Prompt; `.venv-main/bin/python` and `mv` on a Mac or Linux):

   ```
   .venv-main\Scripts\python -m sweepkit.cli make-synthetic --generator classification --n-rows 3000 --n-features 12 --n-groups 3 --noise 1.5 --seed 11 --name classify --chunk-rows 1000 --sample-rows 300
   .venv-main\Scripts\python -m sweepkit.cli make-synthetic --generator regression_nonlinear --n-rows 2000 --n-features 8 --n-groups 1 --noise 1.0 --seed 12 --name regress --chunk-rows 1000 --sample-rows 300
   .venv-main\Scripts\python -m sweepkit.cli make-synthetic --generator blobs --n-rows 3000 --n-features 8 --n-groups 4 --noise 1.0 --seed 13 --name blobs --chunk-rows 1000 --sample-rows 300
   .venv-main\Scripts\python -m sweepkit.cli make-synthetic --generator outliers --n-rows 2000 --n-features 5 --n-groups 3 --noise 1.0 --seed 14 --name outliers --chunk-rows 1000 --sample-rows 300
   ```

   Each writes `data/synthetic/<name>.csv` (`row_id`, the `f` columns, `group`, `published_split`, and `truth`,
   `target` or `is_outlier`), a sample and a record. Push sends one folder of `data/`, so move the twelve files into
   `data/flow/`: `mkdir data\flow && move data\synthetic\classify* data\flow\ && move data\synthetic\regress*
   data\flow\ && move data\synthetic\blobs* data\flow\ && move data\synthetic\outliers* data\flow\`.
2. Look at the data before any model, with the reference labels hidden: `profile-data --target flow/blobs.csv
   --row-id row_id --labels none --label-key none --max-rows all --top-values 5 --hide truth`, and the same for
   `flow/outliers.csv` with `--hide is_outlier,truth`. Then `check` on each of the six sweep files (it lists every
   configuration with its hash) and `smoke --device cpu --rows 200` on the five whose `fit.device` is `auto`;
   `flow_mlp.json` asks for the GPU (`cuda`), so its smoke test needs `--device cuda` on a machine with a GPU, or is
   left to the node, where the round's plan smoke-tests every sweep on one of the round's GPUs.
3. The round, on the computer, with `<date>` today (YYYYMMDD), `<g>` the GPU the node test may use and `<keep>` the
   free disk agreed in step 3 of the node test:

   ```
   .venv-main\Scripts\python -m sweepkit.cli round-new --round flow --date <date> --question "Does the whole flow work for every kind of method?" --sweep-files sweeps/checks/flow_estimators.json sweeps/checks/flow_mlp.json sweeps/checks/flow_regression.json sweeps/checks/flow_cluster.json sweeps/checks/flow_embedding.json sweeps/checks/flow_outliers.json --workers-per-gpu 3 --gpus <g> --end-time "now + 45 minutes" --environment main --profiles core torch unsupervised boosting --min-free-gb <keep>
   ```

   It writes `artifacts/analyses/flow_prep_<date>/` and the plan `sweeps/rounds/flow_prep_<date>/plan.txt`, which
   travels to the node with the code. Fill the RULE's four sections (what runs; the reading: analyze each sweep,
   select the configuration with the best value of its sweep's `select` criterion, then `test-report` for the three
   supervised choices, `reveal-report` for the three unsupervised ones, and `test-compare`; the stop condition: any
   failed run, sweep not enqueued or refusal stops the check; what it cannot show: how good the methods are). Write
   the comparison families into `flow_families.json` in the same folder, and add it to the README's list of files:

   ```json
   {"families": [
     {"name": "estimators_against_control", "files": "test", "metric": "balanced_accuracy",
      "comparisons": [{"a": {"selection": "flow_estimators_choice"}, "b": {"sweep": "check_flow_estimators", "config": "8b8b105d27"}}]},
     {"name": "regression_against_control", "files": "test", "metric": "rmse",
      "comparisons": [{"a": {"selection": "flow_regression_choice"}, "b": {"sweep": "check_flow_regression", "config": "3c1b489a5f"}}]}
   ]}
   ```

   (the hashes of the two shuffled controls, as `check` lists them). Then `rule-fix --name flow --rule-file
   artifacts/analyses/flow_prep_<date>/RULE_flow.txt --sweeps check_flow_estimators check_flow_mlp
   check_flow_regression check_flow_cluster check_flow_embedding check_flow_outliers --families
   artifacts/analyses/flow_prep_<date>/flow_families.json --late-reason none`, and `round-check --round flow --date
   <date>`, which should end with `ROUND OK`.
4. On the node, once: the main environment with those profiles (steps 4 and 5 above; `setup_env.sh main core torch
   unsupervised boosting`).
5. Push with the data instead of line 01: `powershell -ExecutionPolicy Bypass -File node\pc\push.ps1 -Data flow`
   (`bash node/pc/push.sh --data flow` on a Mac or Linux). On the node, the data's `unpack.sh` line push prints
   (it puts the folder in `~/tmpltest/data/flow/`), then lines 02 to 05 of `artifacts/analyses/flow_prep_<date>/lines/`,
   one at a time, each output compared with its `# expect:`: 02 installs the code and runs every test on a free GPU
   of the round's (`check.sh exit 0`); 03 starts the chain, whose first step checks the schedule
   (`logs/flow/1_schedule.txt`: `SCHEDULE OK`), whose second smoke-tests the six sweeps on GPU `<g>` and enqueues them
   (`logs/flow/2_smoke_enqueue.txt`: `SMOKE OK` for each, the jobs of each and `ALL DONE`) and whose third starts
   three workers on GPU `<g>` until the end time (`logs/flow/3_schedule.txt`); 04 shows the progress; 05 packs `runs/`
   and `sweeps/` of the six sweeps once the workers are gone.
   Line 06, on the computer, pulls the archive and checks every file.
6. The reading, on the computer, in the fixed order: `analyze` each sweep; `select` each choice (named
   `flow_estimators_choice`, `flow_mlp_choice`, `flow_regression_choice`, `flow_cluster_choice`,
   `flow_embedding_choice`, `flow_outliers_choice`); then `test-report` of the first three and `reveal-report` of
   the other three; then `test-compare --rule flow_<utc>`; then `export` and `cluster-report` of the clustering
   choice. Each choice should beat or match its sweep's base on the reading files, the two choices should be better
   than their shuffled controls, the identity check fold should agree, and the clustering choice should find the
   four planted groups (adjusted Rand index near 1).

## What ran where

Where each part of the template has run. The node test (2026-10-27 to 29) ran on a shared Linux node with NVIDIA H100
GPUs and Python 3.12, one free GPU at a time; every failure it found was fixed, and each became a test.

| Part | Where it ran | Result |
|---|---|---|
| The unit tests, core profile (Python 3.13, numpy, SciPy, pandas, scikit-learn, matplotlib, networkx) | the build container (Linux, CPU, no package index) | 621 tests, all passed, 128 skipped (packages not installed there, no GPU, the large-data and download tests off) |
| Every test, with every profile but RAPIDS and TPOT (`test_all.sh` with main: core torch unsupervised boosting search singlecell pretrained graph extras) | the node, GPU 3 | 592 tests (the code of 2026-10-28): one error, FLASC's test, redesigned since and passed on the node with the main environment of the flow check; the CPU and GPU agreement tests, `smoke-all` on cuda and on the CPU, and the large-data test at 1,000,000 rows passed |
| Every test with RAPIDS (`test_all.sh` with rapids: rapids rapids-singlecell) | the node, GPU 3 | 593 tests passed (110 skipped: the packages of other profiles); the agreement tests and `smoke-all` on both devices passed: cuML, cuGraph and rapids-singlecell match the CPU versions within the tests' bounds |
| Every test with TPOT (`test_all.sh` with automl: core automl) | the node, GPU 3 | passed, with agreement and `smoke-all` on both devices |
| Every test with the flow check's environment (main: core torch unsupervised boosting), through line 02 | the node, GPU 3 | the code of the flow check: 619 tests, one failure (`test_round_flow`: its workers' check that their GPU is visible asked torch, which sees no stand-in GPU; fixed); then the fixed code: 620 tests, all passed, `ENV OK` |
| `check-additions` | the build container | 200 components, metrics and stage kinds, 53 example sweeps: ADDITIONS OK |
| `smoke-all` on the CPU, core profile | the build container | 86 components ok, 0 failed, 48 skipped (29 need torch, the rest optional packages), in 15 s |
| The large-data test at 50,000 rows | the build container | passed in 16 s; the estimate predicted 332 MB of memory for the full run, which used 327 MB |
| The large-data test at a million rows | the build container | passed in 49 s; the estimate predicted 721 MB and 12.6 s for the full run, which used 1,006 MB and 10.1 s (two earlier runs: 1,005 MB against 720 and 745 MB) |
| docs/WALKTHROUGH.md and the notebook | the build container | every command of the walkthrough ran in a fresh copy and printed what the page shows; the notebook's code cells ran in a fresh copy |
| The worked examples of docs/EXTENDING.md (an estimator, a clusterer, a preparation step, a metric, a dataset, a split, a stage kind) | the build container | each scaffolded, edited as the page shows, its test passed, `check-additions` and `smoke` of its example passed |
| The example round (docs/EXAMPLE_ROUND.md) | the build container; the node | written by running it; a test compares it byte for byte where the core pins match. On the node (2026-10-28) it matched the page of that code byte for byte |
| The flow check (above) | the node, GPU 3, and the build container for the reading | 216 jobs of six sweeps through the round's own lines: 216 done, 0 failed, three workers on GPU 3 in about a minute; every reading command in the RULE's order ran without a refusal; both supervised choices beat their shuffled controls on test; the identity check fold agreed; the clustering choice found the four planted groups (adjusted Rand index 1.0000) |
| The overnight queue check (above) | the node, GPU 3 | 24 of 24 jobs on cuda, 0 failed |
| The laptop-to-GPU check (above) | the node, GPU 3 | no step fell back to the CPU; the script on the CPU and on the GPU agree (adjusted Rand index 1.000000) and the template's run gives the CPU script's labels exactly; on 1,000,000 rows the GPU run recovers the planted groups (adjusted Rand index 1.000000) |
| The node scripts | the build container, with stand-ins for tmux, nvidia-smi and an environment; then the node | in the container: push, unpack, the tree digest, pack, pull, the schedule (its checks before it starts anything, `--check`, and the settings it passes to its workers), the chain and its step logs, smoke_enqueue (a free GPU of the project, all sweeps or none, devices), the swap (each refusal before any change, the install and the schedule that replaces the old one, the timeout, the wait for the project's Python, an end that passes during the wait, an unpack that changes nothing), test_all's refusals and its time limit, check.sh's choice of a free GPU, and the read-only first line (it writes nothing). On the node: push, unpack, the tree digest, setup_env, check, test_all, the schedule, chain, smoke_enqueue, the workers, pack and pull, through the flow check's lines; not the swap |
| A whole round through its own lines (tests/test_round_flow.py) | the build container, and the node through line 02: a temporary home folder as the node, real tmux, stand-ins for nvidia-smi and systemctl | round-new, the RULE and round-check on the computer; push with the data; lines 02 to 05 as the preparation folder holds them (45 jobs by two workers on a GPU no machine has, so on no GPU); the pull line pack prints, pasted as it is; analyze, select, test-report and test-compare, in about 30 s |
| The payload tool (`src/sweepkit/payload.py`) | the build container, on Python 3.13 and 3.11; the Cowork shell on the person's computer (Linux, Python 3.10) | snapshot, check, make and install, with every refusal leaving the files unchanged; every install of the template and the test copy during the build |
| The setup scripts (env/setup.sh, env/setup.ps1, node/setup_env.sh) | `node/setup_env.sh` on the node; the other two not yet | `setup_env.sh` built main, rapids and automl on the node with the exact pins (sizes in step 3's table); env/setup.sh and env/setup.ps1 have not run: the build container could not reach the package index, and neither it nor the Cowork shell has PowerShell |
| Windows | the person's computer | `node/pc/push.ps1` and `node/pc/pull.ps1`, from Command Prompt with Windows PowerShell, at every step of the node test; not the setup script or the template's Python there |
| macOS, and a Linux computer | not yet | the node scripts' Mac and Linux versions (`push.sh`, `pull.sh`) ran in the build container's tests, and the tree digest with stand-ins for a Mac (shasum without sha256sum, an xargs that refuses `-r`) |
| Anything in the Linux shell a Claude Cowork session gets on the person's computer | not yet: when the template was built, that shell had Python 3.10 and no package index, and the template needs Python 3.12 or 3.13 | the person runs the lines in their own terminal instead (above) |

The bounds of the tests were set before any run, from what each method guarantees on well separated synthetic data.
Where a test failed on the node, the cause was found and fixed, and the test ran again there.

<!-- The testing matrix below is written by `python -m sweepkit.cli test-matrix --write yes` from the code; edit the code or the tests, not this section. -->

## The testing matrix

Every component of the template, and every method of a preparation step, with: the tests that name it (the most specific first; `file.Class.method`) and the bounds each asserts; its smoke entry (the example sweep whose first configuration using it `smoke-all` runs); and, for a GPU path that is a separate implementation (cuML, cuGraph, rapids-singlecell, XGBoost's CUDA), the tests that run the CPU and GPU versions on the same data and compare them. PyTorch models run the same code on either device. A bound is shown as the test asserts it, so its left side is code: `ari(truth, labels)` is the adjusted Rand index with the known clusters.

### Clustering

| Model | Profile | GPU path | Smoke entry | Tests and their bounds | CPU and GPU agreement |
|---|---|---|---|---|---|
| `affinity_propagation` | core | none | `cluster_graph` | test_cluster_models.TestGraph.test_affinity_propagation_on_blobs: `ari(truth, model.labels())` ≥ 0.95 | no GPU path |
| `agglomerative` | core | cuml | `cluster_hierarchical` | test_distances.TestPrecomputedEqualsComputed.test_agglomerative: 3 exact checks<br>test_cluster_models.TestHierarchical.test_ward_on_blobs_writes_the_tree: `ari(truth, model.labels())` ≥ 0.95<br>test_cluster_models.TestHierarchical.test_single_linkage_separates_circles_and_moons: `ari(truth, model.labels())` ≥ 0.95<br>and more | test_cluster_models.TestGraphOnTheGPU.test_single_linkage_on_cuml_matches_scipy: `ari(cpu.labels(), gpu.labels())` ≥ 0.99 |
| `autoencoder_kmeans` | torch | torch | `cluster_deep` | test_deep_models.TestDeclarations.test_every_example_configuration_passes_its_checks: `summary["configs"]` > 3<br>test_deep_models.TestDeclarations.test_registry_entries: 1 exact check<br>test_deep_models.TestDeclarations.test_settings_the_checks_refuse: no numeric check<br>and more | the same code on both devices |
| `bayesian_gmm` | core | none | `cluster_gmm` | test_unsupervised_models.TestCentroidAndMixtures.test_bayesian_gmm_switches_off_the_components_it_does_not_need: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>test_criteria.TestClusteringCriteria.test_aic_and_bic_come_from_the_mixture_model: 2 exact checks | no GPU path |
| `birch` | core | none | `cluster_birch` | test_cluster_models.TestOtherMethods.test_birch: `ari(truth, model.labels())` ≥ 0.95 | no GPU path |
| `bisecting_kmeans` | core | none | `cluster_kmeans` | test_unsupervised_models.TestCentroidAndMixtures.test_bisecting_kmeans: `ari(columns["truth"], model.labels())` ≥ 0.95; `model.info()["inertia"]` > 0<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95 | no GPU path |
| `consensus` | core | none | `cluster_consensus` | test_cluster_models.TestConsensus.test_co_association_and_clusters: `ari(truth, model.labels())` ≥ 0.95; `info["within_co_association"]` > 0.95<br>test_cluster_models.TestConsensus.test_a_consensus_of_consensus_is_refused: no numeric check<br>test_cluster_models.TestConsensus.test_noise_joins_no_cluster: 1 exact check | no GPU path |
| `dbscan` | core | cuml | `cluster_density` | test_cluster_models.TestConsensus.test_noise_joins_no_cluster: 1 exact check<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>test_distances.TestPrecomputedEqualsComputed.test_density_methods: 1 exact check | test_gpu_agreement.TestClusterersAndEmbedders.test_dbscan: `ari(cpu, gpu)` ≥ 0.99 |
| `dec` | torch | torch | `cluster_deep` | test_deep_models.TestDeclarations.test_every_example_configuration_passes_its_checks: `summary["configs"]` > 3<br>test_deep_models.TestDeclarations.test_registry_entries: 1 exact check<br>test_deep_models.TestDeclarations.test_settings_the_checks_refuse: no numeric check<br>and more | the same code on both devices |
| `dtw_kmeans` | unsupervised | none | `cluster_series` | test_cluster_models.TestOtherMethods.test_series_by_dtw_and_kshape: `ari(truth, dtw.labels())` ≥ 0.75; `ari(truth, shape.labels())` > 0.3 | no GPU path |
| `fast_hdbscan` | unsupervised | none | `cluster_hdbscan` | test_unsupervised_models.TestDensity.test_fast_hdbscan: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_unsupervised_models.TestDensity.test_plscan_picks_its_own_resolution: `ari(columns["truth"], model.labels())` ≥ 0.9<br>test_unsupervised_models.TestDensity.test_flasc_branches_split_a_y_into_its_arms: `on_the_y.size` ≥ 3<br>and more | no GPU path |
| `flasc_branches` | unsupervised | none | `cluster_hdbscan` | test_unsupervised_models.TestDensity.test_flasc_branches_split_a_y_into_its_arms: `on_the_y.size` ≥ 3<br>test_unsupervised_models.TestDensity.test_flasc_branches_says_why_when_hdbscan_finds_no_cluster: no numeric check | no GPU path |
| `flowsom` | core | none | `cluster_som` | test_cluster_models.TestOtherMethods.test_flowsom: `ari(truth, model.labels())` ≥ 0.95 | no GPU path |
| `fuzzy_cmeans` | core | none | `cluster_kmeans` | test_unsupervised_models.TestCentroidAndMixtures.test_fuzzy_cmeans_memberships_sum_to_one: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95 | no GPU path |
| `genie` | extras | none | `cluster_hierarchical` | test_cluster_models.TestHierarchical.test_genie_on_moons: `ari(truth, model.labels())` ≥ 0.9 | no GPU path |
| `gmm` | core | none | `cluster_gmm` | test_unsupervised_models.TestCentroidAndMixtures.test_gmm_finds_the_groups_and_bic_is_lowest_at_three: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>test_criteria.TestClusteringCriteria.test_aic_and_bic_come_from_the_mixture_model: 2 exact checks | no GPU path |
| `hdbscan` | unsupervised | cuml | `cluster_hdbscan` | test_unsupervised_models.TestDensity.test_hdbscan_package_gives_outlier_scores_and_assigns_new_rows: `ari(columns["truth"], labels)` ≥ 0.95; `(model.assign(x) == labels).mean()` ≥ 0.95<br>test_unsupervised_models.TestDetectors.test_hdbscan_outliers_glosh_scores: `auroc(columns, scores)` ≥ 0.9<br>test_checks.TestLaptopStandin.test_the_script_clusters_a_small_table: 1 exact check | test_gpu_agreement.TestClusterersAndEmbedders.test_hdbscan_both_implementations: `ari(cpu, gpu)` ≥ 0.95; `ari(truth, gpu)` ≥ 0.95<br>and more |
| `hdbscan_sklearn` | core | cuml | `cluster_hdbscan` | test_unsupervised_models.TestDensity.test_hdbscan_sklearn: `ari(columns["truth"], labels)` ≥ 0.95; `np.unique(labels[labels >= 0]).size` > 0<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>test_unsupervised_models.TestDensity.test_auto_records_why_a_setting_ran_on_the_cpu: 2 exact checks<br>and more | test_gpu_agreement.TestClusterersAndEmbedders.test_hdbscan_both_implementations: `ari(cpu, gpu)` ≥ 0.95; `ari(truth, gpu)` ≥ 0.95<br>and more |
| `kmeans` | core | cuml | `cluster_kmeans` | test_unsupervised_models.TestCentroidAndMixtures.test_kmeans_and_minibatch_kmeans_find_the_groups: `ari(columns["truth"], model.labels())` ≥ 0.95; `model.info()["inertia"]` > 0<br>test_deep_models.TestDeepClustering.test_kmeans_on_the_codes_finds_the_planted_groups: `ari(truth, model.labels())` ≥ 0.9<br>test_large_data.TestLargeData.test_large_data_path: `reveal["metrics"]["ari"]` ≥ 0.95<br>and more | test_gpu_agreement.TestClusterersAndEmbedders.test_kmeans: `ari(cpu, gpu)` ≥ 0.99; `ari(truth, gpu)` ≥ 0.99<br>and more |
| `kmedoids` | core | none | `cluster_kmedoids` | test_unsupervised_models.TestCentroidAndMixtures.test_kmedoids_medoids_are_rows_at_a_fixed_point: `ari(columns["truth"], labels)` ≥ 0.95<br>test_unsupervised_models.TestCentroidAndMixtures.test_kmedoids_with_gower_on_mixed_columns: `ari(columns["truth"], model.labels())` ≥ 0.9<br>test_unsupervised_models.TestCentroidAndMixtures.test_kmedoids_fasterpam: `ari(columns["truth"], model.labels())` ≥ 0.95<br>and more | no GPU path |
| `kprototypes` | core | none | `cluster_mixed` | test_unsupervised_models.TestCentroidAndMixtures.test_kprototypes_on_mixed_columns: `ari(columns["truth"], model.labels())` ≥ 0.85<br>test_unsupervised_models.TestThroughTheFitProcedures.test_kprototypes_receives_the_categorical_columns: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.9 | no GPU path |
| `kshape` | unsupervised | none | `cluster_series` | test_cluster_models.TestOtherMethods.test_series_by_dtw_and_kshape: `ari(truth, dtw.labels())` ≥ 0.75; `ari(truth, shape.labels())` > 0.3 | no GPU path |
| `lda` | core | none | `cluster_topics` | test_cluster_models.TestOtherMethods.test_lda_topics: `ari(truth, model.labels())` ≥ 0.9; `model.info()["perplexity"]` > 0 | no GPU path |
| `leiden` | core | cugraph | `cluster_graph` | test_cluster_models.TestGraph.test_leiden_networkx: `ari(truth, model.labels())` ≥ 0.95; `ari(truth, cpm.labels())` ≥ 0.95<br>test_cluster_models.TestGraph.test_leiden_with_leidenalg: `ari(truth, model.labels())` ≥ 0.9<br>test_cluster_models.TestGraph.test_graph_settings_checked: no numeric check | test_cluster_models.TestGraphOnTheGPU.test_leiden_and_louvain_on_cugraph: `ari(truth, gpu.labels())` ≥ 0.85; `ari(cpu.labels(), gpu.labels())` ≥ 0.85 |
| `louvain` | core | cugraph | `cluster_graph` | test_cluster_models.TestGraph.test_louvain_on_jaccard_weights: `ari(truth, model.labels())` ≥ 0.95; `model.info()["modularity"]` > 0.5 | test_cluster_models.TestGraphOnTheGPU.test_leiden_and_louvain_on_cugraph: `ari(truth, gpu.labels())` ≥ 0.85; `ari(cpu.labels(), gpu.labels())` ≥ 0.85 |
| `lumbermark` | extras | none | `cluster_lumbermark` | test_cluster_models.TestPackagesFromExtrasAndSingleCell.test_lumbermark_on_blobs_and_moons: `ari(truth, model.labels())` ≥ 0.9 | no GPU path |
| `markov` | core | none | `cluster_graph` | test_cluster_models.TestGraph.test_markov_finds_the_groups_and_keeps_columns_stochastic: `ari(truth, model.labels())` ≥ 0.95; `info["column_sum_error"]` < 1e-09 | no GPU path |
| `mean_shift` | core | none | `cluster_density` | test_unsupervised_models.TestDensity.test_mean_shift_estimates_its_bandwidth: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95 | no GPU path |
| `minibatch_kmeans` | core | none | `cluster_kmeans` | test_unsupervised_models.TestCentroidAndMixtures.test_kmeans_and_minibatch_kmeans_find_the_groups: `ari(columns["truth"], model.labels())` ≥ 0.95; `model.info()["inertia"]` > 0<br>test_sweep.TestOverrides.test_paths_through_prep_and_whole_sections: 4 exact checks | no GPU path |
| `optics` | core | none | `cluster_density` | test_unsupervised_models.TestDensity.test_optics_by_xi_and_by_a_dbscan_cut: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_unsupervised_models.TestDensity.test_optics_settings_belong_to_their_extraction_method: no numeric check<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>and more | no GPU path |
| `plscan` | unsupervised | none | `cluster_hdbscan` | test_unsupervised_models.TestDensity.test_plscan_picks_its_own_resolution: `ari(columns["truth"], model.labels())` ≥ 0.9 | no GPU path |
| `scanpy_leiden` | singlecell | rapids_singlecell | `singlecell_workflow` | test_cluster_models.TestPackagesFromExtrasAndSingleCell.test_scanpy_leiden_on_simulated_counts: `ari(truth, model.labels())` ≥ 0.8 | test_cluster_models.TestPackagesFromExtrasAndSingleCell.test_scanpy_leiden_on_the_gpu_finds_the_cell_groups: `ari(truth, labels)` ≥ 0.8<br>and more |
| `som` | core | none | `cluster_som` | test_cluster_models.TestOtherMethods.test_som_quantization_error_falls: `ari(truth, model.labels())` ≥ 0.95<br>test_cluster_models.TestOtherMethods.test_som_is_deterministic_for_a_seed: 1 exact check | no GPU path |
| `spectral` | core | none | `cluster_graph` | test_cluster_models.TestGraph.test_spectral_on_circles: `ari(truth, model.labels())` ≥ 0.95 | no GPU path |
| `spectral_biclustering` | core | none | `cluster_bicluster` | test_cluster_models.TestOtherMethods.test_biclustering_finds_rows_and_columns: `ari(truth, model.labels())` ≥ 0.95; `ari(columns, model.extra_files()["column_labels.npy"])` ≥ 0.95 | no GPU path |
| `spectral_coclustering` | core | none | `cluster_bicluster` | test_cluster_models.TestOtherMethods.test_biclustering_finds_rows_and_columns: `ari(truth, model.labels())` ≥ 0.95; `ari(columns, model.extra_files()["column_labels.npy"])` ≥ 0.95 | no GPU path |

### Dimensionality reduction and embeddings

| Model | Profile | GPU path | Smoke entry | Tests and their bounds | CPU and GPU agreement |
|---|---|---|---|---|---|
| `autoencoder` | torch | torch | `embed_deep` | test_deep_models.TestDeepEmbedders.test_the_autoencoder_keeps_planted_groups_apart: `label_agreement(e, truth)` ≥ 0.9; `np.mean((rebuilt - x) ** 2)` < 1<br>test_deep_models.TestDeclarations.test_every_example_configuration_passes_its_checks: `summary["configs"]` > 3<br>test_deep_models.TestDeepClustering.test_the_example_runs_through_the_clusterer_runner: 1 exact check<br>and more | the same code on both devices |
| `diffusion_map` | core | none | `embed_diffusion` | test_embed_models.TestManifold.test_diffusion_map_recovers_the_angle_and_places_new_rows: `angle_recovery(e, angle)` ≥ 0.95; `angle_recovery(placed, other_columns["reference"].astype(...` ≥ 0.95<br>test_embed_models.TestManifold.test_diffusion_map_solvers_agree: no numeric check<br>test_embed_models.TestThroughTheRunner.test_smoke_of_a_small_sweep: 1 exact check<br>and more | no GPU path |
| `factor_analysis` | core | none | `embed_linear` | test_unsupervised_models.TestLinearEmbedders.test_factor_analysis_finds_the_low_rank_subspace: `smallest_cosine(model.model.components_, vt[:2])` ≥ 0.99<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01 | no GPU path |
| `ica` | core | none | `embed_linear` | test_unsupervised_models.TestLinearEmbedders.test_ica_recovers_independent_sources_up_to_sign_and_order: `r.max(axis=1).min()` ≥ 0.95<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01 | no GPU path |
| `isomap` | core | none | `embed_manifold` | test_embed_models.TestManifold.test_isomap_unrolls_and_places_fit_rows_exactly: `trust(roll, e)` ≥ 0.95<br>test_embed_models.TestThroughTheRunner.test_smoke_of_a_small_sweep: 1 exact check | no GPU path |
| `kernel_pca` | core | none | `embed_linear` | test_unsupervised_models.TestLinearEmbedders.test_kernel_pca_with_a_linear_kernel_is_pca: `((rebuilt - x) ** 2).mean() / (centred ** 2).mean()` ≤ 0.05; `abs(np.corrcoef(e[:, j], scores[:, j])[0, 1])` ≥ 0.999<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01 | no GPU path |
| `lle` | core | none | `embed_manifold` | test_embed_models.TestManifold.test_lle_variants: `trust(clean_roll, standard.fit_transform(clean_roll))` ≥ 0.95; `trust(clean_roll, model.fit_transform(clean_roll))` ≥ 0.8<br>test_embed_models.TestChecksAtExpansion.test_refusals: no numeric check<br>test_embed_models.TestManifold.test_same_seed_same_embedding: no numeric check | no GPU path |
| `mds` | core | none | `embed_manifold` | test_embed_models.TestManifold.test_spectral_embedding_and_mds: `trust(roll, spectral.fit_transform(roll))` ≥ 0.93; `trust(circle, mds.fit_transform(circle))` ≥ 0.95<br>test_distances.TestPrecomputedEqualsComputed.test_pcoa_and_mds: no numeric check<br>test_prep.TestReduce.test_reduce_without_a_transform_needs_the_evaluated_rows_to_be_the_fit_rows: 3 exact checks | no GPU path |
| `nmf` | core | none | `embed_linear` | test_unsupervised_models.TestLinearEmbedders.test_nmf_on_a_non_negative_product: `error` ≤ 0.02<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01 | no GPU path |
| `opentsne` | unsupervised | none | `embed_neighbors` | test_embed_models.TestOpenTSNE.test_blobs_and_new_rows: `label_agreement(e, truth[:500])` ≥ 0.9; `label_agreement(combined, truth)` ≥ 0.9 | no GPU path |
| `pacmap` | unsupervised | none | `embed_neighbors` | test_embed_models.TestPaCMAP.test_blobs_and_transform: `label_agreement(e, columns["truth"][:500])` ≥ 0.9 | no GPU path |
| `pca` | core | cuml | `embed_linear` | test_prep.TestReduce.test_reduce_with_pca_fitted_on_the_fit_rows: `record["explained_variance"]` > 0.99<br>test_settings.TestScalarAndSections.test_registry_section_validates_against_the_named_component: 2 exact checks<br>test_criteria.TestEmbeddingCriteria.test_reconstruction_error_as_the_embedder_runner_computes_it: 2 exact checks | test_gpu_agreement.TestClusterersAndEmbedders.test_pca_and_truncated_svd_span_the_same_subspace: `np.linalg.svd(q1.T @ q2, compute_uv=False).min()` ≥ 0.999 |
| `pcoa` | core | none | `embed_manifold` | test_embed_models.TestManifold.test_pcoa_reports_negative_eigenvalues: `model.info()["negative_eigenvalue_share"]` > 0.05<br>test_embed_models.TestManifold.test_pcoa_on_euclidean_distances_is_pca: 1 exact check<br>test_distances.TestPrecomputedEqualsComputed.test_pcoa_and_mds: no numeric check<br>and more | no GPU path |
| `phate` | extras | none | `embed_neighbors` | test_embed_models.TestPHATE.test_blobs: `label_agreement(e, columns["truth"])` ≥ 0.85 | no GPU path |
| `spectral_embedding` | core | cuml | `embed_manifold` | test_embed_models.TestManifold.test_spectral_embedding_and_mds: `trust(roll, spectral.fit_transform(roll))` ≥ 0.93; `trust(circle, mds.fit_transform(circle))` ≥ 0.95<br>test_embed_models.TestChecksAtExpansion.test_refusals: no numeric check | test_embed_models.TestCuML.test_umap_tsne_and_spectral_on_the_gpu: `label_agreement(build("umap", UMAP, device="cuda").fit_tr...` ≥ 0.9; `label_agreement(tsne.fit_transform(x), truth)` ≥ 0.9 |
| `trimap` | extras | torch | `embed_neighbors` | test_embed_models.TestTriMap.test_blobs: `label_agreement(e, columns["truth"])` ≥ 0.9 | the same code on both devices |
| `truncated_svd` | core | cuml | `embed_linear` | test_unsupervised_models.TestLinearEmbedders.test_truncated_svd_keeps_a_low_rank_table: `model.info()["explained_variance"]` ≥ 0.99; `((rebuilt - x) ** 2).mean() / (x ** 2).mean()` ≤ 0.01<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01 | test_gpu_agreement.TestClusterersAndEmbedders.test_pca_and_truncated_svd_span_the_same_subspace: `np.linalg.svd(q1.T @ q2, compute_uv=False).min()` ≥ 0.999 |
| `tsne` | core | cuml | `embed_neighbors` | test_embed_models.TestManifold.test_tsne_keeps_blobs_apart: `label_agreement(e, columns["truth"])` ≥ 0.9<br>test_embed_models.TestChecksAtExpansion.test_refusals: no numeric check<br>test_embed_models.TestManifold.test_same_seed_same_embedding: no numeric check | test_embed_models.TestCuML.test_umap_tsne_and_spectral_on_the_gpu: `label_agreement(build("umap", UMAP, device="cuda").fit_tr...` ≥ 0.9; `label_agreement(tsne.fit_transform(x), truth)` ≥ 0.9 |
| `umap` | unsupervised | cuml | `embed_neighbors` | test_prep.TestReduce.test_reduce_with_umap: 2 exact checks<br>test_embed_models.TestUMAP.test_blobs_and_transform: `label_agreement(e, columns["truth"][:500])` ≥ 0.9<br>test_embed_models.TestChecksAtExpansion.test_refusals: no numeric check | test_embed_models.TestCuML.test_umap_tsne_and_spectral_on_the_gpu: `label_agreement(build("umap", UMAP, device="cuda").fit_tr...` ≥ 0.9; `label_agreement(tsne.fit_transform(x), truth)` ≥ 0.9 |
| `vae` | torch | torch | `embed_deep` | test_deep_models.TestDeepEmbedders.test_the_vae_keeps_planted_groups_apart: `label_agreement(e, truth)` ≥ 0.85; `model.info()["final_kl"]` > 0<br>test_deep_models.TestDeclarations.test_every_example_configuration_passes_its_checks: `summary["configs"]` > 3<br>test_deep_models.TestDeclarations.test_registry_entries: 1 exact check<br>and more | the same code on both devices |

### Outlier detection

| Model | Profile | GPU path | Smoke entry | Tests and their bounds | CPU and GPU agreement |
|---|---|---|---|---|---|
| `elliptic_envelope` | core | none | `outlier_detectors` | test_unsupervised_models.TestDetectors.test_elliptic_envelope_on_one_gaussian_cloud: `auroc(columns, scores)` ≥ 0.97<br>test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_prep.TestOutliers.test_outliers_remove_by_detector_takes_flagged_fit_rows_out: 3 exact checks | no GPU path |
| `hdbscan_outliers` | unsupervised | none | `outlier_detectors` | test_unsupervised_models.TestDetectors.test_hdbscan_outliers_glosh_scores: `auroc(columns, scores)` ≥ 0.9 | no GPU path |
| `isolation_forest` | core | cuml | `outlier_detectors` | test_unsupervised_models.TestDetectors.test_isolation_forest: `auroc(columns, scores)` ≥ 0.98<br>test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_the_task_needs_every_em_and_mv_option: no numeric check | test_gpu_agreement.TestDetector.test_isolation_forest_ranks_the_same_outliers: `spearmanr(scores[0], scores[1])[0]` ≥ 0.8; `roc_auc_score(flag, scores[-1])` ≥ 0.95 |
| `lof` | core | none | `outlier_detectors` | test_unsupervised_models.TestDetectors.test_local_outlier_factor_scores_new_rows: `auroc(columns, scores)` ≥ 0.98<br>test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9 | no GPU path |
| `one_class_svm` | core | none | `outlier_detectors` | test_unsupervised_models.TestDetectors.test_one_class_svm_flags_about_nu: `auroc(columns, scores)` ≥ 0.93<br>test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9 | no GPU path |

### Estimators (classification and regression)

| Model | Profile | GPU path | Smoke entry | Tests and their bounds | CPU and GPU agreement |
|---|---|---|---|---|---|
| `decision_tree` | core | none | `classify_trees` | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more | no GPU path |
| `extra_trees` | core | none | `classify_trees` | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more | no GPU path |
| `hist_gradient_boosting` | core | none | `classify_boosting` | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestClassification.test_log_loss_needs_probabilities: `files["metrics_val.json"]["metrics"]["log_loss"]` > 0<br>test_estimators.TestClassification.test_every_classifier_learns_the_planted_classes: `metrics["balanced_accuracy"]` ≥ 0.85; `metrics["roc_auc"]` ≥ 0.9<br>and more | no GPU path |
| `knn` | core | cuml | `classify_knn` | test_prep.TestImpute.test_impute_knn_and_iterative_follow_scikit_learn_fitted_on_the_fit_rows: 2 exact checks<br>test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>and more | test_gpu_agreement.TestEstimators.test_classifiers_agree: `np.mean(cpu.astype(int) == gpu.astype(int))` ≥ 0.95; `np.mean(gpu.astype(int) == y)` ≥ 0.8 |
| `lightgbm` | boosting | none | `classify_boosting` | test_estimators.TestOptionalLibraries.test_lightgbm: `metrics["balanced_accuracy"]` ≥ 0.85; `run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.j...` > 0.5<br>test_estimators.TestSettingChecks.test_classification_settings: no numeric check<br>test_estimators.TestSettingChecks.test_tasks_a_model_does_not_do: no numeric check | no GPU path |
| `linear` | core | cuml | `regress_linear` | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestSettingChecks.test_tasks_a_model_does_not_do: no numeric check<br>test_estimators.TestSettingChecks.test_regression_settings: no numeric check<br>and more | test_gpu_agreement.TestEstimators.test_linear_regression_agrees: `np.corrcoef(cpu, gpu)[0, 1]` ≥ 0.999 |
| `linear_svm` | core | cuml | `classify_svm` | test_estimators.TestMultilabel.test_one_linear_svm_per_label_gives_decision_values: no numeric check<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more | test_gpu_agreement.TestEstimators.test_classifiers_agree: `np.mean(cpu.astype(int) == gpu.astype(int))` ≥ 0.95; `np.mean(gpu.astype(int) == y)` ≥ 0.8 |
| `logistic_regression` | core | cuml | `classify_logistic` | test_torch_family.TestKnownStructure.test_mlp_on_tabular_classes_and_regression: `mlp` > 0.75; `held_out(folder, "r2")` > 0.8<br>test_estimators.TestSettingChecks.test_classification_settings: no numeric check<br>test_estimators.TestSettingChecks.test_regression_settings: no numeric check<br>and more | test_gpu_agreement.TestEstimators.test_classifiers_agree: `np.mean(cpu.astype(int) == gpu.astype(int))` ≥ 0.95; `np.mean(gpu.astype(int) == y)` ≥ 0.8 |
| `naive_bayes` | core | none | `classify_bayes` | test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>test_estimators.TestClassification.test_log_loss_needs_probabilities: `files["metrics_val.json"]["metrics"]["log_loss"]` > 0<br>and more | no GPU path |
| `random_forest` | core | cuml | `classify_trees` | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more | test_gpu_agreement.TestEstimators.test_classifiers_agree: `np.mean(cpu.astype(int) == gpu.astype(int))` ≥ 0.95; `np.mean(gpu.astype(int) == y)` ≥ 0.8 |
| `svm` | core | cuml | `classify_svm` | test_estimators.TestMultilabel.test_one_linear_svm_per_label_gives_decision_values: no numeric check<br>test_estimators.TestClassification.test_svm_without_probabilities_raises_no_deprecation_warning: 1 exact check<br>test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>and more | test_estimators.TestSVMWithoutGPUProbabilities.test_cuda_is_refused_and_auto_runs_on_the_cpu: 2 exact checks<br>and more |
| `tpot` | automl | none | `classify_tpot` | test_estimators.TestOptionalLibraries.test_tpot_with_a_short_budget: `metrics["balanced_accuracy"]` ≥ 0.7<br>test_estimators.TestSettingChecks.test_tasks_a_model_does_not_do: no numeric check | no GPU path |
| `xgboost` | boosting | xgboost | `classify_boosting` | test_estimators.TestOptionalLibraries.test_xgboost: `metrics["balanced_accuracy"]` ≥ 0.85; `run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.j...` > 0.5<br>test_estimators.TestOptionalLibraries.test_lightgbm: `metrics["balanced_accuracy"]` ≥ 0.85; `run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.j...` > 0.5<br>test_estimators.TestSettingChecks.test_classification_settings: no numeric check<br>and more | test_gpu_agreement.TestXGBoostOnCUDA.test_xgboost_cuda_agrees: `np.mean(cpu == gpu)` ≥ 0.97; `np.mean(gpu.astype(int) == y)` ≥ 0.8 |

### Neural networks (PyTorch)

| Model | Profile | GPU path | Smoke entry | Tests and their bounds | CPU and GPU agreement |
|---|---|---|---|---|---|
| `cnn1d` | torch | torch | `sequence_models` | test_torch_models.TestConvolutional.test_every_pooling_gives_finite_outputs_on_sequences_and_images: 3 exact checks<br>test_torch_models.TestConvolutional.test_steps_after_blocks_match_the_network: 2 exact checks<br>test_torch_family.TestKnownStructure.test_convolutional_recurrent_and_transformer_models_on_sequences: no numeric check<br>and more | the same code on both devices |
| `cnn2d` | torch | torch | `image_cnn` | test_torch_family.TestKnownStructure.test_cnn2d_on_images: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_models.TestModelChecks.test_cnn2d_shapes_norms_and_heads: no numeric check<br>test_torch_family.TestKnownStructure.test_spectrogram_models_on_audio: `held_out(folder, "balanced_accuracy")` > 0.8<br>and more | the same code on both devices |
| `cnn_lif` | torch | torch | `spiking` | test_torch_family.TestKnownStructure.test_spiking_models_on_sequences: `held_out(folder, "balanced_accuracy")` > 0.6<br>test_torch_models.TestSpikingModels.test_recording_through_a_model: `report["mac_per_sequence"]` > 0<br>test_lif_reference.TestSpikingSettings.test_a_sweep_over_both_models_expands: 2 exact checks<br>and more | the same code on both devices |
| `cnn_rnn` | torch | torch | `audio_logmel` | test_torch_models.TestSequenceModels.test_cnn_rnn_packing_keeps_padding_out: 4 exact checks<br>test_torch_family.TestKnownStructure.test_spectrogram_models_on_audio: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_family.TestChecks.test_inputs_and_augmentations: no numeric check<br>and more | the same code on both devices |
| `cnn_transformer` | torch | torch | `audio_logmel` | test_torch_models.TestConvolutional.test_cnn_transformer_and_its_padding_mask: 1 exact check<br>test_torch_family.TestKnownStructure.test_spectrogram_models_on_audio: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_family.TestChecks.test_inputs_and_augmentations: no numeric check<br>and more | the same code on both devices |
| `composite` | torch | torch | `sequence_models` | test_torch_models.TestModelChecks.test_composite_takes_exactly_the_blocks_it_names: no numeric check<br>test_torch_models.TestSequenceModels.test_composite_models: 5 exact checks<br>test_torch_family.TestKnownStructure.test_convolutional_recurrent_and_transformer_models_on_sequences: no numeric check<br>and more | the same code on both devices |
| `gat` | torch | torch | `graph_networks` | test_graph_models.TestRuns.test_gcn_and_gat_find_the_planted_communities_and_the_control_does_not: `np.mean(control)` < 0.6; `read(folder, "metrics_test.json")["metrics"]["balanced_ac...` ≥ 0.8<br>test_graph_models.TestLayers.test_one_gat_layer_is_a_softmax_over_incoming_edges: no numeric check<br>test_graph_models.TestRuns.test_the_graph_datasets_example_runs_on_the_example_files: `read(folder, "metrics_val.json")["select"]["value"]` ≥ 0.6<br>and more | the same code on both devices |
| `gcn` | torch | torch | `graph_networks` | test_graph_models.TestRuns.test_gcn_and_gat_find_the_planted_communities_and_the_control_does_not: `np.mean(control)` < 0.6; `read(folder, "metrics_test.json")["metrics"]["balanced_ac...` ≥ 0.8<br>test_graph_models.TestLayers.test_one_gcn_layer_is_the_dense_propagation_rule: no numeric check<br>test_graph_models.TestRuns.test_the_graph_datasets_example_runs_on_the_example_files: `read(folder, "metrics_val.json")["select"]["value"]` ≥ 0.6<br>and more | the same code on both devices |
| `hf_finetune` | pretrained | torch | `text_finetune` | test_pretrained.TestDeclarationAndChecks.test_registry_entry: 3 exact checks<br>test_pretrained.TestDeclarationAndChecks.test_settings_the_checks_refuse: no numeric check<br>test_pretrained.TestFinetune.test_each_training_mode_through_the_runner: 12 exact checks<br>and more | the same code on both devices |
| `mlp` | torch | torch | `classify_mlp` | test_torch_family.TestKnownStructure.test_mlp_on_tabular_classes_and_regression: `mlp` > 0.75; `held_out(folder, "r2")` > 0.8<br>test_torch_models.TestModelChecks.test_mlp: no numeric check<br>test_torch_models.TestSequenceModels.test_mlp: 4 exact checks<br>and more | the same code on both devices |
| `rnn` | torch | torch | `sequence_models` | test_torch_models.TestStep.test_extra_heads_and_the_penalty_add_their_weighted_losses: `head.out.weight.grad.abs().sum().item()` > 0<br>test_torch_family.TestKnownStructure.test_sequence_heads_on_per_step_features: `accuracy` > 0.8<br>test_torch_models.TestStep.test_the_step_is_its_parts: 2 exact checks<br>and more | the same code on both devices |
| `sequence_lif` | torch | torch | `spiking` | test_torch_models.TestStep.test_extra_heads_and_the_penalty_add_their_weighted_losses: `head.out.weight.grad.abs().sum().item()` > 0<br>test_torch_family.TestKnownStructure.test_spiking_models_on_sequences: `held_out(folder, "balanced_accuracy")` > 0.6<br>test_torch_models.TestSpikingModels.test_recording_through_a_model: `report["mac_per_sequence"]` > 0<br>and more | the same code on both devices |
| `sequence_pool` | torch | torch | `sequence_models` | test_torch_family.TestKnownStructure.test_sequence_heads_on_per_step_features: `accuracy` > 0.8<br>test_torch_models.TestSequenceModels.test_widths: 6 exact checks<br>test_torch_models.TestSequenceModels.test_padding_changes_nothing: 2 exact checks<br>and more | the same code on both devices |
| `snn` | torch | torch | `spiking` | test_torch_family.TestKnownStructure.test_spiking_models_on_sequences: `held_out(folder, "balanced_accuracy")` > 0.6<br>test_torch_models.TestSpikingModels.test_recording_through_a_model: `report["mac_per_sequence"]` > 0<br>test_torch_models.TestSpikingModels.test_fully_spiking_both_encodings: no numeric check<br>and more | the same code on both devices |
| `transformer` | torch | torch | `sequence_models` | test_torch_family.TestKnownStructure.test_convolutional_recurrent_and_transformer_models_on_sequences: no numeric check<br>test_torch_family.TestKnownStructure.test_sequence_heads_on_per_step_features: `accuracy` > 0.8<br>test_torch_models.TestSequenceModels.test_widths: 6 exact checks<br>and more | the same code on both devices |

### Preparation steps

| Step | Method | Profile | Smoke entry | Tests and their bounds |
|---|---|---|---|---|
| `batch_correct` | `center` | core | `prep_batch` | test_prep.TestBatchCorrect.test_batch_correct_center_moves_each_batch_mean_to_the_fit_mean: 3 exact checks<br>test_prep.TestBatchCorrect.test_batch_correct_refuses_settings_of_another_method: no numeric check |
| `batch_correct` | `combat` | core | `prep_batch` | test_prep.TestBatchCorrect.test_batch_correct_combat_removes_additive_and_multiplicative_effects: 1 exact check<br>test_prep.TestBatchCorrect.test_batch_correct_refuses_settings_of_another_method: no numeric check<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks |
| `batch_correct` | `harmony` | core | `prep_batch` | test_prep.TestBatchCorrect.test_batch_correct_harmony_corrects_the_fit_rows_themselves: 1 exact check<br>test_prep.TestBatchCorrect.test_batch_correct_refuses_settings_of_another_method: no numeric check |
| `drop_constant` | all | core | `prep_clean` | test_prep.TestDropConstant.test_drop_constant_refusals: no numeric check<br>test_prep.TestDropConstant.test_drop_constant_removes_the_column_constant_on_the_fit_rows: 4 exact checks<br>test_prep.TestDropConstant.test_drop_constant_variance_floor_and_near_constant_rule: 2 exact checks<br>and more |
| `drop_duplicates` | all | core | `prep_clean` | test_prep.TestDropDuplicates.test_drop_duplicates_keeps_the_first_of_repeated_fit_rows: 5 exact checks<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks<br>test_prep.TestEveryStep.test_declarations_match_what_the_steps_do: 2 exact checks |
| `impute` | `drop_rows` | core | none | test_prep.TestImpute.test_impute_drop_rows_takes_rows_out_of_the_fit_only: 5 exact checks<br>test_prep.TestImpute.test_a_step_that_removes_every_fit_row_is_refused: no numeric check |
| `impute` | `mean` | core | none | test_prep.TestImpute.test_impute_mean_median_and_constant_use_the_fit_rows: 3 exact checks<br>test_prep.TestImpute.test_impute_refuses_a_column_without_a_fit_value: no numeric check<br>test_prep.TestImpute.test_impute_fills_categorical_codes_with_the_most_frequent_fit_code: 2 exact checks<br>and more |
| `impute` | `median` | core | `prep_clean` | test_prep.TestImpute.test_impute_mean_median_and_constant_use_the_fit_rows: 3 exact checks<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks |
| `impute` | `constant` | core | none | test_prep.TestImpute.test_impute_mean_median_and_constant_use_the_fit_rows: 3 exact checks<br>test_prep.TestImpute.test_impute_refuses_settings_of_another_method: no numeric check |
| `impute` | `knn` | core | `prep_clean` | test_prep.TestImpute.test_impute_knn_and_iterative_follow_scikit_learn_fitted_on_the_fit_rows: 2 exact checks<br>test_prep.TestImpute.test_impute_refuses_a_column_without_a_fit_value: no numeric check<br>test_prep.TestImpute.test_impute_refuses_settings_of_another_method: no numeric check |
| `impute` | `iterative` | core | `prep_clean` | test_prep.TestImpute.test_impute_knn_and_iterative_follow_scikit_learn_fitted_on_the_fit_rows: 2 exact checks<br>test_prep.TestImpute.test_impute_iterative_gives_a_row_the_same_values_whatever_rows_come_with_it: 1 exact check<br>test_prep.TestImpute.test_impute_refuses_a_column_without_a_fit_value: no numeric check |
| `onehot` | all | core | `prep_mixed` | test_prep.TestOneHot.test_onehot_one_column_per_category_of_the_dataset: 6 exact checks<br>test_prep.TestOneHot.test_onehot_refuses_unknown_codes_and_data_without_categories: no numeric check<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks<br>and more |
| `outliers` | `clip_quantiles` | core | `prep_clean` | test_prep.TestOutliers.test_outliers_clip_quantiles_learned_on_the_fit_rows: 3 exact checks<br>test_prep.TestOutliers.test_outliers_refuse_settings_of_another_method: no numeric check<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks |
| `outliers` | `clip_sd` | core | `prep_clean` | test_prep.TestOutliers.test_outliers_clip_sd: 3 exact checks<br>test_prep.TestOutliers.test_outliers_refuse_settings_of_another_method: no numeric check |
| `outliers` | `remove_by_detector` | core | `prep_clean` | test_prep.TestOutliers.test_outliers_remove_by_detector_takes_flagged_fit_rows_out: 3 exact checks<br>test_prep.TestOutliers.test_outliers_refuse_settings_of_another_method: no numeric check |
| `reduce` | all | core | `cluster_umap_hdbscan` | test_prep.TestReduce.test_reduce_with_pca_fitted_on_the_fit_rows: `record["explained_variance"]` > 0.99<br>test_prep.TestReduce.test_reduce_without_a_transform_needs_the_evaluated_rows_to_be_the_fit_rows: 3 exact checks<br>test_prep.TestReduce.test_reduce_with_umap: 2 exact checks<br>and more |
| `scale` | `zscore` | core | `prep_transform` | test_prep.TestScale.test_scale_zscore_gives_mean_zero_and_sd_one_on_the_fit_rows: 5 exact checks<br>test_distances.TestPrecomputedThroughTheRunner.test_each_subsample_gets_its_block_and_criteria_follow: `reveal["metrics"]["ari"]` ≥ 0.9<br>test_large_data.TestLargeData.test_large_data_path: `reveal["metrics"]["ari"]` ≥ 0.95<br>and more |
| `scale` | `minmax` | core | `prep_transform` | test_prep.TestScale.test_scale_minmax_robust_and_maxabs: 2 exact checks<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01<br>test_prep.TestScale.test_scale_named_columns_only_and_never_categorical_codes: 3 exact checks |
| `scale` | `robust` | core | `prep_transform` | test_prep.TestScale.test_scale_minmax_robust_and_maxabs: 2 exact checks<br>test_graph_models.TestDeclarations.test_preparation_steps_act_on_node_features: no numeric check<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks |
| `scale` | `maxabs` | core | `prep_transform` | test_prep.TestScale.test_scale_minmax_robust_and_maxabs: 2 exact checks |
| `scale` | `l1_rows` | core | none | test_prep.TestScale.test_scale_rows_by_their_l1_or_l2_norm: 3 exact checks |
| `scale` | `l2_rows` | core | `prep_transform` | test_prep.TestScale.test_scale_rows_by_their_l1_or_l2_norm: 3 exact checks<br>test_prep.TestScale.test_scale_refuses_per_group_with_per_row_norms: no numeric check |
| `select_columns` | all | core | `prep_transform` | test_prep.TestSelectColumns.test_select_columns_by_name_and_pattern: 6 exact checks<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks<br>test_prep.TestEveryStep.test_declarations_match_what_the_steps_do: 2 exact checks |
| `select_features` | `variance_top` | core | `prep_transform` | test_prep.TestSelectFeatures.test_select_features_by_fit_row_variance: 3 exact checks<br>test_prep.TestSelectFeatures.test_select_features_keeps_categorical_columns_and_checks_its_settings: 1 exact check<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks |
| `select_features` | `variance_threshold` | core | `prep_transform` | test_prep.TestSelectFeatures.test_select_features_by_fit_row_variance: 3 exact checks |
| `select_features` | `highly_variable` | core | `prep_transform` | test_prep.TestSelectFeatures.test_select_features_highly_variable_finds_the_overdispersed_counts: 2 exact checks<br>test_prep.TestSelectFeatures.test_select_features_keeps_categorical_columns_and_checks_its_settings: 1 exact check |
| `select_features` | `correlation` | core | `prep_transform` | test_prep.TestSelectFeatures.test_select_features_drops_a_column_correlated_with_an_earlier_one: 1 exact check<br>test_prep.TestSelectFeatures.test_select_features_keeps_categorical_columns_and_checks_its_settings: 1 exact check |
| `subsample` | `random` | core | none | test_prep.TestSubsample.test_subsample_keeps_a_fixed_share_of_the_fit_rows: 4 exact checks<br>test_prep.TestSubsample.test_subsample_depends_on_the_row_ids_only: 2 exact checks<br>test_prep.TestSubsample.test_subsample_stratified_keeps_each_strata_share: 1 exact check<br>and more |
| `subsample` | `stratified` | core | `prep_clean` | test_prep.TestSubsample.test_subsample_stratified_keeps_each_strata_share: 1 exact check<br>test_prep.TestSubsample.test_subsample_refusals: no numeric check |
| `transform` | `log` | core | `prep_transform` | test_prep.TestTransform.test_transform_fixed_formulas: 2 exact checks<br>test_prep.TestTransform.test_transform_refuses_values_and_settings_it_cannot_use: no numeric check |
| `transform` | `log1p` | core | `prep_transform` | test_prep.TestTransform.test_transform_fixed_formulas: 2 exact checks |
| `transform` | `arcsinh` | core | `prep_transform` | test_prep.TestTransform.test_transform_fixed_formulas: 2 exact checks<br>test_prep.TestTransform.test_transform_refuses_values_and_settings_it_cannot_use: no numeric check |
| `transform` | `sqrt` | core | `prep_transform` | test_prep.TestTransform.test_transform_fixed_formulas: 2 exact checks<br>test_prep.TestTransform.test_transform_refuses_values_and_settings_it_cannot_use: no numeric check |
| `transform` | `boxcox` | core | `prep_transform` | test_prep.TestTransform.test_transform_power_lambdas_learned_on_the_fit_rows: 2 exact checks |
| `transform` | `yeojohnson` | core | `prep_transform` | test_prep.TestTransform.test_transform_power_lambdas_learned_on_the_fit_rows: 2 exact checks<br>test_prep.TestEveryStep.test_every_template_step_learns_from_the_fit_rows_only: 3 exact checks |
| `transform` | `clr` | core | `prep_transform` | test_prep.TestTransform.test_transform_compositions_and_total_counts: 6 exact checks |
| `transform` | `alr` | core | `prep_transform` | test_prep.TestTransform.test_transform_compositions_and_total_counts: 6 exact checks<br>test_prep.TestTransform.test_transform_refuses_values_and_settings_it_cannot_use: no numeric check |
| `transform` | `total_count` | core | `prep_transform` | test_prep.TestTransform.test_transform_compositions_and_total_counts: 6 exact checks<br>test_prep.TestTransform.test_transform_refuses_values_and_settings_it_cannot_use: no numeric check |
| `transform` | `quantile` | core | `prep_transform` | test_prep.TestTransform.test_transform_quantile_and_rank_against_the_fit_rows: 4 exact checks |
| `transform` | `rank` | core | `prep_transform` | test_prep.TestTransform.test_transform_quantile_and_rank_against_the_fit_rows: 4 exact checks |

### Datasets

| Name | Smoke entry | Tests and their bounds |
|---|---|---|
| `anndata` | `data_anndata` | test_datasets.TestAnnData.test_cells_by_genes: 5 exact checks |
| `graph` | `data_graph` | test_graph_models.TestRuns.test_the_graph_datasets_example_runs_on_the_example_files: `read(folder, "metrics_val.json")["select"]["value"]` ≥ 0.6<br>test_datasets.TestItemsAndGraph.test_graph_nodes_and_edges: 4 exact checks<br>and more |
| `items` | `data_items` | test_datasets.TestItemsAndGraph.test_items_text_csv_and_wav: 3 exact checks<br>test_datasets.TestItemsAndGraph.test_items_refusals: no numeric check<br>and more |
| `store` | `data_store` | test_datasets.TestStore.test_refusals: no numeric check<br>test_datasets.TestStore.test_rows_come_back_in_the_order_asked: 6 exact checks<br>and more |
| `synthetic` | `classify_bayes` | test_datasets.TestSynthetic.test_items_and_graph_kinds: `graph.graph.edges.shape[1]` > 0<br>test_datasets.TestSynthetic.test_tabular_rows_ids_and_columns: 5 exact checks<br>and more |
| `table` | `data_table` | test_datasets.TestTable.test_categorical_columns_four_ways: 4 exact checks<br>test_datasets.TestTable.test_choosing_rows_and_columns: 5 exact checks<br>and more |

### Splits

| Name | Smoke entry | Tests and their bounds |
|---|---|---|
| `all` | `cluster_final` | test_splits.TestUnsupervised.test_all_fits_and_evaluates_every_row: 4 exact checks<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>and more |
| `fit_holdout` | `embed_diffusion` | test_splits.TestUnsupervised.test_fit_holdout_needs_assign_predict: no numeric check<br>test_splits.TestUnsupervised.test_fit_holdout_partitions_the_rows: 3 exact checks<br>and more |
| `fixed` | `classify_trees` | test_splits.TestFixedAndHoldout.test_fixed_reads_the_published_split_column: 3 exact checks<br>test_splits.TestFixedAndHoldout.test_fixed_refuses_other_values: no numeric check<br>and more |
| `group_kfold` | `classify_knn` | test_splits.TestGroupKFold.test_group_kfold_is_stable_and_refuses_too_few_groups: no numeric check<br>test_splits.TestGroupKFold.test_group_kfold_split_report_shows_no_group_overlap: `sum(overlap.values())` > 0<br>and more |
| `holdout` | `classify_svm` | test_splits.TestFixedAndHoldout.test_holdout_refuses_fractions_that_leave_no_training_rows: no numeric check<br>test_splits.TestFixedAndHoldout.test_holdout_fractions_within_each_stratum: 2 exact checks<br>and more |
| `kfold` | `classify_logistic` | test_splits.TestKFold.test_kfold_refuses_k_below_three: no numeric check<br>test_splits.TestKFold.test_kfold_roles_partition_and_the_next_group_validates: `max(sizes) - min(sizes)` ≤ 1<br>and more |
| `leave_one_group_out` | `classify_bayes` | test_splits.TestLeaveOneGroupOut.test_leave_one_group_out_needs_every_value_listed_once: no numeric check<br>test_splits.TestLeaveOneGroupOut.test_fold_names_are_cleaned_and_must_stay_distinct: 1 exact check<br>and more |
| `subsamples` | `cluster_bicluster` | test_splits.TestUnsupervised.test_subsamples_refusals: no numeric check<br>test_splits.TestUnsupervised.test_subsamples_are_fixed_distinct_and_of_the_given_size: 4 exact checks<br>and more |

### Tasks

| Name | Smoke entry | Tests and their bounds |
|---|---|---|
| `classification` | `classify_bayes` | test_estimators.TestClassification.test_svm_without_probabilities_raises_no_deprecation_warning: 1 exact check<br>test_graph_models.TestRuns.test_the_graph_datasets_example_runs_on_the_example_files: `read(folder, "metrics_val.json")["select"]["value"]` ≥ 0.6<br>and more |
| `clustering` | `cluster_bicluster` | test_deep_models.TestDeepClustering.test_the_example_runs_through_the_clusterer_runner: 1 exact check<br>test_distances.TestPrecomputedThroughTheRunner.test_each_subsample_gets_its_block_and_criteria_follow: `reveal["metrics"]["ari"]` ≥ 0.9<br>and more |
| `embedding` | `embed_diffusion` | test_criteria.TestEmbeddingCriteria.test_reconstruction_error_as_the_embedder_runner_computes_it: 2 exact checks<br>test_criteria.TestEmbeddingCriteria.test_reference_silhouette_by_hand: 2 exact checks<br>and more |
| `multilabel` | `classify_multilabel` | test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more |
| `outlier` | `outlier_detectors` | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_the_task_needs_every_em_and_mv_option: no numeric check<br>and more |
| `regression` | `regress_linear` | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_torch_family.TestKnownStructure.test_mlp_on_tabular_classes_and_regression: `mlp` > 0.75; `held_out(folder, "r2")` > 0.8<br>and more |

### Features

| Name | Smoke entry | Tests and their bounds |
|---|---|---|
| `image` | `image_cnn` | test_torch_family.TestKnownStructure.test_cnn2d_on_images: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_family.TestRuns.test_audio_images_and_a_spiking_model: 2 exact checks<br>and more |
| `logmel` | `audio_logmel` | test_torch_family.TestKnownStructure.test_spectrogram_models_on_audio: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_family.TestChecks.test_inputs_and_augmentations: no numeric check<br>and more |
| `none` | `classify_bayes` | test_deep_models.TestDeclarations.test_every_example_configuration_passes_its_checks: `summary["configs"]` > 3<br>test_distances.TestPrecomputedThroughTheRunner.test_each_subsample_gets_its_block_and_criteria_follow: `reveal["metrics"]["ari"]` ≥ 0.9<br>and more |
| `sequence` | `sequence_models` | test_torch_family.TestKnownStructure.test_spiking_models_on_sequences: `held_out(folder, "balanced_accuracy")` > 0.6<br>test_torch_family.TestRuns.test_sequences_with_every_training_option_and_externals: 4 exact checks<br>and more |

### Augmentations

| Name | Smoke entry | Tests and their bounds |
|---|---|---|
| `mask` | `audio_logmel` | test_torch_family.TestKnownStructure.test_spectrogram_models_on_audio: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_family.TestKnownStructure.test_cnn2d_on_images: `held_out(folder, "balanced_accuracy")` > 0.8<br>and more |
| `mixup` | `classify_mlp` | test_torch_family.TestRuns.test_multilabel_and_regression: `np.std(values)` > 0.5<br>test_torch_family.TestKnownStructure.test_mlp_on_tabular_classes_and_regression: `mlp` > 0.75; `held_out(folder, "r2")` > 0.8<br>and more |
| `noise` | `classify_mlp` | test_torch_family.TestKnownStructure.test_cnn2d_on_images: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_graph_models.TestDeclarations.test_registry_entries_and_the_transductive_marking: 5 exact checks<br>and more |
| `speed` | `audio_logmel` | test_torch_family.TestKnownStructure.test_spectrogram_models_on_audio: `held_out(folder, "balanced_accuracy")` > 0.8<br>test_torch_family.TestChecks.test_externals: no numeric check<br>and more |

### Fit procedures

| Name | Smoke entry | Tests and their bounds |
|---|---|---|
| `clusterer` | `cluster_bicluster` | test_deep_models.TestDeepClustering.test_the_example_runs_through_the_clusterer_runner: 1 exact check<br>test_gpu_agreement.TestClusterersAndEmbedders.test_kmeans: `ari(cpu, gpu)` ≥ 0.99; `ari(truth, gpu)` ≥ 0.99<br>and more |
| `detector` | `outlier_detectors` | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_prep.TestOutliers.test_outliers_remove_by_detector_takes_flagged_fit_rows_out: 3 exact checks<br>and more |
| `embedder` | `embed_diffusion` | test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01<br>test_deep_models.TestDeepEmbedders.test_the_example_runs_through_the_embedder_runner: 2 exact checks<br>and more |
| `estimator` | `classify_bayes` | test_torch_family.TestKnownStructure.test_mlp_on_tabular_classes_and_regression: `mlp` > 0.75; `held_out(folder, "r2")` > 0.8<br>test_sweep.TestOverrides.test_family_and_task_checks: no numeric check<br>and more |
| `torch` | `audio_logmel` | test_torch_models.TestDeclarations.test_every_torch_component_is_declared_and_in_its_example: `len(named)` ≥ 5<br>test_graph_models.TestRuns.test_the_graph_datasets_example_runs_on_the_example_files: `read(folder, "metrics_val.json")["select"]["value"]` ≥ 0.6<br>and more |

### Metrics and criteria

Metrics have no smoke entry of their own: every run scores the metrics and criteria its task names.

| Metric | Tasks | Tests and their bounds |
|---|---|---|
| `accuracy` | classification | test_torch_family.TestRuns.test_a_job_writes_every_file_and_reloads_its_best_epoch: `params["extra_heads"]["aux"]` > 0<br>test_estimators.TestClassification.test_every_classifier_learns_the_planted_classes: `metrics["balanced_accuracy"]` ≥ 0.85; `metrics["roc_auc"]` ≥ 0.9<br>and more |
| `aic` | clustering | test_criteria.TestClusteringCriteria.test_aic_and_bic_come_from_the_mixture_model: 2 exact checks<br>test_unsupervised_models.TestCentroidAndMixtures.test_gmm_finds_the_groups_and_bic_is_lowest_at_three: `ari(columns["truth"], model.labels())` ≥ 0.95<br>and more |
| `ami` | clustering | test_criteria.TestRevealMetrics.test_against_scikit_learn_with_noise_as_one_more_cluster: 2 exact checks<br>test_criteria.TestRevealMetrics.test_noise_counts_as_a_cluster: 1 exact check<br>and more |
| `ari` | clustering | test_unsupervised_models.TestThroughTheFitProcedures.test_kprototypes_receives_the_categorical_columns: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.9<br>test_distances.TestPrecomputedThroughTheRunner.test_each_subsample_gets_its_block_and_criteria_follow: `reveal["metrics"]["ari"]` ≥ 0.9<br>and more |
| `auroc` | outlier | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_ranking_metrics_by_hand: 6 exact checks<br>and more |
| `average_precision` | classification | test_metrics_supervised.TestClassification.test_roc_auc_and_average_precision_binary_by_hand: 2 exact checks<br>test_metrics_supervised.TestClassification.test_multiclass_one_against_the_rest: 2 exact checks<br>and more |
| `balanced_accuracy` | classification | test_graph_models.TestRuns.test_gcn_and_gat_find_the_planted_communities_and_the_control_does_not: `np.mean(control)` < 0.6; `read(folder, "metrics_test.json")["metrics"]["balanced_ac...` ≥ 0.8<br>test_pretrained.TestFinetune.test_the_class_in_the_words_is_learned: `read(folder, "metrics_test.json")["metrics"]["balanced_ac...` ≥ 0.8<br>and more |
| `bic` | clustering | test_unsupervised_models.TestCentroidAndMixtures.test_gmm_finds_the_groups_and_bic_is_lowest_at_three: `ari(columns["truth"], model.labels())` ≥ 0.95<br>test_criteria.TestClusteringCriteria.test_aic_and_bic_come_from_the_mixture_model: 2 exact checks<br>and more |
| `brier` | classification | test_metrics.TestBatchedForms.test_every_batched_form_matches_its_metric: 2 exact checks |
| `calinski_harabasz` | clustering | test_criteria.TestClusteringCriteria.test_inertia_calinski_harabasz_and_davies_bouldin_by_hand: 5 exact checks<br>test_distances.TestPrecomputedThroughTheRunner.test_each_subsample_gets_its_block_and_criteria_follow: `reveal["metrics"]["ari"]` ≥ 0.9<br>and more |
| `ccc` | regression | test_torch_models.TestHeadsAndLosses.test_regression_losses_and_concordance: 4 exact checks<br>test_metrics.TestBatchedForms.test_every_batched_form_matches_its_metric: 2 exact checks |
| `cohen_kappa` | classification | test_metrics.TestBatchedForms.test_every_batched_form_matches_its_metric: 2 exact checks |
| `completeness` | clustering | test_criteria.TestRevealMetrics.test_homogeneity_completeness_v_measure_and_nmi_by_hand: 7 exact checks<br>test_criteria.TestRevealMetrics.test_purity_rewards_many_small_clusters: `value("completeness", ctx)` < 0.5<br>and more |
| `continuity` | embedding | test_embed_models.TestThroughTheRunner.test_smoke_of_a_small_sweep: 1 exact check |
| `davies_bouldin` | clustering | test_criteria.TestClusteringCriteria.test_inertia_calinski_harabasz_and_davies_bouldin_by_hand: 5 exact checks<br>test_criteria.TestClusteringCriteria.test_noise_rows_are_left_out: 1 exact check<br>and more |
| `dbcv` | clustering | test_criteria.TestClusteringCriteria.test_planted_blobs_beat_a_random_labelling: no numeric check |
| `em_auc` | outlier | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_excess_mass_known_answers: 2 exact checks<br>and more |
| `explained_variance` | embedding | test_prep.TestReduce.test_reduce_with_pca_fitted_on_the_fit_rows: `record["explained_variance"]` > 0.99<br>test_unsupervised_models.TestLinearEmbedders.test_truncated_svd_keeps_a_low_rank_table: `model.info()["explained_variance"]` ≥ 0.99; `((rebuilt - x) ** 2).mean() / (x ** 2).mean()` ≤ 0.01<br>and more |
| `f1_macro` | classification | test_estimators.TestClassification.test_every_classifier_learns_the_planted_classes: `metrics["balanced_accuracy"]` ≥ 0.85; `metrics["roc_auc"]` ≥ 0.9<br>test_estimators.TestControlAndRefit.test_shuffled_labels_sit_near_chance_on_average: `abs(value["shuffled"] - 1 / 3)` < 0.12; `value["base"]` ≥ 0.9<br>and more |
| `f1_micro` | classification | test_metrics_supervised.TestClassification.test_f1_micro_and_recall_macro_by_hand: 6 exact checks<br>test_metrics_supervised.TestClassification.test_constant_predictions_and_a_missing_class: 3 exact checks<br>and more |
| `f1_weighted` | classification | test_metrics.TestClassificationByHand.test_counts: 6 exact checks |
| `flagged_fraction` | outlier | test_criteria.TestOutlierCriteria.test_flagged_fraction: 1 exact check<br>test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>and more |
| `fowlkes_mallows` | clustering | test_criteria.TestRevealMetrics.test_fowlkes_mallows_by_counting_pairs: 1 exact check<br>test_criteria.TestRevealMetrics.test_against_scikit_learn_with_noise_as_one_more_cluster: 2 exact checks<br>and more |
| `gap` | clustering | test_criteria.TestClusteringCriteria.test_gap_statistic: 1 exact check<br>test_criteria.TestClusteringCriteria.test_the_task_needs_the_options_its_criteria_read: no numeric check<br>and more |
| `hamming_loss` | multilabel | test_torch_family.TestRuns.test_multilabel_and_regression: `np.std(values)` > 0.5<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>and more |
| `homogeneity` | clustering | test_criteria.TestRevealMetrics.test_homogeneity_completeness_v_measure_and_nmi_by_hand: 7 exact checks<br>test_criteria.TestRevealMetrics.test_against_scikit_learn_with_noise_as_one_more_cluster: 2 exact checks<br>and more |
| `inertia` | clustering | test_criteria.TestClusteringCriteria.test_inertia_is_the_sum_kmeans_minimizes: 1 exact check<br>test_criteria.TestClusteringCriteria.test_inertia_calinski_harabasz_and_davies_bouldin_by_hand: 5 exact checks<br>and more |
| `knn_label_agreement` | embedding | test_embed_models.TestThroughTheRunner.test_smoke_of_a_small_sweep: 1 exact check |
| `knn_preservation` | embedding | test_embed_models.TestThroughTheRunner.test_smoke_of_a_small_sweep: 1 exact check |
| `label_average_precision` | multilabel | test_metrics_supervised.TestMultilabel.test_undefined_without_both_sides: `metric("label_average_precision", always)` > 0<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>and more |
| `label_f1_macro` | multilabel | test_torch_family.TestRuns.test_multilabel_and_regression: `np.std(values)` > 0.5<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>and more |
| `label_f1_micro` | multilabel | test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more |
| `label_roc_auc` | multilabel | test_metrics_supervised.TestMultilabel.test_undefined_without_both_sides: `metric("label_average_precision", always)` > 0<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>and more |
| `largest_cluster_fraction` | clustering | test_criteria.TestClusteringCriteria.test_descriptive_counts: 5 exact checks<br>test_criteria.TestClusteringCriteria.test_descriptive_criteria_cannot_choose: no numeric check<br>and more |
| `log_loss` | classification | test_estimators.TestClassification.test_log_loss_needs_probabilities: `files["metrics_val.json"]["metrics"]["log_loss"]` > 0<br>test_metrics_supervised.TestClassification.test_decision_values_rank_like_probabilities: 1 exact check<br>and more |
| `mae` | regression | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestOptionalLibraries.test_xgboost: `metrics["balanced_accuracy"]` ≥ 0.85; `run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.j...` > 0.5<br>and more |
| `max_error` | regression | test_metrics_supervised.TestRegression.test_max_error: 2 exact checks<br>test_metrics_supervised.TestRegression.test_row_weights_count_rows: 2 exact checks<br>and more |
| `mcc` | classification | test_metrics.TestBatchedForms.test_every_batched_form_matches_its_metric: 2 exact checks |
| `median_ae` | regression | test_metrics_supervised.TestRegression.test_median_absolute_error_by_hand: 5 exact checks<br>test_metrics_supervised.TestRegression.test_median_absolute_error_with_weights_against_scikit_learn: 2 exact checks<br>and more |
| `mse` | regression | test_torch_models.TestHeadsAndLosses.test_regression_losses_and_concordance: 4 exact checks<br>test_metrics.TestClassificationByHand.test_regression_by_hand: 4 exact checks<br>and more |
| `mv_auc` | outlier | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_mass_volume_known_answers: 3 exact checks<br>and more |
| `n_clusters` | clustering | test_cluster_models.TestPackagesFromExtrasAndSingleCell.test_lumbermark_on_blobs_and_moons: `ari(truth, model.labels())` ≥ 0.9<br>test_unsupervised_models.TestCentroidAndMixtures.test_bisecting_kmeans: `ari(columns["truth"], model.labels())` ≥ 0.95; `model.info()["inertia"]` > 0<br>and more |
| `nmi` | clustering | test_criteria.TestRevealMetrics.test_homogeneity_completeness_v_measure_and_nmi_by_hand: 7 exact checks<br>test_criteria.TestRevealMetrics.test_against_scikit_learn_with_noise_as_one_more_cluster: 2 exact checks<br>and more |
| `noise_fraction` | clustering | test_unsupervised_models.TestThroughTheFitProcedures.test_every_clusterer_through_the_clusterer_fit: `f["reveal.json"]["metrics"]["ari"]` ≥ 0.95<br>test_criteria.TestClusteringCriteria.test_descriptive_counts: 5 exact checks<br>and more |
| `outlier_average_precision` | outlier | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_ranking_metrics_by_hand: 6 exact checks<br>and more |
| `pearson` | regression | test_metrics.TestBatchedForms.test_every_batched_form_matches_its_metric: 2 exact checks |
| `precision_at_n` | outlier | test_unsupervised_models.TestOutlierSweep.test_detector_sweep_end_to_end: `reveal["metrics"]["auroc"]` ≥ 0.9<br>test_criteria.TestOutlierCriteria.test_ranking_metrics_by_hand: 6 exact checks<br>and more |
| `precision_macro` | classification | test_metrics.TestClassificationByHand.test_counts: 6 exact checks |
| `purity` | clustering | test_criteria.TestRevealMetrics.test_purity_rewards_many_small_clusters: `value("completeness", ctx)` < 0.5<br>test_criteria.TestRevealMetrics.test_purity_of_a_two_by_two_table: 1 exact check<br>and more |
| `r2` | regression | test_torch_family.TestKnownStructure.test_mlp_on_tabular_classes_and_regression: `mlp` > 0.75; `held_out(folder, "r2")` > 0.8<br>test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>and more |
| `recall_macro` | classification | test_metrics_supervised.TestClassification.test_f1_micro_and_recall_macro_by_hand: 6 exact checks<br>test_metrics_supervised.TestClassification.test_constant_predictions_and_a_missing_class: 3 exact checks<br>and more |
| `reconstruction_error` | embedding | test_criteria.TestEmbeddingCriteria.test_reconstruction_error_as_the_embedder_runner_computes_it: 2 exact checks<br>test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01<br>and more |
| `reference_silhouette` | embedding | test_criteria.TestEmbeddingCriteria.test_reference_silhouette_by_hand: 2 exact checks<br>test_criteria.TestEmbeddingCriteria.test_reference_silhouette_reads_the_sample_rows_that_have_a_label: 1 exact check<br>and more |
| `rmse` | regression | test_estimators.TestRegression.test_linear_models_explain_a_linear_target_best: `r2(label, k)` ≥ 0.8; `r2(label, k)` > 0.3<br>test_estimators.TestOptionalLibraries.test_xgboost: `metrics["balanced_accuracy"]` ≥ 0.85; `run_jobs(reg, {}, ["fold1"])["base"][0][2]["metrics_val.j...` > 0.5<br>and more |
| `roc_auc` | classification | test_metrics_supervised.TestClassification.test_roc_auc_and_average_precision_binary_by_hand: 2 exact checks<br>test_estimators.TestClassification.test_every_classifier_learns_the_planted_classes: `metrics["balanced_accuracy"]` ≥ 0.85; `metrics["roc_auc"]` ≥ 0.9<br>and more |
| `sample_f1` | multilabel | test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>test_estimators.TestMultilabel.test_shuffled_labels_lose_the_ranking: `abs(float(np.mean(metric("shuffled", "label_roc_auc"))) -...` < 0.15<br>and more |
| `silhouette` | clustering | test_distances.TestPrecomputedThroughTheRunner.test_each_subsample_gets_its_block_and_criteria_follow: `reveal["metrics"]["ari"]` ≥ 0.9<br>test_large_data.TestLargeData.test_large_data_path: `reveal["metrics"]["ari"]` ≥ 0.95<br>and more |
| `spearman` | regression | test_metrics_supervised.TestRegression.test_spearman_by_hand_and_against_scipy: 3 exact checks<br>test_metrics_supervised.TestRegression.test_row_weights_count_rows: 2 exact checks<br>and more |
| `subset_accuracy` | multilabel | test_torch_family.TestRuns.test_multilabel_and_regression: `np.std(values)` > 0.5<br>test_estimators.TestMultilabel.test_every_multilabel_model_learns_the_labels: `min(metric(label, "label_f1_macro"))` ≥ 0.55; `min(metric(label, "label_roc_auc"))` ≥ 0.65<br>and more |
| `trustworthiness` | embedding | test_unsupervised_models.TestThroughTheFitProcedures.test_every_linear_embedder_through_the_embedder_fit: `svd["explained_variance"]` ≥ 0.99; `svd["reconstruction_error"]` < 0.01<br>test_embed_models.TestChecksAtExpansion.test_refusals: no numeric check<br>and more |
| `v_measure` | clustering | test_criteria.TestRevealMetrics.test_homogeneity_completeness_v_measure_and_nmi_by_hand: 7 exact checks<br>test_criteria.TestRevealMetrics.test_against_scikit_learn_with_noise_as_one_more_cluster: 2 exact checks<br>and more |

### Gaps

Models whose tests assert no numeric bound (they are run and checked in other ways, such as shapes, exact values or refusals): `cnn1d`, `composite`.

Preparation methods that no test names: none.

GPU paths without a test that compares the CPU and GPU versions: none.

<!-- End of the testing matrix. -->
