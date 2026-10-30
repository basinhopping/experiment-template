# An example round

Written by `python -m sweepkit.cli example-round --write yes`, which runs every command below in a temporary copy of
the template on synthetic data and records what it printed and wrote. Times are shown as `<utc>`, the temporary
folder as `<repo>`, and SHA-256 values (the code's tree digest changes with every edit) as `<sha256>`; every other
character is the commands' own output.

A round is one night on a shared node or one batch on one machine (docs/HOW_WE_WORK.md). On a node, the person pastes
the round's lines one at a time (`lines/`, below) and the workers run the sweeps; here `run` does the same work in one
process. The order never changes: prepare, fix the RULE before any result exists, run, read the choosing files,
record the choice, and only then read the files that judge it.

## Round 1: how many clusters, and which method (unsupervised)

### 1. Prepare

The sweep file `sweeps/project/round1_screen.json` holds the base configuration and the stages that vary it. `check` expands it without queueing anything:

```json
{
 "sweep_id": "round1_screen",
 "purpose": "screen",
 "configs": 5,
 "jobs": 120,
 "seeds": [
  1,
  2,
  3
 ]
}
```

`smoke` runs every configuration once on a few rows, through the real runner:

```
$ python -m sweepkit.cli smoke --sweep-file sweeps/project/round1_screen.json --device cpu --rows 60
sweep round1_screen: 5 configurations, 120 jobs
  stage base (base): 1 configurations x 3 seeds, priority 9
  stage methods (variants): 2 configurations x 3 seeds, priority 8
  stage k (one_factor_at_a_time): 2 configurations x 3 seeds, priority 7
SMOKE OK
```

`round-new` makes the round's preparation folder:

```
$ python -m sweepkit.cli round-new --round round1 --date 20261101 --question "How many clusters do the data hold, and which method finds them best?" --sweep-files sweeps/project/round1_screen.json --workers-per-gpu 3 --gpus 0,1 --end-time "2026-11-02 08:00" --environment main --profiles core unsupervised --min-free-gb 15
made <repo>/artifacts/analyses/round1_prep_20261101 and the plan sweeps/rounds/round1_prep_20261101/plan.txt (it travels with the code): fill in RULE_round1.txt, fix it with rule-fix, then run round-check
```

`artifacts/analyses/round1_prep_20261101/README.txt`:

```
Round round1, prepared <utc> (UTC) for project myproject.

Question: How many clusters do the data hold, and which method finds them best?

Sweeps (each checked by expand; files under sweeps/, SHA-256 in sweep_files.sha256):
  sweeps/project/round1_screen.json: sweep round1_screen (screen), 5 configurations, 120 jobs, seeds [1, 2, 3]

On the node: the environment main (~/venvs/myproject-main, profiles core unsupervised); the schedule
starts 3 workers on each of GPUs 0,1 (only those no one else is using) and stops them
at 2026-11-02 08:00, the node's local time,
or earlier when no job is left (the round's session then ends) or less than 15 GB is free in the home
folder.

Plan: sweeps/rounds/round1_prep_20261101/plan.txt
  the steps node/chain.sh runs on the node in one tmux session that ends by itself. It lives under sweeps/ so that
  it travels to the node with the code; the tree digest below covers it, and its SHA-256 is in sweep_files.sha256.

Cost: round1_screen: not estimated yet (run estimate) (120 jobs in all)

Code: tree digest <sha256> (node/tree_digest.sh on the node prints the same number for the same code).

Files of this folder:
  RULE_round1.txt: what runs and how it is read, fixed with rule-fix before any result exists. Fixed at:
      not fixed yet (write rule-fix's time and SHA-256 here)
  sweep_files.sha256: the SHA-256 of each sweep file and of the plan at preparation.
  lines/: one file per step, numbered in the order they run: 01_push.txt, 02_install_test.txt, 03_start.txt, 04_status.txt, 05_pack.txt, 06_pull.txt.

Changes after preparation (each with the time, why, and whether it changes a job of the RULE):
(none yet)
```

The lines the person pastes on a node, one file per step; the start line, `artifacts/analyses/round1_prep_20261101/lines/03_start.txt`, for example:

```
# 3 of 6: check the schedule, smoke-test and enqueue the round's sweeps, then the workers until the queue is empty (at the latest the stop time), in one tmux session that ends by itself (node)
# shell: bash on the node
# expect: "myproject_round1 started" and the round's GPUs that are free; within minutes, logs/round1/1_schedule.txt shows "SCHEDULE OK", logs/round1/2_smoke_enqueue.txt "SMOKE OK" for each sweep and "ALL DONE", and logs/round1/3_schedule.txt "GPU <g> started" (line 04 shows these); when no job is left (at the latest at the stop time) logs/round1/chain.txt ends with "CHAIN DONE" and the session is gone
# stop if: "the tree is not", "slurmd is running", "a session of myproject is running", "jobs in queue/running", "GB free, the round's floor" or "no free GPU": stop here and paste the output
cd ~/myproject && mkdir -p logs/round1 && { D=$(bash node/tree_digest.sh .) && G=$(bash node/free_gpus.sh 2>/dev/null | grep -xE '0|1' | paste -sd, -) && if [ "$D" != <sha256> ]; then echo "the tree is not <sha256> but $D: stop here and paste this"; elif systemctl is-active --quiet slurmd 2>/dev/null; then echo "Slurm's slurmd is running, so Slurm may give these GPUs to batch jobs: stop here and paste this"; elif tmux ls -F '#{session_name}' 2>/dev/null | grep -q '^myproject_'; then echo "a session of myproject is running: stop here and paste this"; tmux ls; elif [ "$(ls queue/running 2>/dev/null | wc -l)" != 0 ]; then echo "jobs in queue/running: stop here and paste this"; elif ! bash node/disk_check.sh 15 > /dev/null; then echo "less than 15 GB free, the round's floor: stop here and paste this"; df -h ~ | tail -1; elif [ -z "$G" ]; then echo "no free GPU among the round's (0,1): stop here and paste this"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; else rm -f queue/STOP queue/STOP_gpu*; tmux new -d -s myproject_round1 "bash node/chain.sh sweeps/rounds/round1_prep_20261101/plan.txt" && echo "myproject_round1 started; free GPUs now: $G"; df -h ~ | tail -1; fi; } 2>&1 | tee -a logs/round1/03_start.txt
```

### 2. Fix the RULE before any result exists

The RULE's four sections are filled in (by Claude, approved by the person), then fixed:

```
RULE_round1.txt: what round round1 runs and how its results are read. Project myproject.
Fixed with rule-fix before any result of this round exists; the time and SHA-256 are in the record rule-fix wrote
(artifacts/rules/round1_<utc>/record.json). It does not change once a file of these runs has been read. Jobs
still queued from earlier rounds keep running under their own rules.

1. What runs
   Sweep round1_screen (sweeps/project/round1_screen.json): k-means with 4 clusters (the base), a
   Gaussian mixture and Ward's agglomerative clustering with 4, and k-means with 3 and 5 clusters, each
   on 8 fixed subsamples of 80% of the rows with seeds 1, 2 and 3, z-scored, on the CPU. No earlier
   result chose any of these settings.
   (The sweeps by file and sweep ID, what each asks, seeds, folds, priorities, the device, and the code version by
   tree digest. Say what earlier choice each setting comes from, and that no test or reveal file chose it.)

2. Reading
   Label-free files only first: analyze ranks the configurations by silhouette (larger is better) and
   compares each with the base. select takes the highest, read with its stability across seeds and
   subsamples and the number of clusters it finds: among the configurations whose intervals overlap
   the highest's, the most stable; if a configuration more stable by over 0.05 ranks below the
   choice, the round stops for a look at the data before any choice. Only then reveal-report reads
   that one configuration's agreement with the planted groups (ARI, AMI). No comparison family is
   read on the reveal files. Runs not finished at the reading are left out and named.
   (Which files are read, in which order: validation or label-free files first, with analyze; then select; only then
   the test or reveal files of the selected configurations, with test-report or reveal-report. Which comparisons
   are made, as families (the families file given to rule-fix: paired bootstrap, Holm within each family), on which
   seeds, and what happens to runs not finished at the reading time. How select chooses: the criterion, and for
   label-free work the stability and the number of groups found, read together (a floor fixed in advance can turn
   away the best: in the template's dry run a stability floor of 0.8 turned away, in two of ten sweeps, the
   configuration nearest the planted groups); what decides a tie (for example, of two whose intervals overlap, the
   more stable, then the simpler); and what happens when the most stable is not the one the criterion ranks first.)

3. Stop condition
   The choice goes to a final run on every row with five seeds. If its stability is below 0.9, or
   the number of clusters it finds differs between seeds, stop and look at the data before any
   final run.
   (Written before the results: which result is a plain substitution, for example "a configuration that beats the
   baseline on validation goes to confirmation on five new seeds", and which result changes the plan, for example
   "if the leak control is above chance by more than its seed spread, stop and check the split".)

4. What this cannot show
   One synthetic dataset with round clusters, which favours silhouette and k-means; settings chosen on
   the same subsamples they are judged on (optimistic until the final run); no other criterion read.
   (What the design of this round leaves open: one dataset, one split, seeds that are not new, settings chosen on
   validation and therefore optimistic until confirmed, transductive steps, and anything not tested.)
```

```
$ python -m sweepkit.cli rule-fix --name round1 --rule-file artifacts/analyses/round1_prep_20261101/RULE_round1.txt --sweeps round1_screen --families none --late-reason none
RULE round1 fixed at <utc>, SHA-256 <sha256>, covering round1_screen; 0 comparison families
saved <repo>/artifacts/rules/round1_<utc>
```

`round-check` passes before the first line is handed over:

```
$ python -m sweepkit.cli round-check --round round1 --date 20261101
round-check round1 (<repo>/artifacts/analyses/round1_prep_20261101):
  note: RULE fixed <utc>, no result yet
  note: lines/01_push.txt holds a part to fill in when the line is handed over
  note: lines/06_pull.txt holds a part to fill in when the line is handed over
ROUND OK
```

### 3. Run

On a node the lines smoke-test and enqueue the sweep and start the workers; here `run` does the same work in one process. Its last line:

```
$ python -m sweepkit.cli run --sweep-file sweeps/project/round1_screen.json --device cpu --max-consecutive-failures 3
{'sweep_id': 'round1_screen', 'jobs': 120, 'done': 120, 'skipped_complete': 0, 'failed': 0}
```

### 4. Read the label-free files, then choose

```
$ python -m sweepkit.cli analyze --sweep round1_screen --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 --top 5
sweep round1_screen (clustering): 5 of 5 configurations, 120 of 120 runs complete; chosen by silhouette (larger is better); label-free files only; 95% intervals, 2000 bootstrap draws
NOTE label-free comparisons take the larger of the bootstrap's p and a sign-flip test's over the 8 folds, so no p here falls below 0.0078
baseline 87132aa14cf0: seeds [1, 2, 3], silhouette 0.5158 [0.5158, 0.5158], SD 0.0000
top 5 of 5 by silhouette; difference = configuration minus baseline:
  0.5158 [0.5158, 0.5158]  (baseline)  | 87132aa14cf0 base: base
      calinski_harabasz 959.7177  n_clusters 4.0000
      stability (adjusted Rand index) 1.000 over 108 pairs: across seeds 1.000 (24 pairs), across folds 1.000 (84 pairs)
  0.5157 [0.5157, 0.5157]  diff -0.0001188 bootstrap [-0.0001996, -0.00004240] fold range [-0.0002763, +0.0000]  P(better) 0.19  Holm p 0.062  no detectable difference  | 34b2805e9948 methods: gmm
      calinski_harabasz 959.4963  n_clusters 4.0000
      stability (adjusted Rand index) 0.999 over 108 pairs: across seeds 1.000 (24 pairs), across folds 0.998 (84 pairs)
  0.5155 [0.5155, 0.5155]  diff -0.0003014 bootstrap [-0.0006409, -0.00009572] fold range [-0.0014, +0.0000]  P(better) 0.06  Holm p 0.031  worse  | 2d33b7605625 methods: ward
      calinski_harabasz 958.7874  n_clusters 4.0000
      stability (adjusted Rand index) 0.998 over 108 pairs: across seeds 1.000 (24 pairs), across folds 0.998 (84 pairs)
  0.4846 [0.4833, 0.4859]  diff -0.0312 bootstrap [-0.0329, -0.0293] fold range [-0.0343, -0.0273]  P(better) 0.00  Holm p 0.031  worse  | 7c82ceec2901 k: model.n_clusters=3
      calinski_harabasz 686.3530  n_clusters 3.0000
      stability (adjusted Rand index) 0.837 over 108 pairs: across seeds 0.911 (24 pairs), across folds 0.815 (84 pairs)
  0.4418 [0.4388, 0.4448]  diff -0.0740 bootstrap [-0.0769, -0.0715] fold range [-0.0796, -0.0704]  P(better) 0.00  Holm p 0.031  worse  | e8bfec93548b k: model.n_clusters=5
      calinski_harabasz 788.7254  n_clusters 5.0000
      stability (adjusted Rand index) 0.892 over 108 pairs: across seeds 0.906 (24 pairs), across folds 0.888 (84 pairs)
saved <repo>/artifacts/analysis/round1_screen_<utc>
```

A difference can be detectable and still too small to matter: read the difference and its interval, not only the verdict.

The RULE says: the highest silhouette, read with its stability and the number of clusters it finds. That is `87132aa14cf0` (base: base), stability 1.000, 4.0 clusters on average; no configuration below it is more stable by over 0.05. `select` records it:

```
$ python -m sweepkit.cli select --sweep round1_screen --config 87132aa14c --name round1_choice --reason "highest silhouette, read with its stability and number of clusters, as the RULE says"
selected 87132aa14cf0 of round1_screen (base), silhouette 0.5158, rank 1 of 5, from round1_screen_<utc>
saved <repo>/artifacts/selections/round1_choice
```

### 5. Only now, the files that judge the choice

```
$ python -m sweepkit.cli reveal-report --selection round1_choice --confidence 0.95
reveal results of selection round1_choice: round1_screen / 87132aa14cf0 (base), seeds [1, 2, 3]
chosen because: highest silhouette, read with its stability and number of clusters, as the RULE says (recorded <utc>; RULE round1_<utc>)
  ari 1.0000 [1.0000, 1.0000] (95% t interval over seeds), SD 0.0000
  ami 1.0000 [1.0000, 1.0000] (95% t interval over seeds), SD 0.0000
saved <repo>/artifacts/reveal_reports/round1_choice_<utc>
```

`export` writes what the person takes home (plots left out here); its files:

```
$ python -m sweepkit.cli export --selection round1_choice --runs all --reveal no --plots no --top-features 3 --formats csv
15360 rows from 24 runs; saved <repo>/artifacts/exports/round1_choice_<utc>
```

```
README.txt
cluster_summary.csv
distinguishing.csv
packages.json
provenance.json
report.json
rows.csv
settings.json
sizes.csv
```

## Round 2: does a change beat the baseline (supervised)

### 1. Prepare

The sweep file `sweeps/project/round2_screen.json` holds the base configuration and the stages that vary it. `check` expands it without queueing anything:

```json
{
 "sweep_id": "round2_screen",
 "purpose": "screen",
 "configs": 4,
 "jobs": 72,
 "seeds": [
  1,
  2,
  3
 ]
}
```

`smoke` runs every configuration once on a few rows, through the real runner:

```
$ python -m sweepkit.cli smoke --sweep-file sweeps/project/round2_screen.json --device cpu --rows 60
sweep round2_screen: 4 configurations, 72 jobs
  stage base (base): 1 configurations x 3 seeds, priority 9
  stage changes (variants): 2 configurations x 3 seeds, priority 8
  stage control (variants): 1 configurations x 3 seeds, priority 7
SMOKE OK
```

`round-new` makes the round's preparation folder:

```
$ python -m sweepkit.cli round-new --round round2 --date 20261101 --question "Does a more regularized model or a random forest beat the baseline?" --sweep-files sweeps/project/round2_screen.json --workers-per-gpu 3 --gpus 0,1 --end-time "2026-11-02 08:00" --environment main --profiles core unsupervised --min-free-gb 15
made <repo>/artifacts/analyses/round2_prep_20261101 and the plan sweeps/rounds/round2_prep_20261101/plan.txt (it travels with the code): fill in RULE_round2.txt, fix it with rule-fix, then run round-check
```

`artifacts/analyses/round2_prep_20261101/README.txt` (with the families file added to its list):

```
Round round2, prepared <utc> (UTC) for project myproject.

Question: Does a more regularized model or a random forest beat the baseline?

Sweeps (each checked by expand; files under sweeps/, SHA-256 in sweep_files.sha256):
  sweeps/project/round2_screen.json: sweep round2_screen (screen), 4 configurations, 72 jobs, seeds [1, 2, 3]

On the node: the environment main (~/venvs/myproject-main, profiles core unsupervised); the schedule
starts 3 workers on each of GPUs 0,1 (only those no one else is using) and stops them
at 2026-11-02 08:00, the node's local time,
or earlier when no job is left (the round's session then ends) or less than 15 GB is free in the home
folder.

Plan: sweeps/rounds/round2_prep_20261101/plan.txt
  the steps node/chain.sh runs on the node in one tmux session that ends by itself. It lives under sweeps/ so that
  it travels to the node with the code; the tree digest below covers it, and its SHA-256 is in sweep_files.sha256.

Cost: round2_screen: not estimated yet (run estimate) (72 jobs in all)

Code: tree digest <sha256> (node/tree_digest.sh on the node prints the same number for the same code).

Files of this folder:
  RULE_round2.txt: what runs and how it is read, fixed with rule-fix before any result exists. Fixed at:
      not fixed yet (write rule-fix's time and SHA-256 here)
  sweep_files.sha256: the SHA-256 of each sweep file and of the plan at preparation.
  round2_families.json: the comparison families test-compare runs, fixed with the RULE.
  lines/: one file per step, numbered in the order they run: 01_push.txt, 02_install_test.txt, 03_start.txt, 04_status.txt, 05_pack.txt, 06_pull.txt.

Changes after preparation (each with the time, why, and whether it changes a job of the RULE):
(none yet)
```

The lines the person pastes on a node, one file per step; the start line, `artifacts/analyses/round2_prep_20261101/lines/03_start.txt`, for example:

```
# 3 of 6: check the schedule, smoke-test and enqueue the round's sweeps, then the workers until the queue is empty (at the latest the stop time), in one tmux session that ends by itself (node)
# shell: bash on the node
# expect: "myproject_round2 started" and the round's GPUs that are free; within minutes, logs/round2/1_schedule.txt shows "SCHEDULE OK", logs/round2/2_smoke_enqueue.txt "SMOKE OK" for each sweep and "ALL DONE", and logs/round2/3_schedule.txt "GPU <g> started" (line 04 shows these); when no job is left (at the latest at the stop time) logs/round2/chain.txt ends with "CHAIN DONE" and the session is gone
# stop if: "the tree is not", "slurmd is running", "a session of myproject is running", "jobs in queue/running", "GB free, the round's floor" or "no free GPU": stop here and paste the output
cd ~/myproject && mkdir -p logs/round2 && { D=$(bash node/tree_digest.sh .) && G=$(bash node/free_gpus.sh 2>/dev/null | grep -xE '0|1' | paste -sd, -) && if [ "$D" != <sha256> ]; then echo "the tree is not <sha256> but $D: stop here and paste this"; elif systemctl is-active --quiet slurmd 2>/dev/null; then echo "Slurm's slurmd is running, so Slurm may give these GPUs to batch jobs: stop here and paste this"; elif tmux ls -F '#{session_name}' 2>/dev/null | grep -q '^myproject_'; then echo "a session of myproject is running: stop here and paste this"; tmux ls; elif [ "$(ls queue/running 2>/dev/null | wc -l)" != 0 ]; then echo "jobs in queue/running: stop here and paste this"; elif ! bash node/disk_check.sh 15 > /dev/null; then echo "less than 15 GB free, the round's floor: stop here and paste this"; df -h ~ | tail -1; elif [ -z "$G" ]; then echo "no free GPU among the round's (0,1): stop here and paste this"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; else rm -f queue/STOP queue/STOP_gpu*; tmux new -d -s myproject_round2 "bash node/chain.sh sweeps/rounds/round2_prep_20261101/plan.txt" && echo "myproject_round2 started; free GPUs now: $G"; df -h ~ | tail -1; fi; } 2>&1 | tee -a logs/round2/03_start.txt
```

### 2. Fix the RULE before any result exists

The RULE's four sections are filled in (by Claude, approved by the person), then fixed:

```
RULE_round2.txt: what round round2 runs and how its results are read. Project myproject.
Fixed with rule-fix before any result of this round exists; the time and SHA-256 are in the record rule-fix wrote
(artifacts/rules/round2_<utc>/record.json). It does not change once a file of these runs has been read. Jobs
still queued from earlier rounds keep running under their own rules.

1. What runs
   Sweep round2_screen (sweeps/project/round2_screen.json): logistic regression (the base), a random
   forest of 200 trees, k-nearest neighbours with 15 neighbours, and the base with shuffled training
   labels (the leak control), on 5 stratified folds and the identity check fold, seeds 1, 2 and 3,
   on the CPU. No earlier result chose any of these settings.
   (The sweeps by file and sweep ID, what each asks, seeds, folds, priorities, the device, and the code version by
   tree digest. Say what earlier choice each setting comes from, and that no test or reveal file chose it.)

2. Reading
   Validation files only first: analyze ranks by balanced accuracy and compares each configuration with
   the base (paired bootstrap over seeds and rows, Holm). The highest is selected with select. Then
   test-report reads that configuration's test files, and test-compare runs family main of
   round2_families.json on the test files: the selection against the base, and against k-nearest
   neighbours, paired bootstrap with 2,000 draws and Holm's correction within the family.
   (Which files are read, in which order: validation or label-free files first, with analyze; then select; only then
   the test or reveal files of the selected configurations, with test-report or reveal-report. Which comparisons
   are made, as families (the families file given to rule-fix: paired bootstrap, Holm within each family), on which
   seeds, and what happens to runs not finished at the reading time. How select chooses: the criterion, and for
   label-free work the stability and the number of groups found, read together (a floor fixed in advance can turn
   away the best: in the template's dry run a stability floor of 0.8 turned away, in two of ten sweeps, the
   configuration nearest the planted groups); what decides a tie (for example, of two whose intervals overlap, the
   more stable, then the simpler); and what happens when the most stable is not the one the criterion ranks first.)

3. Stop condition
   The control must sit near chance (balanced accuracy near 0.5); if it is above 0.6, stop and
   check the split before reading any test file. The identity fold's validation and test numbers
   must agree. A selection that beats the base on test goes to confirmation on five new seeds.
   (Written before the results: which result is a plain substitution, for example "a configuration that beats the
   baseline on validation goes to confirmation on five new seeds", and which result changes the plan, for example
   "if the leak control is above chance by more than its seed spread, stop and check the split".)

4. What this cannot show
   One synthetic dataset; three seeds (a screen, not a confirmation); the selection is optimistic until
   confirmed on new seeds.
   (What the design of this round leaves open: one dataset, one split, seeds that are not new, settings chosen on
   validation and therefore optimistic until confirmed, transductive steps, and anything not tested.)
```

The comparison families test-compare will run, `artifacts/analyses/round2_prep_20261101/round2_families.json`:

```json
{
 "families": [
  {
   "name": "main",
   "files": "test",
   "metric": "balanced_accuracy",
   "comparisons": [
    {
     "a": {
      "selection": "round2_choice"
     },
     "b": {
      "sweep": "round2_screen",
      "config": "42f4b79a3b"
     }
    },
    {
     "a": {
      "selection": "round2_choice"
     },
     "b": {
      "sweep": "round2_screen",
      "config": "a33d5539d5"
     }
    }
   ]
  }
 ]
}
```

```
$ python -m sweepkit.cli rule-fix --name round2 --rule-file artifacts/analyses/round2_prep_20261101/RULE_round2.txt --sweeps round2_screen --families artifacts/analyses/round2_prep_20261101/round2_families.json --late-reason none
RULE round2 fixed at <utc>, SHA-256 <sha256>, covering round2_screen; 1 comparison families
saved <repo>/artifacts/rules/round2_<utc>
```

`round-check` passes before the first line is handed over:

```
$ python -m sweepkit.cli round-check --round round2 --date 20261101
round-check round2 (<repo>/artifacts/analyses/round2_prep_20261101):
  note: RULE fixed <utc>, no result yet
  note: lines/01_push.txt holds a part to fill in when the line is handed over
  note: lines/06_pull.txt holds a part to fill in when the line is handed over
ROUND OK
```

### 3. Run

On a node the lines smoke-test and enqueue the sweep and start the workers; here `run` does the same work in one process. Its last line:

```
$ python -m sweepkit.cli run --sweep-file sweeps/project/round2_screen.json --device cpu --max-consecutive-failures 3
{'sweep_id': 'round2_screen', 'jobs': 72, 'done': 72, 'skipped_complete': 0, 'failed': 0}
```

### 4. Read the validation files, then choose

```
$ python -m sweepkit.cli analyze --sweep round2_screen --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 --top 5
sweep round2_screen (classification): 4 of 4 configurations, 72 of 72 runs complete; chosen by balanced_accuracy (larger is better); validation files only; 95% intervals, 2000 bootstrap draws
NOTE the identity fold validates on test rows: its runs are a check of the evaluation paths, never read here; test-report shows whether its validation and test numbers agree
baseline 42f4b79a3bd0: seeds [1, 2, 3], balanced_accuracy 0.4954 [0.4954, 0.4954], SD 0.0000
control shuffled_labels: balanced_accuracy 0.4969 over seeds [1, 2, 3]
top 3 of 3 by balanced_accuracy; difference = configuration minus baseline:
  0.8038 [0.7911, 0.8166]  diff +0.3085 bootstrap [+0.2607, +0.3551] seeds [+0.2957, +0.3212]  P(better) 1.00  Holm p 0.002  better  | e7678cf98d70 changes: random_forest
      accuracy 0.8044  f1_macro 0.8036  roc_auc 0.8789
  0.7893 [0.7893, 0.7893]  diff +0.2939 bootstrap [+0.2485, +0.3395] seeds [+0.2939, +0.2939]  P(better) 1.00  Holm p 0.002  better  | a33d5539d50f changes: knn
      accuracy 0.7917  f1_macro 0.7897  roc_auc 0.8790
  0.4954 [0.4954, 0.4954]  (baseline)  | 42f4b79a3bd0 base: base
      accuracy 0.5050  f1_macro 0.4691  roc_auc 0.4988
saved <repo>/artifacts/analysis/round2_screen_<utc>
```

The highest validation balanced accuracy is `e7678cf98d70` (changes: random_forest). `select` records it:

```
$ python -m sweepkit.cli select --sweep round2_screen --config e7678cf98d --name round2_choice --reason "highest validation balanced accuracy, as the RULE says"
selected e7678cf98d70 of round2_screen (random_forest), balanced_accuracy 0.8038, rank 1 of 3, from round2_screen_<utc>
saved <repo>/artifacts/selections/round2_choice
```

### 5. Only now, the test files

```
$ python -m sweepkit.cli test-report --selection round2_choice --confidence 0.95
test results of selection round2_choice: round2_screen / e7678cf98d70 (random_forest), seeds [1, 2, 3]
chosen because: highest validation balanced accuracy, as the RULE says (recorded <utc>; RULE round2_<utc>)
  balanced_accuracy 0.7888 [0.7537, 0.8238] (95% t interval over seeds), SD 0.0141
  accuracy 0.7894 [0.7547, 0.8242] (95% t interval over seeds), SD 0.0140
  f1_macro 0.7886 [0.7537, 0.8235] (95% t interval over seeds), SD 0.0140
  roc_auc 0.8721 [0.8657, 0.8784] (95% t interval over seeds), SD 0.0026
  recall c0 0.778  c1 0.800
  seed 1: fold1 0.7692  fold2 0.7734  fold3 0.7756  fold4 0.8559  fold5 0.8066
  seed 2: fold1 0.7606  fold2 0.7319  fold3 0.7497  fold4 0.8383  fold5 0.7820
  seed 3: fold1 0.7692  fold2 0.7912  fold3 0.7750  fold4 0.8551  fold5 0.7978
identity check fold: validation and test numbers agree in all 3 runs
saved <repo>/artifacts/test_reports/round2_choice_<utc>
```

`test-compare` runs exactly the families the RULE fixed:

```
$ python -m sweepkit.cli test-compare --rule round2_<utc> --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75
test-compare of RULE round2_<utc> (fixed <utc>): 1 families, 95% intervals, 2000 bootstrap draws, Holm within each family
family main (test files, balanced_accuracy, larger is better):
  random_forest (e7678cf98d70, round2_screen) 0.7888 against base (42f4b79a3bd0, round2_screen) 0.4936
    diff +0.2952 [+0.2473, +0.3440]  P(A better) 1.00 [1.00, 1.00]  p 0.0010 (the least 2000 draws can give)  Holm p 0.0020  better
  random_forest (e7678cf98d70, round2_screen) 0.7888 against knn (a33d5539d50f, round2_screen) 0.7774
    diff +0.0113 [-0.0220, +0.0438]  P(A better) 0.67 [0.00, 1.00]  p 0.5197  Holm p 0.5197  no detectable difference
saved <repo>/artifacts/comparisons/round2_<utc>_<utc>
```

## After the reading

The reading folder `artifacts/analyses/<round>_<date>/` gets a README with what ran, what was read when and what it found; findings.csv, VERIFICATION_LOG.md and STATE.md are updated; and a results doc is written from `templates/results_doc.md` (docs/REPORTING.md). Round 1's choice goes to a final run on every row (`sweeps/examples/cluster_final.json` shows one); round 2's to a confirmation on five new seeds.
