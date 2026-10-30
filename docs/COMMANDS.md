# Commands

Written by `python -m sweepkit.cli commands-doc --write yes` from the commands' own definitions; edit
`src/sweepkit/cli.py` or `src/sweepkit/commands_doc.py`, not this file.

Every command is `python -m sweepkit.cli <command> --argument value ...`, run from the repository folder with an
environment's Python (START_HERE.md; it finds the package by itself), or any Python with `PYTHONPATH=src` set. On a
node, `bash node/run.sh <command> ...` runs it with the project's environment and keeps its output in `logs/`. Every
argument is required: no command has a default, so a command line says everything it does. Results go to fixed places under `artifacts/`; no command takes an
output path. Exit codes: 0 passed, 1 failed or refused, 3 not finished (run the same command again).
`python -m sweepkit.cli <command> --help` prints the same arguments.

The order below is the order of the work: sweeps and the queue, choosing then judging, data, results, rounds,
extending, environments, code. docs/CONCEPTS.md explains the ideas; docs/WALKTHROUGH.md runs one result end to end.

## Sweeps and the queue

### check

Expand a sweep file and print what it would run, with each configuration's stage, label, hash and fit.device (nothing is queued or recorded).

| Argument | What it is |
|---|---|
| `--sweep-file` | the sweep file |

```
python -m sweepkit.cli check --sweep-file sweeps/examples/cluster_kmeans.json
```

### smoke

Run every configuration of a sweep file once on a few rows, before enqueueing it.

| Argument | What it is |
|---|---|
| `--sweep-file` | the sweep file |
| `--device` | what this machine offers: cpu, cuda or mps |
| `--rows` | rows per role in each smoke run, for example 200; a whole number |

```
python -m sweepkit.cli smoke --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --rows 200
```

### enqueue

Expand sweep files, record them, and put their jobs in the queue: all of them, or none when any is refused.

| Argument | What it is |
|---|---|
| `--sweep-file` | the sweep file, or several; one or more values |

```
python -m sweepkit.cli enqueue --sweep-file sweeps/examples/cluster_kmeans.json
```

### run

Run every job of a sweep file in this process (local mode; resumable).

| Argument | What it is |
|---|---|
| `--sweep-file` | the sweep file |
| `--device` | what this machine offers: cpu, cuda or mps |
| `--max-consecutive-failures` | stop after this many failed jobs in a row; a whole number |

```
python -m sweepkit.cli run --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --max-consecutive-failures 3
```

### worker

Take jobs from the queue until nothing is left that this worker can run.

| Argument | What it is |
|---|---|
| `--worker-id` | <group>_<n>, for example gpu0_1 or cpu_1 (STOP_<group> stops a group) |
| `--device` | what this worker offers: cpu, cuda or mps |
| `--min-free-gb` | stop when the disk has less free space than this; a number |
| `--max-consecutive-failures` | stop after this many failed jobs in a row; a whole number |

```
python -m sweepkit.cli worker --worker-id gpu0_1 --device cuda --min-free-gb 20 --max-consecutive-failures 5
```

### local-workers

Start N worker processes on this machine and wait for them (no tmux, any system).

| Argument | What it is |
|---|---|
| `--n` | how many workers; a whole number |
| `--device` | what they offer: cpu, cuda or mps |
| `--min-free-gb` | each stops below this much free disk; a number |
| `--max-consecutive-failures` | each stops after this many failed jobs in a row; a whole number |

```
python -m sweepkit.cli local-workers --n 4 --device cpu --min-free-gb 5 --max-consecutive-failures 5
```

### status

Queue, failures and the ranking of complete runs of a sweep (choosing files only).

| Argument | What it is |
|---|---|
| `--sweep` | the sweep ID |
| `--top` | how many (configuration, seed) pairs to list; a whole number |
| `--since` | a UTC time such as 2026-10-27T18:00:00, or all |

```
python -m sweepkit.cli status --sweep example_cluster_kmeans --top 10 --since all
```

### park

Move a sweep's pending jobs aside (queue/parked/<name>/), where no worker looks.

| Argument | What it is |
|---|---|
| `--sweep` | the sweep ID |
| `--name` | a name for the parked set |

```
python -m sweepkit.cli park --sweep example_cluster_kmeans --name kmeans_later
```

### unpark

Put a parked set of jobs back in the queue.

| Argument | What it is |
|---|---|
| `--name` | the parked set |

```
python -m sweepkit.cli unpark --name kmeans_later
```

### requeue-running

Move jobs left in queue/running back to pending, after every worker stopped.

| Argument | What it is |
|---|---|
| `--all-workers-stopped` | say yes once no worker runs (tmux ls shows none; local workers exited); one of `yes` |

```
python -m sweepkit.cli requeue-running --all-workers-stopped yes
```

### requeue-failed

Move a sweep's failed jobs back to pending (after fixing what failed them).

| Argument | What it is |
|---|---|
| `--sweep` | the sweep ID |

```
python -m sweepkit.cli requeue-failed --sweep example_cluster_kmeans
```

## Choosing, then judging

### analyze

Compare a sweep's configurations with its baseline on the choosing files only.

| Argument | What it is |
|---|---|
| `--sweep` | the sweep ID |
| `--confidence` | for example 0.95; a number |
| `--n-boot` | bootstrap draws, for example 2000; a whole number |
| `--seed` | seed of the bootstrap's random generator; a whole number |
| `--meaningful` | P(better) threshold whose interval's upper end counts as meaningful, for example 0.75; a number |
| `--top` | how many configurations to print; a whole number |

```
python -m sweepkit.cli analyze --sweep example_cluster_kmeans --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 --top 10
```

### select

Record the chosen configuration of an analyzed sweep, before any test or reveal file is read.

| Argument | What it is |
|---|---|
| `--sweep` | the sweep ID |
| `--config` | the configuration hash, or its first 8 or more characters |
| `--name` | a name for this selection, for example screen1_best |
| `--reason` | why this configuration, in one sentence |

```
python -m sweepkit.cli select --sweep example_cluster_kmeans --config 1a2b3c4d5e --name kmeans_best --reason "highest silhouette and stable across seeds"
```

### rule-fix

Store a RULE file with its UTC time and SHA-256, before the results it governs exist.

| Argument | What it is |
|---|---|
| `--name` | a name for the RULE, for example round3 |
| `--rule-file` | the RULE text (templates/RULE.txt is a starting point) |
| `--sweeps` | the sweep IDs it covers; one or more values |
| `--families` | a JSON file of comparison families for test-compare, or none |
| `--late-reason` | why the RULE is fixed after runs of its sweeps exist (recorded, and printed with every report that uses it), or none |

```
python -m sweepkit.cli rule-fix --name round1 --rule-file artifacts/analyses/round1_prep_20261101/RULE_round1.txt --sweeps example_cluster_kmeans --families none --late-reason none
```

### test-report

The test results of a supervised selection (reads only its configuration's test files).

| Argument | What it is |
|---|---|
| `--selection` | the selection name |
| `--confidence` | for example 0.95; a number |

```
python -m sweepkit.cli test-report --selection logistic_best --confidence 0.95
```

### reveal-report

Agreement with the reference labels of an unsupervised selection (reads only its configuration's reveal files).

| Argument | What it is |
|---|---|
| `--selection` | the selection name |
| `--confidence` | for example 0.95; a number |

```
python -m sweepkit.cli reveal-report --selection kmeans_best --confidence 0.95
```

### test-compare

Run exactly the comparison families a RULE fixed, on test or reveal files.

| Argument | What it is |
|---|---|
| `--rule` | the RULE folder name rule-fix printed (under artifacts/rules/) |
| `--confidence` | for example 0.95; a number |
| `--n-boot` | bootstrap draws, for example 2000; a whole number |
| `--seed` | seed of the bootstrap's random generator; a whole number |
| `--meaningful` | P(better) threshold, for example 0.75; a number |

```
python -m sweepkit.cli test-compare --rule round1_20261028T020000_000000Z --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75
```

### compare-runs

A sweep's runs against every copy of the same runs in other sweeps (exact reproduction).

| Argument | What it is |
|---|---|
| `--sweep` | the sweep ID |
| `--differences-shown` | how many differences to list per copy; a whole number |

```
python -m sweepkit.cli compare-runs --sweep example_cluster_kmeans --differences-shown 5
```

### atlas

Every sweep, configuration, setting and value tried, as tables.

| Argument | What it is |
|---|---|
| `--format` | csv, or xlsx for a workbook as well (needs openpyxl); one of `csv`, `xlsx` |

```
python -m sweepkit.cli atlas --format csv
```

## Data

### make-synthetic

Write a synthetic table with planted structure to data/synthetic/<name>.csv (and a sample).

| Argument | What it is |
|---|---|
| `--generator` | for example blobs, moons, mixed, classification, regression_linear, outliers |
| `--n-rows` | rows; a whole number |
| `--n-features` | columns; a whole number |
| `--n-groups` | clusters, classes or topics, as the generator documents; a whole number |
| `--noise` | the generator's spread; a number |
| `--seed` | the data's seed; a whole number |
| `--name` | the file's name under data/synthetic/ |
| `--chunk-rows` | rows written at a time; a whole number |
| `--sample-rows` | rows in the sample file beside it; a whole number |

```
python -m sweepkit.cli make-synthetic --generator blobs --n-rows 1000000 --n-features 10 --n-groups 5 --noise 1.0 --seed 1 --name blobs_1m --chunk-rows 100000 --sample-rows 1000
```

### convert

Turn a large CSV under data/ into a memory-mapped store (or Parquet), once, in chunks.

| Argument | What it is |
|---|---|
| `--source` | the CSV or TSV, relative to data/ |
| `--name` | the store's name: data/stores/<name>/ |
| `--sep` | the column separator: comma or tab; one of `comma`, `tab` |
| `--row-id` | the row ID column, or none (row numbers) |
| `--features` | input columns: names, re:<regex>, or * for every other column; one or more values |
| `--exclude` | columns that are never inputs, or none; one or more values |
| `--keep` | columns carried along as text (targets, groups, labels), or none; one or more values |
| `--chunk-rows` | rows read at a time, for example 100000; a whole number |
| `--to` | store (memory-mapped float32) or parquet; one of `store`, `parquet` |

```
python -m sweepkit.cli convert --source cells/cells.csv --name cells --sep comma --row-id cell_id --features "*" --exclude none --keep cell_type batch --chunk-rows 100000 --to store
```

### extract-features

Turn every row of text, images or audio into one vector with a frozen pretrained encoder, written once as a store (data/stores/<name>/) for the store dataset.

| Argument | What it is |
|---|---|
| `--data-file` | a JSON file holding one dataset section (its kind chooses text, image or audio) |
| `--source` | hub (a Hub model pinned to a commit) or local (a folder under data/); one of `hub`, `local` |
| `--model-id` | hub: <owner>/<name>; local: the folder, relative to data/ |
| `--revision` | hub: the commit's 40 hexadecimal characters; local: none |
| `--pooling` | cls (the first position), mean (over the real positions) or pooler (the model's pooler output); one of `cls`, `mean`, `pooler` |
| `--max-length` | text: tokens per row, longer rows cut; images and audio: none |
| `--batch-size` | rows per forward pass (audio: 1); a whole number |
| `--dtype` | the encoder's weights: float32 or bfloat16 (values are stored as float32); one of `float32`, `bfloat16` |
| `--device` | where the encoder runs: cpu, cuda or mps; one of `cpu`, `cuda`, `mps` |
| `--keep` | dataset columns carried along as text (targets, groups), or none; one or more values |
| `--name` | the store's name: data/stores/<name>/ |

```
python -m sweepkit.cli extract-features --data-file sweeps/project/notes_data.json --source hub --model-id sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --pooling mean --max-length 128 --batch-size 64 --dtype float32 --device cuda --keep label group --name notes_minilm
```

### ingest

Copy a received folder into data/received/<name>/ with a manifest; run again until finalized.

| Argument | What it is |
|---|---|
| `--name` | the copy's name under data/received/ |
| `--source` | the received folder (it is only read) |
| `--max-seconds` | stop cleanly after this long; the next call continues; a number |

```
python -m sweepkit.cli ingest --name cells_v1 --source /path/to/the/received/folder --max-seconds 3600
```

### verify-data

Re-hash an ingested copy against its manifest.

| Argument | What it is |
|---|---|
| `--name` | the copy's name |

```
python -m sweepkit.cli verify-data --name cells_v1
```

### profile-data

Look at a data file or folder before any model runs (sizes, types, missing values, joins).

| Argument | What it is |
|---|---|
| `--target` | a file or folder, relative to data/ |
| `--row-id` | the row ID column of a table, or none |
| `--labels` | a label table to join on the row IDs (relative to data/), or none |
| `--label-key` | the label table's key column, or none |
| `--max-rows` | rows to read, or all |
| `--top-values` | how many most frequent values to list per text column; a whole number |
| `--hide` | columns whose values must not be shown, comma-separated (reference labels and any other column that judges a choice), or none |

```
python -m sweepkit.cli profile-data --target cells/cells.csv --row-id cell_id --labels none --label-key none --hide cell_type --max-rows all --top-values 10
```

### estimate

Measure one configuration on growing parts of the data and predict the full run's cost.

| Argument | What it is |
|---|---|
| `--sweep-file` | the sweep file |
| `--config` | base, or 8 or more characters of a configuration hash |
| `--device` | what this machine offers: cpu, cuda or mps |
| `--fractions` | increasing fractions of the fold's fitting rows, for example 0.01 0.02 0.05; one or more values; a number |
| `--workers` | how many jobs will run at once (memory check); a whole number |

```
python -m sweepkit.cli estimate --sweep-file sweeps/project/full_run.json --config base --device cuda --fractions 0.01 0.02 0.05 --workers 3
```

## Results to read and take home

### export

The results of a selection to take home: rows with IDs, cluster summaries, plots, settings.

| Argument | What it is |
|---|---|
| `--selection` | the selection name |
| `--runs` | all, or one run ID of the selected configuration |
| `--reveal` | yes to include the files that judge the choice (reference labels, test predictions; needs the RULE discipline), no otherwise; one of `yes`, `no` |
| `--plots` | yes or no; one of `yes`, `no` |
| `--top-features` | how many distinguishing columns to list per cluster; a whole number |
| `--formats` | csv, or csv+parquet (needs pyarrow); one of `csv`, `csv+parquet` |

```
python -m sweepkit.cli export --selection kmeans_best --runs all --reveal no --plots yes --top-features 10 --formats csv+parquet
```

### cluster-report

Print what a selected clustering found: sizes, distinguishing columns, agreement tables.

| Argument | What it is |
|---|---|
| `--selection` | the selection name |
| `--runs` | all, or one run ID of the selected configuration |
| `--reveal` | yes to add the agreement with the reference labels (needs the RULE discipline), or no; one of `yes`, `no` |
| `--top-features` | how many distinguishing columns to print per cluster; a whole number |

```
python -m sweepkit.cli cluster-report --selection kmeans_best --runs all --reveal no --top-features 5
```

### describe-model

A configuration exactly as a job builds it: settings, preparation, the network or the library object's every parameter.

| Argument | What it is |
|---|---|
| `--source` | a sweep file, or selection:<name> |
| `--config` | base or 8 or more characters of a configuration hash (sweep file), or selected |

```
python -m sweepkit.cli describe-model --source selection:kmeans_best --config selected
```

### propose

The next configurations from Bayesian search over a finished sweep, as a new sweep file (needs the search profile).

| Argument | What it is |
|---|---|
| `--sweep` | the finished sweep |
| `--stage` | the stage whose factors and levels are the search space |
| `--n-configs` | how many new configurations; a whole number |
| `--seed` | the sampler's seed; a whole number |
| `--n-startup` | finished trials before TPE replaces random sampling (Optuna's default is 10); a whole number |
| `--name` | the new sweep's sweep_id |
| `--seeds` | the new sweep's seeds; one or more values; a whole number |
| `--priority` | the new sweep's priority; a whole number |

```
python -m sweepkit.cli propose --sweep round1_screen --stage search --n-configs 20 --seed 1 --n-startup 10 --name round2_proposed --seeds 1 2 3 --priority 5
```

## Rounds

### round-new

Make a round's preparation folder (README, RULE, plan, lines) from the templates.

| Argument | What it is |
|---|---|
| `--round` | the round's name, for example round3 |
| `--date` | the round's date, YYYYMMDD |
| `--question` | what the round asks, in one or two sentences |
| `--sweep-files` | the round's sweep files, under sweeps/; one or more values |
| `--workers-per-gpu` | worker processes per GPU in the schedule; a whole number |
| `--gpus` | the GPUs the schedule starts with, for example 0,1,2,3 |
| `--end-time` | when the schedule stops every worker: "YYYY-MM-DD HH:MM" in the node's local time, for example "2026-10-29 08:00", on the round's date or at most 6 days after it, or "now + 45 minutes" (at least 30) or "now + 10 hours" (at most 168) from the start |
| `--environment` | the node's environment the round runs in, as built by node/setup_env.sh, for example main or rapids |
| `--profiles` | the profiles installed in that environment, for example core unsupervised (line 02 tests the environment against them); one or more values |
| `--min-free-gb` | the free disk in GB the round keeps in the home folder, the floor agreed for the project: line 03 does not start below it, and the workers stop below it; a whole number |

```
python -m sweepkit.cli round-new --round round1 --date 20261028 --question "Which number of clusters is most stable?" --sweep-files sweeps/project/round1_screen.json --workers-per-gpu 3 --gpus 0,1 --end-time "2026-10-29 08:00" --environment main --profiles core unsupervised --min-free-gb 15
```

### round-check

Check a round's preparation folder: RULE fixed before results, lines, plan, README.

| Argument | What it is |
|---|---|
| `--round` | the round's name |
| `--date` | the round's date, YYYYMMDD |

```
python -m sweepkit.cli round-check --round round1 --date 20261028
```

## Extending

### scaffold

Write a new component: its module, catalog line, example sweep and test.

| Argument | What it is |
|---|---|
| `--kind` | clusterer, embedder, detector, estimator, torch_model, prep, metric, split, dataset or stage_kind |
| `--name` | its registered name: lowercase letters, digits and _ |
| `--where` | project (src/project/, the usual place) or template (the template's own code); one of `project`, `template` |

```
python -m sweepkit.cli scaffold --kind clusterer --name my_method --where project
```

### check-additions

Every registered component and stage kind has an example sweep that uses it, and each (metrics included) a test that names it.

| Argument | What it is |
|---|---|
| `--show` | how many problems to print; a whole number |

```
python -m sweepkit.cli check-additions --show 50
```

### smoke-all

Run every component of a profile once, from its example sweep, on a few rows.

| Argument | What it is |
|---|---|
| `--profile` | a profile name (core, torch, unsupervised, ...) or all |
| `--device` | what this machine offers: cpu, cuda or mps |
| `--rows` | rows per role, for example 100; a whole number |

```
python -m sweepkit.cli smoke-all --profile core --device cpu --rows 100
```

### make-example-data

Write the small files the example sweeps of file datasets read (data/synthetic/).

| Argument | What it is |
|---|---|
| `--confirm` | yes; one of `yes` |

```
python -m sweepkit.cli make-example-data --confirm yes
```

## Environments and documentation

### env-check

Installed package versions against the profiles' pins, and the devices this process sees.

| Argument | What it is |
|---|---|
| `--profiles` | the profiles this environment was built from, for example core torch unsupervised; one or more values |

```
python -m sweepkit.cli env-check --profiles core torch unsupervised
```

### required-modules

The import names an environment built from these profiles must have (for SWEEPKIT_REQUIRE).

| Argument | What it is |
|---|---|
| `--profiles` | profiles; one or more values |

```
python -m sweepkit.cli required-modules --profiles core torch unsupervised
```

### profiles-doc

Write docs/PROFILES.md from the profile table and the registry's declarations.

| Argument | What it is |
|---|---|
| `--write` | yes to write the file, no to print it; one of `yes`, `no` |

```
python -m sweepkit.cli profiles-doc --write yes
```

### commands-doc

Write docs/COMMANDS.md from these commands' own definitions and their examples.

| Argument | What it is |
|---|---|
| `--write` | yes to write the file, no to print it; one of `yes`, `no` |

```
python -m sweepkit.cli commands-doc --write yes
```

### test-matrix

Write the testing matrix, the last section of TESTING.md, from the components, the example sweeps and the tests.

| Argument | What it is |
|---|---|
| `--write` | yes to write the section into TESTING.md, no to print it; one of `yes`, `no` |

```
python -m sweepkit.cli test-matrix --write yes
```

### example-round

Write docs/EXAMPLE_ROUND.md by running two complete rounds in a temporary copy.

| Argument | What it is |
|---|---|
| `--write` | yes to write the file, no to print it; one of `yes`, `no` |

```
python -m sweepkit.cli example-round --write yes
```

## Code

### tree-digest

One SHA-256 over the code tree, the same number node/tree_digest.sh prints.

| Argument | What it is |
|---|---|
| `--root` | the repository folder, for example . |

```
python -m sweepkit.cli tree-digest --root .
```
