# Walkthrough: one result, end to end

One result, from a sweep file to a number someone else can check, on the example data, in a couple of minutes on a
laptop. It clusters synthetic data with three planted groups (`sweeps/examples/cluster_kmeans.json`), the
unsupervised case; the supervised differences are at the end. docs/EXAMPLE_ROUND.md runs a complete round with every
file it writes, and a test keeps it current; this page is the short version.

Run the commands from the repository folder with an environment's Python (START_HERE.md), written here as
`python -m sweepkit.cli ...`. The outputs below are what they printed when this page was written. Times and paths
differ on another machine, and so does the RULE's SHA-256 (it is the hash of your RULE's text); a configuration's
hash is the hash of its settings, the same on every machine. The numbers can differ with another code version.

## 1. The sweep file

A sweep file holds one complete configuration, the base, and the stages that vary it. The base of
`sweeps/examples/cluster_kmeans.json`, section by section:

| Section | Here | What it says |
|---|---|---|
| `data` | `synthetic`, generator `blobs`, 600 rows, 5 columns, 3 groups, noise 1.0, seed 1 | what to read: here data made by code, with the planted group in a `truth` column |
| `split` | `subsamples`, k 3, fraction 0.8 | three fixed subsamples of 80% of the rows; each run fits and evaluates one |
| `task` | `clustering`; criteria silhouette, DBCV, Calinski-Harabasz, Davies-Bouldin, number of clusters; `select` silhouette; `reference` truth; reveal metrics ARI, AMI, NMI | which label-free number chooses, and which column holds the known labels, read only after the choice |
| `prep` | `scale` with `zscore` | each column z-scored with the fit rows' mean and SD |
| `features` | `none` | the table's columns as they are |
| `model` | `kmeans`, 3 clusters, k-means++, 4 starts | the method and every setting it takes |
| `fit` | `clusterer`, device `auto` | the runner, and the device: the GPU where the method has a GPU path and one is offered |

Nothing has a default: a setting that is not written is an error, so the file says everything a run does. The stages:

```json
{"name": "base", "kind": "base", "priority": 9, "seeds": [1, 2, 3]},
{"name": "methods", "kind": "variants", "priority": 8, "seeds": [1, 2, 3],
 "variants": {"minibatch_kmeans": {"model": {...}}, "bisecting_kmeans": {"model": {...}}, "fuzzy_cmeans": {"model": {...}}}},
{"name": "k", "kind": "one_factor_at_a_time", "priority": 7, "seeds": [1, 2, 3],
 "factors": {"model.n_clusters": [2, 4, 5]}}
```

The base, three other methods with the same number of clusters, and k-means with 2, 4 and 5 clusters: seven
configurations, each on three subsamples with three seeds.

## 2. check: what will run

```
$ python -m sweepkit.cli check --sweep-file sweeps/examples/cluster_kmeans.json
{
 "sweep_id": "example_cluster_kmeans",
 ...
 "configs": 7,
 "jobs": 63,
 ...
}
```

Seven configurations times three subsamples times three seeds is 63 jobs. Each configuration is named by the hash of
its settings (the first 12 characters appear below), so the same settings always get the same name. Nothing is queued
or run.

## 3. smoke: does every configuration run

```
$ python -m sweepkit.cli smoke --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --rows 100
sweep example_cluster_kmeans: 7 configurations, 63 jobs
  stage base (base): 1 configurations x 3 seeds, priority 9
  stage methods (variants): 3 configurations x 3 seeds, priority 8
  stage k (one_factor_at_a_time): 3 configurations x 3 seeds, priority 7
     1 ok    1.01s  base: base (silhouette 0.7589)
     2 ok    0.05s  methods: minibatch_kmeans (silhouette 0.7589)
     3 ok    0.09s  methods: bisecting_kmeans (silhouette 0.7589)
     7 ok    0.04s  k: model.n_clusters=5 (silhouette 0.3341)
slowest: ea732305b146 1.01s; 12568e239e4d 0.09s; eb7375211756 0.05s
saved <repo>/artifacts/smoke/example_cluster_kmeans_<utc>
SMOKE OK
```

Every configuration once, through the real runner, on a few rows (it prints the first three, the last, and every
25th). A setting that changes nothing would show here as `NOTE identical models`.

## 4. The RULE, fixed before any result exists

Before running, write down what will be read and how: copy `templates/RULE.txt` to `artifacts/RULE_walk.txt`
(`artifacts/` is kept out of git) and replace every `{{...}}` blank in it: the round's name (`walk`), the project's
name, and the four sections (what runs; the reading, in order; the stop condition; what the round cannot show). Then
fix it:

```
$ python -m sweepkit.cli rule-fix --name walk --rule-file artifacts/RULE_walk.txt --sweeps example_cluster_kmeans --families none --late-reason none
RULE walk fixed at <utc>, SHA-256 <sha256>, covering example_cluster_kmeans; 0 comparison families
saved <repo>/artifacts/rules/walk_<utc>
```

The time and the SHA-256 prove the RULE came before the results. Later, `reveal-report` refuses to open the files
that judge a choice unless a RULE covering the sweep was fixed before the choice was recorded. `rule-fix` refuses a
RULE whose blanks are not filled in, and one fixed after runs of its sweeps exist unless `--late-reason` says why
(then every report that uses it prints that it came late, with the reason).

## 5. run

```
$ python -m sweepkit.cli run --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --max-consecutive-failures 3
data synthetic: 600 rows, kind tabular, 5 columns, loaded in 0.0s
  prep 0 scale: 480 fit rows, 5 -> 5 columns
run ea732305b146_sub1_seed1: fold sub1, seed 1, fit 480, evaluated 480 of 480, 5 columns, kmeans on cpu
  3 clusters, 0 noise rows
  criteria {'silhouette': 0.7604, 'dbcv': 0.6932, 'calinski_harabasz': 2968.1339, 'davies_bouldin': 0.3473, 'n_clusters': 3.0}
JOB DONE ea732305b146_sub1_seed1 (1 of 63) silhouette 0.7604, 1.1s
...
```

Each job writes one folder, `artifacts/runs/example_cluster_kmeans/<hash>_<fold>_seed<k>/`, once: its settings, the
cluster of every row, the criteria, what the preparation learned, and its provenance (the code's hash, the package
versions, the device that ran). Run the same command again after an interruption: finished jobs are skipped. On a
node, `enqueue` and workers do the same work (docs/NODE.md).

## 6. status

```
$ python -m sweepkit.cli status --sweep example_cluster_kmeans --top 10 --since all
sweep example_cluster_kmeans
queue: {'pending': 0, 'running': 0, 'done': 0, 'failed': 0, 'parked': 0}
runs: 63 complete of 63, 0 failed, 0 in progress; median time per run 0.0s; devices used cpu
baseline: 3 seeds, fold-mean silhouette 0.7597, SD 0.0000, range 0.7597 to 0.7597
...
```

## 7. analyze: compare the configurations on the choosing files only

```
$ python -m sweepkit.cli analyze --sweep example_cluster_kmeans --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 --top 10
sweep example_cluster_kmeans (clustering): 7 of 7 configurations, 63 of 63 runs complete; chosen by silhouette (larger is better); label-free files only; 95% intervals, 2000 bootstrap draws
NOTE label-free comparisons take the larger of the bootstrap's p and a sign-flip test's over the 3 folds, so no p here falls below 0.2500; a difference needs about 8 or more subsamples to be detectable
baseline ea732305b146: seeds [1, 2, 3], silhouette 0.7597 [0.7597, 0.7597], SD 0.0000
top 7 of 7 by silhouette; difference = configuration minus baseline:
  0.7597 [0.7597, 0.7597]  (baseline)  | ea732305b146 base: base
      dbcv 0.7036  calinski_harabasz 2968.9631  davies_bouldin 0.3477  n_clusters 3.0000
      stability (adjusted Rand index) 1.000 over 18 pairs
  0.7597 [0.7597, 0.7597]  diff +0.0000 ... no detectable difference  | eb7375211756 methods: minibatch_kmeans
  ...
  0.5862 [0.5862, 0.5862]  diff -0.1735 bootstrap [-0.1751, -0.1726] ... Holm p 1.000  no detectable difference  | 7086f817534b k: model.n_clusters=2
  ...
```

How to read it:

- Only the label-free criteria are read; the agreement with the planted groups is not opened. A clustering is chosen
  as a model is chosen on validation data, before its test.
- The three other methods find the same three clusters as k-means (the same silhouette, stability 1.000 across seeds
  and subsamples). Two, four and five clusters score lower.
- With three subsamples no difference can be called detected, however large: the NOTE says why and what to do (eight
  or more subsamples; docs/UNSUPERVISED.md, "How many subsamples").
- The report is kept in `artifacts/analysis/example_cluster_kmeans_<utc>/` (`report.txt` and `report.json`).

## 8. select: record the choice and its reason

```
$ python -m sweepkit.cli select --sweep example_cluster_kmeans --config ea732305b1 --name kmeans_best --reason "the highest silhouette, stable across seeds and subsamples"
selected ea732305b146 of example_cluster_kmeans (base), silhouette 0.7597, rank 1 of 7, from example_cluster_kmeans_<utc>
saved <repo>/artifacts/selections/kmeans_best
```

`--config` takes a prefix of the configuration's hash from the analysis. The selection records which analysis it came
from, the reason and the time.

## 9. reveal-report: the agreement with the known groups, once

```
$ python -m sweepkit.cli reveal-report --selection kmeans_best --confidence 0.95
reveal results of selection kmeans_best: example_cluster_kmeans / ea732305b146 (base), seeds [1, 2, 3]
chosen because: the highest silhouette, stable across seeds and subsamples (recorded <utc>; RULE walk_<utc>)
  ari 1.0000 [1.0000, 1.0000] (95% t interval over seeds), SD 0.0000
  ami 1.0000 [1.0000, 1.0000] (95% t interval over seeds), SD 0.0000
  nmi 1.0000 [1.0000, 1.0000] (95% t interval over seeds), SD 0.0000
saved <repo>/artifacts/reveal_reports/kmeans_best_<utc>
```

Only now are the files that judge the choice opened, for the selected configuration only, and the reading is
recorded (`reading.json`), so it is known later which judging files were read and when. Here k-means recovers the
planted groups exactly: an adjusted Rand index of 1.

## 10. Take the results home

```
$ python -m sweepkit.cli export --selection kmeans_best --runs all --reveal no --plots yes --top-features 5 --formats csv
4320 rows from 9 runs; saved <repo>/artifacts/exports/kmeans_best_<utc>
```

The folder holds `rows.csv` (every row's ID with its cluster, for each run), `sizes.csv`, `cluster_summary.csv`,
`distinguishing.csv` (the columns that most set each cluster apart), `settings.json`, `packages.json` and `plots/`,
with a README that says what each file holds. `cluster-report` prints the same summaries. From a node, `pack` and
`pull` bring the folder home with every file checked (docs/NODE.md).

## What comes next

- The final run: the chosen configuration on every row (split `all`) with five seeds,
  `sweeps/examples/cluster_final.json`, and its reveal report.
- Eight or more subsamples when the question is which of several methods is better (docs/UNSUPERVISED.md).
- On a node, the same steps as a round: lines the person pastes one at a time (docs/EXAMPLE_ROUND.md).

## The supervised version

`sweeps/examples/classify_logistic.json` classifies three planted classes:

- the task is `classification` with a `target` column; `select` names the validation metric that chooses
  (balanced accuracy);
- the split is `kfold` with five folds and the identity check fold (each fold tests one part and validates on the
  next; the identity fold's validation rows are its test rows, so no step that chooses may read it);
- the stages vary the penalty one factor at a time, the class weights, and a shuffled-label control, which should sit
  at chance.

The commands are the same until step 9, which becomes `test-report`: it opens the test files of the selected
configuration, and checks that the identity fold's validation and test numbers agree. A RULE with comparison families
(`--families` with a JSON file) lets `test-compare` compare configurations on the test files, with a paired
bootstrap over seeds and rows and Holm's correction within each family.
