# Concepts

How the template turns settings into runs and runs into results you can trust. docs/COMMANDS.md lists every command;
docs/WALKTHROUGH.md runs one result end to end.

```
sweep file ──check/smoke──> jobs ──enqueue/run──> run folders ──analyze──> ranking ──select──> selection
                                                                                               │
                                     RULE file (rule-fix, before the results) ─────────────────┤
                                                                                               v
                                              test-report / reveal-report / test-compare (the judging files)
```

## A job

A job is one configuration on one fold with one seed. Its settings have seven sections, each naming a registered
component and giving every setting that component takes. Nothing has a default: a settings file says everything a
run did, and a typo or a missing setting is an error when the sweep is expanded, not a surprise overnight.

| Section | What it says | Examples |
|---|---|---|
| `data` | where the rows come from | `synthetic`, `table`, `store`, `items`, `graph`, `anndata` |
| `split` | which rows play which role in each fold | `kfold`, `group_kfold`, `leave_one_group_out`, `fixed`, `holdout`, `all`, `subsamples`, `fit_holdout` |
| `task` | what is predicted or found, and the numbers that choose and judge | `classification`, `multilabel`, `regression`, `clustering`, `embedding`, `outlier` |
| `prep` | a list of preparation steps, fitted on the fit or training rows only | `impute`, `transform`, `scale`, `select_features`, `batch_correct`, `reduce`, `subsample`, ... |
| `features` | per-batch input transforms of neural networks (`none` otherwise) | `none`, `sequence`, `image`, `logmel` |
| `model` | the method and its settings | `logistic_regression`, `random_forest`, `kmeans`, `hdbscan`, `umap`, `cnn1d`, ... |
| `fit` | how the method is fitted: the family's procedure and its settings | `estimator`, `clusterer`, `embedder`, `detector`, `torch` |

docs/PROFILES.md lists every registered component with the environment profile it needs.

## Components and registries

Every dataset, split, task, preparation step, feature, augmentation, model, fit procedure and metric is a registered
component: a frozen settings class with no defaults and a registry decorator, imported once by `catalog.py`. The
registry carries what the rest of the code needs to know: the family, the tasks and input kinds a model accepts, its
GPU path, its environment profile and packages, a one-line description and its example sweep. A project's own
components live in `src/project/` and are registered the same way (`scaffold --where project`, docs/EXTENDING.md).

Optional packages (torch, cuML, hdbscan, umap, transformers, ...) are imported inside functions, never at the top of a
module, so every settings check works with the core profile alone.

## Sweep files

A sweep file (JSON) holds:

| Key | What it holds |
|---|---|
| `sweep_id` | letters, digits, `_` and `-`; names the sweep's records and run folders |
| `description` | one or two sentences: what the sweep asks |
| `purpose` | `learning`, `smoke`, `screen`, `confirm`, `final`, `control`, `protocol` or `regression` |
| `follows` | the sweep IDs this one builds on (a `confirm` or `final` sweep may not reuse their seeds) |
| `folds` | the folds every configuration runs on (names from the split scheme) |
| `base` | a complete configuration: all seven sections |
| `stages` | each with a `name`, a `kind`, a `priority` (0 to 999, higher runs first) and its `seeds` |

Stage kinds:

| Kind | Configurations |
|---|---|
| `base` | the base alone (with five seeds, it measures seed noise) |
| `variants` | named configurations, each a set of overrides of the base: alternatives and controls |
| `one_factor_at_a_time` | each level of each factor changed alone from the base |
| `grid` | every combination of the factors' levels (the one equal to the base is left out) |
| `random` | `n_configs` configurations drawing every factor's level at random from `space`, with a `sampling_seed`; `skip_draws` extends a random stage so that the extension draws exactly what one larger stage would have |

An override names a setting by its path: `{"model.min_cluster_size": 50}`, `{"prep.scale.method": "robust"}` (a prep
step by name when it occurs once, or by position, `prep.1.method`), or a whole section, `{"model": {"name": ...}}`.
Every override must name a setting that exists in the base. A factor `{"model.n_clusters": [3, 4, 5]}` is short for
three levels.

**One protocol per sweep.** `data`, `split` and `task` are the same in every configuration of a sweep, so every
configuration is compared on the same validation (or evaluated) rows.

`check` expands a sweep file and prints what it would run; `smoke` runs every configuration once on a few rows
through the real runner, and flags configurations that built identical models and outputs (a setting that changes
nothing); `enqueue` records the sweep and queues its jobs; `run` runs them in one process. A sweep meant to rank or
report with one seed in a stage gets a warning: screen with three seeds, report with five.

## Configurations, hashes and run IDs

A configuration is named by the SHA-256 of its validated settings (`config_hash`), so the same settings always have
the same name, on any machine, in any sweep. A run's ID is `<first 12 characters of the hash>_<fold>_seed<k>`.
Settings added to a component later are added so that every earlier configuration keeps its hash.

Every random draw comes from a named stream derived from the run's seed (`sweepkit.streams`): the model's
initialization, the batches, the augmentations, the shuffled-label control. Adding a new random component does not
change the draws of the existing ones, so old configurations reproduce exactly.

## Folds and roles

A split scheme makes folds; each fold gives every row a role:

- supervised tasks: `train` (fitting), `val` (choosing) and `test` (judging the choice), never sharing a row;
- unsupervised tasks: `fit` (the rows a method is fitted on) and `eval` (the rows scored; often the same rows).

| Split | Folds |
|---|---|
| `kfold` | rows dealt into k groups (stratified by a column if named); fold i tests group i, validates on group i+1, trains on the rest |
| `group_kfold` | the same with whole groups (subjects, patients, sites), so no group is in two roles |
| `leave_one_group_out` | one fold per value of a group column, in the order given; the next value validates |
| `fixed` | a column says `train`, `val` or `test` for every row (a published split) |
| `holdout` | one fold with stratified validation and test fractions |
| `all` | one fold: fit on every row, evaluate every row (the final run of an unsupervised round) |
| `subsamples` | k fixed subsamples, each fitted and evaluated: choose settings, measure stability |
| `fit_holdout` | fit on a fraction, place the other rows with the method's predict |

Row orders come from SHA-256 of a fixed salt and the row ID, never from a random generator, so the folds are the same
on every machine. Every run records the folds' hash.

**The identity fold.** The supervised schemes take `identity_fold`: `true` adds the fold `identity`, whose validation
rows are its test rows. It checks that the validation and test paths give the same numbers. Its numbers are test
numbers, so no step that chooses reads them: `analyze`, `status`, `select`, `atlas` and `propose` leave it out,
and `test-report` reports whether its validation and test numbers agree (a disagreement makes it exit 1).

## Tasks and metrics

A task names the numbers recorded for every run and the one that chooses, with its direction:

- `classification`: the target column and its classes, metrics such as `balanced_accuracy`, `f1_macro`, `log_loss`,
  `roc_auc`; `multilabel`: one 0/1 column per label; `regression`: `rmse`, `mae`, `r2`, `ccc`, ... Each may also
  name group columns, whose groups get their own metrics.
- `clustering`, `embedding` and `outlier`: label-free criteria (`silhouette`, `dbcv`, `calinski_harabasz`, `gap`,
  `bic`, `trustworthiness`, `em_auc`, ...), one of which chooses, or `stability`; a fixed sample of rows for the costly
  ones; and optionally a `reference` column of known labels with the reveal metrics (`ari`, `nmi`, `purity`,
  `auroc`, ...) that are read only after the choice (docs/UNSUPERVISED.md).

A criterion with no value for a run (a silhouette with one cluster) is recorded as missing with its reason, and
`analyze` ranks such configurations last. Descriptive numbers (`n_clusters`, `noise_fraction`) are recorded but
cannot choose.

## Model families and run folders

Each family has one runner. A run folder is written into `<id>.tmp/` and renamed when complete; a complete folder is
never touched again, and a failed one becomes `<id>.failed-<utc>/` with its error, kept as evidence.

| Family | Fits | Choosing files | Judging files |
|---|---|---|---|
| `estimator` | any scikit-learn-style model on the training rows | `metrics_val.json`, `predictions_val.csv` (and `_train` with `eval_train`) | `metrics_test.json`, `predictions_test.csv` |
| `torch` | a network, choosing the epoch on validation | the same, with each epoch's curve in `metrics_val.json` | the same, and `metrics_external_<name>.json` for external datasets |
| `clusterer` | a clustering method on the fit rows | `criteria.json`, `assignments.csv` (cluster, probability, outlier score) | `reveal.json`, `reveal_reference.csv` |
| `embedder` | a reduction on the fit rows | `criteria.json`, `embedding.npy`, `embedding_rows.csv` | the same reveal files |
| `detector` | an outlier detector on the fit rows | `criteria.json`, `scores.csv` | the same reveal files |

The torch family reads tables, sequences, images, audio, text (`hf_finetune` fine-tunes a pretrained encoder) and
graphs (`gcn` and `gat` classify the nodes of one graph, and are transductive: below). The autoencoders and DEC are
torch networks in the embedder and clusterer families.

Every run folder also holds `spec.json` (the job), `model_params.json` (every parameter the library used, defaults
included), `prep_record.json` (what each preparation step learned and changed), `efficiency.json` (times, peak
memory), `provenance.json` (code hash, package versions, device requested and used, CPU fallbacks, folds' hash, data
fingerprint) and `log.txt`. No log line carries a test number.

## Running

- **Local mode**: `run` runs a sweep file in one process, highest priority first, and resumes (a complete run folder is
  skipped). `enqueue` then `local-workers --n N` runs N worker processes on one machine, on any system.
- **The queue**: one JSON file per job in `queue/pending/`, named so that higher priorities sort first. A worker
  takes a job by renaming its file into `queue/running/` (a rename is atomic, so two workers never take the same job),
  then moves it to `done/` or `failed/`. `queue/STOP` stops every worker after its current job; `queue/STOP_<group>`
  stops one group (a GPU's workers). `park` moves a sweep's pending jobs aside, `requeue-running` and
  `requeue-failed` recover jobs.
- **Devices**: `fit.device` is `cpu`, `cuda`, `mps` (torch) or `auto` (the method's GPU path when the worker offers
  a GPU, the package is installed and the GPU path takes the settings: with `auto`, cuML's HDBSCAN with a metric
  other than Euclidean runs on the CPU, while `cuda` refuses it). The device that actually ran, and any CPU fallback
  with its reason, is recorded in every run.
- **A shared node**: the same queue, with tmux workers on free GPUs and a schedule that ends by itself (docs/NODE.md).

## Reading the results

### Which files a step may open

| Step | Opens | Never opens |
|---|---|---|
| `status`, `analyze`, `select`, `atlas`, `propose`, `smoke` | the choosing files | any judging file, and any result of the identity fold |
| `compare-runs` | the choosing files; the judging files and the identity fold's results only as bytes, to compare their SHA-256 | the values in any judging file |
| `test-report`, `reveal-report` | the judging files of one selected configuration | anything of other configurations |
| `test-compare` | the judging files of the configurations a RULE names | anything else |
| `export --reveal yes`, `cluster-report --reveal yes` | a selection's judging files | |

Every analysis step opens run files through one reader that takes its purpose ("choose" or "judge") and refuses a
judging file to a step that chooses (`sweepkit.analysis.records`); the tests check that `analyze`, `select`,
`status` and `smoke` open no judging file at all.

### analyze

For every configuration: the choosing metric's fold mean per seed, the mean over seeds with a t interval, the other
metrics, per-group values, and (unsupervised) stability. Every configuration is compared with the sweep's baseline
(the `base` stage):

- supervised metrics that take row weights: a **paired bootstrap** over seeds and rows: each draw resamples each
  configuration's seeds, and within every fold resamples the rows once, the same rows for both configurations;
- label-free criteria and stability: the **folds resampled (paired) and the seeds**, with a p that is the larger of
  the bootstrap's and a **sign-flip test** over the folds. A handful of folds cannot establish a difference: three
  subsamples can never give p below 0.25, so a comparison needs about eight or more subsamples (docs/UNSUPERVISED.md).
  One seed is enough over two or more folds, so a deterministic method (the same result for every seed: HDBSCAN,
  DBSCAN, agglomerative clustering) is screened on one seed; supervised metrics need two seeds or more;
- **P(better)**, the probability that a run of one configuration beats a run of the other, with an interval: over the
  seeds for supervised metrics, within each fold and averaged over the folds for label-free criteria; `--meaningful`
  sets the level at which a difference counts as meaningful (for example 0.75);
- **Holm's correction** across the configurations compared with the baseline, and a verdict word: better, worse, or
  no detectable difference.

Controls (a `fit.control` other than `none`) are listed apart and never ranked. Configurations whose output files are
byte-identical (they built the same model) are grouped. For random stages, every factor level's mean choosing value
is listed. A comparison that cannot be made honestly is reported as not compared, with the reason: a metric undefined
in some bootstrap draw (a class with very few rows), or no spread at all to estimate from.

### select

`select` records the chosen configuration with the reason, its choosing values, the analysis report it came from
(by SHA-256) and the UTC time. It refuses if runs completed after the analysis, so a choice always sees every result.

### RULE files and the discipline

A RULE file, written before the results exist, says what runs, which judging files will be read and which
comparisons will be made, as families of comparisons with Holm's correction within each family, what happens to
unfinished runs, and what result would change the plan (`templates/RULE.txt`). `rule-fix` stores it with its UTC
time and SHA-256, with the sweeps it covers and an optional families file:

```
{"families": [
  {"name": "main", "files": "test", "metric": "balanced_accuracy",
   "comparisons": [{"a": {"selection": "final_model"}, "b": {"sweep": "round2_screen", "config": "3fa9c2d1"}}]}
]}
```

`files` is `test` (supervised) or `reveal` (unsupervised); a side is a recorded selection, or a sweep and 8 or more
characters of a configuration hash.

With `"discipline": "enforced"` in `project.json` (the default):

- `rule-fix` records how many runs of each covered sweep exist when the RULE is fixed, and refuses when some do
  unless `--late-reason` says why; the reason is recorded, and every report that uses the RULE prints "RULE <name> was
  fixed after <n> runs of <sweep> existed" with it. A confirmation sweep whose RULE is fixed before its runs is not
  affected;
- `test-report` and `reveal-report` read a selection's judging files only when a RULE covering its sweep was fixed
  before the selection was recorded;
- `test-compare` runs a RULE's families only when no configuration they name had its judging files read before the
  RULE was fixed.

Every judging read writes `reading.json` beside its report, so these checks read the record, not memory. A refusal
prints REFUSED and exits 1. `"advisory"` runs anyway and puts a warning at the top of the report.

### Transductive settings

A setting that uses the inputs (never the labels) of rows outside the fit rows, such as normalization statistics per
subject over every row of that subject, or a graph network whose message passing sees the test nodes, is marked
transductive by its component. `analyze`, `select`, `test-report`, `reveal-report` and `export` say so beside every
configuration that uses it.

## Controls and checks

- **Shuffled labels** (`fit.control: shuffle_train_labels`): the same configuration fitted on permuted training
  labels should sit at chance. Read it as the mean over seeds and folds: with well separated classes a single run can
  land far from chance in either direction, and only the mean is near it.
- **The identity fold** checks the two evaluation paths (above).
- **compare-runs** compares a sweep's runs with every copy of the same run IDs in other sweeps: the choosing files
  by SHA-256 and value (timing left out), the judging files by SHA-256 only, so no test or reveal value is read or
  shown. A run counts as reproduced when a copy made by other code is identical; it is how a reproduction is proved.
  The verdict is SAME, DIFFERENT, INCOMPLETE (runs not finished: run it again later) or NO COPIES (no run of the sweep
  has a copy, so nothing was compared).
- **atlas** lists every sweep, configuration, setting and value tried, so nothing is tried twice by accident.
