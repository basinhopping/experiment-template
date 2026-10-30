# experiment-template

A template for machine-learning experiments in any domain, supervised or unsupervised, on one computer or a shared
GPU node: sweeps of configurations described in JSON, run from a file queue, each run written once with its
provenance, and read in a fixed order so that the numbers can be trusted. It is built to be adapted by a Claude
session to a new project (CLAUDE.md), and to be extended in minutes when a method is missing.

New here? Read [START_HERE.md](START_HERE.md).

## What it does

- **One settings schema for every kind of model.** A job's settings have seven sections (`data`, `split`, `task`,
  `prep`, `features`, `model`, `fit`), each naming a registered component with every setting given: no hidden
  defaults. A sweep file holds a base configuration and stages that vary it (variants, one factor at a time, grids,
  random search), with seeds, folds and priorities; each configuration is named by the hash of its settings.
- **Model families**: neural networks in PyTorch (MLPs, CNNs, RNNs, Transformers, hybrids, spiking networks, graph
  networks, autoencoders, fine-tuned pretrained encoders), scikit-learn-style estimators (logistic regression, SVMs, k-nearest neighbours, naive
  Bayes, trees, forests, gradient boosting, XGBoost, LightGBM, linear models, TPOT), clustering (centroid, mixture,
  hierarchical, density, graph, self-organizing maps, biclustering, topics, consensus, time series, single-cell,
  deep clustering), dimensionality reduction (PCA to UMAP, t-SNE, PaCMAP, diffusion maps, autoencoders and VAEs) and
  outlier detection.
- **Unsupervised learning as a first-class task**: a label-free criterion chooses (silhouette, DBCV, the gap
  statistic, BIC and more), stability is measured across seeds and subsamples, and agreement with any known labels is
  revealed only after the choice is recorded, as a test set is.
- **Data preparation as settings**: imputation, cleaning, transformations (log, arcsinh, centred log-ratio,
  total-count and more), scaling, feature selection, batch correction, dimensionality reduction and subsampling,
  fitted on the right rows only and recorded in every run.
- **Large data**: chunked conversion of a large CSV to a memory-mapped store, settings chosen on subsamples, and a
  measured estimate of time and memory before the full run.
- **The result discipline**: `analyze` on validation files only, `select` to record the choice, a RULE file fixed
  before the results, then `test-report`, `reveal-report` and `test-compare` (paired bootstrap, Holm's correction).
  The code refuses a judging read that the discipline does not allow.
- **Devices**: `cpu`, `cuda`, `mps` or `auto` per job, RAPIDS cuML and cuGraph on NVIDIA GPUs where a GPU path
  exists, and a record of the device that actually ran.
- **A shared node** (`node/`, deletable): push and pull with SHA-256, installs that never break running workers, tmux
  workers on free GPUs only, a night schedule that ends by itself, and the lines a person pastes, one at a time.
- **Extending**: `scaffold` writes a new component with its catalog line, example sweep and test;
  `check-additions` refuses a component without them; `smoke-all` runs every component.

## Quick start

```
python -m unittest discover -s tests -t .
python -m sweepkit.cli make-example-data --confirm yes
python -m sweepkit.cli smoke --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --rows 200
python -m sweepkit.cli run --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --max-consecutive-failures 3
python -m sweepkit.cli analyze --sweep example_cluster_kmeans --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 --top 10
```

Run them with an environment's Python from START_HERE.md (it finds the package by itself), or with `PYTHONPATH=src`.
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) continues from there to a recorded choice and its reading.

## Layout

| Path | What it holds |
|---|---|
| `src/sweepkit/` | the package: settings, registries, sweeps and the queue, runners, models, preparation, metrics, analysis |
| `src/project/` | a project's own components, written by `scaffold --where project` (made when first needed) |
| `sweeps/examples/` | one example sweep for every model, step, split and dataset; `sweeps/project/` for a project's own |
| `tests/` | `python -m unittest discover -s tests -t .` |
| `env/` | pinned requirements per profile and the setup scripts |
| `node/` | the shared-node layer (deletable) |
| `templates/` | the round's RULE, README and lines; the project's brief, plan, state, results doc and summaries |
| `notebooks/` | one thin example: a sweep run and its analysis read from a notebook |
| `docs/` | the documentation below |
| `data/`, `artifacts/`, `queue/`, `logs/`, `reports/` | data and results, never in git |

## Documentation

| For | Read |
|---|---|
| a new user | [START_HERE.md](START_HERE.md) |
| Claude | [CLAUDE.md](CLAUDE.md) |
| how it works | [docs/CONCEPTS.md](docs/CONCEPTS.md) |
| every command | [docs/COMMANDS.md](docs/COMMANDS.md) |
| data and preparation | [docs/DATA.md](docs/DATA.md) |
| clustering, reduction and outliers | [docs/UNSUPERVISED.md](docs/UNSUPERVISED.md) |
| adding a component | [docs/EXTENDING.md](docs/EXTENDING.md) |
| environments and which profile each component needs | [docs/PROFILES.md](docs/PROFILES.md) |
| a shared GPU node | [docs/NODE.md](docs/NODE.md) |
| the working rules and their reasons | [docs/HOW_WE_WORK.md](docs/HOW_WE_WORK.md) |
| writing results | [docs/REPORTING.md](docs/REPORTING.md) |
| one result end to end | [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) |
| a complete round with every file | [docs/EXAMPLE_ROUND.md](docs/EXAMPLE_ROUND.md) |
| checking a fresh copy | [TESTING.md](TESTING.md) |

## Licence

MIT ([LICENSE](LICENSE)). The default profiles install only permissively licensed packages. Some optional packages are
GPL, AGPL or non-commercial; they are in the `extras` profile, installed only on request, and docs/PROFILES.md lists
each licence.
