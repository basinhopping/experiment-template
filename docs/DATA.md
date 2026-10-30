# Data

Where data lives, how it gets in, the datasets that read it, the preparation steps that change it, and the distances
between rows. docs/CONCEPTS.md has the job schema these settings belong to.

## Where data lives

Everything is under `data/`, which git never sees. Paths in settings are relative to `data/` and use forward slashes
(`"path": "cells/cells_v1.csv"`): an absolute path or `..` is refused, so a sweep file works unchanged on every
machine. Licensed or private data never leaves the machines it is allowed on.

| Folder | What it holds |
|---|---|
| `data/received/<name>/` | a verified copy of a folder someone sent (`ingest`) |
| `data/manifests/` | the SHA-256 of every file of each copy |
| `data/stores/<name>/` | a large CSV converted once for memory-mapped reading (`convert`) |
| `data/synthetic/` | data made by code: `make-synthetic`, `make-example-data` |

## Getting data in, and looking at it first

- **ingest** copies a received folder into `data/received/<name>/`, hashing every byte as it is read, reading every
  copy back, and listing the source again to show nothing changed during the copy. It writes a manifest
  (`data/manifests/<name>.csv`). It resumes: run it until it reports finalized. **verify-data** re-hashes a copy
  against its manifest later.
- **profile-data** looks at a file or folder before any model code: for a folder, every file with its size, empty
  and unreadable files, editor and notebook leftovers, names that differ only in case; for a table, rows, columns,
  types, missing values, ranges, the most frequent values of text columns, duplicate row IDs, malformed lines, and
  the join with a label table (`--labels`; only its key column is reported). `--hide` names the columns whose values
  must stay unseen (the reference labels of unsupervised work, and any column that judges a choice): they show only
  how many values are present and missing. Its findings go into VERIFICATION_LOG.md entry 1 before any model runs.

## Datasets

| Dataset | Reads | Main settings |
|---|---|---|
| `synthetic` | data with known structure, made by code | `generator`, `n_rows`, `n_features`, `n_groups`, `noise`, `seed` |
| `table` | one table file: CSV, TSV, Parquet, Excel, HDF5, Feather | `path`, `format`, `sheet_or_key`, `row_id`, `features`, `exclude`, `keep`, `categorical`, `filter_column`, `filter_values` |
| `store` | a store made by `convert`: memory-mapped float32, for data larger than memory | `store`, `features`, `keep`, `filter_column`, `filter_values` |
| `items` | one file per row (NumPy or CSV arrays, 16-bit WAV audio, UTF-8 text) listed in an index CSV | `index`, `path_column`, `row_id`, `keep`, `kind`, `loader`, `sample_rate` |
| `graph` | a node table and an edge table | `nodes`, `edges`, `row_id`, `features`, `keep`, `source`, `target`, `undirected` |
| `anndata` | a single-cell AnnData file (`.h5ad`; the singlecell profile) | `path`, `layer`, `keep` (columns of `obs`), `var_filter` |

Columns are chosen by name, by pattern (`"re:^gene_"`), or `"*"` for every column not used otherwise (not the row
ID, not kept, not excluded). **Kept** columns travel with every row as text: targets, groups, batches, reference
labels, metadata. Non-numeric input columns are refused, dropped, turned into category codes (for k-prototypes and
Gower distance) or one-hot columns (`categorical`). Row IDs are kept from the first read to the last export; without
a row ID column, the row number in the file is the ID.

A **precomputed distance matrix** is a `table` with one row per row ID and one column per row ID (the header names
the row IDs; the order may differ from the rows'), and `"features": ["*"]`. Methods that take precomputed distances
are listed below.

Every run records a fingerprint of what it read (the file's SHA-256 and size), so a result names its data exactly.

`make-synthetic` writes a synthetic table to `data/synthetic/<name>.csv` in chunks (any size, for example a million
rows for the large-data test) with a small sample beside it. Each synthetic dataset also has `group` (five groups,
for group splits) and `published_split` (train, val or test, for the `fixed` split). `make-example-data` writes the
small files the example sweeps of the file datasets read.

## Large data

1. `convert` turns a large CSV into a store once, in chunks (`--chunk-rows`), reading the file twice so it is never
   whole in memory: `values.npy` (float32, memory-mapped), `row_ids.txt`, `columns.json`, the kept columns in
   `meta.csv`, and `record.json` (the source's SHA-256, counts, missing values per column). `--to parquet` writes
   Parquet instead, for other tools.
2. The `store` dataset memory-maps it: a job reads only the rows it needs.
3. Settings are chosen on subsamples (the `subsamples` split, or a `subsample` step), then the chosen configuration
   runs once on everything.
4. Before the full run, `estimate` runs the chosen configuration on growing parts of the data, fits time and peak
   memory against the rows, and predicts the full run, the whole sweep, and what N workers at once need, against this
   machine's free disk and memory. Its figures are approximate and the memory tends to be low when it extrapolates
   far: in three runs of the template's own test at a million rows, from parts of up to 50,000, the full run took 1.35
   to 1.4 times the predicted memory and 0.8 to 1.5 times the predicted time. So it plans workers with a margin of 1.5
   times the memory, says how far it extrapolates, and larger parts (`--fractions 0.05 0.1 0.2`) predict better when
   the run is long.

`tests/test_large_data.py` runs this path end to end at any size (TESTING.md).

## Features from a pretrained model

`extract-features` runs a frozen pretrained encoder (the pretrained profile) over every row of a text, image or audio
dataset (`items`) and writes one vector per row as a store, which the `store` dataset then reads like any table:

```
python -m sweepkit.cli extract-features --data-file sweeps/project/notes_data.json --source hub --model-id sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --pooling mean --max-length 128 --batch-size 64 --dtype float32 --device cuda --keep label group --name notes_minilm
```

A model from the Hugging Face Hub is pinned to one commit (`--revision`, 40 hexadecimal characters), or read from a
folder under `data/` (`--source local`), and the store's record names the model, the commit and a fingerprint of its
weights, so the features name exactly what made them. Fine-tuning the encoder itself is the torch model
`hf_finetune` (full, the top layers, or LoRA).

## Preparation steps

`prep` is a list of steps applied in order. Each is fitted on the fit (or training) rows only and then applied
unchanged to every role's rows, so a value in a held-out row never changes what a step learns; every step's test
checks this. Each run writes what every step learned and changed to `prep_record.json`, and a sweep can vary the
steps like any other setting (`{"prep.scale.method": "robust"}`).

| Step | What it does | Settings |
|---|---|---|
| `impute` | missing values: drop the fit rows that have them, or fill with the mean, median, a constant, k nearest neighbours or iterative modelling | `method`, `fill_value`, `n_neighbors`, `max_iter` |
| `drop_duplicates` | takes repeated rows out of the fit, keeping the first | `decimals` |
| `drop_constant` | drops constant and near-constant columns | `min_variance`, `max_frequency_ratio`, `min_unique_fraction` |
| `outliers` | clips each column to quantiles (winsorizing) or to mean plus or minus n SD, or removes the fit rows a detector flags | `method`, `low_quantile`, `high_quantile`, `n_sd`, `detector` |
| `subsample` | fits on a fixed share or number of the fit rows, at random or within strata, by SHA-256 of the row IDs | `method`, `fraction`, `n_rows`, `stratify`, `salt` |
| `transform` | log, log1p, arcsinh with a cofactor (cytometry), square root, Box-Cox, Yeo-Johnson, centred and additive log-ratio (compositional data), total-count normalization (single-cell counts), quantile normalization, ranks | `method`, `columns`, `offset`, `cofactor`, `pseudocount`, `reference`, `target_sum`, `n_quantiles`, `output` |
| `scale` | z-score, min-max, robust (median and IQR), max-abs, per-row L1 or L2, optionally within each value of a group column | `method`, `columns`, `per_group` |
| `select_columns` | keeps the columns named | `keep` |
| `select_features` | keeps columns by variance, by a variance threshold, the highly variable features of count data, or drops a column too correlated with an earlier one | `method`, `n`, `threshold`, `n_bins` |
| `onehot` | turns category code columns into one 0/1 column per category | `drop_first` |
| `batch_correct` | removes batch effects: centring per batch, ComBat (the template's own) or Harmony (the extras profile) | `method`, `batch`, `tolerance`, `max_iter`, `theta` |
| `reduce` | replaces the columns with an embedder's components (PCA, UMAP, ...), fitted on the fit rows | `method` (an embedder's settings) |

Steps that take rows out of the fit (`impute` with drop_rows, `drop_duplicates`, `outliers` with a detector,
`subsample`) record which rows, and unsupervised runs report those rows as removed. A step that has to see rows other
than the fit rows is declared transductive, and every report that uses it says so (docs/CONCEPTS.md).

Order matters and is part of the configuration: for single-cell counts, for example, `transform` total_count, then
log1p, then `select_features` highly_variable, then `scale`, then `reduce` with PCA.

## Distances

Methods that work from distances between rows take a `metric` (`sweepkit.models.distances`):

| Metric | For |
|---|---|
| `euclidean`, `manhattan`, `chebyshev` | the usual geometric distances |
| `cosine`, `correlation` | direction or shape rather than size |
| `canberra` | a weighted Manhattan distance, sensitive near zero |
| `braycurtis` | counts and abundances (ecology, microbiome) |
| `mahalanobis` | Euclidean after whitening by the fit rows' covariance, learned from the fit rows only |
| `jaccard`, `hamming` | binary data: a value is on when it is not 0 |
| `gower` | mixed numeric and categorical columns, numeric ranges from the fit rows |
| `precomputed` | the data are already a distance matrix (above) |

`kmedoids`, `agglomerative`, `pcoa` and `mds` take every one of these; `dbscan`, `hdbscan_sklearn`, `hdbscan` and
`optics` take the geometric ones and `precomputed`. With precomputed distances, the runner gives each method the block
of the rows it fits (and, to place other rows, their distances to the fit rows), refuses preparation steps (they would
change the distances), and the criteria follow: silhouette and the neighbour criteria use the distances, and criteria
that need coordinates (Calinski-Harabasz, Davies-Bouldin, DBCV, the gap statistic) are recorded as undefined with that
reason. `sweeps/examples/cluster_precomputed.json` runs one.

Label-free criteria other than these use Euclidean distances on the prepared columns, whatever distance the method
used: they are a fixed yardstick for comparing methods, which is worth remembering when methods with different
distances are compared on silhouette.
