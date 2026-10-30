# Unsupervised learning

Clustering, dimensionality reduction and outlier detection, chosen without labels and judged, when labels exist,
only after the choice is recorded. The same sweeps, runners and discipline as supervised work (docs/CONCEPTS.md);
this page is about the choices that are different.

## The flow

1. **Look at the data** (`profile-data`, with `--hide` naming the reference labels so their values stay unseen until
   the reveal), and decide its preparation: the transformations, scaling and feature selection the field uses
   (docs/DATA.md), and for many columns a reduction first (below).
2. **Screen on subsamples**: the `subsamples` split fits and evaluates each configuration on k fixed subsamples of
   the rows, with three seeds. A **label-free criterion chooses** (written in the RULE before the results), and
   **stability** across seeds and subsamples is measured for every configuration.
3. **Choose**: `analyze` ranks the configurations by the criterion and compares each with the baseline; `select`
   records the choice.
4. **The final run**: the chosen configuration on every row (the `all` split; `sweeps/examples/cluster_final.json`),
   with several seeds. Its stability is then across seeds only.
5. **Reveal**, if reference labels exist (cell types, known classes, planted outliers): `reveal-report` reads the
   agreement with them for the selected configuration only, as a test set is read, once a RULE covers it.
6. **Take the results home**: `export` and `cluster-report` (below), then `pack` and `pull` from a node.

## Choosing without labels

The task names its criteria; one of them chooses, with its direction:

| Criterion | Use it for | Direction |
|---|---|---|
| `silhouette` | compact, separated clusters (k-means and the like); on the fixed sample | larger |
| `dbcv` | density-based clusters of any shape (HDBSCAN, DBSCAN, OPTICS); Moulavi et al. 2014, on the fixed sample | larger |
| `calinski_harabasz` | between- over within-cluster dispersion | larger |
| `davies_bouldin` | how alike each cluster is to its most similar one | smaller |
| `gap` | the gap statistic against uniform reference data (Tibshirani 2001); needs `criteria_options.gap_references` | larger |
| `bic`, `aic` | mixture models (`gmm`, `bayesian_gmm`) | smaller |
| `inertia` | within-cluster sum of squares; an elbow curve over the number of clusters (a `one_factor_at_a_time` stage, then `atlas` or the analysis table) | smaller |
| `stability` | agreement of the results across seeds and subsamples (below) | larger |
| `trustworthiness`, `continuity`, `knn_preservation` | embeddings: are neighbours kept? | larger |
| `reconstruction_error`, `explained_variance` | embeddings with an inverse (PCA, autoencoders) | smaller, larger |
| `em_auc`, `mv_auc` | outlier detectors: excess mass and mass volume (Goix 2016); need their `criteria_options` | larger, smaller |

Descriptive numbers are recorded but never choose: `n_clusters`, `noise_fraction`, `largest_cluster_fraction`,
`flagged_fraction`. A criterion with no value for a run (one cluster, every row noise) is recorded as missing with its
reason, and the configuration ranks last.

No criterion is right for every method: silhouette favours round clusters and will score density-based results
poorly even when they are right, and DBCV is the one made for them. Name the criterion in the RULE before the results,
with the reason, and do not switch after seeing them.

Label-free criteria are computed with Euclidean distances on the prepared columns, whatever distance the method used
(docs/DATA.md, "Distances").

## Stability

Every unsupervised configuration's stability is measured from its own output files, never from reference labels:

- **across seeds**: within each subsample, every pair of seeds, on the subsample's rows;
- **across subsamples**: for each seed, every pair of subsamples, on the rows both evaluated (they overlap).

Agreement is the adjusted Rand index for clusters (noise counts as one more group), Spearman's correlation for outlier
scores, and the mean overlap of each row's nearest neighbours for embeddings. A clustering that changes with the seed
or with a fifth of the rows is not a finding, whatever its silhouette. `stability` can also be the criterion that
chooses.

## How many subsamples

A label-free criterion is one number per run; it cannot be resampled row by row as a supervised metric can. So
comparisons resample the subsamples (paired) and the seeds, and the p-value is the larger of the bootstrap's and a
sign-flip test's over the subsamples. With k subsamples that all agree, the sign-flip test gives p = 2 / 2^k:

| Subsamples | Smallest possible p | Enough for |
|---|---|---|
| 3 | 0.25 | stability, a first look, a ranking; never a detected difference |
| 5 | 0.0625 | still no difference at 0.05 |
| 8 | 0.0078 | a difference among a few comparisons (Holm's correction multiplies p) |
| 10 | 0.0020 | a difference among ten or more comparisons |

The examples use three subsamples to stay quick. A screen meant to compare methods uses eight to ten. A
deterministic method (the same result for every seed: HDBSCAN, DBSCAN, agglomerative clustering) varies only across
subsamples, so more seeds do not help it; more subsamples do. Screen it on one seed: `analyze` compares
configurations on one seed when they share two or more subsamples. A method with randomness (k-means starts, UMAP,
the neural networks) keeps three seeds, so that its seed-to-seed spread is measured.

## Reveal metrics: agreement with known labels

When the data have reference labels, the task names the column (`reference`) and the reveal metrics: `ari`, `ami`,
`nmi`, `homogeneity`, `completeness`, `v_measure`, `fowlkes_mallows`, `purity` (clusters); `auroc`,
`outlier_average_precision`, `precision_at_n` (outliers); `knn_label_agreement`, `reference_silhouette`
(embeddings). Every run writes them to `reveal.json`, which no step that chooses may open. `reveal-report` reads them
for a selection once a RULE fixed before the selection covers its sweep; `test-compare` runs the RULE's comparison
families on them.

## Methods

Every method is a registered model with an example sweep and a test that it recovers known structure in synthetic
data. docs/PROFILES.md says which profile each needs; methods marked GPU have a cuML, cuGraph or RAPIDS path used
with `fit.device` `cuda` or `auto`. Some settings have no GPU path (cuML's HDBSCAN takes Euclidean distances only,
its DBSCAN euclidean, cosine or precomputed ones, its agglomerative clustering single linkage only, cuGraph's Leiden
modularity only): `cuda` refuses them, and `auto` runs them on the CPU and records why among the run's fallbacks.

**Clustering**

| Family | Models |
|---|---|
| centroid | `kmeans` (GPU), `minibatch_kmeans`, `bisecting_kmeans`, `fuzzy_cmeans`, `kmedoids` (any distance) |
| mixtures | `gmm`, `bayesian_gmm` |
| hierarchical | `agglomerative` (Ward, complete, average, single; cut by count or height; dendrogram saved; GPU single linkage), `genie` (extras) |
| density | `dbscan` (GPU), `hdbscan_sklearn` (GPU), `hdbscan` (GPU; outlier scores), `fast_hdbscan`, `plscan` (picks its own resolution), `flasc_branches` (branches of clusters), `optics`, `mean_shift` |
| graph | `leiden`, `louvain` (on a nearest-neighbour graph; GPU through cuGraph), `spectral`, `affinity_propagation`, `markov` |
| other | `birch`, `som`, `flowsom` (cytometry), `kprototypes` (mixed columns), `dtw_kmeans`, `kshape` (time series), `spectral_biclustering`, `spectral_coclustering` (rows and columns together), `lda` (topics of counts), `consensus` (co-association over subsamples) |
| deep | `autoencoder_kmeans` (an autoencoder, then k-means on its codes), `dec` (deep embedded clustering, Xie et al. 2016); the torch profile, on the GPU through PyTorch |
| newer | `lumbermark` (extras); Torque Clustering as a recipe (docs/EXTENDING.md) |
| single-cell | `scanpy_leiden` (scanpy, or rapids-singlecell on the GPU), and the workflow below |

**Dimensionality reduction**: `pca` (full, randomized or incremental; GPU), `truncated_svd` (GPU), `nmf`, `ica`,
`factor_analysis`, `kernel_pca`, `mds`, `pcoa` (any distance, the ecology standard with Bray-Curtis), `isomap`,
`lle`, `spectral_embedding` (GPU), `umap` (GPU), `tsne` (GPU), `opentsne`, `pacmap`, `trimap` (extras), `phate`
(extras), `diffusion_map`, and the deep ones, `autoencoder` and `vae` (a variational autoencoder; the torch profile,
on the GPU through PyTorch), which also reconstruct their input, so `reconstruction_error` can choose between them.

**Features from a pretrained model**: text, images or audio turned into one vector per row by a frozen encoder
(`extract-features`, docs/DATA.md), then clustered or reduced like any table.

**Outlier detection**: `isolation_forest` (GPU), `lof`, `one_class_svm`, `elliptic_envelope`, `hdbscan_outliers`
(GLOSH scores). Higher scores are more unusual.

## Many columns: reduce first

HDBSCAN slows down and finds less structure as columns grow, because density needs ever more points. The UMAP
documentation recommends reducing to about 10 dimensions with UMAP before HDBSCAN, with more neighbours than for
pictures and `min_dist` 0, and warns that UMAP can tear clusters, so clusters found that way need checking
(`sweeps/examples/cluster_umap_hdbscan.json`). A `reduce` step with PCA is the plainer alternative, and stability
across seeds and subsamples is the check. fast_hdbscan is a multicore HDBSCAN for low-dimensional Euclidean data.

A two-dimensional UMAP or t-SNE picture is for looking, not for measuring: distances and cluster sizes in it are not
the data's. Clustering on UMAP output uses other settings than the picture.

## Single-cell data

`sweeps/examples/singlecell_workflow.json` runs the standard workflow as one configuration: total-count
normalization, log1p, highly variable genes, PCA, a neighbour graph, Leiden, with scanpy on the CPU or
rapids-singlecell on the GPU (the singlecell and rapids-singlecell profiles). `sweeps/examples/data_anndata.json`
reads an `.h5ad` file. Cell types, when known, are reference labels: revealed after the choice.

## The GPU

On an NVIDIA GPU (the rapids environment of a node), `kmeans`, `dbscan`, `hdbscan`, `hdbscan_sklearn`, single-linkage
`agglomerative`, `pca`, `truncated_svd`, `umap`, `tsne`, `spectral_embedding` and `isolation_forest` run on cuML, and
`leiden` and `louvain` on cuGraph, with `fit.device` `cuda` (or `auto` on a GPU worker). Each run records the device
that ran and any fallback to the CPU. A person's own scikit-learn, UMAP or HDBSCAN script runs unchanged on the GPU
through cuML's zero-code-change mode (`node/rapids_run.sh`, docs/NODE.md), with a summary of what fell back to the CPU.

## Results to take home

- `export --selection <name> --runs all --reveal no --plots yes --top-features 10 --formats csv+parquet` writes, for
  the selected configuration's runs: every row's ID with its cluster, probability and outlier score where the method
  gives them (or its embedding coordinates, or its anomaly score), cluster sizes, each cluster's summary of every
  column, the columns that most distinguish each cluster (effect size), plots (the clusters on the first two
  principal components, sizes, a heatmap of cluster means, the dendrogram), the settings and package versions, as
  CSV, Parquet and PNG, with a README. With `--reveal yes` (and a RULE) it adds the agreement tables with the
  reference labels.
- `cluster-report` prints the same summaries.
- From a node, `pack` and `pull` bring the export folder home, every file checked.
