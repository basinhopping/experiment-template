# Environment profiles

Written by `python -m sweepkit.cli profiles-doc --write yes` from `src/sweepkit/env.py` and the registry's declarations; do not edit by hand (a test checks it is current).

A profile is a set of pinned packages (`env/requirements-<profile>.txt`). An environment is built from one or more profiles: `main` (core, torch and unsupervised by default, and any on-request profile but rapids and automl), `rapids` (the node's GPU libraries) and `automl`. At home an environment is `.venv-<environment>` in the repository folder (`env/setup.sh` or `env/setup.ps1`); on the node it is `~/venvs/<project>-<environment>` (`node/setup_env.sh`). Install only with these scripts, never with an unpinned `pip install`.

## Which profiles a problem needs

| Problem | Profiles |
|---|---|
| Tables, scikit-learn models, k-means, GMM, hierarchical, spectral, DBSCAN, PCA, t-SNE | core |
| Neural networks on tables, sequences, images or audio | core, torch |
| HDBSCAN (the hdbscan package), UMAP, openTSNE, PaCMAP, time-series clustering | core, unsupervised |
| Large data on the node's GPU (cuML) | rapids (its own environment) |
| Single-cell data | core, singlecell (CPU); rapids, rapids-singlecell (GPU) |
| Gradient boosting | core, boosting |
| Pretrained Hugging Face models | core, torch, pretrained |
| Bayesian search (propose) | core, search |
| Genie, Lumbermark, leidenalg, FasterPAM, Harmony, PHATE, TriMap; igraph for scanpy_leiden on the CPU; the FlowSOM and ComBat packages (the template's own FlowSOM and ComBat need only core) | core, extras (read the licences first) |

## The profiles

### core

For: everything: settings, sweeps, the queue, analysis, data preparation, scikit-learn and SciPy models, and the template's own algorithms.

Environment: main. Installed: everywhere. Licences: BSD, MIT, Apache-2.0 (permissive). Platforms: Windows, Mac, Linux; Python 3.12 or 3.13.

| Package | Pin |
|---|---|
| numpy | ==2.5.3 |
| scipy | ==1.18.1 |
| pandas | ==3.0.5 |
| scikit-learn | ==1.9.1 |
| matplotlib | ==3.11.2 |
| networkx | ==3.7 |
| joblib | ==1.6.0 |
| threadpoolctl | ==3.7.0 |
| pyarrow | ==26.0.0 |
| h5py | ==3.16.0 |
| openpyxl | ==3.1.5 |
| psutil | ==7.2.2 |
| polars | ==2.1.0 |

Components that need it: graph (dataset), items (dataset), store (dataset), synthetic (dataset), table (dataset), none (feature), affinity_propagation (model), agglomerative (model; GPU cuml), bayesian_gmm (model), birch (model), bisecting_kmeans (model), consensus (model), dbscan (model; GPU cuml), decision_tree (model), diffusion_map (model), elliptic_envelope (model), extra_trees (model), factor_analysis (model), flowsom (model), fuzzy_cmeans (model), gmm (model), hdbscan_sklearn (model; GPU cuml), hist_gradient_boosting (model), ica (model), isolation_forest (model; GPU cuml), isomap (model), kernel_pca (model), kmeans (model; GPU cuml), kmedoids (model), knn (model; GPU cuml), kprototypes (model), lda (model), leiden (model; GPU cugraph), linear (model; GPU cuml), linear_svm (model; GPU cuml), lle (model), lof (model), logistic_regression (model; GPU cuml), louvain (model; GPU cugraph), markov (model), mds (model), mean_shift (model), minibatch_kmeans (model), naive_bayes (model), nmf (model), one_class_svm (model), optics (model), pca (model; GPU cuml), pcoa (model), random_forest (model; GPU cuml), som (model), spectral (model), spectral_biclustering (model), spectral_coclustering (model), spectral_embedding (model; GPU cuml), svm (model; GPU cuml), truncated_svd (model; GPU cuml), tsne (model; GPU cuml), batch_correct (prep step), drop_constant (prep step), drop_duplicates (prep step), impute (prep step), onehot (prep step), outliers (prep step), reduce (prep step), scale (prep step), select_columns (prep step), select_features (prep step), subsample (prep step), transform (prep step).

### torch

For: neural networks (the torch family): MLPs, CNNs, RNNs, Transformers, spiking models, autoencoders.

Environment: main. Installed: everywhere but an Intel Mac. Licences: BSD-3-Clause. Platforms: Linux (CUDA 13 from PyPI; cu126 index for GPUs older than Turing), Windows (cu130 index for an NVIDIA GPU, else CPU), a Mac with Apple silicon (CPU and MPS; PyTorch has published no build for an Intel Mac since 2.2).

| Package | Pin |
|---|---|
| torch | ==2.14.1 |

Components that need it: mask (augmentation), mixup (augmentation), noise (augmentation), speed (augmentation), image (feature), logmel (feature), sequence (feature), autoencoder (model; GPU torch), autoencoder_kmeans (model; GPU torch), cnn1d (model; GPU torch), cnn2d (model; GPU torch), cnn_lif (model; GPU torch), cnn_rnn (model; GPU torch), cnn_transformer (model; GPU torch), composite (model; GPU torch), dec (model; GPU torch), gat (model; GPU torch), gcn (model; GPU torch), mlp (model; GPU torch), rnn (model; GPU torch), sequence_lif (model; GPU torch), sequence_pool (model; GPU torch), snn (model; GPU torch), transformer (model; GPU torch), vae (model; GPU torch).

### unsupervised

For: HDBSCAN (hdbscan, fast_hdbscan), UMAP, openTSNE, PaCMAP, FAISS and pynndescent neighbours, tslearn.

Environment: main. Installed: everywhere but an Intel Mac. Licences: BSD, MIT, Apache-2.0 (permissive). Platforms: Windows, Linux, and a Mac with Apple silicon (numba and llvmlite publish no build of these versions for an Intel Mac; FAISS needs macOS 14).

| Package | Pin |
|---|---|
| numba | ==0.68.0 |
| llvmlite | ==0.50.0 |
| hdbscan | ==0.8.44 |
| fast-hdbscan | ==0.3.2 |
| umap-learn | ==0.5.12 |
| pynndescent | ==0.6.0 |
| openTSNE | ==1.0.4 |
| pacmap | ==0.9.1 |
| faiss-cpu | ==1.15.1 |
| tslearn | ==0.10.0 |

Components that need it: dtw_kmeans (model), fast_hdbscan (model), flasc_branches (model), hdbscan (model; GPU cuml), hdbscan_outliers (model), kshape (model), opentsne (model), pacmap (model), plscan (model), umap (model; GPU cuml).

### rapids

For: cuML, cuDF and cuGraph: k-means, DBSCAN, HDBSCAN, UMAP, PCA, t-SNE, random forests, Leiden and Louvain and more on the GPU, and someone's own scikit-learn script unchanged under cuml.accel (node/rapids_run.sh).

Environment: rapids. Installed: on the node. Licences: Apache-2.0. Platforms: Linux with an NVIDIA GPU only; its own environment (numpy 2.4, numba 0.64).

| Package | Pin |
|---|---|
| cuml-cu13 | ==26.10.0 |
| cudf-cu13 | ==26.10.2 |
| cugraph-cu13 | ==26.10.0 |
| numpy | ==2.4.6 |
| numba | ==0.64.0 |
| llvmlite | ==0.46.0 |
| scipy | ==1.18.1 |
| pandas | ==3.0.5 |
| scikit-learn | ==1.9.1 |
| umap-learn | ==0.5.12 |
| pynndescent | ==0.6.0 |
| hdbscan | ==0.8.44 |
| matplotlib | ==3.11.2 |
| networkx | ==3.7 |
| joblib | ==1.6.0 |
| threadpoolctl | ==3.7.0 |
| pyarrow | ==26.0.0 |
| h5py | ==3.16.0 |
| openpyxl | ==3.1.5 |
| psutil | ==7.2.2 |

### rapids-singlecell

For: the single-cell workflow on the GPU (rapids_singlecell), added to the rapids environment.

Environment: rapids. Installed: on request. Licences: MIT, BSD-3-Clause. Platforms: Linux with an NVIDIA GPU.

| Package | Pin |
|---|---|
| rapids-singlecell-cu13 | ==0.18.1 |
| scanpy | ==1.12.4 |
| anndata | ==0.13.4 |

### boosting

For: XGBoost (CPU or CUDA) and LightGBM.

Environment: main. Installed: on request. Licences: Apache-2.0, MIT. Platforms: Windows, Mac (brew install libomp), Linux.

| Package | Pin |
|---|---|
| xgboost | ==3.4.1 |
| lightgbm | ==4.7.0 |

Components that need it: lightgbm (model), xgboost (model; GPU xgboost).

### pretrained

For: Hugging Face models: fine-tuning (full, top layers, LoRA) and frozen encoders for features.

Environment: main. Installed: on request. Licences: Apache-2.0. Platforms: all; downloads weights.

| Package | Pin |
|---|---|
| transformers | ==5.21.0 |
| tokenizers | ==0.23.3 |
| safetensors | ==0.8.0 |
| peft | ==0.21.2 |
| huggingface-hub | ==2.3.0 |
| accelerate | ==1.15.0 |
| timm | ==1.0.30 |
| torchvision | ==0.29.1 |

Components that need it: hf_finetune (model; GPU torch).

### graph

For: PyTorch Geometric (optional: the template's own GCN and GAT need only torch).

Environment: main. Installed: on request. Licences: MIT. Platforms: all.

| Package | Pin |
|---|---|
| torch-geometric | ==2.8.1 |

### singlecell

For: scanpy, anndata and scvi-tools on the CPU (scanpy_leiden also needs igraph, in extras: GPL).

Environment: main. Installed: on request. Licences: BSD-3-Clause. Platforms: all; Python 3.12 or 3.13.

| Package | Pin |
|---|---|
| scanpy | ==1.12.4 |
| anndata | ==0.13.4 |
| scvi-tools | ==1.5.1 |

Components that need it: anndata (dataset), scanpy_leiden (model; GPU rapids_singlecell).

### search

For: Optuna, for propose (Bayesian search that writes the next sweep file).

Environment: main. Installed: on request. Licences: MIT. Platforms: all.

| Package | Pin |
|---|---|
| optuna | ==5.0.0 |

### automl

For: TPOT: genetic programming over scikit-learn pipelines.

Environment: automl. Installed: on request. Licences: LGPL-3.0. Platforms: all; its own environment.

| Package | Pin |
|---|---|
| tpot | ==1.1.0 |
| xgboost | ==3.4.1 |
| lightgbm | ==4.7.0 |
| optuna | ==5.0.0 |
| setuptools | <82 |

Components that need it: tpot (model).

### extras

For: Genie and Lumbermark (genieclust, lumbermark), leidenalg and igraph, FasterPAM k-medoids, FlowSOM, Harmony, ComBat (inmoose), PHATE, TriMap.

Environment: main. Installed: on request, knowing the licences. Licences: AGPL-3.0 (genieclust, lumbermark, deadwood), GPL-3.0 (leidenalg, igraph is GPL-2.0+, kmedoids, flowsom, harmonypy, inmoose), PHATE non-commercial (Yale) by its repository, Apache-2.0 (trimap). Platforms: where they install; annoy (trimap) and inmoose build from source, harmonypy has no Windows wheel.

| Package | Pin |
|---|---|
| genieclust | ==1.3.0 |
| lumbermark | ==0.9.1 |
| leidenalg | ==0.12.0 |
| igraph | ==1.0.0 |
| kmedoids | ==0.5.5 |
| flowsom | ==0.2.2 |
| harmonypy | ==2.1.0 |
| inmoose | ==0.9.1 |
| phate | ==2.0.0 |
| trimap | ==1.2.0 |

Components that need it: genie (model), lumbermark (model), phate (model), trimap (model; GPU torch).

## Setup lines

Home, Linux or Mac: `bash env/setup.sh main core torch unsupervised`.

Home, Windows: `powershell -ExecutionPolicy Bypass -File env\setup.ps1 -Environment main -Profiles "core,torch,unsupervised" -TorchIndex cpu` (or `-TorchIndex cu130` with an NVIDIA GPU; add `-Python 3.13` when the py launcher lists 3.13 and not 3.12: `py -0`).

Node: `bash node/setup_env.sh main core torch unsupervised`, then `bash node/setup_env.sh rapids rapids`.
