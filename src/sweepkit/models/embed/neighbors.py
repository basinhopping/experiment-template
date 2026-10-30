"""Neighbour-graph embeddings: t-SNE (scikit-learn; cuML on the GPU), openTSNE, UMAP (umap-learn; cuML on the GPU),
PaCMAP, TriMap and PHATE.

Read their pictures with care. A two-dimensional UMAP or t-SNE picture is for looking, not for measuring distances:
these methods keep neighbourhoods, not distances, so the size of a cluster and the gap between two clusters in the
picture mean little. To cluster on UMAP output, run UMAP with other settings than for a picture (more neighbours,
min_dist 0, about 10 dimensions; see prep reduce and the example cluster_umap_hdbscan), and check the clusters,
since UMAP can tear a cluster apart.

Which can embed rows they were not fitted on: opentsne (TSNEEmbedding.transform), umap (transform) and pacmap
(transform with the fit rows as its basis). tsne, trimap and phate cannot: their evaluated rows must be the fit rows.

Licences and installs (docs/PROFILES.md): umap-learn, openTSNE and PaCMAP are in the unsupervised profile; TriMap
1.2 needs torch and annoy (annoy has no wheel for Python 3.12 or 3.13, so it compiles), and PHATE's repository now
carries a non-commercial licence, so both are in the extras profile.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Embedder, LibraryEmbedder, _host, library_params, merged_kwargs, plain_params
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

EMBED = ("embedding",)
TAB = ("tabular",)
NEIGHBOUR_METRICS = Literal["euclidean", "manhattan", "cosine", "chebyshev"]


@dataclasses.dataclass(frozen=True)
class TSNESettings:
    n_components: int               # barnes_hut: at most 3; cuML: 2 only
    perplexity: float               # about the number of neighbours each row attends to; below the rows
    early_exaggeration: float
    learning_rate: float | None     # None: "auto", max(rows / early_exaggeration / 4, 50) on the CPU; on cuML its
                                    # adaptive method, which also tunes the exaggeration and neighbours by data size
    max_iter: int
    metric: NEIGHBOUR_METRICS
    init: Literal["pca", "random"]
    method: Literal["barnes_hut", "exact", "fft"]   # fft: cuML only (its fast default); exact: CPU only
    angle: float                    # barnes_hut's accuracy trade-off (0.5 is usual)
    extra: dict[str, Scalar]


@register_model("tsne", settings=TSNESettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="cuml", profile="core",
                packages=("sklearn",),
                description="t-SNE (scikit-learn; cuML on the GPU); for looking at neighbourhoods; no transform",
                example="embed_neighbors")
class TSNE(LibraryEmbedder):
    """scikit-learn 1.9.1's TSNE (max_iter, renamed from n_iter in 1.5) on the CPU; cuML's TSNE on the GPU (API as
    documented for cuML 26.12: 2 components only, methods fft, barnes_hut and exact, init random or pca; runs are
    reproducible from random_state for the fft method)."""

    @staticmethod
    def check_job(job):
        s, device = job.model, job.fit.device
        if s.method == "barnes_hut" and s.n_components > 3:
            raise ValueError("tsne barnes_hut: n_components at most 3 (use method exact)")
        if s.method == "fft" and device != "cuda":
            raise ValueError("tsne fft runs on cuML only: fit.device cuda, or method barnes_hut")
        if device == "cuda" and s.n_components != 2:
            raise ValueError("tsne on cuML: n_components 2 only")

    def make(self, s, seed, device):
        if device == "cuda":
            if s.n_components != 2:
                raise ValueError("tsne on cuML: n_components 2 only")
            from cuml.manifold import TSNE as T

            kwargs = {"n_components": s.n_components, "perplexity": s.perplexity,
                      "early_exaggeration": s.early_exaggeration, "max_iter": s.max_iter,
                      "metric": s.metric, "init": s.init, "method": s.method, "angle": s.angle,
                      "random_state": seed,
                      "learning_rate_method": "adaptive" if s.learning_rate is None else "none"}
            if s.learning_rate is not None:
                kwargs["learning_rate"] = s.learning_rate
            return T(**merged_kwargs(kwargs, s.extra, "tsne"))
        if s.method == "fft":
            raise ValueError("tsne fft runs on cuML only")
        from sklearn.manifold import TSNE as T

        return T(**merged_kwargs({"n_components": s.n_components, "perplexity": s.perplexity,
                                  "early_exaggeration": s.early_exaggeration,
                                  "learning_rate": "auto" if s.learning_rate is None else s.learning_rate,
                                  "max_iter": s.max_iter, "metric": s.metric, "init": s.init, "method": s.method,
                                  "angle": s.angle, "random_state": seed}, s.extra, "tsne"))

    def can_transform(self):
        return False

    def can_inverse(self):
        return False

    def info(self):
        out = {"kl_divergence": float(self.model.kl_divergence_)}
        if hasattr(self.model, "learning_rate_"):
            out["learning_rate_used"] = float(self.model.learning_rate_)
        return out


@dataclasses.dataclass(frozen=True)
class OpenTSNESettings:
    n_components: int
    perplexity: float
    learning_rate: float | None     # None: "auto", openTSNE's rows / exaggeration rule
    early_exaggeration: float | None   # None: "auto"
    early_exaggeration_iter: int
    n_iter: int                     # iterations after the early exaggeration phase (openTSNE's name)
    exaggeration: float | None      # None: none after the early phase; above 1 tightens clusters
    dof: float                      # degrees of freedom of the t kernel: 1 is standard t-SNE, below 1 heavier tails
    initialization: Literal["pca", "spectral", "random"]
    metric: NEIGHBOUR_METRICS
    neighbors: Literal["auto", "exact", "annoy", "pynndescent", "hnsw", "approx"]
    negative_gradient_method: Literal["auto", "bh", "fft"]
    transform_perplexity: float     # placing new rows: the perplexity of their neighbourhoods in the fitted map
    transform_n_iter: int           # placing new rows: optimization steps
    extra: dict[str, Scalar]


@register_model("opentsne", settings=OpenTSNESettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none",
                profile="unsupervised", packages=("openTSNE",),
                description="openTSNE: fast t-SNE for many rows that can place new rows in a fitted map",
                example="embed_neighbors")
class OpenTSNE(Embedder):
    """openTSNE 1.0.4: TSNE(...).fit(x) returns a TSNEEmbedding, an array that also places new rows with
    transform(x_new, perplexity, ..., n_iter). Parameters of transform not in the settings keep openTSNE 1.0.4's
    defaults (initialization median, k 25, learning_rate 0.1, early_exaggeration 4, exaggeration 1.5), which
    params() records. Threads: openTSNE's n_jobs (1 by default) can be raised through extra."""

    def fit_transform(self, x):
        from openTSNE import TSNE as T

        s = self.s
        kwargs = {"n_components": s.n_components, "perplexity": s.perplexity,
                  "learning_rate": "auto" if s.learning_rate is None else s.learning_rate,
                  "early_exaggeration": "auto" if s.early_exaggeration is None else s.early_exaggeration,
                  "early_exaggeration_iter": s.early_exaggeration_iter, "n_iter": s.n_iter,
                  "exaggeration": s.exaggeration, "dof": s.dof, "initialization": s.initialization,
                  "metric": s.metric, "neighbors": s.neighbors, "negative_gradient_method": s.negative_gradient_method,
                  "random_state": self.seed, "verbose": False}
        self.kwargs = merged_kwargs(kwargs, s.extra, "opentsne")
        self.embedding = T(**self.kwargs).fit(np.asarray(x, dtype=np.float64))
        return np.asarray(self.embedding, dtype=np.float32)

    def can_transform(self):
        return True

    def transform(self, x_new):
        placed = self.embedding.transform(np.asarray(x_new, dtype=np.float64), perplexity=self.s.transform_perplexity,
                                          n_iter=self.s.transform_n_iter)
        return np.asarray(placed, dtype=np.float32)

    def info(self):
        kl = getattr(self.embedding, "kl_divergence", None)
        return {} if kl is None else {"kl_divergence": float(kl)}

    def params(self):
        return {**plain_params(self.kwargs), "transform_perplexity": self.s.transform_perplexity,
                "transform_n_iter": self.s.transform_n_iter,
                "transform_defaults_openTSNE_1_0_4": "initialization median, k 25, learning_rate 0.1, "
                                                     "early_exaggeration 4, early_exaggeration_iter 0, "
                                                     "exaggeration 1.5"}


@dataclasses.dataclass(frozen=True)
class UMAPSettings:
    n_components: int               # about 10 to cluster on, 2 to look at
    n_neighbors: int                # larger keeps more of the global structure (30 or more to cluster on)
    min_dist: float                 # 0 to cluster on (rows packed tightly), about 0.1 for pictures
    spread: float
    metric: Literal["euclidean", "manhattan", "cosine", "correlation", "chebyshev"]
    n_epochs: int | None            # None: the library chooses from the number of rows
    learning_rate: float
    init: Literal["spectral", "random", "pca", "tswspectral"]   # cuML: spectral or random only
    set_op_mix_ratio: float
    local_connectivity: float
    repulsion_strength: float
    negative_sample_rate: int
    densmap: bool                   # densMAP (CPU only): keep local density as well
    gpu_build_algo: Literal["brute_force_knn", "nn_descent", "auto"]
    # cuML's neighbour graph: brute_force_knn keeps GPU runs reproducible; nn_descent (and auto above 50,000 rows)
    # is faster but not reproducible. Not used on the CPU.
    extra: dict[str, Scalar]


@register_model("umap", settings=UMAPSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="cuml",
                profile="unsupervised", packages=("umap",),
                description="UMAP (umap-learn; cuML on the GPU); for looking, or before clustering with other settings",
                example="embed_neighbors")
class UMAP(LibraryEmbedder):
    """umap-learn 0.5.12 (the version pinned for cuml.accel 26.10) on the CPU: a seed forces one thread, so n_jobs is
    set to 1 explicitly. cuML's UMAP on the GPU (API as documented for cuML 26.12): init spectral or random, no
    densMAP, the neighbour graph from build_algo. UMAP's inverse_transform generates new points rather than
    reconstructing rows, so no reconstruction error is computed from it."""

    @staticmethod
    def check_job(job):
        s = job.model
        if job.fit.device == "cuda":
            if s.init not in ("spectral", "random"):
                raise ValueError("umap on cuML: init spectral or random")
            if s.densmap:
                raise ValueError("umap on cuML: densmap is CPU only")

    def make(self, s, seed, device):
        if device == "cuda":
            if s.init not in ("spectral", "random") or s.densmap:
                raise ValueError("umap on cuML: init spectral or random, and no densmap")
            from cuml.manifold import UMAP as U

            return U(**merged_kwargs({"n_components": s.n_components, "n_neighbors": s.n_neighbors,
                                      "min_dist": s.min_dist, "spread": s.spread, "metric": s.metric,
                                      "n_epochs": s.n_epochs, "learning_rate": s.learning_rate, "init": s.init,
                                      "set_op_mix_ratio": s.set_op_mix_ratio,
                                      "local_connectivity": s.local_connectivity,
                                      "repulsion_strength": s.repulsion_strength,
                                      "negative_sample_rate": s.negative_sample_rate, "random_state": seed,
                                      "build_algo": s.gpu_build_algo}, s.extra, "umap"))
        from umap import UMAP as U

        return U(**merged_kwargs({"n_components": s.n_components, "n_neighbors": s.n_neighbors,
                                  "min_dist": s.min_dist, "spread": s.spread, "metric": s.metric,
                                  "n_epochs": s.n_epochs, "learning_rate": s.learning_rate, "init": s.init,
                                  "set_op_mix_ratio": s.set_op_mix_ratio, "local_connectivity": s.local_connectivity,
                                  "repulsion_strength": s.repulsion_strength,
                                  "negative_sample_rate": s.negative_sample_rate, "densmap": s.densmap,
                                  "random_state": seed, "n_jobs": 1}, s.extra, "umap"))

    def can_inverse(self):
        return False


@dataclasses.dataclass(frozen=True)
class PaCMAPSettings:
    n_components: int
    n_neighbors: int                # near pairs per row
    mn_ratio: float                 # mid-near pairs per near pair
    fp_ratio: float                 # further pairs per near pair
    distance: Literal["euclidean", "manhattan", "angular"]
    lr: float
    num_iters: int                  # iterations of the last phase (the first two run 100 each, PaCMAP's schedule)
    apply_pca: bool                 # reduce to 100 dimensions with PCA first when the data have more
    knn_backend: Literal["faiss", "annoy", "voyager"]   # annoy and voyager are optional extras of pacmap
    extra: dict[str, Scalar]


@register_model("pacmap", settings=PaCMAPSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none",
                profile="unsupervised", packages=("pacmap",),
                description="PaCMAP: keeps local and global structure; places new rows given the fit rows",
                example="embed_neighbors")
class PaCMAP(Embedder):
    """PaCMAP 0.9.1: PaCMAP(...).fit_transform(x); transform(x_new, basis=x_fit) places new rows (the fit rows are
    the basis it rebuilds its neighbour index from). random_state must be an int; it makes the sampling seeded."""

    def _make(self):
        from pacmap import PaCMAP as P

        s = self.s
        return P(**merged_kwargs({"n_components": s.n_components, "n_neighbors": s.n_neighbors,
                                  "MN_ratio": s.mn_ratio, "FP_ratio": s.fp_ratio, "distance": s.distance,
                                  "lr": s.lr, "num_iters": s.num_iters, "apply_pca": s.apply_pca,
                                  "knn_backend": s.knn_backend, "random_state": int(self.seed), "verbose": False,
                                  "save_tree": False}, s.extra, "pacmap"))

    def fit_transform(self, x):
        self.model = self._make()
        self.x_fit = np.asarray(x, dtype=np.float32)
        return np.asarray(self.model.fit_transform(self.x_fit), dtype=np.float32)

    def can_transform(self):
        return True

    def transform(self, x_new):
        return np.asarray(self.model.transform(np.asarray(x_new, dtype=np.float32), basis=self.x_fit),
                          dtype=np.float32)

    def params(self):
        return library_params(self.model)


@dataclasses.dataclass(frozen=True)
class TriMapSettings:
    n_components: int
    n_inliers: int                  # nearest neighbours that form each triplet's near point
    n_outliers: int                 # far points per near point
    n_random: int                   # random triplets per row
    distance: Literal["euclidean", "manhattan", "cosine", "chebyshev"]
    lr: float
    n_iters: int
    apply_pca: bool
    opt_method: Literal["dbd", "momentum", "sd"]
    init: Literal["pca", "random"]
    knn_backend: Literal["auto", "torch", "faiss", "faiss-flat", "faiss-ivf", "cuvs-cagra"]
    # auto: cuVS or Faiss when installed and suitable, else exact blocked search in torch; faiss and cuVS take
    # euclidean and cosine only
    extra: dict[str, Scalar]


@register_model("trimap", settings=TriMapSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="torch",
                profile="extras", packages=("trimap", "torch"),
                description="TriMap (TorchTRIMAP: CPU, CUDA or Apple MPS); keeps global structure; no transform",
                example="embed_neighbors")
class TriMap(Embedder):
    """TriMap 1.2.0's TorchTRIMAP (read at the repository's commit of 2026-08-19): seeded by random_state, run on the
    device the job resolved (cpu, cuda or mps); fit_transform returns a torch tensor on that device, copied to the
    CPU here. The older numba TRIMAP class takes no seed, so it is not used."""

    def fit_transform(self, x):
        from trimap import TorchTRIMAP

        s = self.s
        if s.knn_backend.startswith(("faiss", "cuvs")) and s.distance not in ("euclidean", "cosine"):
            raise ValueError("trimap: the faiss and cuVS neighbour searches take euclidean or cosine only")
        self.kwargs = merged_kwargs({"n_dims": s.n_components, "n_inliers": s.n_inliers, "n_outliers": s.n_outliers,
                                     "n_random": s.n_random, "distance": s.distance, "lr": s.lr, "n_iters": s.n_iters,
                                     "apply_pca": s.apply_pca, "opt_method": s.opt_method, "init": s.init,
                                     "knn_backend": s.knn_backend, "device": self.device,
                                     "random_state": int(self.seed), "verbose": False}, s.extra, "trimap")
        self.model = TorchTRIMAP(**self.kwargs)
        out = self.model.fit_transform(np.asarray(x, dtype=np.float32))
        return out.detach().cpu().numpy().astype(np.float32)

    def info(self):
        history = getattr(self.model, "loss_history_", None)
        if history is None or len(history) == 0:
            return {}
        return {"final_loss": float(history[-1])}

    def params(self):
        return {**plain_params(self.kwargs),
                "knn_backend_used": str(getattr(self.model, "knn_backend_", self.s.knn_backend))}


@dataclasses.dataclass(frozen=True)
class PHATESettings:
    n_components: int
    knn: int                        # neighbours of the adaptive kernel
    decay: float | None             # alpha-decay of the kernel (None: a plain k-nearest-neighbour graph)
    n_landmark: int | None          # landmarks that speed up large data (None: no landmarks, every row)
    t: int | None                   # diffusion time (None: chosen at the knee of the von Neumann entropy, "auto")
    gamma: float                    # informational distance: 1 is log potential, 0 square root
    n_pca: int | None               # principal components before the kernel (None: none)
    mds_solver: Literal["sgd", "smacof"]
    mds: Literal["classic", "metric", "nonmetric"]
    knn_dist: Literal["euclidean", "cosine", "manhattan", "correlation"]
    mds_dist: Literal["euclidean", "cosine"]
    knn_max: int | None             # most neighbours searched for the kernel (None: the library's choice)
    extra: dict[str, Scalar]


@register_model("phate", settings=PHATESettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none",
                profile="extras", packages=("phate",),
                description="PHATE: diffusion potentials for trajectories and branches; no transform; licence: "
                            "non-commercial (see docs/PROFILES.md)",
                example="embed_neighbors")
class PHATE(LibraryEmbedder):
    """PHATE 2.0.0 (constructor as on its main branch and in the 1.0.11 documentation). Its transform does not
    place new data (the documentation says PHATE does not embed rows not given to fit), so this embedder has no
    transform. The licence of the repository changed to a non-commercial one in 2024 while PyPI still says GPL-2:
    treat it as non-commercial."""

    def make(self, s, seed, device):
        import phate

        return phate.PHATE(**merged_kwargs({"n_components": s.n_components, "knn": s.knn, "decay": s.decay,
                                            "n_landmark": s.n_landmark, "t": "auto" if s.t is None else s.t,
                                            "gamma": s.gamma, "n_pca": s.n_pca, "mds_solver": s.mds_solver,
                                            "mds": s.mds, "knn_dist": s.knn_dist, "mds_dist": s.mds_dist,
                                            "knn_max": s.knn_max, "random_state": seed, "verbose": 0},
                                           s.extra, "phate"))

    def fit_transform(self, x):
        self.model = self.make(self.s, self.seed, self.device)
        return _host(self.model.fit_transform(np.asarray(x, dtype=np.float64))).astype(np.float32)

    def can_transform(self):
        return False

    def can_inverse(self):
        return False
