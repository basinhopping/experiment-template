"""Linear and matrix-factorization reductions: PCA (full, randomized, incremental), truncated SVD, NMF, ICA, factor
analysis, kernel PCA. PCA and truncated SVD have GPU paths in cuML. Each reports explained_variance where defined;
methods with an inverse transform get a reconstruction error from the runner."""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import LibraryEmbedder, merged_kwargs
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

EMBED = ("embedding",)
TAB = ("tabular",)


@dataclasses.dataclass(frozen=True)
class PCASettings:
    n_components: int
    solver: Literal["full", "randomized", "incremental"]   # incremental: in batches, for many rows
    whiten: bool
    batch_size: int | None          # incremental only
    extra: dict[str, Scalar]


@register_model("pca", settings=PCASettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="cuml", profile="core",
                packages=("sklearn",), description="principal component analysis (cuML on the GPU)",
                example="embed_linear")
class PCA(LibraryEmbedder):
    @staticmethod
    def check_job(job):
        if (job.model.solver == "incremental") != (job.model.batch_size is not None):
            raise ValueError("pca: batch_size with solver incremental only")

    def make(self, s, seed, device):
        if s.solver == "incremental":
            if device == "cuda":
                from cuml.decomposition import IncrementalPCA
            else:
                from sklearn.decomposition import IncrementalPCA
            return IncrementalPCA(**merged_kwargs({"n_components": s.n_components, "whiten": s.whiten,
                                                   "batch_size": s.batch_size}, s.extra, "pca"))
        if device == "cuda":
            from cuml.decomposition import PCA as P

            return P(**merged_kwargs({"n_components": s.n_components, "whiten": s.whiten,
                                      "svd_solver": "full" if s.solver == "full" else "auto"}, s.extra, "pca"))
        from sklearn.decomposition import PCA as P

        return P(**merged_kwargs({"n_components": s.n_components, "whiten": s.whiten, "svd_solver": s.solver,
                                  "random_state": seed}, s.extra, "pca"))

    def info(self):
        ratio = np.asarray(getattr(self.model, "explained_variance_ratio_", []), dtype=np.float64)
        return {"explained_variance": float(ratio.sum())} if ratio.size else {}


@dataclasses.dataclass(frozen=True)
class TruncatedSVDSettings:
    n_components: int
    n_iter: int
    extra: dict[str, Scalar]


@register_model("truncated_svd", settings=TruncatedSVDSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="cuml",
                profile="core", packages=("sklearn",), description="truncated SVD (no centring; sparse-friendly)",
                example="embed_linear")
class TruncatedSVD(LibraryEmbedder):
    def make(self, s, seed, device):
        if device == "cuda":
            from cuml.decomposition import TruncatedSVD as T

            return T(**merged_kwargs({"n_components": s.n_components, "n_iter": s.n_iter}, s.extra, "truncated_svd"))
        from sklearn.decomposition import TruncatedSVD as T

        return T(**merged_kwargs({"n_components": s.n_components, "n_iter": s.n_iter, "random_state": seed}, s.extra,
                                 "truncated_svd"))

    def info(self):
        return {"explained_variance": float(np.sum(self.model.explained_variance_ratio_))}


@dataclasses.dataclass(frozen=True)
class NMFSettings:
    n_components: int
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar"]
    beta_loss: Literal["frobenius", "kullback-leibler"]
    max_iter: int
    alpha_w: float                  # regularization of the row factors
    l1_ratio: float
    extra: dict[str, Scalar]


@register_model("nmf", settings=NMFSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none", profile="core",
                packages=("sklearn",), description="non-negative matrix factorization (non-negative data)",
                example="embed_linear")
class NMF(LibraryEmbedder):
    def make(self, s, seed, device):
        from sklearn.decomposition import NMF as N

        return N(**merged_kwargs({"n_components": s.n_components, "init": s.init, "beta_loss": s.beta_loss,
                                  "solver": "mu" if s.beta_loss != "frobenius" else "cd", "max_iter": s.max_iter,
                                  "alpha_W": s.alpha_w, "l1_ratio": s.l1_ratio, "random_state": seed}, s.extra, "nmf"))


@dataclasses.dataclass(frozen=True)
class ICASettings:
    n_components: int
    algorithm: Literal["parallel", "deflation"]
    fun: Literal["logcosh", "exp", "cube"]
    max_iter: int
    extra: dict[str, Scalar]


@register_model("ica", settings=ICASettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none", profile="core",
                packages=("sklearn",), description="independent component analysis (FastICA)", example="embed_linear")
class ICA(LibraryEmbedder):
    def make(self, s, seed, device):
        from sklearn.decomposition import FastICA

        return FastICA(**merged_kwargs({"n_components": s.n_components, "algorithm": s.algorithm, "fun": s.fun,
                                        "max_iter": s.max_iter, "whiten": "unit-variance", "random_state": seed},
                                       s.extra, "ica"))


@dataclasses.dataclass(frozen=True)
class FactorAnalysisSettings:
    n_components: int
    rotation: Literal["none", "varimax", "quartimax"]
    max_iter: int
    extra: dict[str, Scalar]


@register_model("factor_analysis", settings=FactorAnalysisSettings, family="embedder", tasks=EMBED, kinds=TAB,
                gpu="none", profile="core", packages=("sklearn",), description="factor analysis",
                example="embed_linear")
class FactorAnalysis(LibraryEmbedder):
    def make(self, s, seed, device):
        from sklearn.decomposition import FactorAnalysis as F

        return F(**merged_kwargs({"n_components": s.n_components,
                                  "rotation": None if s.rotation == "none" else s.rotation, "max_iter": s.max_iter,
                                  "random_state": seed}, s.extra, "factor_analysis"))

    def can_inverse(self):
        return False


@dataclasses.dataclass(frozen=True)
class KernelPCASettings:
    n_components: int
    kernel: Literal["rbf", "poly", "sigmoid", "cosine", "linear"]
    gamma: float | None
    degree: int
    fit_inverse_transform: bool     # learn an approximate inverse (reconstruction error)
    extra: dict[str, Scalar]


@register_model("kernel_pca", settings=KernelPCASettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="kernel PCA", example="embed_linear")
class KernelPCA(LibraryEmbedder):
    def make(self, s, seed, device):
        from sklearn.decomposition import KernelPCA as K

        return K(**merged_kwargs({"n_components": s.n_components, "kernel": s.kernel, "gamma": s.gamma,
                                  "degree": s.degree, "fit_inverse_transform": s.fit_inverse_transform,
                                  "random_state": seed}, s.extra, "kernel_pca"))

    def can_inverse(self):
        return self.s.fit_inverse_transform
