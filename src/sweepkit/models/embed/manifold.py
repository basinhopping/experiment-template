"""Manifold and distance-based reductions: metric and non-metric MDS, principal coordinates analysis (PCoA), Isomap,
locally linear embedding (standard, modified, Hessian, LTSA) and spectral embedding.

Which can embed rows they were not fitted on (fit.assign predict, or a reduce step whose evaluated rows are not its
fit rows):
  isomap, lle   yes, through the library's transform
  pcoa          yes, by Gower's (1968) formula for adding a point to a classical scaling (exact for Euclidean
                distances, an approximation for others; not for gower, whose column ranges come from the fit rows)
  mds, spectral_embedding   no: the evaluated rows must be the fit rows
MDS and PCoA hold every pairwise distance of the fit rows, so their memory grows with the square of the rows
(10,000 rows: about 800 MB in float64); screen on subsamples.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Embedder, LibraryEmbedder, merged_kwargs, plain_params
from sweepkit.models.distances import DISTANCE_OR_PRECOMPUTED, matrix
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

EMBED = ("embedding",)
TAB = ("tabular",)
@dataclasses.dataclass(frozen=True)
class MDSSettings:
    n_components: int
    metric_mds: bool                # True: metric MDS (SMACOF on the distances); False: non-metric (ranks only)
    metric: DISTANCE_OR_PRECOMPUTED  # the dissimilarity between rows (sweepkit.models.distances), or precomputed
    init: Literal["random", "classical_mds"]   # classical_mds: start from PCoA (n_init is then ignored)
    n_init: int                     # restarts from random starts; the lowest stress is kept
    max_iter: int
    eps: float                      # stop when the relative change of stress is below this
    normalized_stress: bool         # report Stress-1 (normalized) rather than raw stress
    extra: dict[str, Scalar]


@register_model("mds", settings=MDSSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none", profile="core",
                packages=("sklearn",),
                description="multidimensional scaling, metric or non-metric (scikit-learn SMACOF; no transform)",
                example="embed_manifold")
class MDS(LibraryEmbedder):
    """scikit-learn 1.9.1's MDS: `metric_mds` chooses metric or non-metric scaling (renamed from `metric` in 1.8),
    `metric` names the dissimilarity (it replaced `dissimilarity`, deprecated in 1.8), and `init` is set explicitly
    because its default changes to classical_mds in 1.10. The distances come from sweepkit.models.distances (or are
    the data, precomputed) and reach scikit-learn as a precomputed matrix, so every distance there works here."""

    def make(self, s, seed, device):
        from sklearn.manifold import MDS as M

        return M(**merged_kwargs({"n_components": s.n_components, "metric_mds": s.metric_mds,
                                  "metric": "precomputed", "init": s.init, "n_init": s.n_init,
                                  "max_iter": s.max_iter, "eps": s.eps, "normalized_stress": s.normalized_stress,
                                  "random_state": seed}, s.extra, "mds"))

    def fit_transform(self, x):
        d, _ = matrix(np.asarray(x, dtype=np.float64), self.s.metric, self.categorical)
        return super().fit_transform(d)

    def can_transform(self):
        return False

    def can_inverse(self):
        return False

    def info(self):
        return {"stress": float(self.model.stress_), "n_iter": int(self.model.n_iter_)}


@dataclasses.dataclass(frozen=True)
class PCoASettings:
    n_components: int
    metric: DISTANCE_OR_PRECOMPUTED  # any distance of sweepkit.models.distances (gower: mixed numeric and categorical
                                     # columns), or precomputed: the data are a distance matrix


@register_model("pcoa", settings=PCoASettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none", profile="core",
                packages=("scipy",),
                description="principal coordinates analysis (classical scaling) on any distance (own implementation)",
                example="embed_manifold")
class PCoA(Embedder):
    """Principal coordinates analysis (Gower 1966), the template's own: the squared distances are double-centred,
    B = -1/2 J D^2 J, and the rows' coordinates are the eigenvectors of B scaled by the square roots of their
    eigenvalues. A Euclidean distance gives the same coordinates as PCA, up to the sign of each component.

    A distance that is not Euclidean (Bray-Curtis, for example) can give negative eigenvalues: they have no
    coordinates, and info() reports their share of the total absolute eigenvalue mass (negative_eigenvalue_share).
    The share of variance of each kept component is its eigenvalue over the sum of the positive eigenvalues."""

    def fit_transform(self, x):
        from scipy.linalg import eigh

        x = np.asarray(x, dtype=np.float64)
        d, self.distances = matrix(x, self.s.metric, self.categorical)
        n = d.shape[0]
        k = self.s.n_components
        if k >= n:
            raise ValueError(f"pcoa: n_components {k} must be below the rows ({n})")
        squared = d ** 2
        row_means = squared.mean(axis=1)
        b = -0.5 * (squared - row_means[:, None] - row_means[None, :] + squared.mean())
        values, vectors = eigh(b)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        positive = values[values > 1e-12 * max(1.0, abs(values[0]))]
        if positive.size < k:
            raise ValueError(f"pcoa: only {positive.size} positive eigenvalues, fewer than n_components {k}")
        self.eigenvalues = values[:k]
        self.vectors = vectors[:, :k]
        self.diag_b = np.diag(b).copy()
        self.x_fit = x
        self.positive_sum = float(positive.sum())
        negative = values[values < 0]
        self.negative_share = float(-negative.sum() / np.abs(values).sum()) if negative.size else 0.0
        self.coordinates = self.vectors * np.sqrt(self.eigenvalues)
        return self.coordinates.astype(np.float32)

    def can_transform(self):
        return True

    def transform(self, x_new):
        """Gower's add-a-point formula: y = 1/2 L^(-1/2) V' (diag(B) - d^2), with d the new row's distances to the
        fit rows (with precomputed distances, x_new holds them). For a fit row it gives that row's own coordinates
        back."""
        x_new = np.asarray(x_new, dtype=np.float64)
        d = x_new if self.s.metric == "precomputed" else self.distances.between(x_new, self.x_fit)
        squared = d ** 2
        out = 0.5 * (self.diag_b[None, :] - squared) @ self.vectors / np.sqrt(self.eigenvalues)[None, :]
        return out.astype(np.float32)

    def info(self):
        shares = self.eigenvalues / self.positive_sum
        return {"explained_variance": float(shares.sum()), "eigenvalues": [float(v) for v in self.eigenvalues],
                "component_shares": [float(v) for v in shares], "negative_eigenvalue_share": self.negative_share}

    def params(self):
        return plain_params(dataclasses.asdict(self.s))


@dataclasses.dataclass(frozen=True)
class IsomapSettings:
    n_components: int
    n_neighbors: int                # the neighbourhood graph whose shortest paths approximate geodesic distances
    metric: Literal["euclidean", "manhattan", "cosine", "chebyshev"]
    eigen_solver: Literal["auto", "arpack", "dense"]
    path_method: Literal["auto", "FW", "D"]   # Floyd-Warshall or Dijkstra for the shortest paths
    extra: dict[str, Scalar]


@register_model("isomap", settings=IsomapSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",), description="Isomap: geodesic distances on a neighbour graph",
                example="embed_manifold")
class Isomap(LibraryEmbedder):
    def make(self, s, seed, device):
        from sklearn.manifold import Isomap as I

        return I(**merged_kwargs({"n_components": s.n_components, "n_neighbors": s.n_neighbors,
                                  "metric": s.metric, "eigen_solver": s.eigen_solver, "path_method": s.path_method},
                                 s.extra, "isomap"))

    def can_inverse(self):
        return False

    def info(self):
        # Isomap's own measure: the error of the kernel's low-rank approximation, not a reconstruction of the input
        return {"kernel_reconstruction_error": float(self.model.reconstruction_error())}


@dataclasses.dataclass(frozen=True)
class LLESettings:
    n_components: int
    n_neighbors: int
    method: Literal["standard", "modified", "hessian", "ltsa"]
    reg: float                      # regularization of the local weights
    eigen_solver: Literal["auto", "arpack", "dense"]
    hessian_tol: float              # hessian only
    modified_tol: float             # modified only
    extra: dict[str, Scalar]


@register_model("lle", settings=LLESettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="none", profile="core",
                packages=("sklearn",),
                description="locally linear embedding: standard, modified, Hessian (HLLE) or LTSA",
                example="embed_manifold")
class LLE(LibraryEmbedder):
    @staticmethod
    def check_job(job):
        s = job.model
        if s.method == "hessian" and s.n_neighbors <= s.n_components * (s.n_components + 3) / 2:
            raise ValueError(f"lle hessian: n_neighbors must be above n_components * (n_components + 3) / 2 = "
                             f"{s.n_components * (s.n_components + 3) / 2:g}")
        if s.method == "modified" and s.n_neighbors < s.n_components:
            raise ValueError("lle modified: n_neighbors must be at least n_components")

    def make(self, s, seed, device):
        from sklearn.manifold import LocallyLinearEmbedding as L

        return L(**merged_kwargs({"n_components": s.n_components, "n_neighbors": s.n_neighbors, "method": s.method,
                                  "reg": s.reg, "eigen_solver": s.eigen_solver, "hessian_tol": s.hessian_tol,
                                  "modified_tol": s.modified_tol, "random_state": seed}, s.extra, "lle"))

    def can_inverse(self):
        return False

    def info(self):
        # LLE's own embedding cost, not a reconstruction of the input
        return {"lle_cost": float(self.model.reconstruction_error_)}


@dataclasses.dataclass(frozen=True)
class SpectralEmbeddingSettings:
    n_components: int
    affinity: Literal["nearest_neighbors", "rbf"]   # cuML (the GPU): nearest_neighbors only
    n_neighbors: int | None         # nearest_neighbors: the graph's neighbours (None: rows / 10)
    gamma: float | None             # rbf: the kernel's coefficient (None: 1 / columns)
    eigen_solver: Literal["arpack", "lobpcg"]
    extra: dict[str, Scalar]


@register_model("spectral_embedding", settings=SpectralEmbeddingSettings, family="embedder", tasks=EMBED,
                kinds=TAB, gpu="cuml", profile="core", packages=("sklearn",),
                description="spectral embedding (Laplacian eigenmaps; cuML on the GPU; no transform)",
                example="embed_manifold")
class SpectralEmbedding(LibraryEmbedder):
    """cuML's SpectralEmbedding (API as documented for cuML 26.12: n_components, affinity, random_state,
    n_neighbors) takes the nearest_neighbors affinity only."""

    @staticmethod
    def check_job(job):
        s = job.model
        if s.affinity == "rbf" and s.n_neighbors is not None:
            raise ValueError("spectral_embedding rbf: n_neighbors is None (it is for nearest_neighbors)")
        if s.affinity == "nearest_neighbors" and s.gamma is not None:
            raise ValueError("spectral_embedding nearest_neighbors: gamma is None (it is for rbf)")
        if job.fit.device == "cuda" and s.affinity != "nearest_neighbors":
            raise ValueError("spectral_embedding on cuML: affinity nearest_neighbors only")

    def make(self, s, seed, device):
        if device == "cuda":
            if s.affinity != "nearest_neighbors":
                raise ValueError("spectral_embedding on cuML: affinity nearest_neighbors only")
            from cuml.manifold import SpectralEmbedding as S

            return S(**merged_kwargs({"n_components": s.n_components, "affinity": s.affinity,
                                      "n_neighbors": s.n_neighbors, "random_state": seed}, s.extra,
                                     "spectral_embedding"))
        from sklearn.manifold import SpectralEmbedding as S

        return S(**merged_kwargs({"n_components": s.n_components, "affinity": s.affinity, "gamma": s.gamma,
                                  "n_neighbors": s.n_neighbors, "eigen_solver": s.eigen_solver,
                                  "random_state": seed}, s.extra, "spectral_embedding"))

    def can_transform(self):
        return False

    def can_inverse(self):
        return False
