"""Diffusion maps (Coifman and Lafon 2006), the template's own implementation.

A Gaussian kernel between the fit rows, either with one bandwidth for every pair (fixed: exp(-d^2 / bandwidth)) or
with each row's own scale, its distance to its n_neighbors-th nearest fit row counting itself (adaptive:
exp(-d^2 / (s_i s_j)), Zelnik-Manor and Perona 2004). The kernel is normalized by the rows' densities to the power
alpha (0: the graph Laplacian; 0.5: Fokker-Planck; 1: Laplace-Beltrami, which removes the effect of the sampling
density), then turned into a Markov matrix P. Its right eigenvectors after the constant one, each scaled by its
eigenvalue to the power t (the diffusion time), are the coordinates.

New rows are placed by the Nystrom extension: a new row's transition probabilities to the fit rows, times the fit
rows' eigenvectors, divided by the eigenvalue. A fit row passed through transform gets its own coordinates back.

The kernel holds every pair of fit rows, so memory grows with the square of the rows (10,000 rows: about 800 MB);
solver dense computes every eigenvector (rows^3 time, for up to a few thousand rows), solver arpack only the leading
ones (seeded start vector).
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Embedder, plain_params
from sweepkit.registry import register_model
from sweepkit.streams import generator


@dataclasses.dataclass(frozen=True)
class DiffusionMapSettings:
    n_components: int
    kernel: Literal["fixed", "adaptive"]
    bandwidth: float | None         # fixed: the kernel's bandwidth (in squared distance units); adaptive: None
    n_neighbors: int | None         # adaptive: the neighbour whose distance is each row's scale; fixed: None
    alpha: float                    # density normalization, from 0 to 1
    t: int                          # diffusion time: coordinates scaled by eigenvalue^t (0: the eigenvectors)
    solver: Literal["dense", "arpack"]


@register_model("diffusion_map", settings=DiffusionMapSettings, family="embedder", tasks=("embedding",),
                kinds=("tabular",), gpu="none", profile="core", packages=("scipy",),
                description="diffusion maps with a fixed or adaptive Gaussian kernel (own implementation; places new "
                            "rows by the Nystrom extension)", example="embed_diffusion")
class DiffusionMap(Embedder):
    @staticmethod
    def check_job(job):
        s = job.model
        if s.kernel == "fixed" and (s.bandwidth is None or s.n_neighbors is not None):
            raise ValueError("diffusion_map fixed: set bandwidth, and n_neighbors None")
        if s.kernel == "adaptive" and (s.n_neighbors is None or s.bandwidth is not None):
            raise ValueError("diffusion_map adaptive: set n_neighbors, and bandwidth None")
        if s.bandwidth is not None and not s.bandwidth > 0:
            raise ValueError("diffusion_map: bandwidth above 0")
        if s.n_neighbors is not None and s.n_neighbors < 2:
            raise ValueError("diffusion_map: n_neighbors at least 2 (a row counts itself as its first neighbour)")
        if not 0 <= s.alpha <= 1 or s.t < 0 or s.n_components < 1:
            raise ValueError("diffusion_map: alpha from 0 to 1, t at least 0, n_components at least 1")

    def _scales(self, squared: np.ndarray) -> np.ndarray:
        """Each row's adaptive scale: the distance to its n_neighbors-th nearest fit row, counting itself."""
        k = self.s.n_neighbors
        if k > squared.shape[1]:
            raise ValueError(f"diffusion_map: n_neighbors {k} is above the fit rows ({squared.shape[1]})")
        scale = np.sqrt(np.partition(squared, k - 1, axis=1)[:, k - 1])
        return np.maximum(scale, np.finfo(np.float64).tiny)

    def _kernel(self, squared: np.ndarray, row_scales, column_scales) -> np.ndarray:
        if self.s.kernel == "fixed":
            return np.exp(-squared / self.s.bandwidth)
        return np.exp(-squared / np.outer(row_scales, column_scales))

    def fit_transform(self, x):
        from scipy.spatial.distance import cdist

        x = np.asarray(x, dtype=np.float64)
        n, k = x.shape[0], self.s.n_components
        if k + 1 >= n:
            raise ValueError(f"diffusion_map: n_components {k} needs more than {k + 1} fit rows")
        squared = cdist(x, x, metric="sqeuclidean")
        scales = self._scales(squared) if self.s.kernel == "adaptive" else None
        kernel = self._kernel(squared, scales, scales)
        density = kernel.sum(axis=1)
        normalized = kernel / np.outer(density ** self.s.alpha, density ** self.s.alpha)
        degree = normalized.sum(axis=1)
        symmetric = normalized / np.sqrt(np.outer(degree, degree))
        if self.s.solver == "dense":
            from scipy.linalg import eigh

            values, vectors = eigh(symmetric)
        else:
            from scipy.sparse.linalg import eigsh

            start = generator(self.seed, "diffusion map start").normal(size=n)
            values, vectors = eigsh(symmetric, k=k + 1, which="LA", v0=start)
        order = np.argsort(values)[::-1][:k + 1]
        values, vectors = values[order], vectors[:, order]
        if not (values[1:] > 0).all():
            raise ValueError(f"diffusion_map: fewer than {k} positive eigenvalues after the first; use fewer "
                             "components or a wider kernel")
        right = vectors / np.sqrt(degree)[:, None]           # right eigenvectors of the Markov matrix
        right = right[:, 1:]                                 # the first is constant (eigenvalue 1)
        signs = np.sign(right[np.argmax(np.abs(right), axis=0), np.arange(k)])
        right = right * np.where(signs == 0, 1.0, signs)[None, :]
        self.eigenvalues = values[1:]
        self.leading = float(values[0])
        self.psi = right
        self.x_fit, self.scales, self.density = x, scales, density
        return (right * self.eigenvalues[None, :] ** self.s.t).astype(np.float32)

    def can_transform(self):
        return True

    def transform(self, x_new):
        from scipy.spatial.distance import cdist

        squared = cdist(np.asarray(x_new, dtype=np.float64), self.x_fit, metric="sqeuclidean")
        new_scales = self._scales(squared) if self.s.kernel == "adaptive" else None
        kernel = self._kernel(squared, new_scales, self.scales)
        density = kernel.sum(axis=1)
        if (density <= 0).any():
            raise ValueError("diffusion_map: a new row is too far from every fit row for the kernel (all weights 0)")
        normalized = kernel / np.outer(density ** self.s.alpha, self.density ** self.s.alpha)
        transition = normalized / normalized.sum(axis=1, keepdims=True)
        psi = transition @ self.psi / self.eigenvalues[None, :]
        return (psi * self.eigenvalues[None, :] ** self.s.t).astype(np.float32)

    def info(self):
        return {"eigenvalues": [float(v) for v in self.eigenvalues], "leading_eigenvalue": self.leading}

    def params(self):
        return plain_params(dataclasses.asdict(self.s))
