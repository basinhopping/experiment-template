"""The single-cell workflow's clustering step: a neighbour graph of the cells, then Leiden communities, with scanpy on
the CPU and rapids-singlecell on the GPU (profile singlecell).

The steps before it are ordinary preparation steps in the sweep, so each is recorded and can be varied: total-count
normalization and log1p (transform), highly variable genes (select_features), principal components (reduce with
pca). This step reads the prepared matrix as it is (use_rep "X"), builds scanpy's neighbour graph (UMAP
connectivities) and runs Leiden on it.

scanpy's Leiden runs in igraph (flavor igraph, the faster and the future default) or in leidenalg (flavor
leidenalg); igraph and leidenalg are GPL packages that scanpy's leiden extra installs. On the GPU, rapids-singlecell
builds the graph with cuVS and runs cuGraph's Leiden; flavor is then not used.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, merged_kwargs, plain_params
from sweepkit.registry import register_model
from sweepkit.settings import Scalar


@dataclasses.dataclass(frozen=True)
class ScanpyLeidenSettings:
    n_neighbors: int                # cells in each cell's neighbourhood (scanpy's usual 15)
    metric: Literal["euclidean", "cosine", "manhattan"]
    resolution: float               # larger gives more, smaller clusters
    flavor: Literal["igraph", "leidenalg"]   # the CPU implementation
    n_iterations: int               # Leiden iterations; -1 until no improvement (igraph's flavor suggests 2)
    extra: dict[str, Scalar]        # passed to the Leiden call (scanpy tl.leiden or rapids-singlecell tl.leiden)


@register_model("scanpy_leiden", settings=ScanpyLeidenSettings, family="clusterer", tasks=("clustering",),
                kinds=("tabular",), gpu="rapids_singlecell", profile="singlecell", packages=("scanpy", "anndata"),
                cpu_packages=("igraph",),
                description="single-cell Leiden clustering of a neighbour graph (scanpy, which needs igraph from the "
                            "extras profile on the CPU; rapids-singlecell on the GPU)", example="singlecell_workflow")
class ScanpyLeiden(Clusterer):
    def fit(self, x):
        import anndata

        adata = anndata.AnnData(X=np.asarray(x, dtype=np.float32))
        if self.device == "cuda":
            # rapids-singlecell 0.18.1: pp.neighbors(adata, n_neighbors, n_pcs, *, use_rep, rng, algorithm, metric,
            # ...); tl.leiden(adata, resolution, *, rng, theta, ..., n_iterations, use_weights, ...)
            import rapids_singlecell as rsc

            rsc.pp.neighbors(adata, n_neighbors=self.s.n_neighbors, use_rep="X", rng=self.seed, metric=self.s.metric)
            # rapids-singlecell hands n_iterations to cuGraph's Leiden, where it caps the levels the algorithm
            # aggregates (default 100), while scanpy and igraph mean full passes (default 2): with 2 the cells stayed
            # in many small groups on the node (ARI 0.053, 2026-10-28). So the GPU keeps cuGraph's own limit, and the
            # run records that n_iterations applied on the CPU only.
            self.call = merged_kwargs({"resolution": self.s.resolution, "rng": self.seed}, self.s.extra,
                                      "scanpy_leiden")
            rsc.tl.leiden(adata, **self.call)
            self.library = "rapids_singlecell"
            self.notes = [f"n_iterations {self.s.n_iterations} is scanpy's on the CPU; on the GPU cuGraph aggregates "
                          "levels up to its own limit"]
        else:
            # scanpy 1.12.4: pp.neighbors(adata, n_neighbors, n_pcs, *, use_rep, ..., metric, random_state, ...);
            # tl.leiden(adata, resolution, *, random_state, key_added, directed, use_weights, n_iterations, ...,
            # flavor); flavor igraph needs directed False
            import scanpy as sc

            sc.pp.neighbors(adata, n_neighbors=self.s.n_neighbors, use_rep="X", metric=self.s.metric,
                            random_state=self.seed)
            fixed = {"resolution": self.s.resolution, "random_state": self.seed, "n_iterations": self.s.n_iterations,
                     "flavor": self.s.flavor}
            if self.s.flavor == "igraph":
                fixed["directed"] = False
            self.call = merged_kwargs(fixed, self.s.extra, "scanpy_leiden")
            sc.tl.leiden(adata, **self.call)
            self.library = "scanpy"
        raw = np.asarray(adata.obs["leiden"].astype(str))
        _, labels = np.unique(raw, return_inverse=True)
        self._labels = labels.astype(np.int64)
        self.modularity = adata.uns.get("leiden", {}).get("modularity")

    def labels(self):
        return self._labels

    def fallbacks(self):
        return list(getattr(self, "notes", []))

    def info(self):
        return {"modularity": None if self.modularity is None else float(self.modularity)}

    def params(self):
        return plain_params({**{k: v for k, v in dataclasses.asdict(self.s).items() if k != "extra"},
                             **{f"leiden_{k}": v for k, v in self.call.items()}, "library": self.library})
