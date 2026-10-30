"""Graph clustering on a k-nearest-neighbour graph of the rows (sweepkit.models.cluster.knn_graph), and the methods
scikit-learn builds on similarities: Leiden, Louvain, Markov clustering, spectral clustering and affinity propagation.

Leiden and Louvain find communities in the neighbour graph, PhenoGraph style (shared-neighbour Jaccard weights, or
plain connectivity). On the CPU they run in networkx 3.7, which is pure Python and so suits up to some tens of
thousands of rows; Leiden can also use the leidenalg package (profile extras, GPL). On the GPU (fit.device cuda) the
graph comes from cuML's neighbour search and the communities from cuGraph, which optimizes modularity.

Markov clustering (MCL, van Dongen 2000) is the template's own, on SciPy sparse matrices: add self loops, make the
columns sum to 1, then repeat expansion (the matrix to a power), inflation (each entry to a power, columns rescaled)
and pruning (entries below a threshold set to 0) until the matrix stops changing. Each row is then assigned to the
attractor (a row that keeps mass on its own diagonal entry) holding most of its column.

Spectral clustering and affinity propagation are scikit-learn's. Both hold a rows by rows similarity matrix, so they
suit a few thousand rows.

None of these can label rows they were not fitted on, except affinity propagation (its predict); use fit.assign
fit_rows.
"""
import dataclasses
import inspect
from typing import Literal

import numpy as np

from sweepkit.models.base import Clusterer, LibraryClusterer, _host, merged_kwargs, plain_params
from sweepkit.models.cluster.knn_graph import adjacency, check_search, neighbours
from sweepkit.registry import register_model
from sweepkit.settings import Scalar

CLUSTER = ("clustering",)
TAB = ("tabular",)


def _labels_from_communities(communities, n: int) -> np.ndarray:
    """A label per node from a list of node sets, largest community first (ties by smallest member)."""
    ordered = sorted((sorted(c) for c in communities), key=lambda c: (-len(c), c[0]))
    labels = np.full(n, -1, dtype=np.int64)
    for k, members in enumerate(ordered):
        labels[np.asarray(members, dtype=np.int64)] = k
    if (labels < 0).any():
        raise RuntimeError("the community detection left some rows without a community")
    return labels


def _called(fn, args: dict) -> dict:
    """Every parameter of a library call, its defaults included, for model_params.json (keyword arguments the
    function collects in **kwargs are listed by their own names)."""
    signature = inspect.signature(fn)
    bound = signature.bind_partial(**args)
    bound.apply_defaults()
    out = {}
    for name, value in bound.arguments.items():
        if name in ("G", "graph"):
            continue
        if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        else:
            out[name] = value
    return out


class _GraphClusterer(Clusterer):
    """The neighbour graph shared by Leiden, Louvain and Markov clustering."""

    def graph(self, x):
        idx = neighbours(x, self.s.n_neighbors, self.s.metric, self.s.graph_search, self.device, self.seed)
        self.graph_ = adjacency(idx, self.s.graph_weights)
        return self.graph_

    def labels(self):
        return np.asarray(self._labels, dtype=np.int64)

    def info(self):
        out = {"graph_edges": int(self.graph_.nnz // 2)}
        out.update(getattr(self, "_info", {}))
        return out

    def params(self):
        return plain_params({**{k: v for k, v in dataclasses.asdict(self.s).items() if k != "extra"},
                             **{f"call_{k}": v for k, v in getattr(self, "_call", {}).items()}})


def _cugraph(adj):
    """A cuGraph graph from a SciPy adjacency (cuGraph 26.x: Graph().from_pandas_edgelist(pdf, source,
    destination, weight=..., renumber=...))."""
    import cugraph
    import pandas as pd

    coo = adj.tocoo()
    keep = coo.row < coo.col
    frame = pd.DataFrame({"src": coo.row[keep].astype(np.int32), "dst": coo.col[keep].astype(np.int32),
                          "weight": coo.data[keep].astype(np.float32)})
    g = cugraph.Graph()
    g.from_pandas_edgelist(frame, source="src", destination="dst", weight="weight", renumber=False)
    return g


def _from_parts(parts, n: int) -> np.ndarray:
    frame = parts.to_pandas() if hasattr(parts, "to_pandas") else parts
    labels = np.full(n, -1, dtype=np.int64)
    labels[frame["vertex"].to_numpy(dtype=np.int64)] = frame["partition"].to_numpy(dtype=np.int64)
    if (labels < 0).any():
        raise RuntimeError("cuGraph returned no community for some rows")
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def _check_graph(job) -> None:
    check_search(job)
    if job.model.n_neighbors < 1:
        raise ValueError(f"{job.model_name}: n_neighbors at least 1")


@dataclasses.dataclass(frozen=True)
class LeidenSettings:
    n_neighbors: int                # neighbours per row in the graph (PhenoGraph uses 30)
    metric: Literal["euclidean", "manhattan", "cosine"]
    graph_search: Literal["exact", "pynndescent", "faiss"]
    graph_weights: Literal["connectivity", "jaccard"]
    resolution: float               # larger gives more, smaller communities
    quality: Literal["modularity", "cpm"]   # what is optimized (cpm: the constant Potts model)
    backend: Literal["networkx", "leidenalg"]   # the CPU implementation; on the GPU, cuGraph (modularity)
    extra: dict[str, Scalar]        # passed to the backend's call: networkx leiden_communities (max_level, theta),
                                    # leidenalg find_partition (n_iterations), cuGraph leiden (max_iter, theta)


@register_model("leiden", settings=LeidenSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="cugraph",
                profile="core", packages=("networkx",),
                description="Leiden communities of a nearest-neighbour graph (networkx; leidenalg; cuGraph on the "
                            "GPU)", example="cluster_graph")
class Leiden(_GraphClusterer):
    @staticmethod
    def gpu_unsupported(s):
        """Why cuGraph cannot take these settings, or None (fit.device auto then runs them on the CPU)."""
        if s.quality != "modularity":
            return f"cuGraph's Leiden optimizes modularity, not {s.quality}"
        return None

    @staticmethod
    def check_job(job):
        _check_graph(job)
        reason = Leiden.gpu_unsupported(job.model)
        if job.fit.device == "cuda" and reason:
            raise ValueError(f"leiden: {reason}; use quality modularity, or fit.device cpu or auto")

    def fit(self, x):
        adj = self.graph(x)
        n = adj.shape[0]
        if self.device == "cuda":
            import cugraph

            args = merged_kwargs({"resolution": self.s.resolution, "random_state": self.seed}, self.s.extra,
                                 "leiden")
            parts, modularity = cugraph.leiden(_cugraph(adj), **args)
            self._labels = _from_parts(parts, n)
            self._info = {"modularity": float(modularity)}
            self._call = _called(cugraph.leiden, args)
        elif self.s.backend == "networkx":
            import networkx as nx

            graph = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
            args = merged_kwargs({"weight": "weight", "resolution": self.s.resolution, "metric": self.s.quality,
                                  "seed": self.seed}, self.s.extra, "leiden")
            communities = nx.community.leiden_communities(graph, **args)
            self._labels = _labels_from_communities(communities, n)
            self._call = _called(nx.community.leiden_communities, args)
        else:
            # leidenalg 0.12: find_partition(graph, partition_type, initial_membership, weights, n_iterations,
            # max_comm_size, seed, **kwargs) with resolution_parameter for RBConfiguration and CPM partitions
            import igraph
            import leidenalg

            coo = adj.tocoo()
            keep = coo.row < coo.col
            graph = igraph.Graph(n=n, edges=list(zip(coo.row[keep].tolist(), coo.col[keep].tolist())),
                                 directed=False, edge_attrs={"weight": coo.data[keep].tolist()})
            kind = (leidenalg.RBConfigurationVertexPartition if self.s.quality == "modularity"
                    else leidenalg.CPMVertexPartition)
            args = merged_kwargs({"weights": "weight", "resolution_parameter": self.s.resolution, "seed": self.seed},
                                 self.s.extra, "leiden")
            part = leidenalg.find_partition(graph, kind, **args)
            _, labels = np.unique(np.asarray(part.membership, dtype=np.int64), return_inverse=True)
            self._labels = labels.astype(np.int64)
            self._info = {"quality": float(part.quality())}
            self._call = {**_called(leidenalg.find_partition, {**args, "partition_type": kind.__name__}),
                          "partition_type": kind.__name__}


@dataclasses.dataclass(frozen=True)
class LouvainSettings:
    n_neighbors: int
    metric: Literal["euclidean", "manhattan", "cosine"]
    graph_search: Literal["exact", "pynndescent", "faiss"]
    graph_weights: Literal["connectivity", "jaccard"]
    resolution: float               # larger gives more, smaller communities
    threshold: float                # stop a level when modularity gains less than this
    extra: dict[str, Scalar]        # passed to networkx louvain_communities (max_level) or cuGraph louvain


@register_model("louvain", settings=LouvainSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="cugraph",
                profile="core", packages=("networkx",),
                description="Louvain communities of a nearest-neighbour graph (networkx; cuGraph on the GPU)",
                example="cluster_graph")
class Louvain(_GraphClusterer):
    check_job = staticmethod(_check_graph)

    def fit(self, x):
        adj = self.graph(x)
        n = adj.shape[0]
        if self.device == "cuda":
            import cugraph

            args = merged_kwargs({"resolution": self.s.resolution, "threshold": self.s.threshold}, self.s.extra,
                                 "louvain")
            parts, modularity = cugraph.louvain(_cugraph(adj), **args)
            self._labels = _from_parts(parts, n)
            self._info = {"modularity": float(modularity)}
            self._call = _called(cugraph.louvain, args)
            return
        import networkx as nx

        graph = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
        args = merged_kwargs({"weight": "weight", "resolution": self.s.resolution, "threshold": self.s.threshold,
                              "seed": self.seed}, self.s.extra, "louvain")
        communities = nx.community.louvain_communities(graph, **args)
        self._labels = _labels_from_communities(communities, n)
        self._info = {"modularity": float(nx.community.modularity(graph, communities, weight="weight",
                                                                  resolution=self.s.resolution))}
        self._call = _called(nx.community.louvain_communities, args)


def markov_clustering(adj, expansion: int, inflation: float, self_loops: float, prune: float, max_iter: int,
                      tol: float) -> dict:
    """MCL on a symmetric non-negative SciPy sparse matrix. Returns labels, the final matrix, the iterations run,
    whether it converged, and the largest deviation of a column sum from 1 seen after any iteration."""
    from scipy import sparse

    if expansion < 2 or not inflation > 1 or self_loops < 0 or prune < 0 or max_iter < 1 or not tol > 0:
        raise ValueError("markov: expansion at least 2, inflation above 1, self_loops and prune at least 0, "
                         "max_iter at least 1, tol above 0")
    n = adj.shape[0]

    def normalize(m):
        sums = np.asarray(m.sum(axis=0)).ravel()
        sums[sums == 0] = 1.0
        return (m @ sparse.diags(1.0 / sums)).tocsc()

    m = normalize(sparse.csc_matrix(adj, dtype=np.float64) + self_loops * sparse.identity(n, format="csc"))
    worst, converged, iteration = 0.0, False, 0
    for iteration in range(1, max_iter + 1):
        expanded = m
        for _ in range(expansion - 1):
            expanded = expanded @ m
        inflated = normalize(expanded.power(inflation))
        if prune > 0:
            # entries below prune go, except each column's largest, so no column is ever emptied
            column = np.repeat(np.arange(n), np.diff(inflated.indptr))
            largest = np.asarray(inflated.max(axis=0).todense()).ravel()
            inflated.data[(inflated.data < prune) & (inflated.data < largest[column])] = 0.0
            inflated.eliminate_zeros()
            inflated = normalize(inflated)
        sums = np.asarray(inflated.sum(axis=0)).ravel()
        worst = max(worst, float(np.abs(sums[sums > 0] - 1).max()))
        change = abs(inflated - m).max()
        m = inflated
        if change < tol:
            converged = True
            break
    m = m.tocsr()
    diagonal = m.diagonal()
    attractors = np.flatnonzero(diagonal > 0)
    if attractors.size == 0:
        raise RuntimeError("markov: no attractor found; raise max_iter or lower prune")
    owner = np.asarray(m[attractors].argmax(axis=0)).ravel()
    # attractors whose rows are identical form one cluster: number the clusters by their attractor rows' patterns
    patterns = {}
    cluster_of_attractor = np.empty(attractors.size, dtype=np.int64)
    for k, a in enumerate(attractors):
        key = tuple(np.sort(m[a].indices).tolist())
        cluster_of_attractor[k] = patterns.setdefault(key, len(patterns))
    labels = cluster_of_attractor[owner]
    _, labels = np.unique(labels, return_inverse=True)
    return {"labels": labels.astype(np.int64), "matrix": m, "iterations": iteration, "converged": converged,
            "column_sum_error": worst, "attractors": int(attractors.size)}


@dataclasses.dataclass(frozen=True)
class MarkovSettings:
    n_neighbors: int
    metric: Literal["euclidean", "manhattan", "cosine"]
    graph_search: Literal["exact", "pynndescent", "faiss"]
    graph_weights: Literal["connectivity", "jaccard"]
    expansion: int                  # the matrix power of each step (2 is usual)
    inflation: float                # the entry power of each step; larger gives more, smaller clusters (2 is usual)
    self_loops: float               # weight added on the diagonal before the first step (1 is usual)
    prune: float                    # entries below this are set to 0 after each step (0: none)
    max_iter: int
    tol: float                      # stop when no entry changes by more than this


@register_model("markov", settings=MarkovSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("scipy",),
                description="Markov clustering (MCL) of a nearest-neighbour graph (own implementation)",
                example="cluster_graph")
class Markov(_GraphClusterer):
    check_job = staticmethod(_check_graph)

    def fit(self, x):
        adj = self.graph(x)
        result = markov_clustering(adj, self.s.expansion, self.s.inflation, self.s.self_loops, self.s.prune,
                                   self.s.max_iter, self.s.tol)
        self._labels = result["labels"]
        self._info = {k: result[k] for k in ("iterations", "converged", "column_sum_error", "attractors")}


@dataclasses.dataclass(frozen=True)
class SpectralSettings:
    n_clusters: int
    affinity: Literal["rbf", "nearest_neighbors"]
    gamma: float                    # rbf only: exp(-gamma * squared distance)
    n_neighbors: int                # nearest_neighbors only
    assign_labels: Literal["kmeans", "discretize", "cluster_qr"]
    n_init: int                     # k-means restarts when assign_labels is kmeans
    extra: dict[str, Scalar]


@register_model("spectral", settings=SpectralSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="none",
                profile="core", packages=("sklearn",),
                description="spectral clustering (scikit-learn; a rows by rows affinity)", example="cluster_graph")
class Spectral(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.cluster import SpectralClustering

        return SpectralClustering(**merged_kwargs({
            "n_clusters": s.n_clusters, "affinity": s.affinity, "gamma": s.gamma, "n_neighbors": s.n_neighbors,
            "assign_labels": s.assign_labels, "n_init": s.n_init, "random_state": seed}, s.extra, "spectral"))

    def can_assign(self):
        return False


@dataclasses.dataclass(frozen=True)
class AffinityPropagationSettings:
    damping: float                  # between 0.5 and 1
    max_iter: int
    convergence_iter: int           # stop after this many iterations without a change of exemplars
    preference: float | None        # an exemplar's own similarity; None: the median similarity (scikit-learn)
    extra: dict[str, Scalar]


@register_model("affinity_propagation", settings=AffinityPropagationSettings, family="clusterer", tasks=CLUSTER,
                kinds=TAB, gpu="none", profile="core", packages=("sklearn",),
                description="affinity propagation (scikit-learn; finds the number of clusters itself)",
                example="cluster_graph")
class AffinityPropagation(LibraryClusterer):
    def make(self, s, seed, device):
        from sklearn.cluster import AffinityPropagation as A

        return A(**merged_kwargs({"damping": s.damping, "max_iter": s.max_iter, "convergence_iter": s.convergence_iter,
                                  "preference": s.preference, "affinity": "euclidean", "random_state": seed},
                                 s.extra, "affinity_propagation"))

    def labels(self):
        return _host(self.model.labels_).astype(np.int64)

    def info(self):
        return {"n_iter": int(self.model.n_iter_), "exemplars": int(len(self.model.cluster_centers_indices_))}
