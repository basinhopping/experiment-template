"""The k-nearest-neighbour graph that graph clustering (Leiden, Louvain, Markov clustering) runs on.

Neighbours (each row's n_neighbors nearest rows, itself left out) come from one of three searches:
  exact        scikit-learn's NearestNeighbors on the CPU; cuML's NearestNeighbors when the job runs on the GPU
  pynndescent  approximate search with pynndescent (profile unsupervised), for many rows; on the CPU always
  faiss        exact search with FAISS's flat index on the CPU (Euclidean, or cosine through rows scaled to unit
               length)
Edges join a row and each of its neighbours, both ways (the union), with a weight:
  connectivity  1 for every edge
  jaccard       the Jaccard overlap of the two rows' neighbour sets, each set holding the row itself and its
                neighbours, as PhenoGraph weights its graph
The graph is a symmetric SciPy sparse matrix with an empty diagonal.
"""
import numpy as np


def _drop_self(indices: np.ndarray, k: int) -> np.ndarray:
    """Each row's k neighbours without the row itself (a search for k + 1 neighbours returns the row too, usually
    first; with repeated rows it may come later or not at all)."""
    n = indices.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row = indices[i][indices[i] != i]
        if row.size < k:
            raise ValueError("the neighbour search returned too few neighbours; lower n_neighbors")
        out[i] = row[:k]
    return out


def neighbours(x: np.ndarray, n_neighbors: int, metric: str, search: str, device: str, seed: int) -> np.ndarray:
    """[rows, n_neighbors] indices of each row's nearest other rows."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[0]
    if not 1 <= n_neighbors < n:
        raise ValueError(f"n_neighbors {n_neighbors} must be between 1 and the rows minus 1 ({n - 1})")
    k = n_neighbors + 1
    if search == "exact" and device == "cuda":
        # cuML 26.x: NearestNeighbors(*, n_neighbors, algorithm, metric, ...); kneighbors returns distances, indices
        from cuml.neighbors import NearestNeighbors

        from sweepkit.models.base import _host

        _, idx = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(x).kneighbors(x)
        return _drop_self(_host(idx).astype(np.int64), n_neighbors)
    if search == "exact":
        from sklearn.neighbors import NearestNeighbors

        idx = NearestNeighbors(n_neighbors=k, metric=metric).fit(x).kneighbors(x, return_distance=False)
        return _drop_self(np.asarray(idx, dtype=np.int64), n_neighbors)
    if search == "pynndescent":
        # pynndescent 0.6: NNDescent(data, metric, n_neighbors, ..., random_state); neighbor_graph is
        # (indices, distances), each row listing itself first, padded with -1 where fewer were found
        from pynndescent import NNDescent

        idx, _ = NNDescent(x, metric=metric, n_neighbors=k, random_state=seed).neighbor_graph
        idx = np.asarray(idx, dtype=np.int64)
        if (idx < 0).any():
            raise ValueError("pynndescent found fewer neighbours than asked for some rows; lower n_neighbors")
        return _drop_self(idx, n_neighbors)
    if search == "faiss":
        # FAISS: IndexFlatL2(d); add(xb); D, I = search(xq, k) (squared distances, then indices)
        import faiss

        if metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = np.ascontiguousarray(x / np.maximum(norms, np.finfo(np.float32).tiny), dtype=np.float32)
        elif metric != "euclidean":
            raise ValueError(f"the faiss search takes euclidean or cosine, not {metric}")
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        _, idx = index.search(x, k)
        return _drop_self(np.asarray(idx, dtype=np.int64), n_neighbors)
    raise ValueError(f"unknown neighbour search {search!r}")


def adjacency(indices: np.ndarray, weights: str):
    """The symmetric weighted graph of a neighbour list, as a SciPy CSR matrix with an empty diagonal."""
    from scipy import sparse

    n, k = indices.shape
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    union = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    union = ((union + union.T) > 0).astype(np.float64).tocsr()
    union.setdiag(0)
    union.eliminate_zeros()
    if weights == "connectivity":
        return union
    if weights == "jaccard":
        member = sparse.csr_matrix((np.ones(rows.size + n), (np.concatenate([rows, np.arange(n)]),
                                                              np.concatenate([cols, np.arange(n)]))), shape=(n, n))
        member.sum_duplicates()
        member.data[:] = 1.0
        shared = (member @ member.T).tocsr()
        coo = union.tocoo()
        common = np.asarray(shared[coo.row, coo.col]).ravel()
        size = k + 1
        values = common / (2 * size - common)
        graph = sparse.csr_matrix((values, (coo.row, coo.col)), shape=(n, n))
        graph.eliminate_zeros()
        return graph
    raise ValueError(f"unknown edge weights {weights!r}")


def check_search(job) -> None:
    """Cross-checks of the graph settings a sweep can make before anything runs."""
    s = job.model
    if s.graph_search == "faiss" and s.metric not in ("euclidean", "cosine"):
        raise ValueError(f"{job.model_name}: the faiss search takes metric euclidean or cosine")
