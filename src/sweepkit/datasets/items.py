"""Datasets whose rows are files or nodes: items (one file per row: sequences, images, audio, text), graph (a node
table and an edge table), and anndata (a single-cell .h5ad file; the singlecell profile)."""
import dataclasses
import wave
from typing import Literal

import numpy as np

from sweepkit import paths
from sweepkit.data.base import Data, Dataset, Graph, ItemSource, RowTable
from sweepkit.data.readers import read_frame, split_frame, text_column
from sweepkit.datasets.table import file_fingerprint
from sweepkit.registry import register_dataset


def load_item(path, loader: str):
    if loader == "npy":
        return np.load(path, allow_pickle=False).astype(np.float32)
    if loader == "csv":
        return np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if loader == "txt":
        return path.read_text(encoding="utf-8")
    with wave.open(str(path), "rb") as handle:
        width, channels = handle.getsampwidth(), handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is read (got {8 * width}-bit)")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


class FileItems(ItemSource):
    def __init__(self, files: list, loader: str, sample_rate: int | None):
        self.files = files
        self.loader = loader
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def get(self, position):
        item = load_item(self.files[position], self.loader)
        if self.loader == "wav":
            samples, rate = item
            if rate != self.sample_rate:
                raise ValueError(f"{self.files[position]}: {rate} Hz, the settings say {self.sample_rate}")
            return samples
        return item

    def length(self, position):
        item = self.get(position)
        return len(item) if isinstance(item, str) else int(np.asarray(item).shape[0])


@dataclasses.dataclass(frozen=True)
class ItemsSettings:
    index: str                      # a CSV under data/ with one row per item
    path_column: str                # its column holding each item's file path, relative to data/
    row_id: str                     # its column of unique row IDs
    keep: tuple[str, ...]           # its columns carried along (targets, groups)
    kind: Literal["sequence", "image", "audio", "text"]
    loader: Literal["npy", "csv", "wav", "txt"]   # npy and csv arrays, 16-bit wav audio, UTF-8 text
    sample_rate: int | None         # audio: every file's sample rate (checked); None otherwise


@register_dataset("items", settings=ItemsSettings, kinds=("sequence", "image", "audio", "text"), profile="core",
                  packages=("pandas",), description="one file per row (arrays, audio, text) listed in an index CSV",
                  example="data_items")
class Items(Dataset):
    @staticmethod
    def check_job(job) -> None:
        s = job.data
        paths.data_path(s.index)
        if (s.loader == "wav") != (s.sample_rate is not None):
            raise ValueError("data items: wav takes a sample_rate; other loaders take None")
        expected = {"npy": ("sequence", "image", "audio"), "csv": ("sequence",), "wav": ("audio",), "txt": ("text",)}
        if s.kind not in expected[s.loader]:
            raise ValueError(f"data items: loader {s.loader} gives {expected[s.loader]}, not {s.kind}")

    def load(self, s: ItemsSettings) -> Data:
        index_path = paths.data_path(s.index)
        frame = read_frame(index_path, "csv", None)
        frame.columns = [str(c) for c in frame.columns]
        for column in (s.path_column, s.row_id) + tuple(s.keep):
            if column not in frame.columns:
                raise ValueError(f"data items: {column!r} is not a column of {s.index}")
        files = [paths.data_path(p) for p in text_column(frame[s.path_column]).tolist()]
        absent = [str(f) for f in files if not f.is_file()]
        if absent:
            raise FileNotFoundError(f"data items: {len(absent)} files listed in {s.index} are missing, e.g. {absent[:2]}")
        table = RowTable(text_column(frame[s.row_id]), {c: text_column(frame[c]) for c in s.keep})
        return Data(s.kind, table, items=FileItems(files, s.loader, s.sample_rate),
                    fingerprint={"index": s.index, **file_fingerprint(index_path), "items": len(files)})


@dataclasses.dataclass(frozen=True)
class GraphSettings:
    nodes: str                      # a CSV under data/: one row per node (its ID, features and kept columns)
    edges: str                      # a CSV under data/ with a source and a target node ID per edge
    row_id: str                     # the node ID column of nodes
    features: tuple[str, ...]       # node feature columns (names, "re:<regex>", "*")
    keep: tuple[str, ...]           # node columns carried along (targets, groups)
    source: str                     # the edges' source column
    target: str                     # the edges' target column
    undirected: bool                # list every edge in both directions


@register_dataset("graph", settings=GraphSettings, kinds=("graph",), profile="core", packages=("pandas",),
                  description="a node table and an edge table (graph neural networks)", example="data_graph")
class GraphTable(Dataset):
    def load(self, s: GraphSettings) -> Data:
        nodes_path, edges_path = paths.data_path(s.nodes), paths.data_path(s.edges)
        parts = split_frame(read_frame(nodes_path, "csv", None), s.row_id, s.features, (), s.keep, "error", None, (),
                            f"data graph {s.nodes}")
        table = RowTable(parts["ids"], parts["kept"])
        edges = read_frame(edges_path, "csv", None)
        source = table.positions(text_column(edges[s.source]).tolist())
        target = table.positions(text_column(edges[s.target]).tolist())
        e = np.stack([source, target])
        if s.undirected:
            e = np.concatenate([e, e[::-1]], axis=1)
        return Data("graph", table, x=parts["x"], feature_names=parts["names"], graph=Graph(e, len(table)),
                    fingerprint={"nodes": file_fingerprint(nodes_path), "edges": file_fingerprint(edges_path)})


@dataclasses.dataclass(frozen=True)
class AnnDataSettings:
    path: str                       # an .h5ad file under data/
    layer: str | None               # None: X; else a layer name (for example "counts")
    keep: tuple[str, ...]           # columns of obs carried along (cell types, batches, samples)
    var_filter: str | None          # a boolean column of var choosing the genes (for example "highly_variable")


@register_dataset("anndata", settings=AnnDataSettings, kinds=("tabular",), profile="singlecell",
                  packages=("anndata",), description="a single-cell AnnData (.h5ad) file", example="data_anndata")
class AnnData(Dataset):
    def load(self, s: AnnDataSettings) -> Data:
        import anndata

        path = paths.data_path(s.path)
        adata = anndata.read_h5ad(path)
        if s.var_filter is not None:
            adata = adata[:, adata.var[s.var_filter].to_numpy(dtype=bool)]
        matrix = adata.X if s.layer is None else adata.layers[s.layer]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        table = RowTable(np.asarray(adata.obs_names, dtype=str),
                         {c: text_column(adata.obs[c]) for c in s.keep})
        return Data("tabular", table, x=np.asarray(matrix, dtype=np.float32),
                    feature_names=tuple(str(v) for v in adata.var_names), fingerprint={"path": s.path,
                                                                                         **file_fingerprint(path)})
