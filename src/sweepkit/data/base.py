"""What every dataset gives the rest of the package.

RowTable  the rows in a fixed order: one ID per row (text), and named per-row columns (targets, groups, strata,
          reference labels, metadata). Row IDs are kept from the first read to the last export.
Data      the rows plus their inputs, by kind:
            tabular   x, a 2-D float array [rows, features] (in memory or memory-mapped), with feature names; mixed
                      numeric and categorical columns keep the categorical ones as integer codes listed in
                      `categorical` (column index -> category names)
            graph     x holds the node features, `graph` the edges; one row per node
            sequence, image, audio, text   `items`, an ItemSource that loads one row's input at a time
          and a fingerprint: what the data version is (a file's SHA-256, a store's record hash, a generator's
          settings), recorded in every run's provenance.
Dataset   the interface of a registered dataset: load(settings) -> Data.
"""
import dataclasses

import numpy as np


class RowTable:
    """Row IDs and per-row columns, all the same length, in one fixed order."""

    def __init__(self, ids, columns: dict):
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError("row IDs must be one-dimensional")
        ids = ids.astype(str)
        if ids.size and np.unique(ids).size != ids.size:
            values, counts = np.unique(ids, return_counts=True)
            raise ValueError(f"row IDs repeat, for example {values[counts > 1][:3].tolist()}")
        self.ids = ids
        self.columns = {}
        for name, values in columns.items():
            values = np.asarray(values)
            if values.shape[0] != ids.shape[0]:
                raise ValueError(f"column {name!r} has {values.shape[0]} values for {ids.shape[0]} rows")
            self.columns[name] = values
        self._position = None

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def has(self, name: str) -> bool:
        return name in self.columns

    def column(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(f"no column {name!r}; the dataset gives {sorted(self.columns)}")
        return self.columns[name]

    def positions(self, ids) -> np.ndarray:
        """Row positions of the given IDs (each must exist)."""
        if self._position is None:
            self._position = {rid: i for i, rid in enumerate(self.ids.tolist())}
        try:
            return np.array([self._position[str(r)] for r in ids], dtype=np.int64)
        except KeyError as err:
            raise KeyError(f"row ID {err.args[0]!r} is not in the dataset") from None


class ItemSource:
    """Inputs read one row at a time (sequences, images, audio, text). Subclasses implement get and length."""

    def __len__(self) -> int:
        raise NotImplementedError

    def get(self, position: int):
        """The input of one row: a float array ([steps, channels], [channels, height, width] or [samples]) or a
        string for text."""
        raise NotImplementedError

    def length(self, position: int) -> int:
        """Steps (sequences), samples (audio), characters (text) or 1 (images)."""
        raise NotImplementedError


class ArrayItems(ItemSource):
    """Items held in memory: a list of arrays (or strings)."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def get(self, position: int):
        return self.items[position]

    def length(self, position: int) -> int:
        item = self.items[position]
        return len(item) if isinstance(item, str) else int(np.asarray(item).shape[0])


@dataclasses.dataclass
class Graph:
    """Edges between rows (nodes): edges[0] -> edges[1], int64 positions, both directions listed for an undirected
    graph."""

    edges: np.ndarray
    n_nodes: int

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int64)
        if self.edges.ndim != 2 or self.edges.shape[0] != 2:
            raise ValueError("edges must have shape [2, number of edges]")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= self.n_nodes):
            raise ValueError("an edge points outside the nodes")


@dataclasses.dataclass
class Data:
    kind: str
    table: RowTable
    x: object = None
    feature_names: tuple = ()
    items: object = None
    graph: object = None
    categorical: dict = dataclasses.field(default_factory=dict)
    fingerprint: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        n = len(self.table)
        if self.kind in ("tabular", "graph"):
            if self.x is None or getattr(self.x, "ndim", 0) != 2 or self.x.shape[0] != n:
                raise ValueError(f"{self.kind} data need x with one row per row ID ({n})")
            if len(self.feature_names) != self.x.shape[1]:
                raise ValueError(f"{len(self.feature_names)} feature names for {self.x.shape[1]} columns")
        else:
            if self.items is None or len(self.items) != n:
                raise ValueError(f"{self.kind} data need items, one per row ID ({n})")
        if self.kind == "graph" and (self.graph is None or self.graph.n_nodes != n):
            raise ValueError("graph data need a graph with one node per row")
        for index in self.categorical:
            if not 0 <= index < len(self.feature_names):
                raise ValueError(f"categorical column {index} is not a feature")


class Dataset:
    """Interface of a registered dataset. The registry entry's target is the class; it is constructed with no
    arguments and asked to load settings."""

    def load(self, settings) -> Data:
        raise NotImplementedError


def float_matrix(values, where: str) -> np.ndarray:
    """A 2-D float32 array, refusing anything that is not numeric."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{where}: expected a 2-D table of numbers, got shape {array.shape}")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == bool):
        raise ValueError(f"{where}: columns must be numbers, got {array.dtype}")
    return array.astype(np.float32, copy=False)
