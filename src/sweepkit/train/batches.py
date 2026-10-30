"""Batches for the torch family, in numpy: what one training step or one evaluation pass reads.

A Source is one role's rows (training, validation, test, or an external dataset) of one kind:
    tabular   the role's prepared rows [rows, columns] (after the job's preparation steps)
    sequence  one float32 array [steps, channels] per row, steps varying
    image     one float32 array [channels, height, width] per row, all the same shape
    audio     one float32 waveform [samples] per row, samples varying (16-bit integer audio is scaled to [-1, 1))
    text      one string per row
    graph     the role's nodes, with the whole graph (GraphInput: every node's features after the job's preparation
              steps, and the edges), which every role's source shares

Batch layouts (what the feature and the model receive, after torch.from_numpy):
    tabular   x [batch, columns]; lengths None
    sequence  x [batch, channels, steps], zero-padded after each row's own steps; lengths [batch]
    image     x [batch, channels, height, width]; lengths None
    audio     x [batch, samples], zero-padded; lengths [batch]
    text      x a list of the batch rows' strings (the model tokenizes them); lengths None
    graph     x a GraphBatch: the whole graph and the batch's nodes, whose outputs the model returns in that order (a
              graph model runs every node and reads these); lengths None. Evaluation is one batch of every node of
              the source, since each batch runs the whole graph anyway.

Training batches of sequences and audio are cropped to `crop` steps or samples at a random start (one draw per row
longer than the crop, from the job's batch stream, and nothing else), or, with crop None, padded to the longest row of
the batch with no draw at all. Evaluation sees every row whole (longest first, padded to the longest of each batch), or
cut into windows whose outputs are averaged per row.
"""
import numpy as np


class ItemCache:
    """A dataset's items (sweepkit.data.base.ItemSource) as float32 arrays, each read on first use and kept, so files
    are read once per process however many epochs and jobs use them. Indexed by dataset position. Every sequence must
    have the same channels and every image the same shape; the first item read sets them."""

    def __init__(self, items, kind: str):
        if kind not in ("sequence", "image", "audio"):
            raise ValueError(f"the torch family reads sequence, image and audio items, not {kind}")
        self.items, self.kind = items, kind
        self.arrays = [None] * len(items)
        self.shape = None

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, position: int) -> np.ndarray:
        array = self.arrays[position]
        if array is None:
            array = as_float_item(self.items.get(int(position)), self.kind)
            shape = {"sequence": array.shape[1:], "image": array.shape, "audio": ()}[self.kind]
            if self.shape is None:
                self.shape = shape
            elif shape != self.shape:
                raise ValueError(f"item {position} has shape {array.shape}; the first item read set "
                                 f"{'channels' if self.kind == 'sequence' else 'the shape'} {self.shape}")
            self.arrays[position] = array
        return array

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays if a is not None))


class TextCache:
    """A dataset's text items (sweepkit.data.base.ItemSource), each read on first use and kept, indexed by dataset
    position."""

    def __init__(self, items):
        self.items = items
        self.texts = [None] * len(items)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, position: int) -> str:
        text = self.texts[position]
        if text is None:
            text = self.items.get(int(position))
            if not isinstance(text, str):
                raise ValueError(f"item {position} is not text ({type(text).__name__})")
            self.texts[position] = text
        return text

    def nbytes(self) -> int:
        return int(sum(len(t.encode("utf-8")) for t in self.texts if t is not None))


class GraphInput:
    """A graph as the models read it: every node's features (float32 [nodes, features], after the job's preparation
    steps) and the edges (int64 [2, edges], source then target), each listed once per direction, without repeats and
    without self loops (a model adds self loops when its settings ask). Tensors on a device are made once and kept."""

    def __init__(self, x, edges):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        e = np.asarray(edges, dtype=np.int64)
        if e.ndim != 2 or e.shape[0] != 2:
            raise ValueError("edges must have shape [2, number of edges]")
        if e.size:
            pairs = np.unique(e.T, axis=0)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            e = np.ascontiguousarray(pairs.T)
        self.edges = e.reshape(2, -1)
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= self.x.shape[0]):
            raise ValueError("an edge points outside the nodes")
        self._tensors = {}

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    def tensors(self, device) -> tuple:
        """(features, edges) as tensors on the device."""
        import torch

        key = str(device)
        if key not in self._tensors:
            self._tensors[key] = (torch.from_numpy(self.x).to(device), torch.from_numpy(self.edges).to(device))
        return self._tensors[key]


class GraphBatch:
    """What a graph model reads for one batch: the whole graph and the nodes (dataset positions) whose outputs it
    returns, in that order."""

    def __init__(self, graph: GraphInput, nodes):
        self.graph = graph
        self.nodes = np.asarray(nodes, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.nodes.shape[0])


class Source:
    """One role's rows. positions are dataset positions, in the role's order; k below is a place in that order."""

    def __init__(self, kind: str, positions: np.ndarray, table_x=None, items=None, graph=None):
        if kind == "tabular" and (table_x is None or table_x.shape[0] != len(positions)):
            raise ValueError("a tabular source needs one prepared row per position")
        if kind == "graph" and graph is None:
            raise ValueError("a graph source needs the graph (GraphInput)")
        if kind not in ("tabular", "graph") and items is None:
            raise ValueError(f"a {kind} source needs the dataset's items")
        self.kind = kind
        self.positions = np.asarray(positions, dtype=np.int64)
        self.table_x = table_x
        self.items = items
        self.graph = graph

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def item(self, k: int):
        if self.kind == "tabular":
            return self.table_x[k]
        if self.kind == "graph":
            return self.graph.x[int(self.positions[k])]
        return self.items[int(self.positions[k])]

    def length(self, k: int) -> int:
        """Steps (sequence) or samples (audio); 1 for tabular rows and images."""
        if self.kind in ("sequence", "audio"):
            return int(self.item(k).shape[0])
        return 1

    def subset(self, ks) -> "Source":
        ks = np.asarray(ks, dtype=np.int64)
        return Source(self.kind, self.positions[ks], self.table_x[ks] if self.table_x is not None else None,
                      self.items, self.graph)


def concat_sources(first: Source, second: Source) -> Source:
    """The rows of two sources of one dataset, the first's then the second's (the training-plus-validation refit)."""
    if first.kind != second.kind or first.items is not second.items or first.graph is not second.graph:
        raise ValueError("only sources of the same dataset can be joined")
    positions = np.concatenate([first.positions, second.positions])
    if first.kind == "tabular":
        return Source("tabular", positions, table_x=np.concatenate([first.table_x, second.table_x]))
    if first.kind == "graph":
        return Source("graph", positions, graph=first.graph)
    return Source(first.kind, positions, items=first.items)


def as_float_item(value, kind: str) -> np.ndarray:
    """One item as the batches read it: float32, 16-bit integer audio scaled to [-1, 1)."""
    array = np.asarray(value)
    if kind == "audio" and np.issubdtype(array.dtype, np.integer):
        return (array.astype(np.float32) / 32768.0)
    array = array.astype(np.float32, copy=False)
    expected = {"sequence": 2, "image": 3, "audio": 1}[kind]
    if array.ndim != expected:
        raise ValueError(f"a {kind} item must have {expected} axes, got shape {array.shape}")
    return array


def train_order(labels, class_balance: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """Places within the training rows for one epoch: a permutation, or, with the balanced sampler (classification),
    n draws with replacement, each class equally likely."""
    if class_balance == "balanced_sampler":
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        return rng.choice(n, size=n, replace=True, p=weights / weights.sum())
    return rng.permutation(n)


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """n / (classes x count) per class, from the training labels."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"a class has no training rows: counts {counts.tolist()}")
    return (labels.shape[0] / (n_classes * counts)).astype(np.float32)


def positive_weights(targets: np.ndarray) -> np.ndarray:
    """Multi-label: negatives / positives per label, from the training rows (the loss's pos_weight)."""
    positives = targets.sum(axis=0).astype(np.float64)
    if (positives == 0).any() or (positives == targets.shape[0]).any():
        raise ValueError("a label is never on, or always on, in the training rows")
    return ((targets.shape[0] - positives) / positives).astype(np.float32)


def crop_arrays(arrays: list, crop: int | None, rng: np.random.Generator) -> tuple:
    """Time-first arrays ([steps, ...] or [samples]) cropped to `crop` at a random start (one draw per array longer
    than the crop, in order, and nothing else) and zero-padded when shorter; with crop None, padded to the longest
    with no draw. Returns (float32 [batch, length, ...], real lengths [batch])."""
    longest = max(a.shape[0] for a in arrays)
    length = longest if crop is None else crop
    out = np.zeros((len(arrays), length) + tuple(arrays[0].shape[1:]), dtype=np.float32)
    valid = np.empty(len(arrays), dtype=np.int64)
    for j, a in enumerate(arrays):
        if crop is not None and a.shape[0] > crop:
            start = int(rng.integers(0, a.shape[0] - crop + 1))
            out[j] = a[start:start + crop]
            valid[j] = crop
        else:
            out[j, :a.shape[0]] = a
            valid[j] = a.shape[0]
    return out, valid


def _layout(kind: str, x: np.ndarray) -> np.ndarray:
    """Sequences go from [batch, steps, channels] to the models' [batch, channels, steps]."""
    if kind == "sequence":
        return np.ascontiguousarray(x.transpose(0, 2, 1))
    return x


def training_batch(source: Source, picked: np.ndarray, crop: int | None, rng: np.random.Generator,
                   speed=None, speed_rng=None) -> tuple:
    """(x, lengths) of one training batch. speed: the "speed" augmentation's settings for audio, or None; it draws
    from speed_rng only (sweepkit.train.augment.draw_speeds), so the crop stream is the same with or without it."""
    if source.kind == "tabular":
        return np.ascontiguousarray(source.table_x[picked], dtype=np.float32), None
    if source.kind == "image":
        return np.stack([source.item(int(k)) for k in picked]).astype(np.float32, copy=False), None
    if source.kind == "text":
        return [source.item(int(k)) for k in picked], None
    if source.kind == "graph":
        return GraphBatch(source.graph, source.positions[np.asarray(picked, dtype=np.int64)]), None
    arrays = [source.item(int(k)) for k in picked]
    if speed is not None:
        from sweepkit.train.augment import at_speed, draw_speeds

        speeds = draw_speeds(speed.factors, len(arrays), speed_rng)
        arrays = [at_speed(source.items, int(source.positions[k]), float(f)) for k, f in zip(picked, speeds)]
    x, lengths = crop_arrays(arrays, crop, rng)
    return _layout(source.kind, x), lengths


def eval_batches(source: Source, batch_size: int):
    """Every row whole. Yields (places within the source, x, lengths). Sequences and audio go longest first, padded to
    the longest of each batch; tabular rows, images and text go in order; a graph's nodes in one batch."""
    n = len(source)
    if source.kind == "graph":
        if n:
            ks = np.arange(n, dtype=np.int64)
            yield ks, GraphBatch(source.graph, source.positions), None
        return
    if source.kind == "text":
        for start in range(0, n, batch_size):
            ks = np.arange(start, min(n, start + batch_size), dtype=np.int64)
            yield ks, [source.item(int(k)) for k in ks], None
        return
    if source.kind in ("tabular", "image"):
        for start in range(0, n, batch_size):
            ks = np.arange(start, min(n, start + batch_size), dtype=np.int64)
            if source.kind == "tabular":
                yield ks, np.ascontiguousarray(source.table_x[ks], dtype=np.float32), None
            else:
                yield ks, np.stack([source.item(int(k)) for k in ks]).astype(np.float32, copy=False), None
        return
    order = sorted(range(n), key=lambda k: -source.length(k))
    for start in range(0, n, batch_size):
        ks = np.array(order[start:start + batch_size], dtype=np.int64)
        x, lengths = crop_arrays([source.item(int(k)) for k in ks], None, None)
        yield ks, _layout(source.kind, x), lengths


def window_starts(n: int, window: int, hop: int) -> list:
    """Starts of the windows of one row: every hop while a whole window fits, plus one window ending at the last
    step if the regular ones stop short of it. A row of at most `window` steps is a single window starting at 0."""
    if window < 1 or hop < 1:
        raise ValueError("window and hop must be at least one step")
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, hop))
    if starts[-1] + window < n:
        starts.append(n - window)
    return starts


def window_batches(source: Source, batch_size: int, window: int, hop: int):
    """Every window of every row (window_starts), longest first, batch_size windows per batch, padded to the longest
    in the batch. Yields (the place within the source of each window's row, x, lengths)."""
    entries = []
    for k in range(len(source)):
        n = source.length(k)
        for start in window_starts(n, window, hop):
            entries.append((k, start, min(window, n)))
    entries.sort(key=lambda e: -e[2])
    for first in range(0, len(entries), batch_size):
        chunk = entries[first:first + batch_size]
        ks = np.array([k for k, _, _ in chunk], dtype=np.int64)
        pieces = [source.item(k)[start:start + length] for k, start, length in chunk]
        x, lengths = crop_arrays(pieces, None, None)
        yield ks, _layout(source.kind, x), lengths
