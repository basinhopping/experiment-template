"""Synthetic data with known structure, made by code: the template's examples, smoke tests and tests run on these, so
the template ships no data. Every generator is deterministic given its settings (named random streams from the
data's seed), so the same settings give the same rows on every machine.

Each generator returns (x or items, columns, extras):
  x        float32 [rows, features] for tabular kinds
  columns  per-row columns, as text where they are labels: "truth" (the planted cluster or class), "target"
           (regression), "label_<j>" (multi-label), "is_outlier" (0/1), "batch", "reference" (a continuous truth such
           as a manifold coordinate), "group" (five groups g0 to g4, for group splits), "published_split" (train,
           val or test, for the split "fixed")
  extras   graph edges, categorical columns, item lists

n_groups means clusters, classes, topics or cell types, as the generator documents; noise is its spread.
"""
import numpy as np

from sweepkit.streams import generator as stream

TABULAR_GENERATORS = ("blobs", "anisotropic", "varied", "densities", "moons", "circles", "classification", "xor",
                      "regression_linear", "regression_nonlinear", "multilabel", "outliers", "swiss_roll", "circle",
                      "low_rank", "sources", "counts", "mixed", "series", "biclusters", "topics", "y_shape", "batches",
                      "planted_graph_points")
ITEM_GENERATORS = {"sequences": "sequence", "images": "image", "audio": "audio", "text": "text"}
GRAPH_GENERATORS = ("graph",)
GENERATORS = TABULAR_GENERATORS + tuple(ITEM_GENERATORS) + GRAPH_GENERATORS


def audio_sample_rate() -> int:
    """The sample rate of the "audio" generator's waveforms (the log-mel feature's sample_rate must match it)."""
    return 16000


def item_shape(name: str, n_features: int) -> tuple | None:
    """One row's input shape for the item generators, as the torch family's models see it: sequences (channels,)
    with varying steps, images (1, side, side); None for audio (a waveform needs a feature first) and text."""
    if name == "sequences":
        return (n_features,)
    if name == "images":
        side = max(8, n_features)
        return (1, side, side)
    return None


def kind_of(name: str) -> str:
    if name in ITEM_GENERATORS:
        return ITEM_GENERATORS[name]
    if name in GRAPH_GENERATORS:
        return "graph"
    return "tabular"


def _groups(n: int, k: int) -> np.ndarray:
    """Balanced group numbers 0..k-1 for n rows, in a fixed order."""
    return np.arange(n) % k


def _pad(x: np.ndarray, n_features: int, rng, noise: float) -> np.ndarray:
    """Columns beyond the structure's own are noise, so every generator can give any width."""
    if x.shape[1] >= n_features:
        return x[:, :n_features]
    extra = rng.normal(0.0, max(noise, 1e-3), size=(x.shape[0], n_features - x.shape[1]))
    return np.hstack([x, extra])


def _shuffle(rng, n: int) -> np.ndarray:
    return rng.permutation(n)


def make(name: str, n_rows: int, n_features: int, n_groups: int, noise: float, seed: int) -> dict:
    if name not in GENERATORS:
        raise ValueError(f"no synthetic generator {name!r}; choose one of {GENERATORS}")
    if n_rows < 4 or n_features < 1:
        raise ValueError("synthetic data need at least 4 rows and 1 feature")
    rng = stream(seed, f"synthetic {name}")
    out = {"x": None, "columns": {}, "categorical": {}, "items": None, "edges": None, "names": None}
    k = max(1, n_groups)
    if name in ("blobs", "anisotropic", "varied", "batches"):
        truth = _groups(n_rows, k)
        centers = rng.uniform(-10, 10, size=(k, n_features))
        spread = np.full(k, noise) if name != "varied" else noise * np.linspace(0.5, 2.5, k)
        x = centers[truth] + rng.normal(size=(n_rows, n_features)) * spread[truth, None]
        if name == "anisotropic":
            transform = rng.normal(size=(n_features, n_features))
            x = x @ transform / np.sqrt(n_features)
        if name == "batches":
            batch = rng.integers(0, 3, size=n_rows)
            shifts = rng.normal(0, 3.0, size=(3, n_features))
            x = x + shifts[batch]
            out["columns"]["batch"] = np.array([f"b{b}" for b in batch])
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
        out["x"] = x
    elif name == "densities":
        truth = _groups(n_rows, k)
        centers = rng.uniform(-20, 20, size=(k, n_features))
        spreads = noise * (1.0 + 2.0 * np.arange(k) / max(1, k - 1))
        x = centers[truth] + rng.normal(size=(n_rows, n_features)) * spreads[truth, None]
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
        out["x"] = x
    elif name in ("moons", "circles"):
        half = n_rows // 2
        sizes = (half, n_rows - half)
        if name == "moons":
            a = np.linspace(0, np.pi, sizes[0])
            b = np.linspace(0, np.pi, sizes[1])
            outer = np.stack([np.cos(a), np.sin(a)], axis=1)
            inner = np.stack([1 - np.cos(b), 1 - np.sin(b) - 0.5], axis=1)
        else:
            a = np.linspace(0, 2 * np.pi, sizes[0], endpoint=False)
            b = np.linspace(0, 2 * np.pi, sizes[1], endpoint=False)
            outer = np.stack([np.cos(a), np.sin(a)], axis=1)
            inner = 0.4 * np.stack([np.cos(b), np.sin(b)], axis=1)
        x = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n_rows, 2))
        truth = np.array([0] * sizes[0] + [1] * sizes[1])
        order = _shuffle(rng, n_rows)
        out["x"] = _pad(x[order], n_features, rng, noise)
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth[order]])
    elif name == "classification":
        truth = _groups(n_rows, k)
        informative = min(n_features, max(2, k))
        centers = rng.normal(0, 2.0, size=(k, informative))
        x = centers[truth] + rng.normal(0, noise, size=(n_rows, informative))
        out["x"] = _pad(x, n_features, rng, 1.0)
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "xor":
        x = rng.uniform(-1, 1, size=(n_rows, 2))
        truth = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
        out["x"] = _pad(x + rng.normal(0, noise, size=x.shape), n_features, rng, 0.5)
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "regression_linear":
        x = rng.normal(size=(n_rows, n_features))
        coef = np.zeros(n_features)
        informative = max(1, n_features // 2)
        coef[:informative] = rng.uniform(1, 3, size=informative) * rng.choice([-1, 1], size=informative)
        out["x"] = x
        out["columns"]["target"] = x @ coef + rng.normal(0, noise, size=n_rows)
        out["columns"]["coefficients"] = np.tile(",".join(f"{c:.6f}" for c in coef), n_rows)
    elif name == "regression_nonlinear":
        x = rng.uniform(0, 1, size=(n_rows, max(5, n_features)))
        y = (10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4])
        out["x"] = x[:, :n_features] if n_features >= 5 else x
        out["columns"]["target"] = y + rng.normal(0, noise, size=n_rows)
    elif name == "multilabel":
        n_labels = max(2, k)
        x = rng.normal(size=(n_rows, n_features))
        weights = rng.normal(size=(n_features, n_labels))
        logits = x @ weights / np.sqrt(n_features) + rng.normal(0, noise, size=(n_rows, n_labels))
        labels = (logits > 0).astype(int)
        out["x"] = x
        for j in range(n_labels):
            out["columns"][f"label_{j}"] = labels[:, j].astype(str)
    elif name == "outliers":
        n_out = max(2, n_rows // 20)
        n_in = n_rows - n_out
        truth = _groups(n_in, k)
        centers = rng.uniform(-5, 5, size=(k, n_features))
        inliers = centers[truth] + rng.normal(0, noise, size=(n_in, n_features))
        low, high = inliers.min(axis=0) - 4, inliers.max(axis=0) + 4
        outliers = rng.uniform(low, high, size=(n_out, n_features))
        far = np.linalg.norm(outliers[:, None, :] - centers[None, :, :], axis=2).min(axis=1) > 3 * noise
        while not far.all():
            outliers[~far] = rng.uniform(low, high, size=(int((~far).sum()), n_features))
            far = np.linalg.norm(outliers[:, None, :] - centers[None, :, :], axis=2).min(axis=1) > 3 * noise
        x = np.vstack([inliers, outliers])
        flag = np.array([0] * n_in + [1] * n_out)
        order = _shuffle(rng, n_rows)
        out["x"] = x[order]
        out["columns"]["is_outlier"] = flag[order].astype(str)
        out["columns"]["truth"] = np.concatenate([np.array([f"c{t}" for t in truth]), np.array(["out"] * n_out)])[order]
    elif name == "swiss_roll":
        t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_rows))
        height = 21 * rng.uniform(size=n_rows)
        x = np.stack([t * np.cos(t), height, t * np.sin(t)], axis=1) + rng.normal(0, noise, size=(n_rows, 3))
        out["x"] = _pad(x, n_features, rng, noise)
        out["columns"]["reference"] = t.astype(str)
        out["columns"]["truth"] = np.array([f"c{int(v)}" for v in np.digitize(t, np.quantile(t, [0.25, 0.5, 0.75]))])
    elif name == "circle":
        angle = 2 * np.pi * rng.uniform(size=n_rows)
        x = np.stack([np.cos(angle), np.sin(angle)], axis=1) + rng.normal(0, noise, size=(n_rows, 2))
        out["x"] = _pad(x, n_features, rng, noise)
        out["columns"]["reference"] = angle.astype(str)
    elif name == "low_rank":
        rank = max(1, k)
        basis = rng.normal(size=(rank, n_features))
        scores = rng.normal(size=(n_rows, rank)) * np.linspace(3, 1, rank)
        out["x"] = scores @ basis + rng.normal(0, noise, size=(n_rows, n_features))
        out["columns"]["rank"] = np.full(n_rows, str(rank))
    elif name == "sources":
        n_src = max(2, k)
        t = np.linspace(0, 80, n_rows)      # many periods, so the sources are close to independent (ICA's premise)
        bank = [np.sin(2 * t), np.sign(np.sin(3 * t)), ((t * 1.3) % 2) - 1, np.sin(5 * t + 1) ** 3]
        sources = np.stack([bank[j % len(bank)] * (1 + 0.1 * (j // len(bank))) for j in range(n_src)], axis=1)
        sources += rng.normal(0, noise, size=sources.shape)
        mixing = rng.uniform(0.5, 2, size=(n_src, n_features))
        out["x"] = sources @ mixing
        for j in range(n_src):
            out["columns"][f"source_{j}"] = sources[:, j].astype(str)
    elif name == "counts":
        truth = _groups(n_rows, k)
        base = rng.gamma(2.0, 1.0, size=n_features)
        markers = np.ones((k, n_features))
        per_type = max(1, n_features // (2 * k))
        for c in range(k):
            markers[c, c * per_type:(c + 1) * per_type] = 8.0
        depth = rng.lognormal(mean=np.log(2000), sigma=0.3, size=n_rows)
        rates = base[None, :] * markers[truth] * depth[:, None] / (base.sum() * 1.5)
        dispersion = max(noise, 1e-3)
        counts = rng.negative_binomial(1.0 / dispersion, 1.0 / (1.0 + dispersion * rates))
        out["x"] = counts.astype(np.float32)
        out["columns"]["truth"] = np.array([f"type{t}" for t in truth])
        out["names"] = [f"gene_{j}" for j in range(n_features)]
    elif name == "mixed":
        truth = _groups(n_rows, k)
        n_num = max(1, n_features // 2)
        n_cat = max(1, n_features - n_num)
        centers = rng.uniform(-5, 5, size=(k, n_num))
        numeric = centers[truth] + rng.normal(0, noise, size=(n_rows, n_num))
        cats = []
        for j in range(n_cat):
            modes = rng.integers(0, 4, size=k)
            column = np.where(rng.uniform(size=n_rows) < 0.85, modes[truth], rng.integers(0, 4, size=n_rows))
            cats.append(column)
        out["x"] = np.hstack([numeric, np.stack(cats, axis=1)]).astype(np.float32)
        out["categorical"] = {n_num + j: ["a", "b", "c", "d"] for j in range(n_cat)}
        out["names"] = [f"num_{j}" for j in range(n_num)] + [f"cat_{j}" for j in range(n_cat)]
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "series":
        truth = _groups(n_rows, k)
        length = max(8, n_features)
        t = np.linspace(0, 4 * np.pi, length)
        shapes = [np.sin(t), np.sign(np.sin(t)), ((t / np.pi) % 2) - 1, np.sin(t) * np.exp(-t / 8), np.cos(2 * t)]
        shift = rng.integers(0, max(1, length // 8), size=n_rows)
        x = np.stack([np.roll(shapes[c % len(shapes)], s) for c, s in zip(truth, shift)])
        out["x"] = x + rng.normal(0, noise, size=x.shape)
        out["columns"]["truth"] = np.array([f"shape{c}" for c in truth])
    elif name == "biclusters":
        k_b = max(2, k)
        row_groups = _groups(n_rows, k_b)
        col_groups = _groups(n_features, k_b)
        values = rng.uniform(10, 100, size=k_b)
        x = values[row_groups][:, None] * (row_groups[:, None] == col_groups[None, :])
        out["x"] = x + rng.normal(0, noise, size=(n_rows, n_features))
        out["columns"]["truth"] = np.array([f"r{g}" for g in row_groups])
        out["columns"]["column_truth"] = np.tile(",".join(str(g) for g in col_groups), n_rows)
    elif name == "topics":
        n_topics = max(2, k)
        topic_words = rng.dirichlet(np.full(n_features, 0.05), size=n_topics)
        truth = _groups(n_rows, n_topics)
        mix = np.full((n_rows, n_topics), max(noise, 1e-3))
        mix[np.arange(n_rows), truth] += 1.0
        mix = mix / mix.sum(axis=1, keepdims=True)
        lengths = rng.integers(50, 150, size=n_rows)
        counts = np.stack([rng.multinomial(lengths[i], mix[i] @ topic_words) for i in range(n_rows)])
        out["x"] = counts.astype(np.float32)
        out["columns"]["truth"] = np.array([f"topic{t}" for t in truth])
    elif name == "y_shape":
        arms = _groups(n_rows, 3)
        position = rng.uniform(0, 1, size=n_rows)
        directions = np.array([[0.0, -1.0], [-0.87, 0.5], [0.87, 0.5]])
        x = directions[arms] * (position[:, None] * 5) + rng.normal(0, noise, size=(n_rows, 2))
        out["x"] = _pad(x, n_features, rng, noise)
        out["columns"]["truth"] = np.array(["y"] * n_rows)
        out["columns"]["branch"] = np.array([f"arm{a}" for a in arms])
    elif name == "planted_graph_points":
        truth = _groups(n_rows, k)
        centers = rng.uniform(-6, 6, size=(k, n_features))
        out["x"] = centers[truth] + rng.normal(0, noise, size=(n_rows, n_features))
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "graph":
        truth = _groups(n_rows, k)
        p_in, p_out = min(0.5, 20.0 / n_rows), min(0.05, 1.0 / n_rows)
        edges = []
        for i in range(n_rows):
            same = truth == truth[i]
            probs = np.where(same, p_in, p_out)
            probs[i] = 0
            neighbours = np.flatnonzero(rng.uniform(size=n_rows) < probs)
            edges.extend((i, j) for j in neighbours if j > i)
        e = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        out["edges"] = np.concatenate([e, e[::-1]], axis=1)
        centers = rng.normal(0, 1.0, size=(k, n_features))
        out["x"] = centers[truth] + rng.normal(0, max(noise, 1e-3) * 3, size=(n_rows, n_features))
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "sequences":
        truth = _groups(n_rows, k)
        items = []
        for c in truth:
            steps = int(rng.integers(20, 60))
            t = np.linspace(0, 1, steps)
            base = np.stack([np.sin(2 * np.pi * (c + 1) * t + j) for j in range(n_features)], axis=1)
            items.append((base + rng.normal(0, noise, size=base.shape)).astype(np.float32))
        out["items"] = items
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "images":
        truth = _groups(n_rows, k)
        size = item_shape("images", n_features)[1]
        items = []
        for c in truth:
            img = rng.normal(0, noise, size=(1, size, size))
            stripe = (c * size) // k
            if c % 2 == 0:
                img[0, stripe:stripe + max(1, size // k), :] += 1.5
            else:
                img[0, :, stripe:stripe + max(1, size // k)] += 1.5
            items.append(img.astype(np.float32))
        out["items"] = items
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "audio":
        truth = _groups(n_rows, k)
        rate = audio_sample_rate()
        items = []
        for c in truth:
            seconds = float(rng.uniform(0.5, 1.5))
            t = np.arange(int(seconds * rate)) / rate
            wave = 0.3 * np.sin(2 * np.pi * (200 + 150 * c) * t) + rng.normal(0, noise, size=t.size)
            items.append(wave.astype(np.float32))
        out["items"] = items
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    elif name == "text":
        truth = _groups(n_rows, k)
        vocab = [[f"w{c}_{j}" for j in range(5)] for c in range(k)]
        common = [f"the{j}" for j in range(5)]
        items = []
        for c in truth:
            words = list(rng.choice(vocab[c], size=4)) + list(rng.choice(common, size=4))
            items.append(" ".join(rng.permutation(words)))
        out["items"] = items
        out["columns"]["truth"] = np.array([f"c{t}" for t in truth])
    if out["x"] is not None:
        out["x"] = np.asarray(out["x"], dtype=np.float32)
        if out["names"] is None:
            out["names"] = [f"f{j}" for j in range(out["x"].shape[1])]
    n = n_rows
    out["columns"]["group"] = np.array([f"g{g}" for g in np.arange(n) % 5])
    # a published split, as some datasets ship one: train, val or test for every row (60, 20 and 20 percent, from a
    # stream of its own so the other columns do not change), for the split "fixed"
    order = stream(seed, "synthetic published split").permutation(n)
    roles = np.empty(n, dtype=object)
    first, second = int(round(0.6 * n)), int(round(0.8 * n))
    roles[order[:first]], roles[order[first:second]], roles[order[second:]] = "train", "val", "test"
    out["columns"]["published_split"] = roles.astype(str)
    return out
