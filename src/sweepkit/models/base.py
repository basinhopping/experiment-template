"""The interfaces every registered model implements, one per family, and the adapters for scikit-learn-style
libraries. A model's registry entry declares its family; the family's runner (sweepkit.families) calls these.

estimator  supervised, any scikit-learn-style object: make(settings, task, seed, device) returns an object with fit
           and predict, and predict_proba or decision_function for class scores.
clusterer  fit(x); labels() of the fit rows (-1 is noise or "no cluster"); soft() probabilities and outlier scores
           where the method has them; assign(x_new) for rows it was not fitted on, where it can; info() numbers it
           reports about itself (bic, aic, inertia); params() every parameter the library actually used.
embedder   fit_transform(x); transform(x_new) where it can; inverse_transform where it can (reconstruction error);
           info(); params().
detector   fit(x); score(x) anomaly scores, higher is more unusual; flag(x) the rows it calls outliers; info();
           params().
torch      see sweepkit.models.torch.base.

params() returns every parameter the library object used, its own defaults included, so a library default can
never hide: the runner writes it to model_params.json.

Devices. A model declares gpu="cuml" (or "torch", "xgboost") when it has a GPU path. The runner resolves the fit
section's device (cpu, cuda, or auto: cuda when the GPU path, its package and a GPU are all there) and passes cpu or
cuda here; a library that falls back to the CPU on its own must report it in fallbacks().
"""
import json

import numpy as np

from sweepkit.registry import MODELS


def plain_params(params: dict) -> dict:
    """Library parameters as JSON: anything that is not a plain value becomes its text form."""
    out = {}
    for key, value in sorted(params.items()):
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value if not (isinstance(value, float) and not np.isfinite(value)) else str(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (bool, int, float, str)) for v in value):
            out[key] = list(value)
        else:
            out[key] = repr(value)
    json.dumps(out)
    return out


def library_params(obj) -> dict:
    getter = getattr(obj, "get_params", None)
    if getter is None:
        return {"object": repr(obj)}
    try:
        return plain_params(getter(deep=False))
    except TypeError:
        return plain_params(getter())


def merged_kwargs(fixed: dict, extra: dict, where: str) -> dict:
    """The library keyword arguments: the settings' named fields, plus `extra` for parameters the settings class
    does not list. An extra may not repeat a named one."""
    clash = sorted(set(fixed) & set(extra))
    if clash:
        raise ValueError(f"{where}: extra repeats {clash}, which the settings name already")
    return {**fixed, **extra}


class Columns:
    """What every unsupervised model knows about its input columns: their names after preparation, and which hold
    integer category codes (index -> category names). The runner calls set_columns before fit; methods for mixed
    data (Gower distance, k-prototypes) read it, the others ignore it."""

    feature_names: tuple = ()
    categorical: dict = {}

    def set_columns(self, names: tuple, categorical: dict) -> None:
        self.feature_names = tuple(names)
        self.categorical = dict(categorical)


class Clusterer(Columns):
    def __init__(self, settings, seed: int, device: str):
        self.s = settings
        self.seed = seed
        self.device = device

    def fit(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def labels(self) -> np.ndarray:
        raise NotImplementedError

    def soft(self) -> dict:
        return {}

    def can_assign(self) -> bool:
        return False

    def assign(self, x_new: np.ndarray) -> np.ndarray:
        raise ValueError(f"{type(self).__name__} cannot label rows it was not fitted on")

    def info(self) -> dict:
        return {}

    def params(self) -> dict:
        return {}

    def fallbacks(self) -> list:
        return []

    def extra_files(self) -> dict:
        """Files the method adds to the run folder (a dendrogram's linkage, a co-association matrix)."""
        return {}


class LibraryClusterer(Clusterer):
    """A clusterer whose make(settings, seed, device) returns a scikit-learn-style object with fit and labels_."""

    def make(self, s, seed: int, device: str):
        raise NotImplementedError

    def fit(self, x):
        self.model = self.make(self.s, self.seed, self.device)
        self.model.fit(x)

    def labels(self):
        return _host(self.model.labels_).astype(np.int64)

    def can_assign(self):
        return hasattr(self.model, "predict")

    def assign(self, x_new):
        return _host(self.model.predict(x_new)).astype(np.int64)

    def params(self):
        return library_params(self.model)


class Embedder(Columns):
    def __init__(self, settings, seed: int, device: str):
        self.s = settings
        self.seed = seed
        self.device = device

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def can_transform(self) -> bool:
        return False

    def transform(self, x_new: np.ndarray) -> np.ndarray:
        raise ValueError(f"{type(self).__name__} cannot embed rows it was not fitted on")

    def can_inverse(self) -> bool:
        return False

    def inverse_transform(self, embedding: np.ndarray) -> np.ndarray:
        raise ValueError(f"{type(self).__name__} has no inverse transform")

    def info(self) -> dict:
        return {}

    def params(self) -> dict:
        return {}

    def fallbacks(self) -> list:
        return []


class LibraryEmbedder(Embedder):
    def make(self, s, seed: int, device: str):
        raise NotImplementedError

    def fit_transform(self, x):
        self.model = self.make(self.s, self.seed, self.device)
        return _host(self.model.fit_transform(x)).astype(np.float32)

    def can_transform(self):
        return hasattr(self.model, "transform")

    def transform(self, x_new):
        return _host(self.model.transform(x_new)).astype(np.float32)

    def can_inverse(self):
        return hasattr(self.model, "inverse_transform")

    def inverse_transform(self, embedding):
        return _host(self.model.inverse_transform(embedding))

    def params(self):
        return library_params(self.model)


class Detector(Columns):
    def __init__(self, settings, seed: int, device: str):
        self.s = settings
        self.seed = seed
        self.device = device

    def fit(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def flag(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {}

    def params(self) -> dict:
        return {}

    def fallbacks(self) -> list:
        return []


class LibraryDetector(Detector):
    """scikit-learn's convention: score_samples is higher for normal rows, predict is -1 for outliers."""

    def make(self, s, seed: int, device: str):
        raise NotImplementedError

    def fit(self, x):
        self.model = self.make(self.s, self.seed, self.device)
        self.model.fit(x)

    def score(self, x):
        return -_host(self.model.score_samples(x)).astype(np.float64)

    def flag(self, x):
        return _host(self.model.predict(x)) == -1

    def params(self):
        return library_params(self.model)


def _host(values) -> np.ndarray:
    """A numpy array from a numpy, CuPy or cuDF result."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "get") and not isinstance(values, np.ndarray):
        values = values.get()
    return np.asarray(values)


def _build(selected, family: str, seed: int, device: str):
    entry = MODELS.get(selected.name)
    if entry.meta["family"] != family:
        raise ValueError(f"{selected.name} is a {entry.meta['family']}, not a {family}")
    return entry.target(selected.settings, seed, device)


def build_clusterer(selected, seed: int, device: str) -> Clusterer:
    return _build(selected, "clusterer", seed, device)


def build_embedder(selected, seed: int, device: str) -> Embedder:
    return _build(selected, "embedder", seed, device)


def build_detector(selected, seed: int, device: str) -> Detector:
    return _build(selected, "detector", seed, device)
