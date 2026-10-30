# Extending

How to add a model, an algorithm, a preparation step, a metric, a data reader, a split, a stage kind, or a
combination of existing parts, so that it is trusted like the rest: registered with every setting explicit, used by
an example sweep (a metric has none: the runs that compute it use it), tested on data with a known answer, and run
through the real runner (`smoke-all` runs every component; a stage kind runs when its example sweep is checked and
smoked). Each kind has a worked example below; every example on this page was scaffolded, edited as shown, and run
(its test, `check-additions`, and `smoke` of its example sweep) when the page was written.

## The steps, for any addition

1. **Check what exists.** docs/PROFILES.md lists every component; docs/UNSUPERVISED.md and docs/DATA.md describe
   them. Many requests are settings of an existing component, or a combination (the last section), and need no code.
2. **Look up the library's current API**, at the version that will be pinned: its documentation, its changelog, its
   source at that tag. Do not write a call from memory. Pass every parameter explicitly; put the rest in an `extra`
   dictionary (`models.base.merged_kwargs`), so a library's default never decides a result silently.
3. **Scaffold it**: `python -m sweepkit.cli scaffold --kind <kind> --name <name> --where project`. Kinds:
   `clusterer`, `embedder`, `detector`, `estimator`, `torch_model`, `prep`, `metric`, `split`, `dataset`,
   `stage_kind`. The name is lowercase letters, digits and `_`, and must not be taken (scaffold refuses a name that
   is registered already: `lda`, for example, is the topic model). It writes four things and prints their paths:
   - the module, `src/project/<name>.py`: a working starting point (it runs as written) whose docstring says what to
     change;
   - its line in `src/project/catalog.py`, so it is imported with everything else;
   - an example sweep, `sweeps/project/<name>.json`, on synthetic data (metrics have none);
   - a test, `tests/test_project_<name>.py`.
4. **Edit the module**: the settings class (every setting, no defaults), then the body. Then the example sweep and the
   test, to match the new settings.
5. **Set the test's bound from runs.** Run the method on the test's data over several data seeds and set the bound a
   margin below the worst value seen, with that value in a comment (`# 0.970 at worst over seeds 1 to 10`). A bound
   is never set from the answer you hope for.
6. **Check it**, from the repository folder:

   ```
   python -m unittest tests.test_project_<name>
   python -m sweepkit.cli check-additions --show 50
   python -m sweepkit.cli smoke --sweep-file sweeps/project/<name>.json --device cpu --rows 100
   ```

   The test ends with `OK`; `check-additions` with `ADDITIONS OK` (it refuses an addition that no test names in
   quotes, and one that no example sweep uses, except a metric); `smoke` with `SMOKE OK`. Then the whole suite, and
   `smoke-all` for the profile.
7. **A new package** gets an exact pin (`==`) in the profile it belongs to, `env/requirements-<profile>.txt`, and
   the component declares it (`packages=("its_import_name",)` and `profile="<profile>"`), so a machine without it
   skips the component instead of failing, and a machine that must have it (`SWEEPKIT_REQUIRE`) fails.

`--where project` keeps your additions in `src/project/`, `sweeps/project/` and `tests/test_project_*.py`, so the
template's own files stay untouched and `git pull upstream main` brings the template's fixes without conflicts. The
template's generated pages (docs/PROFILES.md, the testing matrix in TESTING.md) list the template's own components
only, so your additions never make them stale. `--where template` is for an addition meant for the template itself;
it then needs `profiles-doc --write yes` and `test-matrix --write yes`.

## A model from a library: an estimator

Linear discriminant analysis, scikit-learn's `LinearDiscriminantAnalysis` (checked against scikit-learn 1.9's
documentation: `solver` is `svd`, `lsqr` or `eigen`; `shrinkage` works with `lsqr` and `eigen` only).

```
python -m sweepkit.cli scaffold --kind estimator --name linear_discriminant --where project
```

`src/project/linear_discriminant.py`, edited:

```python
"""linear_discriminant: linear discriminant analysis (scikit-learn's LinearDiscriminantAnalysis), a linear classifier
from the class means and one covariance shared by the classes, which can be shrunk when there are few rows for many
columns."""
import dataclasses
from typing import Literal

from sweepkit.models.base import merged_kwargs
from sweepkit.models.estimators import register_estimator
from sweepkit.settings import Scalar


@dataclasses.dataclass(frozen=True)
class LinearDiscriminantSettings:
    solver: Literal["svd", "lsqr", "eigen"]
    shrinkage: float | None         # 0 to 1 with solver lsqr or eigen; None for none
    extra: dict[str, Scalar]


@register_estimator("linear_discriminant", LinearDiscriminantSettings, ("classification",), "none", "core",
                    ("sklearn",), "linear discriminant analysis (scikit-learn)", "linear_discriminant")
def build(s: LinearDiscriminantSettings, task: str, seed: int, device: str):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if s.shrinkage is not None and (s.solver == "svd" or not 0 <= s.shrinkage <= 1):
        raise ValueError("model linear_discriminant: shrinkage is between 0 and 1, with solver lsqr or eigen")
    return LinearDiscriminantAnalysis(**merged_kwargs({"solver": s.solver, "shrinkage": s.shrinkage}, s.extra,
                                                      "linear_discriminant"))
```

`register_estimator` takes the name, the settings class, the tasks, the GPU path (`"none"`, or `"cuml"` with a cuML
build when `device` is `"cuda"`), the profile, the packages, a description and the example's name. The estimator
runner does the rest: fitting on the training rows, predictions and class scores (`predict_proba` or
`decision_function`), the controls and the refit. In the example sweep, the model section becomes
`{"name": "linear_discriminant", "solver": "svd", "shrinkage": null, "extra": {}}`, and a variant
`{"model.solver": "lsqr", "model.shrinkage": 0.2}` tries shrinkage. In the test, the settings given to `validate`
become the same, with both solvers; it asserts a balanced accuracy of at least 0.9 on separable classes.

## An algorithm written by hand: a clusterer

Spherical k-means (Dhillon and Modha 2001): rows scaled to length 1 and grouped by cosine similarity, for data whose
direction matters more than its size. Nothing to install: numpy.

```
python -m sweepkit.cli scaffold --kind clusterer --name spherical_kmeans --where project
```

```python
"""spherical_kmeans: k-means on the unit sphere (Dhillon and Modha 2001). Each row is scaled to length 1 and rows are
grouped by cosine similarity to their cluster's mean direction, for data whose direction matters more than its size
(counts, spectra, embeddings)."""
import dataclasses

import numpy as np

from sweepkit.models.base import Clusterer, plain_params
from sweepkit.registry import register_model
from sweepkit.streams import generator


@dataclasses.dataclass(frozen=True)
class SphericalKmeansSettings:
    n_clusters: int
    n_init: int                     # starts; the one with the largest total cosine similarity is kept
    max_iter: int                   # updates per start at most


def unit_rows(x) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("spherical_kmeans: a row of zeros has no direction")
    return x / norms


@register_model("spherical_kmeans", settings=SphericalKmeansSettings, family="clusterer", tasks=("clustering",),
                kinds=("tabular",), gpu="none", profile="core", packages=(),
                description="k-means on the unit sphere (cosine similarity)", example="spherical_kmeans")
class SphericalKmeans(Clusterer):
    @staticmethod
    def check_job(job) -> None:
        if job.model.n_clusters < 2 or job.model.n_init < 1 or job.model.max_iter < 1:
            raise ValueError("model spherical_kmeans: n_clusters at least 2, n_init and max_iter at least 1")

    def fit(self, x):
        u = unit_rows(np.asarray(x, dtype=np.float64))
        rng = generator(self.seed, "spherical_kmeans")
        best = None
        for _ in range(self.s.n_init):
            centers = u[rng.choice(u.shape[0], self.s.n_clusters, replace=False)]
            for _ in range(self.s.max_iter):
                labels = np.argmax(u @ centers.T, axis=1)
                new = unit_rows(np.array([u[labels == c].sum(axis=0) if (labels == c).any() else centers[c]
                                          for c in range(self.s.n_clusters)]))
                if np.allclose(new, centers):
                    break
                centers = new
            labels = np.argmax(u @ centers.T, axis=1)
            similarity = float((u * centers[labels]).sum())
            if best is None or similarity > best[0]:
                best = (similarity, centers, labels)
        self.similarity, self.centers, self._labels = best

    def info(self):
        return {"total_cosine_similarity": self.similarity}

    def labels(self):
        return np.asarray(self._labels, dtype=np.int64)

    def can_assign(self):
        return True

    def assign(self, x_new):
        return np.argmax(unit_rows(np.asarray(x_new, dtype=np.float64)) @ self.centers.T, axis=1)

    def params(self):
        return plain_params(dataclasses.asdict(self.s))
```

What the clusterer runner asks of it: `fit(x)` on the fit rows; `labels()` for them (-1 for noise); `assign(x_new)`
when `can_assign()` (so the evaluated rows can differ from the fit rows); `info()` for numbers it reports about itself;
`params()` for every parameter used. `soft()` gives membership probabilities when a method has them. Randomness comes
from `sweepkit.streams.generator(self.seed, "<name>")` only, so a seed gives the same result on every machine. A
method with a GPU version declares `gpu="cuml"` (or another path in `sweepkit.devices`), builds it when
`self.device == "cuda"`, and gets an agreement test (tests/test_gpu_agreement.py).

The test plants three directions with lengths that vary tenfold within each group, and asserts an adjusted Rand index
of at least 0.95 and that scaling every row by 7 changes no label. The example sweep scales rows to length 1
(`scale` with `l2_rows`) and compares with k-means and with two and four clusters.

<details><summary>The test, <code>tests/test_project_spherical_kmeans.py</code>, as edited</summary>

```python
"""spherical_kmeans groups rows by direction, whatever their length, where k-means groups them by length too."""
import unittest

from tests.support import load_catalog


class TestSphericalKmeans(unittest.TestCase):
    def test_directions_of_any_length(self):
        load_catalog()
        import numpy as np
        from sklearn.metrics import adjusted_rand_score

        from sweepkit.models.base import build_clusterer
        from sweepkit.registry import MODELS
        from sweepkit.settings import Selected, validate

        rng = np.random.default_rng(1)
        directions = rng.normal(size=(3, 5))
        truth = np.repeat([0, 1, 2], 100)
        lengths = rng.uniform(0.5, 5.0, size=(300, 1))           # lengths vary tenfold within each group
        x = directions[truth] / np.linalg.norm(directions[truth], axis=1, keepdims=True) * lengths
        x += rng.normal(scale=0.05, size=x.shape) * lengths
        raw = {"n_clusters": 3, "n_init": 4, "max_iter": 100}
        model = build_clusterer(Selected("spherical_kmeans", validate(
            MODELS.get("spherical_kmeans").settings_class, raw, "model")), 0, "cpu")
        model.fit(x)
        self.assertGreaterEqual(adjusted_rand_score(truth, model.labels()), 0.95)
        self.assertEqual(model.assign(x * 7.0).tolist(), model.labels().tolist())   # length does not matter


if __name__ == "__main__":
    unittest.main()
```

</details>

<details><summary>The example sweep, <code>sweeps/project/spherical_kmeans.json</code>, as edited</summary>

```json
{
 "sweep_id": "example_spherical_kmeans",
 "description": "spherical_kmeans on planted blobs whose rows are scaled to length 1 (directions), against k-means, and its number of clusters around the planted three.",
 "purpose": "learning",
 "follows": [],
 "folds": ["sub1", "sub2", "sub3"],
 "base": {
  "data": {"name": "synthetic", "generator": "blobs", "n_rows": 600, "n_features": 5, "n_groups": 3, "noise": 1.0, "seed": 1},
  "split": {"name": "subsamples", "k": 3, "fraction": 0.8, "salt": "example"},
  "task": {"name": "clustering", "criteria": ["silhouette", "calinski_harabasz", "n_clusters"], "select": "silhouette",
           "sample_rows": 400, "sample_salt": "example", "reference": "truth", "reveal_metrics": ["ari", "nmi"],
           "criteria_options": {}},
  "prep": [{"name": "scale", "method": "l2_rows", "columns": ["*"], "per_group": null}],
  "features": {"name": "none"},
  "model": {"name": "spherical_kmeans", "n_clusters": 3, "n_init": 4, "max_iter": 100},
  "fit": {"name": "clusterer", "device": "cpu", "assign": "fit_rows"}
 },
 "stages": [
  {"name": "base", "kind": "base", "priority": 9, "seeds": [1, 2, 3]},
  {"name": "against", "kind": "variants", "priority": 8, "seeds": [1, 2, 3],
   "variants": {"kmeans": {"model": {"name": "kmeans", "n_clusters": 3, "init": "k-means++", "n_init": 4, "max_iter": 300,
                                     "tol": 0.0001, "extra": {}}}}},
  {"name": "k", "kind": "one_factor_at_a_time", "priority": 7, "seeds": [1, 2, 3], "factors": {"model.n_clusters": [2, 4]}}
 ]
}
```

The scaffold's sweep had the settings of its own body; the edit gives the model's three settings, scales rows to
length 1 first, and adds k-means as a variant and two and four clusters as one factor at a time. Run and analyzed,
spherical k-means and k-means tie (silhouette 0.7492 on these blobs) and two or four clusters score lower.

</details>

## A neural network: a torch model

```
python -m sweepkit.cli scaffold --kind torch_model --name my_net --where project
```

The module's body is a multilayer perceptron, so it trains as written; its docstring lists what to change. A torch
model is only the network: the torch family's fit procedure does the training (epochs, the best validation epoch
kept and reloaded, early stopping, the refit, the controls, mixed precision, augmentations). Change, in order:

- the settings class (every layer size, activation and dropout as a setting);
- `check(settings, input_shape)`: settings that cannot work raise `ValueError` (it runs when a sweep is expanded
  and again when the network is built);
- the module: `embed(x, lengths)` gives each row's summary `[batch, embedding_dim]`, `classify(summary)` the outputs,
  `forward` both; `kinds` says what a row is (`tabular`, `sequence` with lengths, `image`, `text`, `graph`);
- import torch inside functions only, so registering the model and checking a sweep work without torch.

The example sweep compares it with the template's `mlp` and a shuffled-label control. Its test runs the example's base
through the real runner and skips without torch (it fails where `SWEEPKIT_REQUIRE` names torch). Torch models run on
`cuda` and `mps` as well as the CPU (`fit.device`). Read `src/sweepkit/models/torch/` for the template's own networks
before writing a new one: a convolutional front with a recurrent or Transformer block is a combination of existing
parts (the last section), not new code.

## A preparation step

Clipping each column to its median plus or minus k median absolute deviations: a robust alternative to clipping at
the mean plus or minus n standard deviations, which the outliers themselves inflate.

```
python -m sweepkit.cli scaffold --kind prep --name clip_mad --where project
```

```python
"""clip_mad: clips each numeric column to its median plus or minus k median absolute deviations, both learned on the
fit rows: a robust alternative to clipping at the mean plus or minus n standard deviations, which the outliers
themselves inflate."""
import dataclasses

import numpy as np

from sweepkit.prep.base import PrepStep
from sweepkit.registry import register_prep


@dataclasses.dataclass(frozen=True)
class ClipMadSettings:
    k: float                        # how many median absolute deviations from the median a value may lie


@register_prep("clip_mad", settings=ClipMadSettings, kinds=("tabular",), profile="core", packages=(),
               description="clips each column to its median plus or minus k MADs", example="clip_mad")
class ClipMad(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        for name, s in job.prep:
            if name == "clip_mad" and not s.k > 0:
                raise ValueError("prep clip_mad: k above 0")

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        self.columns = self.numeric_columns(x.shape[1])
        part = x[:, self.columns]
        self.median = np.median(part, axis=0)
        self.mad = np.median(np.abs(part - self.median), axis=0)

    def transform(self, x, positions):
        x = np.array(x, dtype=np.float64, copy=True)
        low, high = self.median - self.s.k * self.mad, self.median + self.s.k * self.mad
        x[:, self.columns] = np.clip(x[:, self.columns], low, high)
        return x

    def record(self):
        names = [self.ctx.feature_names[j] for j in self.columns]
        return {"k": self.s.k, "median": dict(zip(names, self.median.tolist())),
                "mad": dict(zip(names, self.mad.tolist()))}
```

`fit` sees the fit rows only and `transform` is applied to every role, so a held-out row never changes what the step
learns. The test checks exactly that, by hand:

```python
        x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [100.0, -1000.0]])
        ...
        step = ("clip_mad", validate(PREP.get("clip_mad").settings_class, {"k": 1.5}, "prep"))
        positions = {"fit": np.arange(5), "eval": np.array([5])}
        out, _, _, _, records = run_prep((step,), {"fit": x[:5], "eval": x[5:]}, positions, "fit", context,
                                         lambda m: None)
        # u: median 3, MAD 1, so [1.5, 4.5]; v: median 30, MAD 10, so [15, 45]
        self.assertTrue(np.allclose(out["fit"], [[1.5, 15], [2, 20], [3, 30], [4, 40], [4.5, 45]]))
        # the held-out row is clipped to the bounds learned without it
        self.assertTrue(np.allclose(out["eval"], [[4.5, 15]]))
```

Other things a step can do: drop or add columns (`names()` and `categorical_after()`), take rows out of the fit
(`removed()`, declared `removes_rows=True`; those rows are recorded by ID), read a per-row column such as a batch
(`self.ctx.table.column(name)` at `positions`). A step that must see rows other than the fit rows is declared
`transductive=True`, and every report on a configuration that uses it says so. A sweep varies a step's setting with
its path, `{"prep.clip_mad.k": 5.0}`.

## A metric or a label-free criterion

The Dunn index (Dunn 1974), a label-free clustering criterion: the smallest distance between two rows of different
clusters over the largest distance within one cluster.

```
python -m sweepkit.cli scaffold --kind metric --name dunn --where project
```

```python
"""dunn: the Dunn index (Dunn 1974), a label-free clustering criterion: the smallest distance between two rows of
different clusters over the largest distance between two rows of one cluster, on the task's fixed sample with noise
rows left out. Larger is better; it is sensitive to single outlying rows."""
import numpy as np

from sweepkit.metrics.unsupervised_context import PRECOMPUTED, Undefined
from sweepkit.registry import register_metric


@register_metric("dunn", tasks=("clustering",), direction="max", needs=("x", "labels"), weighted=False,
                 label_free=True, description="Dunn index on the fixed sample, noise left out")
def dunn(ctx) -> float:
    from scipy.spatial.distance import cdist, pdist

    if ctx.distances is not None:
        raise Undefined(PRECOMPUTED)
    labels = ctx.labels[ctx.sample]
    x, labels = ctx.x[ctx.sample][labels >= 0], labels[labels >= 0]
    groups = np.unique(labels)
    if groups.size < 2:
        raise Undefined(f"{groups.size} cluster(s) outside noise; the Dunn index needs at least 2")
    diameter = max(float(pdist(x[labels == g]).max()) if (labels == g).sum() > 1 else 0.0 for g in groups)
    if diameter == 0:
        raise Undefined("every cluster is a single point")
    separation = min(float(cdist(x[labels == a], x[labels == b]).min())
                     for i, a in enumerate(groups) for b in groups[i + 1:])
    return separation / diameter
```

A label-free criterion reads the run's context (`sweepkit.metrics.unsupervised_context`: the evaluated rows, the
labels, the fixed sample); a result it cannot score raises `Undefined` with the reason, which the run records. A
supervised metric is a function of the outputs and optional row weights, `fn(o, w)` (the scaffold's body, the worst
per-class recall, is one); declaring `weighted=True` lets the paired bootstrap use it. A metric has no example sweep:
a task uses it by name (`"criteria": ["silhouette", "dunn"]`, or `"select": "dunn"` in a RULE written before the
results). Its test computes cases by hand: here, clusters {0, 1} and {5, 7} give 4 / 2 = 2.0, a noise row changes
nothing, and one cluster is `Undefined`.

## A data reader: a dataset

A table stored in a MATLAB `.mat` file: a matrix of rows by columns, and vectors with one value per row.

```
python -m sweepkit.cli scaffold --kind dataset --name mat_file --where project
```

```python
"""mat_file: a table stored in a MATLAB .mat file: one matrix of rows by columns, and vectors with one value per row
kept as text columns (labels, groups). Read with SciPy's loadmat, which reads MATLAB's formats up to version 7.2;
a version 7.3 file is HDF5 (the `table` dataset reads HDF5)."""
import dataclasses

import numpy as np

from sweepkit import paths
from sweepkit.data.base import Data, Dataset, RowTable
from sweepkit.datasets.table import file_fingerprint
from sweepkit.registry import register_dataset


@dataclasses.dataclass(frozen=True)
class MatFileSettings:
    path: str                       # the .mat file, relative to data/
    variable: str                   # the matrix: rows by columns
    keep: tuple[str, ...]           # vectors kept as per-row text columns (targets, groups, reference labels)


@register_dataset("mat_file", settings=MatFileSettings, kinds=("tabular",), profile="core", packages=("scipy",),
                  description="a matrix and per-row vectors from a MATLAB .mat file", example="mat_file")
class MatFile(Dataset):
    EXAMPLE_FILE = "synthetic/mat_file_example.mat"

    @staticmethod
    def write_example(path) -> None:
        """Demonstration rows for the example sweep: three planted blobs (sweepkit.data.synthetic)."""
        from scipy.io import savemat

        from sweepkit.data.synthetic import make

        made = make("blobs", 300, 4, 3, 1.0, 1)
        savemat(path, {"X": made["x"], "truth": made["columns"]["truth"]})

    @staticmethod
    def check_job(job) -> None:
        paths.data_path(job.data.path)

    def load(self, s) -> Data:
        from scipy.io import loadmat

        path = paths.data_path(s.path)
        if not path.is_file():
            raise FileNotFoundError(f"data mat_file: {path} does not exist (paths are relative to data/)")
        arrays = loadmat(path)
        if s.variable not in arrays:
            raise KeyError(f"data mat_file: no variable {s.variable!r} in {s.path}")
        x = np.asarray(arrays[s.variable], dtype=np.float32)
        ids = np.array([f"r{i:07d}" for i in range(x.shape[0])])        # the row number is the row ID
        kept = {}
        for name in s.keep:
            values = np.asarray(arrays[name]).reshape(-1)
            if values.size != x.shape[0]:
                raise ValueError(f"data mat_file: {name} has {values.size} values for {x.shape[0]} rows")
            kept[name] = np.char.strip(values.astype(str))
        names = tuple(f"x{j}" for j in range(x.shape[1]))
        return Data("tabular", RowTable(ids, kept), x=x, feature_names=names,
                    fingerprint={"path": s.path, **file_fingerprint(path)})
```

A dataset returns a `Data`: the row table (row IDs and the kept columns, as text), the inputs, the column names, and a
fingerprint of the file read (its SHA-256), which every run records. Paths are relative to `data/`
(`paths.data_path` refuses anything else). `EXAMPLE_FILE` and `write_example` give the example sweep a small file to
read on any machine: `make-example-data` (which `smoke-all` runs first) writes it when it is missing. Most formats
need no new dataset: `table` reads CSV, TSV, Parquet, Excel, HDF5 and Feather, `items` one file per row, `graph` node
and edge tables, `anndata` single-cell files (docs/DATA.md).

## A split

One supervised fold in time order: the earliest rows train, the next validate, the latest test, so a model is always
judged on rows later than the ones it learned from.

```
python -m sweepkit.cli scaffold --kind split --name time_order --where project
```

```python
"""time_order: one supervised fold in time order: the earliest rows train, the next ones validate, the latest test,
so a model is always judged on rows later than the ones it learned from (forecasting, drifting measurements)."""
import dataclasses

import numpy as np

from sweepkit.registry import register_split
from sweepkit.splits.base import SUPERVISED_ROLES, Fold


@dataclasses.dataclass(frozen=True)
class TimeOrderSettings:
    time_column: str | None         # a kept column with each row's time (numbers, or ISO dates as text); None: the
                                    # order of the row IDs
    val_fraction: float
    test_fraction: float


@register_split("time_order", settings=TimeOrderSettings, roles=SUPERVISED_ROLES,
                description="earliest rows train, the next validate, the latest test")
class TimeOrder:
    @staticmethod
    def fold_names(s) -> tuple:
        return ("split1",)

    @staticmethod
    def make(s, table) -> tuple:
        if not 0 < s.val_fraction < 1 or not 0 < s.test_fraction < 1 or s.val_fraction + s.test_fraction >= 1:
            raise ValueError("split time_order: fractions in (0, 1) that leave training rows")
        if s.time_column is None:
            times = table.ids
        else:
            times = table.column(s.time_column)
            try:
                times = times.astype(np.float64)
            except ValueError:
                pass                        # text, such as ISO dates, which sort as text
        order = np.lexsort((table.ids, times))                       # equal times in order of row ID
        n = order.size
        n_test, n_val = int(round(s.test_fraction * n)), int(round(s.val_fraction * n))
        roles = {"train": np.sort(order[:n - n_val - n_test]), "val": np.sort(order[n - n_val - n_test:n - n_test]),
                 "test": np.sort(order[n - n_test:])}
        return (Fold("split1", roles, "rows in time order: train, then validation, then test", False),)
```

`fold_names` comes from the settings alone, so a sweep file is checked without reading data. Orders come from the data
or from SHA-256 (`splits.base.stable_keys`), never from a random generator, so the folds are the same on every machine.
The last argument of `Fold` says whether the fold chooses on test rows: only the identity check fold does. The test
gives 100 rows over 20 days and checks the role sizes, that every training day comes before every validation day and
every validation day before every test day, and that a second call gives the same folds.

<details><summary>The test, <code>tests/test_project_time_order.py</code>, as edited</summary>

```python
"""time_order puts the earliest rows in training, the latest in test, the same on every run."""
import unittest

from tests.support import load_catalog


class TestTimeOrder(unittest.TestCase):
    def test_roles_follow_time(self):
        load_catalog()
        import numpy as np

        from sweepkit.data.base import RowTable
        from sweepkit.registry import SPLITS
        from sweepkit.settings import validate

        entry = SPLITS.get("time_order")
        days = np.array([f"2026-01-{d:02d}" for d in range(10, 30)] * 5)          # 100 rows, 20 days
        table = RowTable(np.array([f"r{i:03d}" for i in range(100)]), {"day": days})
        s = validate(entry.settings_class, {"time_column": "day", "val_fraction": 0.2, "test_fraction": 0.2}, "split")
        (fold,) = entry.target.make(s, table)
        self.assertEqual({r: len(rows) for r, rows in fold.roles.items()}, {"train": 60, "val": 20, "test": 20})
        train, val, test = (sorted(days[fold.roles[r]].tolist()) for r in ("train", "val", "test"))
        self.assertLess(train[-1], val[0])                  # every training day before every validation day
        self.assertLess(val[-1], test[0])
        (again,) = entry.target.make(s, table)
        self.assertTrue(all(np.array_equal(fold.roles[r], again.roles[r]) for r in fold.roles))


if __name__ == "__main__":
    unittest.main()
```

</details>

<details><summary>The example sweep, <code>sweeps/project/time_order.json</code>, as edited</summary>

```json
{
 "sweep_id": "example_time_order",
 "description": "Logistic regression on synthetic classes with the time_order split (the row IDs taken as the time order), two strengths of regularization.",
 "purpose": "learning",
 "follows": [],
 "folds": ["split1"],
 "base": {
  "data": {"name": "synthetic", "generator": "classification", "n_rows": 600, "n_features": 6, "n_groups": 3, "noise": 1.0, "seed": 1},
  "split": {"name": "time_order", "time_column": null, "val_fraction": 0.2, "test_fraction": 0.2},
  "task": {"name": "classification", "target": "truth", "classes": ["c0", "c1", "c2"],
           "metrics": ["balanced_accuracy", "accuracy"], "select": "balanced_accuracy", "groups": []},
  "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": null}],
  "features": {"name": "none"},
  "model": {"name": "logistic_regression", "c": 1.0, "l1_ratio": 0.0, "class_weight": "none", "max_iter": 500, "extra": {}},
  "fit": {"name": "estimator", "device": "cpu", "control": "none", "refit": "train", "eval_train": false}
 },
 "stages": [
  {"name": "base", "kind": "base", "priority": 9, "seeds": [1, 2, 3]},
  {"name": "c", "kind": "one_factor_at_a_time", "priority": 8, "seeds": [1, 2, 3], "factors": {"model.c": [0.01]}}
 ]
}
```

The synthetic data has no time column, so the example takes the row IDs as the time order (`time_column` null); the
sweep's `folds` lists the one fold that `fold_names` gives.

</details>

## A stage kind

A stage kind turns one stage of a sweep file into configurations. The scaffold's body is a working one: the corners
of a grid, every combination of each factor's first and last level.

```
python -m sweepkit.cli scaffold --kind stage_kind --name corners --where project
```

```python
@register_stage_kind("corners", ("factors",))
def corners(stage, base, context):
    factors = {f: levels_of(f, levels) for f, levels in stage["factors"].items()}
    if not factors:
        raise SweepError(f"stage {stage['name']}: give at least one factor")
    corners = [(levels[0], levels[-1]) if len(levels) > 1 else (levels[0],) for levels in factors.values()]
    out = []
    for combination in itertools.product(*corners):
        overrides = {}
        for level in combination:
            overrides.update(level)
        changed = changed_only(base, overrides)
        if changed:
            out.append((f"corners: {label_of(changed)}", overrides))
    return out
```

It receives the stage (with the keys it declares, here `factors`), the base settings and the expansion's context, and
returns `(label, overrides)` pairs. `expand` validates and hashes each, refuses a configuration identical to the base
or to one another stage already holds (so a stage kind leaves out the base's own combination, as `changed_only` does
here; only a `random` stage drops repeated draws, and counts them), and keeps one evaluation protocol per sweep. The
built-in kinds are `base`, `variants`, `one_factor_at_a_time`, `grid` and `random`
(`sweeps/examples/search_stages.json` uses the last two). A sweep file then uses a new kind like a built-in one:
`{"name": "edges", "kind": "corners", "priority": 7, "seeds": [1, 2, 3], "factors": {"model.n_clusters": [2, 3, 5], "model.n_init": [1, 4]}}`.

## Combinations without new code

Many new methods are existing parts put together in the settings:

- **Reduce, then cluster**: a `reduce` preparation step holds an embedder's whole section (PCA, UMAP, an autoencoder),
  so UMAP to ten dimensions and then HDBSCAN is one configuration
  (`sweeps/examples/cluster_umap_hdbscan.json`).
- **A network from blocks**: the torch model `composite` takes a front (`none`, `conv1d` or `conv2d`) and a block
  (`none`, `rnn`, `transformer` or `lif`), each with its own settings: a convolutional front with a Transformer block
  is a setting (`sweeps/examples/sequence_models.json`).
- **Many runs of one method**: `consensus` clusters by co-association over many runs of any base clusterer
  (`sweeps/examples/cluster_consensus.json`).
- **Preparation as a factor**: a variants stage can give whole `prep` lists, so "log1p then z-scores" against "ranks"
  is two variants of one sweep.

## Changing an existing component

A configuration's hash is the hash of its settings, and runs are compared through it, so a change to a component must
not change what an existing configuration does or is called:

- **A new setting**: add the field, and list it in the settings class's `ADDED_FIELDS` with the date and the value
  that does exactly what the code did before:

  ```python
  ADDED_FIELDS = (("2027-01-15", {"speed_factors": [1.0], "mixup_alpha": 0.0}),)
  ```

  Old sweep files (without the field) still validate, and a configuration whose new field holds that value keeps its
  old hash, so earlier runs stay comparable. A new sweep file that forgets the field still fails.
- **A different behaviour**: a new setting value, or a new component; never a silent change to what an existing value
  does.
- **A fix that changes results**: say so in the change, and expect the runs before and after it to differ
  (`compare-runs` shows which).

## A method from a repository without a package: Torque Clustering

Some methods have no installable package. Torque Clustering (Yang and Lin, IEEE TPAMI 2025) is an example: its
official code is MATLAB under CC BY-NC-SA 4.0 (non-commercial), and its README links an unofficial, community Python
port that is installed by cloning it; no package was found on PyPI when this template was written. The recipe:

1. Read the licence of the code you would run, and write down whether the project's use is allowed.
2. Pin the code: clone the port at one commit into a folder outside the repository (or vendor the needed file into
   `src/project/` with its licence and the commit in its docstring), and write the commit's hash down.
3. `scaffold --kind clusterer --name torque --where project`, and call the port from `fit`, passing every parameter
   explicitly; record the commit in `info()`.
4. Its test on planted blobs (and on two moons, which Torque is meant to handle) with a bound set from runs.
5. Compare it with the established methods in one sweep (HDBSCAN, k-means, agglomerative) before trusting it.

The same recipe fits any research code: pinned, wrapped, tested on known structure, compared.

## After adding: what checks what

| Check | What it covers |
|---|---|
| the component's own test | the known answer, with a bound set from runs |
| `check-additions` | every component and stage kind (the template's and the project's) has an example sweep that uses it, and every one, metrics included, a test that names it in quotes |
| `smoke` of its example, then `smoke-all --profile <profile>` | it runs through the real runner, and records the device it ran on; `smoke-all` runs every component, while a metric runs inside the runs that compute it and a stage kind when its example sweep is checked and smoked |
| the whole suite | nothing else changed |
| TESTING.md's matrix | the template's own components; `--where template` additions need `test-matrix --write yes` |
