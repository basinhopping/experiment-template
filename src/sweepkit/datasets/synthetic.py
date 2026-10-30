"""The synthetic dataset: data with known structure made by code (sweepkit.data.synthetic). Every example sweep, smoke
test and test uses it, so the template runs end to end with no data file."""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.data.base import ArrayItems, Data, Dataset, Graph, RowTable
from sweepkit.data.synthetic import GENERATORS, kind_of, make
from sweepkit.registry import register_dataset


@dataclasses.dataclass(frozen=True)
class SyntheticSettings:
    generator: Literal[GENERATORS]
    # blobs, anisotropic, varied, densities, moons, circles, classification, xor, regression_linear,
    # regression_nonlinear, multilabel, outliers, swiss_roll, circle, low_rank, sources, counts (single-cell-like
    # negative binomial counts), mixed (numeric and categorical columns), series (time series of shapes), biclusters,
    # topics, y_shape, batches, planted_graph_points; items: sequences, images, audio, text; graph
    n_rows: int
    n_features: int                 # columns (series: length; images: side; sequences: channels)
    n_groups: int                   # clusters, classes, topics or cell types, as the generator documents
    noise: float                    # the generator's spread
    seed: int                       # the data's own seed (not the run's)

    @property
    def kind(self) -> str:
        return kind_of(self.generator)


@register_dataset("synthetic", settings=SyntheticSettings,
                  kinds=("tabular", "sequence", "image", "audio", "text", "graph"), profile="core", packages=(),
                  description="data with known structure made by code (examples, smoke tests, tests)")
class Synthetic(Dataset):
    @staticmethod
    def check_job(job) -> None:
        s = job.data
        if s.n_rows < 4 or s.n_features < 1 or s.n_groups < 1 or s.noise < 0:
            raise ValueError("data synthetic: n_rows >= 4, n_features >= 1, n_groups >= 1, noise >= 0")

    def load(self, s: SyntheticSettings) -> Data:
        made = make(s.generator, s.n_rows, s.n_features, s.n_groups, s.noise, s.seed)
        ids = np.array([f"r{i:07d}" for i in range(s.n_rows)])
        table = RowTable(ids, made["columns"])
        fingerprint = {"synthetic": dataclasses.asdict(s)}
        kind = kind_of(s.generator)
        if kind == "tabular":
            return Data(kind, table, x=made["x"], feature_names=tuple(made["names"]), categorical=made["categorical"],
                        fingerprint=fingerprint)
        if kind == "graph":
            return Data(kind, table, x=made["x"], feature_names=tuple(made["names"]),
                        graph=Graph(made["edges"], s.n_rows), fingerprint=fingerprint)
        return Data(kind, table, items=ArrayItems(made["items"]), fingerprint=fingerprint)
