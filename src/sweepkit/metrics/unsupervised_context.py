"""What an unsupervised criterion reads. One context per run, built by the family's runner from the evaluated rows.

x           the evaluated rows after preparation, float [rows, features]
labels      cluster labels of the evaluated rows (-1 is noise or "no cluster"); clustering only
embedding   coordinates of the evaluated rows [rows, components]; embedding only
scores      anomaly scores of the evaluated rows (higher is more unusual); outlier only
model_info  numbers the model reports about itself (bic, aic, inertia, explained_variance, reconstruction_error)
sample      positions into x of the fixed sample the costly criteria use
options     the task's criteria_options
refit       clustering: (x_new) -> labels of the same configuration fitted on x_new (gap statistic)
score_new   outlier: (feature indices or None, the sampled rows restricted to them, x_new) -> anomaly scores of
            x_new from the run's model (features None) or from a model of the same configuration fitted on the fit
            rows restricted to those features (excess mass and mass volume)
reference   known labels of the evaluated rows; reveal metrics only (None elsewhere)
n_neighbors the embedding task's neighbourhood size
distances   when the data are a precomputed distance matrix: the distances between the evaluated rows (x then holds
            their distances to the fit rows, not coordinates). Criteria that need coordinates are then undefined;
            silhouette and the neighbour criteria use these distances.
"""
import dataclasses

import numpy as np


@dataclasses.dataclass
class UnsupervisedContext:
    x: np.ndarray
    labels: np.ndarray | None
    embedding: np.ndarray | None
    scores: np.ndarray | None
    model_info: dict
    sample: np.ndarray
    options: dict
    refit: object
    score_new: object
    reference: np.ndarray | None
    n_neighbors: int | None
    seed: int
    distances: np.ndarray | None = None


PRECOMPUTED = ("the data are a precomputed distance matrix and this criterion needs coordinates (silhouette works "
               "on distances)")


class Undefined(ValueError):
    """A criterion that has no value for this result (for example a silhouette with one cluster). The run records
    the reason and the value as missing; analyze ranks it last."""
