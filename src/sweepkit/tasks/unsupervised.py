"""Unsupervised tasks: clustering, embedding (dimensionality reduction) and outlier detection.

There is no validation metric without labels, so each task names label-free criteria computed on the evaluated rows
(criteria.json), and the one that chooses. "stability" is a choosing criterion too: the agreement (adjusted Rand
index) between the labels of the same configuration over seeds and subsamples, which analyze computes from the
assignment files. Criteria that cost O(rows^2) run on a fixed sample of the evaluated rows (sample_rows, chosen by
SHA-256 with sample_salt, so every configuration is scored on the same rows).

Agreement with known labels (a column named `reference`) is the test set of unsupervised work: each run writes it to
reveal.json, which analyze never opens, and reveal-report reads it only for a recorded selection.
"""
import dataclasses

from sweepkit.registry import METRICS, register_task
from sweepkit.settings import Scalar
from sweepkit.splits.base import UNSUPERVISED_ROLES

STABILITY = "stability"


def _check(task: str, s) -> None:
    if len(set(s.criteria)) != len(s.criteria):
        raise ValueError(f"task {task}: criteria repeat")
    needed = {}
    for name in s.criteria + s.reveal_metrics:
        entry = METRICS.get(name)
        for option, kind in entry["options"]:
            needed[option] = (kind, name)
    for name in s.criteria:
        entry = METRICS.get(name)
        if task not in entry["tasks"] or not entry["label_free"]:
            raise ValueError(f"task {task}: {name!r} is not a label-free criterion of {task}")
    if s.select != STABILITY and s.select not in s.criteria:
        raise ValueError(f"task {task}: select {s.select!r} must be one of its criteria {list(s.criteria)} or "
                         f"{STABILITY!r}")
    if s.select != STABILITY and METRICS.get(s.select)["descriptive"]:
        raise ValueError(f"task {task}: {s.select!r} describes a result but cannot choose one")
    missing = sorted(set(needed) - set(s.criteria_options))
    unknown = sorted(set(s.criteria_options) - set(needed))
    if missing or unknown:
        raise ValueError(f"task {task}: criteria_options missing {missing} (needed by "
                         f"{sorted({needed[m][1] for m in missing})}), unknown {unknown}")
    for option, (kind, name) in needed.items():
        value = s.criteria_options[option]
        ok = type(value) is kind or (kind is float and type(value) is int)
        if not ok:
            raise ValueError(f"task {task}: criteria_options.{option} (for {name}) must be a {kind.__name__}")
    if s.sample_rows is not None and s.sample_rows < 10:
        raise ValueError(f"task {task}: sample_rows is None (every evaluated row) or at least 10")
    for name in s.reveal_metrics:
        entry = METRICS.get(name)
        if task not in entry["tasks"] or entry["label_free"]:
            raise ValueError(f"task {task}: {name!r} is not an agreement-with-labels metric of {task}")
    if (s.reference is None) != (len(s.reveal_metrics) == 0):
        raise ValueError(f"task {task}: name a reference column together with reveal_metrics, or neither")


@dataclasses.dataclass(frozen=True)
class ClusteringSettings:
    criteria: tuple[str, ...]       # label-free criteria recorded, for example ["dbcv", "silhouette", "n_clusters"]
    select: str                     # the one that chooses (one of criteria, or "stability")
    sample_rows: int | None         # evaluated rows the costly criteria use (None: all)
    sample_salt: str                # fixes that sample
    reference: str | None           # a column of known labels, revealed only after the choice (None: no labels)
    reveal_metrics: tuple[str, ...]  # for example ["ari", "ami", "nmi"]; [] without a reference
    criteria_options: dict[str, Scalar]  # parameters the named criteria read, for example {"gap_references": 10};
                                        # {} when none needs any


@register_task("clustering", settings=ClusteringSettings, roles=UNSUPERVISED_ROLES,
               description="group rows without labels")
class Clustering:
    @staticmethod
    def check_job(job) -> None:
        _check("clustering", job.task)


@dataclasses.dataclass(frozen=True)
class EmbeddingSettings:
    criteria: tuple[str, ...]       # for example ["trustworthiness", "continuity", "knn_preservation"]
    select: str
    sample_rows: int | None
    sample_salt: str
    n_neighbors: int                # the neighbourhood the neighbour-based criteria look at
    reference: str | None
    reveal_metrics: tuple[str, ...]  # for example ["knn_label_agreement"]
    criteria_options: dict[str, Scalar]


@register_task("embedding", settings=EmbeddingSettings, roles=UNSUPERVISED_ROLES,
               description="reduce dimensions or learn a representation")
class Embedding:
    @staticmethod
    def check_job(job) -> None:
        _check("embedding", job.task)
        if job.task.n_neighbors < 1:
            raise ValueError("task embedding: n_neighbors at least 1")


@dataclasses.dataclass(frozen=True)
class OutlierSettings:
    criteria: tuple[str, ...]       # for example ["em_auc", "mv_auc", "flagged_fraction"]
    select: str
    sample_rows: int | None
    sample_salt: str
    reference: str | None           # a 0/1 column marking known outliers, revealed only after the choice
    reveal_metrics: tuple[str, ...]  # for example ["auroc", "average_precision"]
    criteria_options: dict[str, Scalar]  # em_auc and mv_auc read their parameters here (Goix 2016's values are
                                        # given in the example sweep)


@register_task("outlier", settings=OutlierSettings, roles=UNSUPERVISED_ROLES,
               description="score how unusual each row is")
class Outlier:
    @staticmethod
    def check_job(job) -> None:
        _check("outlier", job.task)
