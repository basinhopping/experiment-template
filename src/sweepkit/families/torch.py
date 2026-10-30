"""The torch family: one fit procedure ("torch") for every torch model (sweepkit.models.torch), on the supervised tasks
(classification, multi-label, regression), for tabular rows, sequences, images, audio through a feature, text (a
model that tokenizes, such as a fine-tuned pretrained encoder) and a graph's nodes (graph networks).

Per job:
  1. the fold's rows (each role cut to smoke_rows in smoke tests and estimates); tabular rows go through the
     preparation steps fitted on the training rows; sequence, image and audio items are read once per process and kept
     (sweepkit.train.batches.ItemCache), and text items too (TextCache); a graph's node features go through the
     preparation steps fitted on the training nodes and applied to every node, and every role's source shares the
     whole graph (GraphInput);
  2. the feature (sweepkit.features) and its normalization: statistics from the training rows (global), or from every
     row of the job (group: transductive, recorded so in prep_record.json);
  3. the targets as training sees them: shuffled under the leak-test control (stream "control shuffle"; every target
     the loss reads moves with its row's label), smoothed (label_smoothing), standardized with the training rows' mean
     and SD for regression; class weights or positive weights (weighted_loss); the auxiliary target;
  4. the model, built after torch.manual_seed from stream "weights", the auxiliary head after "aux head", the
     projection head after "contrastive head", and torch's generator seeded from "dropout" before training, so a head a
     configuration does not use changes none of its draws;
  5. the epochs (sweepkit.train.loop), each scored on the validation rows with every metric of the task; the best
     epoch by the task's select metric is reloaded and the validation rows predicted again, which must give that
     epoch's value (a NOTE in the log and reload_check.equal false otherwise);
  6. the test rows, predicted by the model the refit setting asks for: "train", the reloaded model; "train_and_val",
     a new model built from the same streams and trained on the training plus validation rows for the epoch count
     validation chose, on the job's own schedule (the learning rate follows the schedule of `epochs` epochs and stops
     at the chosen one), with its class weights and regression scale from those rows and the job's preparation and
     normalization (fitted on the training rows, as the estimator family's refit does). Validation predictions always
     come from the model trained on the training rows alone;
  7. each external dataset (fit.external), scored after the test set by the model that predicted the test rows, never
     trained or chosen on; its files are test-like and a choosing step never reads them.

Files of the run folder: predictions_<role>.csv and metrics_<role>.json for val and test (sweepkit.families
.supervised_io), and for train with eval_train (the best epoch's training rows, true labels); metrics_val.json also
holds the training record (every epoch's losses, learning rate, validation metrics and, with eval_train, training
metrics; the best epoch; the reload check); metrics_external_<name>.json and predictions_external_<name>.csv;
model_params.json (the model's settings, the input shape, parameter counts per part, the extra heads, the module tree,
the optimizer's settings with its own defaults, and model_record when the model has one: a pretrained model's
source, revision and weights' SHA-256); prep_record.json (preparation steps, feature, normalization);
efficiency.json (times ending in _seconds, steps per epoch, memory, spike counts of spiking models); provenance.json
(device requested and used, torch and CUDA versions).

Hooks other commands call: smoke_settings(settings) shortens a job for sweepkit.sweep.smoke, and describe(job, data)
gives the network as training builds it for describe-model.

Extending: TASK_RUNNERS maps a task to the function that runs it. The supervised tasks are here; a module that adds
another task to the torch family (an autoencoder's embedding, deep clustering) adds its runner to this mapping, and
the checks and hooks here apply to it too.
"""
import copy
import dataclasses
import math
from typing import Literal

import numpy as np

from sweepkit import devices
from sweepkit.families.common import (Clock, data_key, folds_of, limit_roles, load_data, peak_memory_mb, prepare,
                                      require_finite, run_provenance)
from sweepkit.families.supervised_io import class_names, predictions_csv, role_files, task_targets
from sweepkit.hashing import canonical_json, sha256_text
from sweepkit.registry import AUGMENTS, DATASETS, FEATURES, METRICS, MODELS, TASKS, register_fit
from sweepkit.settings import section_type, to_plain
from sweepkit.streams import generator, int_seed
from sweepkit.tasks.supervised import Outputs, by_group
from sweepkit.train.inputs import dataset_kind

SUPERVISED = ("classification", "multilabel", "regression")
ITEM_KINDS = ("sequence", "image", "audio")
TEXT_KINDS = ("text",)


@dataclasses.dataclass(frozen=True)
class WindowSettings:
    window: int                   # steps (sequences) or samples (audio, before the feature) per window
    hop: int                      # steps or samples between window starts, 1 to window


@dataclasses.dataclass(frozen=True)
class AuxSettings:
    mode: Literal["multitask", "adversarial"]
    # multitask: the shared layers learn the auxiliary target too; adversarial: a gradient reversal between the
    # model's summary and the auxiliary head pushes the shared layers to drop it (sweepkit.models.torch.heads)
    task: Literal["classification", "regression"]
    columns: tuple[str, ...]      # classification: one column of class names; regression: one or more numeric columns
    classes: tuple[str, ...]      # classification: every class of the column, in output order; regression: empty
    weight: float                 # the auxiliary loss's weight in the total loss, above 0
    loss: Literal["cross_entropy", "mse", "ccc"]   # cross_entropy for classification; mse or ccc (on standardized values)


@dataclasses.dataclass(frozen=True)
class ContrastiveSettings:
    weight: float                 # weight of the supervised contrastive loss in the total loss, above 0
    temperature: float            # the loss's temperature, above 0
    dim: int                      # outputs of the projection head


@dataclasses.dataclass(frozen=True)
class ExternalSettings:
    name: str                     # names the files: metrics_external_<name>.json, predictions_external_<name>.csv
    data: section_type("dataset", ())   # a registered dataset with its settings, of the job's kind and target columns
    restrict_classes: bool
    # classification only. True: the dataset may hold fewer of the task's classes; each row gets the most probable of
    # the classes the dataset holds (the others left out, the rest renormalized) and every metric is over those
    # classes; strict_accuracy counts the model's own choice over all classes. False: the classes as the task lists.


@dataclasses.dataclass(frozen=True)
class TorchFitSettings:
    device: Literal["cpu", "cuda", "mps", "auto"]
    epochs: int                   # at most this many passes over the training rows
    batch_size: int               # training rows per step; the last incomplete batch is dropped
    eval_batch_size: int          # rows (or windows) per batch when predicting
    lr: float                     # peak learning rate (AdamW)
    weight_decay: float           # AdamW's decoupled weight decay
    warmup_epochs: int            # linear warmup to lr over this many epochs, below epochs
    schedule: Literal["cosine", "constant"]   # after warmup: cosine decay to 0 at the last epoch, or constant
    early_stopping_patience: int | None   # stop after this many epochs without a better validation value; None: never
    precision: Literal["bf16", "fp32"]    # bf16: the network under bfloat16 autocast (losses in float32)
    class_balance: Literal["none", "weighted_loss", "balanced_sampler"]
    # weighted_loss: classification weights each class by n / (classes x its rows), multi-label weights each label's
    # positives by its negatives / positives; balanced_sampler (classification): each epoch draws n rows with
    # replacement, every class equally likely
    label_smoothing: float        # classification: (1 - e) one-hot + e / classes; multi-label: y (1 - e) + e / 2; 0
    regression_loss: Literal["none", "mse", "l1", "ccc"]   # regression: on standardized targets; none for other tasks
    control: Literal["none", "shuffle_train_labels"]       # shuffle_train_labels: the leak test, expect chance
    eval_train: bool              # also score the training rows after every epoch (over- and underfitting)
    crop: int | None              # sequences and audio: training rows cut to this many steps or samples at a random
                                  # start (shorter ones padded); None: whole rows, padded to the longest of the batch
    eval_window: WindowSettings | None   # sequences and audio: evaluate on windows and average each row's outputs;
                                         # None: whole rows
    augment: tuple[section_type("augment", ()), ...]   # training-batch augmentations (sweepkit.train.augment)
    aux: AuxSettings | None       # an auxiliary target on the model's summary, or None
    contrastive: ContrastiveSettings | None   # a supervised contrastive loss through a projection head, or None
    external: tuple[ExternalSettings, ...]    # datasets scored after the test set (may be empty)
    max_batches_per_epoch: int | None   # None in real runs; smoke tests and estimates set it
    refit: Literal["train", "train_and_val"]
    # the model that predicts the test rows: the best epoch of the model trained on the training rows, or a new model
    # trained on the training plus validation rows for the epoch count validation chose (step 6 above)


# Task runners (see "Extending" above): task name -> run(job, spec, offered, log) -> files.
TASK_RUNNERS = {}


# Checks -----------------------------------------------------------------------------------------------------------

def _check_numbers(s: TorchFitSettings) -> None:
    if s.epochs < 1 or s.batch_size < 2 or s.eval_batch_size < 1:
        raise ValueError("fit torch: epochs >= 1, batch_size >= 2 and eval_batch_size >= 1 are required")
    if not s.lr > 0 or s.weight_decay < 0:
        raise ValueError("fit torch: lr must be above 0 and weight_decay not negative")
    if not 0 <= s.warmup_epochs < s.epochs:
        raise ValueError("fit torch: warmup_epochs must be at least 0 and below epochs")
    if s.early_stopping_patience is not None and s.early_stopping_patience < 1:
        raise ValueError("fit torch: early_stopping_patience must be at least 1, or None")
    if not 0.0 <= s.label_smoothing < 1.0:
        raise ValueError("fit torch: label_smoothing must be in [0, 1)")
    if s.max_batches_per_epoch is not None and s.max_batches_per_epoch < 1:
        raise ValueError("fit torch: max_batches_per_epoch must be at least 1, or None")
    if s.crop is not None and s.crop < 1:
        raise ValueError("fit torch: crop must be at least 1, or None")
    if s.eval_window is not None and not (s.eval_window.window >= 1 and 1 <= s.eval_window.hop <= s.eval_window.window):
        raise ValueError("fit torch: eval_window needs window >= 1 and hop from 1 to window")


def _check_task(job) -> None:
    s, task = job.fit, job.task_name
    if task not in TASK_RUNNERS:
        raise ValueError(f"fit torch runs tasks {sorted(TASK_RUNNERS)}, not {task!r}")
    if task == "regression":
        if s.regression_loss == "none":
            raise ValueError("fit torch: regression needs regression_loss mse, l1 or ccc")
        if s.class_balance != "none" or s.label_smoothing != 0:
            raise ValueError("fit torch: regression takes class_balance none and label_smoothing 0")
    elif s.regression_loss != "none":
        raise ValueError(f"fit torch: regression_loss is for regression; task {task} takes none")
    if s.class_balance == "balanced_sampler" and task != "classification":
        raise ValueError("fit torch: the balanced sampler draws by class; it is for classification")


def _target_columns(job) -> tuple:
    task = job.task
    return tuple(task.targets) if job.task_name == "multilabel" else (task.target,)


def _check_aux(job) -> None:
    a = job.fit.aux
    if a is None:
        return
    if not a.weight > 0:
        raise ValueError("fit torch: aux.weight must be above 0")
    if a.task == "classification":
        if a.loss != "cross_entropy" or len(a.columns) != 1:
            raise ValueError("fit torch: a classification aux target names one column and loss cross_entropy")
        if len(a.classes) < 2 or len(set(a.classes)) != len(a.classes):
            raise ValueError("fit torch: aux.classes must list at least two distinct classes")
    else:
        if a.loss not in ("mse", "ccc") or not a.columns or a.classes:
            raise ValueError("fit torch: a regression aux target names one or more columns, classes [] and loss mse "
                             "or ccc")
        if len(set(a.columns)) != len(a.columns):
            raise ValueError("fit torch: aux.columns repeat")
    shared = sorted(set(a.columns) & set(_target_columns(job)))
    if shared:
        raise ValueError(f"fit torch: aux column {shared} is the task's own target")


def _check_augment(job, kind: str, model_kind: str) -> None:
    s = job.fit
    names = [a.name for a in s.augment]
    if len(set(names)) != len(names):
        raise ValueError(f"fit torch: each augmentation at most once, got {names}")
    for a in s.augment:
        entry = AUGMENTS.get(a.name)
        entry.target.check(a.settings)
        # an augmentation of order 0 acts on the dataset's rows before the feature, the others on the model's input
        where = kind if entry.target.order == 0 else model_kind
        if where not in entry.meta["kinds"]:
            raise ValueError(f"fit torch: augmentation {a.name} works on {list(entry.meta['kinds'])}, not {where} "
                             "input")
    if s.contrastive is not None and "mixup" in names:
        raise ValueError("fit torch: the contrastive loss pairs rows by label, which mixup blends; use one or the other")


def _check_contrastive(job) -> None:
    c = job.fit.contrastive
    if c is None:
        return
    if job.task_name != "classification":
        raise ValueError("fit torch: the supervised contrastive loss pairs rows by class; it is for classification")
    if not c.weight > 0 or not c.temperature > 0 or c.dim < 1:
        raise ValueError("fit torch: contrastive needs weight and temperature above 0 and dim of at least 1")


def _check_external(job, kind: str) -> None:
    s = job.fit
    seen = set()
    for e in s.external:
        name = e.name
        if not name or not name.replace("_", "").isalnum() or name != name.lower() or name[0].isdigit():
            raise ValueError(f"fit torch: external name {name!r}: use lowercase letters, digits and _, starting with a "
                             "letter")
        if name in seen:
            raise ValueError(f"fit torch: external name {name!r} repeats")
        seen.add(name)
        if e.restrict_classes and job.task_name != "classification":
            raise ValueError("fit torch: restrict_classes is for classification")
        other = dataclasses.replace(job, data_name=e.data.name, data=e.data.settings)
        if dataset_kind(other) != kind:
            raise ValueError(f"fit torch: external {name} is {dataset_kind(other)} data; the job's is {kind}")
        for registry, component in ((DATASETS, e.data.name), (FEATURES, job.features_name)):
            hook = getattr(registry.get(component).target, "check_job", None)
            if hook is not None:
                hook(other)
    if s.external and kind == "graph":
        raise ValueError("fit torch: external datasets are scored row by row; a graph's nodes are scored in its own "
                         "folds")
    if s.external:
        if kind == "tabular" and job.prep:
            raise ValueError("fit torch: external datasets of a tabular job need an empty prep list (the steps are "
                             "fitted on the job's own rows)")
        if getattr(job.features, "normalization", "none") == "group":
            raise ValueError("fit torch: group normalization has no statistics for an external dataset's groups")


# Data -------------------------------------------------------------------------------------------------------------

_ITEMS = {}
_TEXTS = {}
_EXTERNAL_DATA = {}


def _cached(cache: dict, key: str, make, keep: int):
    if key not in cache:
        if len(cache) >= keep:
            cache.pop(next(iter(cache)))
        cache[key] = make()
    return cache[key]


def item_cache(key: str, data):
    from sweepkit.train.batches import ItemCache

    return _cached(_ITEMS, key, lambda: ItemCache(data.items, data.kind), 4)


def text_cache(key: str, data):
    from sweepkit.train.batches import TextCache

    return _cached(_TEXTS, key, lambda: TextCache(data.items), 4)


def _external_data(section, log):
    key = sha256_text(canonical_json(to_plain(section)))

    def load():
        data = DATASETS.get(section.name).target().load(section.settings)
        log(f"external data {section.name}: {len(data.table)} rows, kind {data.kind}")
        return data

    return key, _cached(_EXTERNAL_DATA, key, load, 2)


def _sources(job, data, positions: dict, seed: int, device: str, log) -> tuple:
    """({role: Source}, positions after the preparation steps, column names, preparation records)."""
    from sweepkit.train.batches import Source

    if data.kind == "tabular":
        x, positions, columns, categorical, records = prepare(job, data, positions, "train", seed, device, log)
        for role in ("train", "val", "test"):
            require_finite(x[role], f"{role} rows")
        if categorical:
            log(f"  NOTE columns {sorted(categorical)} hold category codes; the network reads them as numbers (a "
                "one-hot step in the prep list turns them into indicator columns)")
        return ({role: Source("tabular", positions[role], table_x=x[role]) for role in ("train", "val", "test")},
                positions, list(columns), records)
    if data.kind == "graph":
        return _graph_sources(job, data, positions, seed, device, log)
    if data.kind in TEXT_KINDS:
        texts = text_cache(f"data {data_key(job)}", data)
        sources = {role: Source("text", positions[role], items=texts) for role in ("train", "val", "test")}
        return sources, positions, [], []
    if data.kind not in ITEM_KINDS:
        raise ValueError(f"the torch family has no batches for {data.kind} data (sweepkit.train.batches)")
    items = item_cache(f"data {data_key(job)}", data)
    return {role: Source(data.kind, positions[role], items=items) for role in ("train", "val", "test")}, positions, [], []


def _graph_sources(job, data, positions: dict, seed: int, device: str, log) -> tuple:
    """Every role's source over one GraphInput: the node features after the preparation steps fitted on the training
    nodes (every node transformed, whatever its role), and the graph's edges."""
    from sweepkit.train.batches import GraphInput, Source

    roles = dict(positions)
    roles["nodes"] = np.arange(len(data.table), dtype=np.int64)
    x, roles, columns, categorical, records = prepare(job, data, roles, "train", seed, device, log)
    require_finite(x["nodes"], "node features")
    if categorical:
        log(f"  NOTE columns {sorted(categorical)} hold category codes; the network reads them as numbers (a "
            "one-hot step in the prep list turns them into indicator columns)")
    graph = GraphInput(x["nodes"], data.graph.edges)
    log(f"  graph: {graph.n_nodes} nodes, {graph.edges.shape[1]} directed edges without repeats or self loops")
    kept = {role: roles[role] for role in ("train", "val", "test")}
    return {role: Source("graph", kept[role], graph=graph) for role in kept}, kept, list(columns), records


def _feature(job, device):
    from sweepkit.features.sequence import Identity

    if job.features_name == "none":
        return Identity()
    return FEATURES.get(job.features_name).target(job.features, device)


def _normalizer(job, data, sources: dict, positions: dict, feature, device):
    """The normalization of the feature's output: statistics of the training rows (global), or of every row of the
    job, labels unused (group)."""
    from sweepkit.features.normalize import Normalizer, group_index, statistics
    from sweepkit.train.batches import Source

    mode = feature.normalization
    batch = job.fit.eval_batch_size
    if mode == "global":
        mean, std = statistics(sources["train"], feature, batch, device)
        return Normalizer("global", mean, std)
    if mode == "group":
        names, group_of = group_index(data.table, feature.group_column)
        rows = np.unique(np.concatenate([positions[r] for r in ("train", "val", "test")]))
        everyone = Source(data.kind, rows, items=sources["train"].items)
        mean, std = statistics(everyone, feature, batch, device, group_of, len(names))
        return Normalizer("group", mean, std, group_of)
    return Normalizer(mode)


def _input_shape(source, feature) -> tuple:
    """One row's input as the model reads it, after the feature."""
    if source.kind == "tabular":
        raw = (int(source.table_x.shape[1]),)
    elif source.kind == "graph":
        raw = (int(source.graph.x.shape[1]),)
    elif source.kind == "text":
        raw = ()
    else:
        # one kind at a time: an audio clip is one-dimensional, so a sequence's shape[1] does not exist for it
        first = source.item(0)
        if source.kind == "sequence":
            raw = (int(first.shape[1]),)
        elif source.kind == "image":
            raw = tuple(int(v) for v in first.shape)
        elif source.kind == "audio":
            raw = (int(first.shape[0]),)
        else:
            raise ValueError(f"no input shape for items of kind {source.kind}")
    return tuple(int(v) for v in feature.out_shape(raw))


# Targets ----------------------------------------------------------------------------------------------------------

def _n_outputs(job) -> int:
    return 1 if job.task_name == "regression" else len(class_names(job))


def _aux_values(job, table, rows: np.ndarray) -> np.ndarray:
    """The auxiliary target of the given rows: class indices [rows] or values [rows, columns], as the data hold them."""
    a = job.fit.aux
    if a.task == "classification":
        raw = table.column(a.columns[0]).astype(str)[rows]
        index = {c: i for i, c in enumerate(a.classes)}
        unknown = sorted(set(raw.tolist()) - set(index))
        if unknown:
            raise ValueError(f"fit torch: aux column {a.columns[0]!r} holds {unknown[:5]}, not in aux.classes")
        return np.array([index[v] for v in raw.tolist()], dtype=np.int64)
    values = np.stack([np.asarray(table.column(c), dtype=np.float64)[rows] for c in a.columns], axis=1)
    if not np.isfinite(values).all():
        raise ValueError(f"fit torch: aux columns {list(a.columns)} hold missing or infinite values")
    return values


@dataclasses.dataclass
class Prepared:
    """The training targets of one training run and what was fitted to them."""

    targets: object                 # sweepkit.train.loop.Targets
    raw: np.ndarray                 # the task's targets of the training rows as training sees them (shuffled under the
                                    # control): class indices, 0/1 labels or values
    scale: tuple | None             # regression: (mean, SD) of the training targets
    aux_scale: tuple | None         # a regression aux target: (means, SDs) of its columns


def _prepare_targets(job, table, y_all: np.ndarray, rows: np.ndarray, permutation) -> Prepared:
    from sweepkit.train.loop import Targets

    s = job.fit
    e = s.label_smoothing
    y = y_all[rows]
    if permutation is not None:
        y = y[permutation]
    scale = None
    labels = None
    if job.task_name == "classification":
        names = class_names(job)
        missing = sorted(set(range(len(names))) - set(np.unique(y).tolist()))
        if missing:
            raise ValueError(f"the training rows hold no row of classes {[names[c] for c in missing]}")
        labels = y.astype(np.int64)
        main = np.eye(len(names), dtype=np.float64)[labels]
        if e > 0:
            main = (1.0 - e) * main + e / len(names)
    elif job.task_name == "multilabel":
        main = y.astype(np.float64)
        if e > 0:
            main = main * (1.0 - e) + e / 2.0
    else:
        values = y.astype(np.float64)
        mean, sd = float(values.mean()), float(values.std())
        if not sd > 0:
            raise ValueError("the training targets are all the same value; a constant cannot be learned")
        main = ((values - mean) / sd)[:, None]
        scale = (mean, sd)
    aux = aux_scale = None
    if s.aux is not None:
        values = _aux_values(job, table, rows)
        if permutation is not None:
            values = values[permutation]
        if s.aux.task == "classification":
            aux = np.eye(len(s.aux.classes), dtype=np.float32)[values]
        else:
            mean, sd = values.mean(axis=0), values.std(axis=0)
            if not (sd > 0).all():
                raise ValueError(f"fit torch: aux columns {list(s.aux.columns)} are constant in the training rows")
            aux = ((values - mean) / sd).astype(np.float32)
            aux_scale = (mean, sd)
    targets = Targets(np.ascontiguousarray(main, dtype=np.float32), labels, aux)
    return Prepared(targets, y, scale, aux_scale)


def _losses(job, prepared: Prepared, device):
    import torch

    from sweepkit.train.batches import class_weights, positive_weights
    from sweepkit.train.loop import Losses

    s = job.fit
    weights = pos_weight = None
    if s.class_balance == "weighted_loss":
        if job.task_name == "classification":
            weights = torch.from_numpy(class_weights(prepared.raw.astype(np.int64), _n_outputs(job))).to(device)
        elif job.task_name == "multilabel":
            pos_weight = torch.from_numpy(positive_weights(prepared.raw.astype(np.int64))).to(device)
    a, c = s.aux, s.contrastive
    return Losses(job.task_name, weights, pos_weight, s.regression_loss,
                  a.task if a is not None else None, a.loss if a is not None else None,
                  a.weight if a is not None else 0.0,
                  c.weight if c is not None else 0.0, c.temperature if c is not None else None)


# The network ------------------------------------------------------------------------------------------------------

def _aux_outputs(job) -> int:
    a = job.fit.aux
    return len(a.classes) if a.task == "classification" else len(a.columns)


def build(job, input_shape: tuple, seed: int, device):
    """(model, auxiliary head or None, projection head or None) on the device, each from its own stream; torch's
    generator is seeded from stream "dropout" afterwards. The refit calls this again and gets the same weights."""
    import torch

    from sweepkit.models.torch.heads import build_aux_head, build_projection_head

    s = job.fit
    torch.manual_seed(int_seed(seed, "weights"))
    model = MODELS.get(job.model_name).target.build(job.model, input_shape, _n_outputs(job)).to(device)
    aux_head = contrastive_head = None
    if s.aux is not None:
        torch.manual_seed(int_seed(seed, "aux head"))
        aux_head = build_aux_head(s.aux.mode, model.embedding_dim, _aux_outputs(job), model.head_dropout).to(device)
    if s.contrastive is not None:
        torch.manual_seed(int_seed(seed, "contrastive head"))
        contrastive_head = build_projection_head(model.embedding_dim, s.contrastive.dim).to(device)
    torch.manual_seed(int_seed(seed, "dropout"))
    return model, aux_head, contrastive_head


def _augments(job) -> dict:
    return {a.name: a.settings for a in job.fit.augment}


def _train_once(job, source, prepared: Prepared, input_shape, feature, normalizer, seed: int, device, epochs: int,
                log, val=None, select=None, train_watch=None, show_val=True) -> tuple:
    """Build the network and train it (sweepkit.train.loop.run_epochs). Returns (network, fit)."""
    from sweepkit.train.loop import Network, Step, Streams, run_epochs

    s = job.fit
    model, aux_head, contrastive_head = build(job, input_shape, seed, device)
    network = Network(model, aux_head, contrastive_head, feature, normalizer, device, s.precision, job.task_name,
                      s.aux.task if s.aux is not None else None)
    log(f"  network: {sum(p.numel() for p in model.parameters())} parameters"
        + "".join(f", {name} head {sum(p.numel() for p in head.parameters())}"
                  for name, head in (("aux", aux_head), ("projection", contrastive_head)) if head is not None))
    step = Step(network, _augments(job), _losses(job, prepared, device))
    fit = run_epochs(network, step, source, prepared.targets, s, epochs, s.epochs, Streams(seed), log, val, select,
                     train_watch, show_val)
    return network, fit


# Outputs and reports ----------------------------------------------------------------------------------------------

def outputs_of(task: str, predictions, y_true: np.ndarray, names: tuple, scale) -> Outputs:
    """A role's Outputs from its predictions: the most probable class, labels at probability 0.5, or the
    standardized value mapped back with the training targets' mean and SD."""
    p = predictions.main
    if task == "classification":
        return Outputs(y_true, p.argmax(axis=1).astype(np.int64), p, "probability", names)
    if task == "multilabel":
        return Outputs(y_true, (p >= 0.5).astype(np.int64), p, "probability", names)
    mean, sd = scale
    return Outputs(y_true, p[:, 0] * sd + mean, None, "none", ())


def metric_values(outputs: Outputs, metrics: tuple) -> dict:
    """Each metric's value, or None where it is undefined on these rows (a class without rows, a constant)."""
    out = {}
    for name in metrics:
        try:
            value = float(METRICS.get(name)["fn"](outputs, None))
        except ValueError:
            value = None
        out[name] = value if value is None or math.isfinite(value) else None
    return out


def aux_report(aux: AuxSettings, true: np.ndarray, predicted: np.ndarray, aux_scale) -> dict:
    """How well the auxiliary head predicts its target on a role's rows. Classification: accuracy, recall per class
    (None for a class without rows) and their mean over the classes present. Regression: per column the concordance
    correlation coefficient, Pearson's r and the RMSE in the column's own units, and the mean coefficient."""
    from sweepkit.train.losses import concordance_numpy

    if aux.task == "classification":
        chosen = predicted.argmax(axis=1)
        recall = {}
        for k, name in enumerate(aux.classes):
            members = true == k
            recall[name] = float(np.mean(chosen[members] == k)) if members.any() else None
        present = [r for r in recall.values() if r is not None]
        return {"task": "classification", "column": aux.columns[0], "mode": aux.mode, "n": int(true.shape[0]),
                "accuracy": float(np.mean(chosen == true)), "recall": recall,
                "mean_recall_present": float(np.mean(present)) if present else None}
    mean, sd = aux_scale
    values = predicted * sd + mean
    ccc = concordance_numpy(values, true)
    out = {"task": "regression", "columns": list(aux.columns), "mode": aux.mode, "n": int(true.shape[0]),
           "train_mean": [float(v) for v in mean], "train_sd": [float(v) for v in sd]}
    for j, column in enumerate(aux.columns):
        a, b = true[:, j], values[:, j]
        out[f"ccc_{column}"] = float(ccc[j])
        out[f"pearson_{column}"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        out[f"rmse_{column}"] = float(np.sqrt(np.mean((b - a) ** 2)))
    out["mean_ccc"] = float(np.mean(ccc))
    return out


def _external_files(job, ext: ExternalSettings, network, scale, spec: dict, main_data, log) -> tuple:
    """(files, seconds) of one external dataset, predicted by the model that predicted the test rows."""
    from sweepkit.train.batches import Source

    s = job.fit
    names = class_names(job)
    key, data = _external_data(ext.data, log)
    table = data.table
    rows = limit_roles({"external": np.arange(len(table), dtype=np.int64)}, table.ids, spec)["external"]
    y_true = task_targets(job, table)[rows]
    if data.kind == "tabular":
        if tuple(data.feature_names) != tuple(main_data.feature_names):
            raise ValueError(f"external {ext.name}: its columns differ from the job's dataset's")
        x = np.asarray(data.x[rows], dtype=np.float32)
        require_finite(x, f"external {ext.name} rows")
        source = Source("tabular", rows, table_x=x)
    elif data.kind in TEXT_KINDS:
        source = Source("text", rows, items=text_cache(f"external {key}", data))
    else:
        source = Source(data.kind, rows, items=item_cache(f"external {key}", data))
    pred = network.predict(source, s.eval_batch_size, s.eval_window)
    outputs = outputs_of(job.task_name, pred, y_true, names, scale)
    extra = {"dataset": {"name": ext.data.name, "fingerprint": data.fingerprint}, "restrict_classes": ext.restrict_classes,
             "seconds": round(pred.seconds, 3)}
    if ext.restrict_classes:
        present = sorted(set(y_true.tolist()))
        if len(present) < 2:
            raise ValueError(f"external {ext.name}: fewer than two of the task's classes")
        shared = pred.main[:, present]
        shared = shared / shared.sum(axis=1, keepdims=True)
        index = {c: i for i, c in enumerate(present)}
        outputs = Outputs(np.array([index[v] for v in y_true.tolist()], dtype=np.int64),
                          shared.argmax(axis=1).astype(np.int64), shared, "probability",
                          tuple(names[c] for c in present))
        extra["classes_scored"] = [names[c] for c in present]
        extra["strict_accuracy"] = float(np.mean(pred.main.argmax(axis=1) == y_true))
    task = TASKS.get(job.task_name).target
    groups = {c: table.column(c).astype(str)[rows] for c in job.task.groups if table.has(c)}
    try:
        detail = task.detail(outputs)
    except ValueError as err:
        detail = {"undefined_because": str(err)}
    body = {"role": f"external_{ext.name}", "n": int(rows.size), "metrics": metric_values(outputs, job.task.metrics),
            "detail": detail, "by_group": by_group(outputs, job.task.metrics, groups, task.detail), **extra}
    files = {f"metrics_external_{ext.name}.json": body,
             f"predictions_external_{ext.name}.csv": predictions_csv(job, table, rows, outputs, groups)}
    return files, pred.seconds


# The supervised runner ----------------------------------------------------------------------------------------------

def _run_supervised(job, spec: dict, offered: str, log) -> dict:
    import torch

    from sweepkit.models.torch.base import deterministic_cudnn, parameter_counts
    from sweepkit.train.augment import cached_bytes
    from sweepkit.train.batches import concat_sources
    from sweepkit.train.loop import Watch

    s = job.fit
    task = job.task_name
    seed = int(spec["seed"])
    data = load_data(job, log)
    folds, fold_hash = folds_of(job, data)
    fold = folds[spec["fold"]]
    device_name = devices.resolve(s.device, offered, MODELS.get(job.model_name).meta["gpu"])
    determinism = deterministic_cudnn(device_name)
    device = torch.device(device_name)
    names = class_names(job)
    y_all = task_targets(job, data.table)
    clock = Clock()
    clock.start()
    sources, positions, columns, records = _sources(job, data, limit_roles(fold.roles, data.table.ids, spec), seed,
                                                    device_name, log)
    feature = _feature(job, device)
    normalizer = _normalizer(job, data, sources, positions, feature, device)
    input_shape = _input_shape(sources["train"], feature)
    clock.stop("prep_seconds")
    permutation = None
    if s.control == "shuffle_train_labels":
        permutation = generator(seed, "control shuffle").permutation(positions["train"].size)
    prepared = _prepare_targets(job, data.table, y_all, positions["train"], permutation)
    select = (job.task.select, METRICS.get(job.task.select)["direction"])

    def watch(role: str):
        y_true = y_all[positions[role]]
        return Watch(sources[role], lambda pred: metric_values(outputs_of(task, pred, y_true, names, prepared.scale),
                                                                job.task.metrics))

    extras = [f"augment {', '.join(a.name for a in s.augment)}"] if s.augment else []
    if s.aux is not None:
        extras.append(f"{s.aux.mode} aux head on {', '.join(s.aux.columns)}, weight {s.aux.weight}")
    if s.contrastive is not None:
        extras.append(f"supervised contrastive loss, weight {s.contrastive.weight}, temperature "
                      f"{s.contrastive.temperature}, projection to {s.contrastive.dim}")
    if s.eval_window is not None:
        extras.append(f"evaluation on windows of {s.eval_window.window} every {s.eval_window.hop}")
    if s.control != "none":
        extras.append(f"control {s.control}")
    log(f"run {spec['run_id']}: fold {spec['fold']}, seed {seed}, train {positions['train'].size}, val "
        f"{positions['val'].size}, test {positions['test'].size}, input {list(input_shape)}, {job.model_name} on "
        f"{device_name}" + "".join(f", {e}" for e in extras))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    clock.start()
    # on the identity check fold the validation rows are test rows: no validation number goes into the log
    show_val = not fold.selection_on_test
    network, fit = _train_once(job, sources["train"], prepared, input_shape, feature, normalizer, seed, device,
                               s.epochs, log, watch("val"), select, watch("train") if s.eval_train else None, show_val)
    clock.stop("train_seconds")
    model = network.model
    counts = parameter_counts(model)
    log(f"  {fit.steps_per_epoch} steps per epoch, best epoch {fit.best_epoch} of {len(fit.curves)}"
        + (" (stopped early)" if fit.stopped_early else ""))

    # the best epoch, reloaded and checked on the validation rows
    clock.start()
    network.load(fit.best_state)
    spiking = hasattr(model, "spike_report") and getattr(model, "is_spiking", True)
    if spiking:
        model.record(True)
    val_pred = network.predict(sources["val"], s.eval_batch_size, s.eval_window, with_aux=True)
    spikes = None
    if spiking:
        model.record(False)
        spikes = model.spike_report()
        rates = ", ".join(f"{k} {v:.3f}" for k, v in spikes["firing_rate"].items() if v is not None)
        log(f"  spikes on validation: {rates}" + (f"; SILENT {spikes['silent_layers']}" if spikes["silent_layers"]
                                                  else ""))
    val_outputs = outputs_of(task, val_pred, y_all[positions["val"]], names, prepared.scale)
    reloaded = metric_values(val_outputs, (select[0],))[select[0]]
    equal = ((reloaded is None and fit.best_value is None) or
             (reloaded is not None and fit.best_value is not None and abs(reloaded - fit.best_value) <= 1e-9))
    if not equal:
        log(f"  NOTE the best epoch's weights reloaded give validation {select[0]} {reloaded}, recorded "
            f"{fit.best_value}" if show_val else "  NOTE the best epoch's weights reloaded do not give the recorded "
            "validation value (numbers not shown on this fold)")
    training = {"select": select[0], "direction": select[1], "best_epoch": fit.best_epoch, "best_value": fit.best_value,
                "epochs_run": len(fit.curves), "stopped_early": fit.stopped_early,
                "steps_per_epoch": fit.steps_per_epoch, "curves": fit.curves,
                "reload_check": {"metric": select[0], "best_epoch_value": fit.best_value,
                                 "reloaded_value": reloaded, "equal": equal}}
    val_extra = {"training": training}
    if s.aux is not None:
        val_extra["aux"] = aux_report(s.aux, _aux_values(job, data.table, positions["val"]), val_pred.aux,
                                      prepared.aux_scale)
    files = role_files(job, "val", data.table, positions["val"], val_outputs, val_extra)
    if s.eval_train:
        train_outputs = outputs_of(task, fit.best_train, y_all[positions["train"]], names, prepared.scale)
        files.update(role_files(job, "train", data.table, positions["train"], train_outputs,
                                {"epoch": fit.best_epoch}))
    clock.stop("predict_seconds")

    # the model that predicts the test rows
    final, final_prepared, refit_record = network, prepared, {"refit": s.refit}
    if s.refit == "train_and_val":
        clock.start()
        rows = np.concatenate([positions["train"], positions["val"]])
        refit_permutation = None
        if s.control == "shuffle_train_labels":
            # the control's refit sees no true label either
            refit_permutation = generator(seed, "control shuffle refit").permutation(rows.size)
        final_prepared = _prepare_targets(job, data.table, y_all, rows, refit_permutation)
        log(f"  refit on train and val: {rows.size} rows, {fit.best_epoch} epochs")
        final, refit_fit = _train_once(job, concat_sources(sources["train"], sources["val"]), final_prepared,
                                       input_shape, feature, normalizer, seed, device, fit.best_epoch, log)
        refit_record.update({"refit_epochs": fit.best_epoch, "refit_rows": int(rows.size),
                             "refit_steps_per_epoch": refit_fit.steps_per_epoch})
        clock.stop("refit_seconds")
    clock.start()
    test_pred = final.predict(sources["test"], s.eval_batch_size, s.eval_window, with_aux=True)
    test_extra = dict(refit_record)
    if s.aux is not None:
        test_extra["aux"] = aux_report(s.aux, _aux_values(job, data.table, positions["test"]), test_pred.aux,
                                       final_prepared.aux_scale)
    files.update(role_files(job, "test", data.table, positions["test"],
                            outputs_of(task, test_pred, y_all[positions["test"]], names, final_prepared.scale),
                            test_extra))
    clock.stop("predict_seconds")
    peak_gpu = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None

    # external datasets, after every number above
    for ext in s.external:
        clock.start()
        ext_files, _ = _external_files(job, ext, final, final_prepared.scale, spec, data, log)
        files.update(ext_files)
        clock.stop("external_seconds")
        log(f"  external {ext.name}: {ext_files[f'metrics_external_{ext.name}.json']['n']} rows scored "
            f"(metrics_external_{ext.name}.json), never trained or chosen on")

    val_select = files["metrics_val.json"]["select"]
    if show_val:
        shown = "undefined" if val_select["value"] is None else f"{val_select['value']:.4f}"
        log(f"  validation {val_select['metric']} {shown}")
    else:
        log("  the identity check fold: its validation numbers are test numbers, so none is shown")
    heads = {"aux": None, "contrastive": None}
    for name, head in (("aux", network.aux_head), ("contrastive", network.contrastive_head)):
        if head is not None:
            heads[name] = int(sum(p.numel() for p in head.parameters()))
    files["model_params.json"] = {"model": job.model_name, "settings": to_plain(job.model),
                                  "input_shape": list(input_shape), "n_outputs": _n_outputs(job),
                                  "parameters_total": counts["total"], "parameters_trainable": counts["trainable"],
                                  "parameters_by_part": counts["by_part"], "extra_heads": heads,
                                  "optimizer": fit.optimizer, "module_tree": str(model)}
    model_record = getattr(model, "model_record", None)
    if callable(model_record):
        files["model_params.json"]["model_record"] = model_record()
    files["prep_record.json"] = {"steps": records, "columns": columns,
                                 "feature": {"name": job.features_name, **to_plain(job.features)},
                                 "normalization": normalizer.record(), "input_shape": list(input_shape)}
    efficiency = {**clock.parts, "n_train": int(positions["train"].size), "input_shape": list(input_shape),
                  "parameters_total": counts["total"], "steps_per_epoch": fit.steps_per_epoch,
                  "epochs_run": len(fit.curves), "seconds_per_epoch": round(fit.seconds / max(1, len(fit.curves)), 3),
                  "ms_per_test_row": round(1000 * test_pred.seconds / max(1, positions["test"].size), 3),
                  "peak_memory_mb": peak_memory_mb(),
                  "peak_gpu_mib": round(peak_gpu, 1) if peak_gpu is not None else None}
    if data.kind in ITEM_KINDS or data.kind in TEXT_KINDS:
        efficiency["item_cache_mib_in_process"] = round(sources["train"].items.nbytes() / 2 ** 20, 1)
    if "speed" in _augments(job):
        efficiency["speed_cache_mib_in_process"] = round(cached_bytes() / 2 ** 20, 1)
    if spikes is not None:
        efficiency["spikes_validation"] = spikes
    files["efficiency.json"] = efficiency
    provenance = run_provenance(job, spec, data, fold_hash, s.device, device_name, [])
    provenance.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                       "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else device_name,
                       "precision": s.precision, **determinism,
                       "external": [{"name": e.name, "dataset": e.data.name} for e in s.external]})
    files["provenance.json"] = provenance
    return files


TASK_RUNNERS.update({task: _run_supervised for task in SUPERVISED})


# The fit procedure ------------------------------------------------------------------------------------------------

@register_fit("torch", settings=TorchFitSettings, family="torch",
              description="train a torch model on the training rows; the best validation epoch is kept")
class TorchFit:
    @staticmethod
    def check_job(job) -> None:
        from sweepkit.runs.spec import input_kind

        _check_numbers(job.fit)
        _check_task(job)
        kind = dataset_kind(job)
        if (job.fit.crop is not None or job.fit.eval_window is not None) and kind not in ("sequence", "audio"):
            raise ValueError(f"fit torch: crop and eval_window cut sequences and audio; {kind} data take None")
        _check_augment(job, kind, input_kind(job))
        _check_aux(job)
        _check_contrastive(job)
        _check_external(job, kind)

    @staticmethod
    def run(job, spec: dict, offered: str, log) -> dict:
        return TASK_RUNNERS[job.task_name](job, spec, offered, log)

    @staticmethod
    def smoke_settings(settings: dict) -> dict:
        """The job as smoke runs it: two epochs of at most five batches each, one of them warmup."""
        out = copy.deepcopy(settings)
        fit = out["fit"]
        limit = fit.get("max_batches_per_epoch")
        fit.update({"epochs": 2, "warmup_epochs": 1, "max_batches_per_epoch": 5 if limit is None else min(limit, 5)})
        return out

    @staticmethod
    def describe(job, data) -> dict:
        return describe(job, data)


# describe-model ---------------------------------------------------------------------------------------------------

def _shape_of(value):
    import torch

    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape_of(v) for v in value]
    if value is None:
        return None
    return type(value).__name__


def describe(job, data) -> dict:
    """The network as training builds it, on the CPU, in eval mode, with no training: the module tree, every leaf
    layer's input and output shapes on one row of the dataset (the first row; for tabular data with preparation steps,
    the first training row of the first fold after the steps fitted on that fold's training rows), the number of calls
    of each leaf (a spiking layer runs once per step), and parameters per part."""
    import torch

    from sweepkit.models.torch.base import parameter_counts
    from sweepkit.train.batches import Source, eval_batches
    from sweepkit.train.loop import to_device

    device = torch.device("cpu")
    feature = _feature(job, device)
    if data.kind == "tabular":
        if job.prep:
            folds, _ = folds_of(job, data)
            fold = next(iter(folds.values()))
            x, _, _, _, _ = prepare(job, data, dict(fold.roles), "train", 1, "cpu", lambda m: None)
            row = x["train"][:1]
            text = (f"{row.shape[1]} columns after the preparation steps fitted on fold {fold.name}'s training rows, "
                    "its first training row")
        else:
            row = np.asarray(data.x[:1], dtype=np.float32)
            text = f"{row.shape[1]} columns, the dataset's first row"
        source = Source("tabular", np.zeros(1, dtype=np.int64), table_x=np.asarray(row, dtype=np.float32))
    elif data.kind in ITEM_KINDS:
        source = Source(data.kind, np.zeros(1, dtype=np.int64), items=item_cache(f"data {data_key(job)}", data))
        text = f"the dataset's first row ({data.kind} of shape {list(source.item(0).shape)})"
    elif data.kind in TEXT_KINDS:
        source = Source("text", np.zeros(1, dtype=np.int64), items=text_cache(f"data {data_key(job)}", data))
        text = f"the dataset's first row (text of {len(source.item(0))} characters)"
    elif data.kind == "graph":
        folds, _ = folds_of(job, data)
        fold = next(iter(folds.values()))
        sources, _, _, _ = _graph_sources(job, data, dict(fold.roles), 1, "cpu", lambda m: None)
        source = sources["train"].subset([0])
        text = (f"the first training node of fold {fold.name}, in a graph of {source.graph.n_nodes} nodes and "
                f"{source.graph.edges.shape[1]} edges")
    else:
        raise ValueError(f"the torch family has no batches for {data.kind} data")
    input_shape = _input_shape(source, feature)
    _, x_np, lengths_np = next(eval_batches(source, 1))
    x = feature(to_device(x_np, device))
    lengths = None
    if lengths_np is not None:
        lengths = feature.lengths(torch.from_numpy(lengths_np)).clamp_max(x.shape[-1])
    model, aux_head, contrastive_head = build(job, input_shape, 1, device)
    model.eval()
    calls, layers, handles = {}, [], []

    def hook(name):
        def record(module, inputs, output):
            entry = calls.setdefault(name, {"calls": 0})
            entry["calls"] += 1
            if entry["calls"] == 1:
                entry["input"] = _shape_of(inputs)
                entry["output"] = _shape_of(output)
        return record

    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(hook(name)))
            layers.append((name, type(module).__name__, int(sum(p.numel() for p in module.parameters(recurse=False)))))
    with torch.no_grad():
        outputs = model(x, lengths)
    for handle in handles:
        handle.remove()
    counts = parameter_counts(model)
    rows = []
    for name, kind, n in layers:
        c = calls.get(name, {"calls": 0})
        rows.append(f"{name:40s} {kind:22s} {c.get('input')} -> {c.get('output')}  {n} parameters  x{c['calls']}")
    heads = {"aux": None, "contrastive": None}
    for name, head in (("aux", aux_head), ("contrastive", contrastive_head)):
        if head is not None:
            heads[name] = int(sum(p.numel() for p in head.parameters()))
    return {"input": text, "input_shape": list(input_shape), "batch_input_shape": _shape_of(x),
            "n_outputs": _n_outputs(job), "outputs_shape": _shape_of(outputs),
            "parameters_total": counts["total"], "parameters_trainable": counts["trainable"],
            "parameters_by_part": counts["by_part"], "extra_heads": heads, "module_tree": str(model), "layers": rows}
