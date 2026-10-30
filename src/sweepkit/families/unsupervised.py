"""The clusterer, embedder and detector families: fit without labels, then score the evaluated rows with the task's
label-free criteria, and write agreement with known labels (if the task names a reference column) where analyze
never looks.

Per job: the fold's fit and eval rows, the preparation steps fitted on the fit rows, the method fitted on the fit
rows, then the evaluated rows' outputs: with assign fit_rows, each evaluated row takes the output of its own fit row
(evaluated rows must be fit rows; a row a preparation step took out of the fit is reported as removed); with assign
predict, the method's predict (or transform, or score) labels them.

Files:
  criteria.json            the label-free criteria (None with the reason where undefined) and the choosing one
  assignments.csv          clusterer: row_id, cluster (-1: noise or none), status, probability and outlier score
                           where the method has them
  embedding.npy            embedder: the evaluated rows' coordinates (float32), in the order of embedding_rows.csv
  embedding_rows.csv       embedder: row_id, status
  scores.csv               detector: row_id, score (higher is more unusual), flagged, status
  reveal.json              agreement with the reference labels: read only after a selection is recorded
  reveal_reference.csv     the evaluated rows' reference labels (the same rule)
  model_params.json, prep_record.json, efficiency.json, provenance.json, and any file the method adds (a
  dendrogram's linkage, a co-association matrix)
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit import devices
from sweepkit.families.common import (Clock, csv_text, folds_of, limit_roles, load_data, number, peak_memory_mb,
                                      prepare, require_finite, run_provenance)
from sweepkit.metrics.unsupervised_context import Undefined, UnsupervisedContext
from sweepkit.models.base import build_clusterer, build_detector, build_embedder
from sweepkit.registry import METRICS, MODELS, register_fit
from sweepkit.settings import Selected
from sweepkit.splits.base import stable_keys
from sweepkit.streams import int_seed
from sweepkit.tasks.unsupervised import STABILITY


@dataclasses.dataclass(frozen=True)
class UnsupervisedFitSettings:
    device: Literal["cpu", "cuda", "auto"]   # cuda: the method's GPU path (cuML); auto: when available
    assign: Literal["fit_rows", "predict"]
    # fit_rows: the evaluated rows are fit rows and take their own outputs; predict: the method labels, embeds or
    # scores the evaluated rows itself (needed when they are not fit rows: split fit_holdout)


def _check(job) -> None:
    if job.fit.device == "cuda" and MODELS.get(job.model_name).meta["gpu"] == "none":
        raise ValueError(f"model {job.model_name} has no GPU path; fit.device cpu or auto")


def _sample(ids: np.ndarray, n: int | None, salt: str) -> np.ndarray:
    """Positions of the fixed sample the costly criteria use: the n rows with the smallest SHA-256 keys."""
    if n is None or n >= ids.size:
        return np.arange(ids.size)
    return np.sort(np.lexsort((ids, stable_keys(ids, salt)))[:n])


def _criteria(task_settings, task_name: str, ctx) -> dict:
    values, undefined = {}, {}
    for name in task_settings.criteria:
        try:
            values[name] = float(METRICS.get(name)["fn"](ctx))
        except Undefined as why:
            values[name] = None
            undefined[name] = str(why)
    select = task_settings.select
    if select == STABILITY:
        chosen = {"metric": STABILITY, "direction": "max", "value": None, "computed_by": "analyze"}
    else:
        chosen = {"metric": select, "direction": METRICS.get(select)["direction"], "value": values[select]}
    return {"criteria": values, "undefined": undefined, "select": chosen, "n_evaluated": int(ctx.x.shape[0]),
            "n_sampled": int(ctx.sample.size)}


def _reveal(task_settings, ctx, ids: np.ndarray) -> dict:
    if task_settings.reference is None:
        return {}
    values, undefined = {}, {}
    for name in task_settings.reveal_metrics:
        try:
            values[name] = float(METRICS.get(name)["fn"](ctx))
        except Undefined as why:
            values[name] = None
            undefined[name] = str(why)
    reference = np.asarray(ctx.reference).astype(str)
    return {"reveal.json": {"reference": task_settings.reference, "metrics": values, "undefined": undefined,
                            "n_with_reference": int((reference != "").sum())},
            "reveal_reference.csv": csv_text(["row_id", "reference"], zip(ids.tolist(), reference.tolist()))}


class _Run:
    """The steps the three families share."""

    def __init__(self, job, spec: dict, offered: str, log):
        self.job, self.spec, self.log = job, spec, log
        self.entry = MODELS.get(job.model_name)
        self.seed = int(spec["seed"])
        self.data = load_data(job, log)
        folds, self.fold_hash = folds_of(job, self.data)
        self.fold = folds[spec["fold"]]
        hook = getattr(self.entry.target, "gpu_unsupported", None)
        unsupported = hook(job.model) if hook is not None else None
        self.device = devices.resolve(job.fit.device, offered, self.entry.meta["gpu"], unsupported)
        self.device_notes = devices.auto_note(job.fit.device, offered, self.device, unsupported)
        self.clock = Clock()
        original = limit_roles(self.fold.roles, self.data.table.ids, spec)
        if job.fit.assign == "fit_rows":
            outside = np.setdiff1d(original["eval"], original["fit"])
            if outside.size:
                raise ValueError(f"{outside.size} evaluated rows are not fit rows; use fit.assign predict")
        self.clock.start()
        self.x, self.positions, self.names, self.categorical, self.records = prepare(job, self.data, original, "fit",
                                                                                     self.seed, self.device, log)
        self.clock.stop("prep_seconds")
        self.precomputed = getattr(job.model, "metric", None) == "precomputed"
        if self.precomputed:
            self._precomputed_columns()
        require_finite(self.x["fit"], "fit rows")
        self.eval_positions = original["eval"]
        if job.fit.assign == "fit_rows":
            index = {int(p): k for k, p in enumerate(self.positions["fit"])}
            self.eval_fit_index = np.array([index.get(int(p), -1) for p in self.eval_positions], dtype=np.int64)
            self.kept = self.eval_fit_index >= 0
        else:
            require_finite(self.x["eval"], "evaluated rows")
            self.kept = np.ones(self.eval_positions.size, dtype=bool)
        self.status = np.where(self.kept, "evaluated", "removed_by_prep")
        self.ids = self.data.table.ids[self.eval_positions]
        log(f"run {spec['run_id']}: fold {spec['fold']}, seed {self.seed}, fit {self.positions['fit'].size}, "
            f"evaluated {int(self.kept.sum())} of {self.eval_positions.size}, {len(self.names)} columns, "
            f"{job.model_name} on {self.device}")

    def _precomputed_columns(self) -> None:
        """The data are a distance matrix, one row and one column per row ID (docs/DATA.md). Every role keeps the
        columns of the fit rows, in the fit rows' order: the method gets the fit rows' square block, and evaluated
        rows their distances to the fit rows."""
        if self.job.prep:
            raise ValueError("precomputed distances take no preparation steps: a step would change the distances")
        ids = self.data.table.ids
        if len(self.names) != ids.size or sorted(self.names) != sorted(ids.tolist()):
            raise ValueError("precomputed distances: the data's columns must be its row IDs, one column for each row "
                             "(a square distance matrix; docs/DATA.md)")
        column_of = {name: j for j, name in enumerate(self.names)}
        self.column_of_row = np.array([column_of[i] for i in ids.tolist()], dtype=np.int64)
        fit_columns = self.column_of_row[self.positions["fit"]]
        for role in list(self.x):
            self.x[role] = np.asarray(self.x[role], dtype=np.float32)[:, fit_columns]
        self.names = tuple(ids[self.positions["fit"]].tolist())

    def eval_distances(self) -> np.ndarray | None:
        """With precomputed distances, the distances between the kept evaluated rows (for the criteria); else None."""
        if not self.precomputed:
            return None
        rows = self.eval_positions[self.kept]
        full = self.data.x
        return np.asarray(full[rows][:, self.column_of_row[rows]], dtype=np.float64)

    def kept_x(self) -> np.ndarray:
        if self.job.fit.assign == "fit_rows":
            return self.x["fit"][self.eval_fit_index[self.kept]]
        return self.x["eval"]

    def from_fit(self, values: np.ndarray) -> np.ndarray:
        """Per-fit-row values mapped to the kept evaluated rows."""
        return np.asarray(values)[self.eval_fit_index[self.kept]]

    def reference(self) -> np.ndarray | None:
        ref = self.job.task.reference
        if ref is None:
            return None
        values = self.data.table.column(ref).astype(str)[self.eval_positions][self.kept]
        return np.where(np.isin(values, ("nan", "None")), "", values)

    def sample(self) -> np.ndarray:
        return _sample(self.ids[self.kept], self.job.task.sample_rows, self.job.task.sample_salt)

    def finish(self, files: dict, model, criteria_ctx) -> dict:
        self.clock.start()
        criteria = _criteria(self.job.task, self.job.task_name, criteria_ctx)
        self.clock.stop("criteria_seconds")
        files["criteria.json"] = criteria
        if self.job.task.reference is not None:
            reveal_ctx = dataclasses.replace(criteria_ctx, reference=self.reference())
            files.update(_reveal(self.job.task, reveal_ctx, self.ids[self.kept]))
        shown = {k: (round(v, 4) if v is not None else None) for k, v in criteria["criteria"].items()}
        self.log(f"  criteria {shown}")
        files["model_params.json"] = {"model": self.job.model_name, "params": model.params()}
        files["prep_record.json"] = {"steps": self.records, "columns": list(self.names)}
        files["efficiency.json"] = {**self.clock.parts, "n_fit": int(self.positions["fit"].size),
                                    "n_evaluated": int(self.kept.sum()), "n_columns": len(self.names),
                                    "peak_memory_mb": peak_memory_mb()}
        files["provenance.json"] = run_provenance(self.job, self.spec, self.data, self.fold_hash, self.job.fit.device,
                                                  self.device, self.device_notes + list(model.fallbacks()))
        return files

    def context(self, **kw) -> UnsupervisedContext:
        base = dict(x=self.kept_x(), labels=None, embedding=None, scores=None, model_info={}, sample=self.sample(),
                    options=dict(self.job.task.criteria_options), refit=None, score_new=None, reference=None,
                    n_neighbors=getattr(self.job.task, "n_neighbors", None), seed=self.seed,
                    distances=self.eval_distances())
        base.update(kw)
        return UnsupervisedContext(**base)


def _selected(job) -> Selected:
    return Selected(job.model_name, job.model)


@register_fit("clusterer", settings=UnsupervisedFitSettings, family="clusterer",
              description="fit a clustering method on the fit rows; label-free criteria choose")
class ClustererFit:
    check_job = staticmethod(_check)

    @staticmethod
    def run(job, spec: dict, offered: str, log) -> dict:
        r = _Run(job, spec, offered, log)
        model = build_clusterer(_selected(job), int_seed(r.seed, "model"), r.device)
        model.set_columns(r.names, r.categorical)
        r.clock.start()
        model.fit(r.x["fit"])
        r.clock.stop("fit_seconds")
        soft = {}
        if job.fit.assign == "fit_rows":
            labels = r.from_fit(model.labels())
            soft = {k: r.from_fit(v) for k, v in model.soft().items()}
        else:
            labels = np.asarray(model.assign(r.x["eval"]), dtype=np.int64)
        all_labels = np.full(r.eval_positions.size, -1, dtype=np.int64)
        all_labels[r.kept] = labels
        header = ["row_id", "cluster", "status"] + sorted(soft)
        rows = []
        k = 0
        for i in range(r.eval_positions.size):
            row = [r.ids[i], int(all_labels[i]), r.status[i] if all_labels[i] >= 0 or not r.kept[i] else "noise"]
            if r.kept[i]:
                row += [number(float(soft[name][k]), 6) for name in sorted(soft)]
                k += 1
            else:
                row += [""] * len(soft)
            rows.append(row)
        files = {"assignments.csv": csv_text(header, rows), **model.extra_files()}

        def refit(x_new):
            other = build_clusterer(_selected(job), int_seed(r.seed, "model"), r.device)
            other.set_columns(r.names, r.categorical)
            other.fit(x_new)
            return other.labels()

        ctx = r.context(labels=labels, model_info=model.info(), refit=refit)
        found = np.unique(labels[labels >= 0]).size
        log(f"  {found} clusters, {int((labels < 0).sum())} noise rows")
        return r.finish(files, model, ctx)


@register_fit("embedder", settings=UnsupervisedFitSettings, family="embedder",
              description="fit a dimensionality reduction on the fit rows; label-free criteria choose")
class EmbedderFit:
    check_job = staticmethod(_check)

    @staticmethod
    def run(job, spec: dict, offered: str, log) -> dict:
        r = _Run(job, spec, offered, log)
        model = build_embedder(_selected(job), int_seed(r.seed, "model"), r.device)
        model.set_columns(r.names, r.categorical)
        r.clock.start()
        fitted = np.asarray(model.fit_transform(r.x["fit"]), dtype=np.float32)
        r.clock.stop("fit_seconds")
        if job.fit.assign == "fit_rows":
            embedding = r.from_fit(fitted)
        else:
            if not model.can_transform():
                raise ValueError(f"{job.model_name} cannot embed rows it was not fitted on; use fit.assign fit_rows")
            embedding = np.asarray(model.transform(r.x["eval"]), dtype=np.float32)
        info = dict(model.info())
        x_kept = r.kept_x()
        if model.can_inverse():
            rebuilt = np.asarray(model.inverse_transform(embedding), dtype=np.float64)
            info["reconstruction_error"] = float(np.mean((rebuilt - x_kept) ** 2))
        files = {"embedding.npy": embedding,
                 "embedding_rows.csv": csv_text(["row_id", "status"], zip(r.ids[r.kept].tolist(),
                                                                         ["evaluated"] * int(r.kept.sum())))}
        removed = r.ids[~r.kept]
        if removed.size:
            files["removed_rows.csv"] = csv_text(["row_id", "status"], [[i, "removed_by_prep"] for i in removed])
        ctx = r.context(embedding=embedding, model_info=info)
        log(f"  embedding {embedding.shape[1]} components")
        return r.finish(files, model, ctx)


@register_fit("detector", settings=UnsupervisedFitSettings, family="detector",
              description="fit an outlier detector on the fit rows; label-free criteria choose")
class DetectorFit:
    check_job = staticmethod(_check)

    @staticmethod
    def run(job, spec: dict, offered: str, log) -> dict:
        r = _Run(job, spec, offered, log)
        model = build_detector(_selected(job), int_seed(r.seed, "model"), r.device)
        model.set_columns(r.names, r.categorical)
        r.clock.start()
        model.fit(r.x["fit"])
        r.clock.stop("fit_seconds")
        x_kept = r.kept_x()
        scores = np.asarray(model.score(x_kept), dtype=np.float64)
        flags = np.asarray(model.flag(x_kept), dtype=bool)
        all_scores = np.full(r.eval_positions.size, np.nan)
        all_scores[r.kept] = scores
        all_flags = np.zeros(r.eval_positions.size, dtype=bool)
        all_flags[r.kept] = flags
        files = {"scores.csv": csv_text(["row_id", "score", "flagged", "status"],
                                        [[r.ids[i], number(all_scores[i], 6), int(all_flags[i]), r.status[i]]
                                         for i in range(r.eval_positions.size)])}
        cache = {}
        x_fit = r.x["fit"]

        def score_new(features, part, x_new):
            if features is None:
                return model.score(np.asarray(x_new, dtype=np.float32))
            key = tuple(int(f) for f in features)
            if key not in cache:
                other = build_detector(_selected(job), int_seed(r.seed, "model"), r.device)
                other.set_columns(tuple(r.names[j] for j in key),
                                  {key.index(j): v for j, v in r.categorical.items() if j in key})
                other.fit(x_fit[:, list(key)])
                cache[key] = other
            return cache[key].score(np.asarray(x_new, dtype=np.float32))

        info = {**model.info(), "flagged": float(flags.mean())}
        ctx = r.context(scores=scores, model_info=info, score_new=score_new)
        log(f"  {int(flags.sum())} of {flags.size} evaluated rows flagged")
        return r.finish(files, model, ctx)
