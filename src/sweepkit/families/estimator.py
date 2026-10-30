"""The estimator family: any scikit-learn-style supervised model, through one runner.

Per job: the fold's rows, the preparation steps fitted on the training rows, the model fitted on the training rows,
then validation predictions (they choose), test predictions (read only after a selection is recorded) from the model
the fit section's refit asks for, and training-row predictions when eval_train is on. The shuffled-label control
fits on permuted training labels (expect chance; a result above chance means something leaks).
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit import devices
from sweepkit.families.common import (Clock, folds_of, limit_roles, load_data, peak_memory_mb, prepare,
                                      require_finite, run_provenance)
from sweepkit.families.supervised_io import class_names, role_files, task_targets
from sweepkit.models.base import library_params
from sweepkit.registry import MODELS, register_fit
from sweepkit.streams import generator, int_seed
from sweepkit.tasks.supervised import Outputs


@dataclasses.dataclass(frozen=True)
class EstimatorFitSettings:
    device: Literal["cpu", "cuda", "auto"]   # cuda: the model's GPU path (cuML, XGBoost); auto: when available
    control: Literal["none", "shuffle_train_labels"]   # shuffle_train_labels: the leak test, expect chance
    refit: Literal["train", "train_and_val"]
    # the model that predicts the test rows: the one fitted on the training rows (validation only chooses), or the
    # same settings fitted again on training plus validation rows. Validation predictions always come from the model
    # fitted on the training rows alone.
    eval_train: bool                # also score the training rows (to see over- and underfitting)


def _scores(model, x: np.ndarray, n_classes: int, task: str) -> tuple:
    """(scores, kind): class probabilities when the model has predict_proba, else decision values, else none."""
    if task == "regression":
        return None, "none"
    if hasattr(model, "predict_proba"):
        try:
            raw = model.predict_proba(x)
        except (AttributeError, NotImplementedError):
            raw = None
        if raw is not None:
            if task == "multilabel":
                if isinstance(raw, list):
                    return np.stack([_positive_column(p, c) for p, c in zip(raw, _label_classes(model, len(raw)))],
                                    axis=1), "probability"
                return np.asarray(raw), "probability"
            raw = np.asarray(raw)
            out = np.zeros((x.shape[0], n_classes))
            out[:, np.asarray(model.classes_, dtype=np.int64)] = raw
            return out, "probability"
    if task == "multilabel" and hasattr(model, "estimators_") and all(hasattr(e, "decision_function")
                                                                       for e in model.estimators_):
        # one model per label without probabilities (a linear SVM per label): each label's decision values
        return np.stack([np.asarray(e.decision_function(x)) for e in model.estimators_], axis=1), "decision"
    if hasattr(model, "decision_function") and task == "classification":
        d = np.asarray(model.decision_function(x))
        if d.ndim == 1:
            return np.stack([-d, d], axis=1), "decision"
        out = np.full((x.shape[0], n_classes), -np.inf)
        out[:, np.asarray(model.classes_, dtype=np.int64)] = d
        return out, "decision"
    return None, "none"


def _label_classes(model, n_labels: int) -> list:
    """Each label's classes as its model saw them: [0, 1], or only one of them when a label had one value in the
    training rows."""
    if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) == n_labels:
        return [np.asarray(e.classes_) for e in model.estimators_]
    return [np.asarray(c) for c in model.classes_]


def _positive_column(probabilities, classes: np.ndarray) -> np.ndarray:
    """P(label = 1) from one label's probability columns (0 where the training rows never had it on)."""
    p = np.asarray(probabilities)
    where = np.flatnonzero(classes.astype(np.int64) == 1)
    return p[:, where[0]] if where.size else np.zeros(p.shape[0])


def _host(values) -> np.ndarray:
    from sweepkit.models.base import _host as host

    return host(values)


def _gpu_unsupported(job):
    """Why the model's GPU path cannot take this job's settings, or None (devices.resolve)."""
    hook = getattr(MODELS.get(job.model_name).target, "gpu_unsupported", None)
    return hook(job.model) if hook is not None else None


@register_fit("estimator", settings=EstimatorFitSettings, family="estimator",
              description="fit a scikit-learn-style model on the training rows; validation chooses")
class EstimatorFit:
    @staticmethod
    def check_job(job) -> None:
        if job.fit.device == "cuda" and MODELS.get(job.model_name).meta["gpu"] == "none":
            raise ValueError(f"model {job.model_name} has no GPU path; fit.device cpu or auto")
        reason = _gpu_unsupported(job)
        if job.fit.device == "cuda" and reason is not None:
            raise ValueError(f"model {job.model_name} on the GPU: {reason}; fit.device cpu, or auto to run these "
                             "settings on the CPU")

    @staticmethod
    def run(job, spec: dict, offered: str, log) -> dict:
        entry = MODELS.get(job.model_name)
        seed = int(spec["seed"])
        data = load_data(job, log)
        folds, fold_hash = folds_of(job, data)
        fold = folds[spec["fold"]]
        unsupported = _gpu_unsupported(job)
        device = devices.resolve(job.fit.device, offered, entry.meta["gpu"], unsupported)
        device_notes = devices.auto_note(job.fit.device, offered, device, unsupported)
        y_all = task_targets(job, data.table)
        names = class_names(job)
        positions = limit_roles(fold.roles, data.table.ids, spec)
        clock = Clock()
        clock.start()
        x, positions, feature_names, _, records = prepare(job, data, positions, "train", seed, device, log)
        clock.stop("prep_seconds")
        for role in ("train", "val", "test"):
            require_finite(x[role], f"{role} rows")
        y_train = y_all[positions["train"]]
        if job.fit.control == "shuffle_train_labels":
            y_train = y_train[generator(seed, "control shuffle").permutation(y_train.shape[0])]
        if job.task_name == "classification":
            missing = sorted(set(range(len(names))) - set(np.unique(y_train).tolist()))
            if missing:
                raise ValueError(f"the training rows hold no row of classes {[names[c] for c in missing]}")
        log(f"run {spec['run_id']}: fold {spec['fold']}, seed {seed}, train {positions['train'].size}, val "
            f"{positions['val'].size}, test {positions['test'].size}, {len(feature_names)} columns, "
            f"{job.model_name} on {device}")
        model = entry.target.make(job.model, job.task_name, int_seed(seed, "model"), device)
        clock.start()
        model.fit(x["train"], y_train)
        clock.stop("fit_seconds")
        files = {}

        def outputs(fitted, role):
            y_pred = _host(fitted.predict(x[role]))
            if job.task_name == "classification":
                y_pred = y_pred.astype(np.int64)
            elif job.task_name == "multilabel":
                y_pred = y_pred.astype(np.int64)
            else:
                y_pred = y_pred.astype(np.float64)
            scores, kind = _scores(fitted, x[role], len(names), job.task_name)
            return Outputs(y_all[positions[role]], y_pred, scores, kind, names)

        clock.start()
        files.update(role_files(job, "val", data.table, positions["val"], outputs(model, "val"), {}))
        if job.fit.eval_train:
            files.update(role_files(job, "train", data.table, positions["train"], outputs(model, "train"), {}))
        final = model
        if job.fit.refit == "train_and_val":
            final = entry.target.make(job.model, job.task_name, int_seed(seed, "model"), device)
            rows = np.concatenate([positions["train"], positions["val"]])
            y_final = np.concatenate([y_all[positions["train"]], y_all[positions["val"]]])
            if job.fit.control == "shuffle_train_labels":
                # the control's refit sees no true label either
                y_final = y_final[generator(seed, "control shuffle refit").permutation(y_final.shape[0])]
            final.fit(np.concatenate([x["train"], x["val"]]), y_final)
            log(f"  refit on train and val: {rows.size} rows")
        files.update(role_files(job, "test", data.table, positions["test"], outputs(final, "test"),
                                {"refit": job.fit.refit}))
        clock.stop("predict_seconds")
        val_select = files["metrics_val.json"]["select"]
        if fold.selection_on_test:
            log("  the identity check fold: its validation numbers are test numbers, so none is shown")
        else:
            log(f"  validation {val_select['metric']} {val_select['value']:.4f}")
        files["model_params.json"] = {"model": job.model_name, "params": library_params(model),
                                      "final_params": library_params(final)}
        files["prep_record.json"] = {"steps": records, "columns": list(feature_names)}
        files["efficiency.json"] = {**clock.parts, "n_train": int(positions["train"].size),
                                    "n_columns": len(feature_names), "peak_memory_mb": peak_memory_mb()}
        files["provenance.json"] = run_provenance(job, spec, data, fold_hash, job.fit.device, device, device_notes)
        return files
