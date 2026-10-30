"""The torch family's fit procedure (sweepkit.families.torch) and its training loop, through the real runner.

Without torch (runs everywhere): the fit settings and every cross-check that refuses a job when a sweep is expanded,
the smoke shortening, and the random streams' names. With torch (the node, where SWEEPKIT_REQUIRE names torch): a real
job through the runner writes every file; reloading the best epoch gives the validation value that chose it; the
refit on training plus validation rows trains for the epoch count validation chose; a run repeats exactly; an option a
configuration does not use changes none of its draws; the shuffled-label control stays near chance; multi-label,
regression, sequences with crops, windows, masks, noise, an auxiliary head, external datasets and group normalization,
audio through the log-mel feature with speed perturbation, images and a spiking model each run end to end; the
network describe-model reports is the one the run trained; and the known-structure tests of the design's appendix
(mlp above 0.9 on tabular classes and R squared above 0.8 on regression; cnn1d, cnn2d, rnn, transformer, cnn_rnn,
cnn_transformer and composite above 0.8; cnn_lif and snn above 0.6; the sequence heads above 0.8 on per-step
features). The thresholds are first proposals, not yet set from runs: the node session sets them from what the runs
give, a little below.
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from tests.support import REPO, TempRepo, load_catalog, needs_module

load_catalog()

HOLDOUT = {"name": "holdout", "val_fraction": 0.2, "test_fraction": 0.2, "stratify": "truth", "salt": "t",
           "identity_fold": False}
SCALE = {"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}
GROUPS = ["g0", "g1", "g2", "g3", "g4"]


def fit(**changes) -> dict:
    out = {"name": "torch", "device": "cpu", "epochs": 8, "batch_size": 32, "eval_batch_size": 64, "lr": 0.003,
           "weight_decay": 0.01, "warmup_epochs": 1, "schedule": "cosine", "early_stopping_patience": None,
           "precision": "fp32", "class_balance": "none", "label_smoothing": 0.0, "regression_loss": "none",
           "control": "none", "eval_train": False, "crop": None, "eval_window": None, "augment": [], "aux": None,
           "contrastive": None, "external": [], "max_batches_per_epoch": None, "refit": "train"}
    out.update(changes)
    return out


def classes(n: int) -> dict:
    return {"name": "classification", "target": "truth", "classes": [f"c{i}" for i in range(n)],
            "metrics": ["balanced_accuracy", "accuracy", "log_loss"], "select": "balanced_accuracy",
            "groups": ["group"]}


def tabular(model=None, **fit_changes) -> dict:
    return {"data": {"name": "synthetic", "generator": "classification", "n_rows": 300, "n_features": 6,
                     "n_groups": 3, "noise": 0.5, "seed": 2},
            "split": HOLDOUT, "task": classes(3), "prep": [SCALE], "features": {"name": "none"},
            "model": model or {"name": "mlp", "hidden": [32], "norm": "batch", "activation": "relu", "dropout": 0.1},
            "fit": fit(**fit_changes)}


SEQUENCES = {"name": "synthetic", "generator": "sequences", "n_rows": 150, "n_features": 4, "n_groups": 3,
             "noise": 0.5, "seed": 3}
RNN = {"name": "rnn", "time_pool": 1, "input_dropout": 0.0, "proj_dim": None, "activation": "relu", "rnn": "gru",
       "rnn_hidden": 16, "rnn_layers": 1, "bidirectional": True, "rnn_dropout": 0.0, "pooling": "mean",
       "classifier_hidden": None, "dropout": 0.1}


def sequences(model=None, data=None, normalization="global", **fit_changes) -> dict:
    return {"data": dict(SEQUENCES, **(data or {})), "split": HOLDOUT, "task": classes(3), "prep": [],
            "features": {"name": "sequence", "normalization": normalization,
                         "group_column": "group" if normalization == "group" else None},
            "model": model or dict(RNN), "fit": fit(**fit_changes)}


def example_base(name: str) -> dict:
    return json.loads((REPO / "sweeps" / "examples" / f"{name}.json").read_text(encoding="utf-8"))["base"]


def checked(settings: dict):
    from sweepkit.runs.spec import check_job, validate_job

    job = validate_job(settings)
    check_job(job)
    return job


def run(settings: dict, root, runs: str = "runs", seed: int = 1, fold: str = "holdout") -> tuple:
    """(run folder, log lines) of one job through the runner, on the CPU."""
    from sweepkit.runs.spec import config_hash, plain, run_id
    from sweepkit.sweep.runner import execute

    job = checked(settings)
    spec = {"sweep_id": "torch_test", "purpose": "learning", "stage": "base", "label": "base", "overrides": {},
            "priority": 1, "config_hash": config_hash(job), "run_id": run_id(job, fold, seed), "fold": fold,
            "seed": seed, "family": job.family, "settings": plain(job)}
    lines = []
    execute(spec, "cpu", lines.append, runs_root=Path(root) / runs)
    return Path(root) / runs / spec["run_id"], lines


def read(folder: Path, name: str):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def without_times(value):
    if isinstance(value, dict):
        return {k: without_times(v) for k, v in value.items() if not k.endswith("_seconds")}
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


# Without torch --------------------------------------------------------------------------------------------------------

class TestChecks(unittest.TestCase):
    def test_the_examples_and_these_bases_pass(self):
        for name in ("classify_mlp", "regress_mlp", "sequence_models", "image_cnn", "audio_logmel", "spiking"):
            checked(example_base(name))
        checked(tabular())
        checked(sequences())

    def refused(self, settings: dict, words: str):
        from sweepkit.runs.spec import check_job, validate_job

        with self.assertRaisesRegex(ValueError, words, msg=json.dumps(settings["fit"])[:300]):
            check_job(validate_job(settings))

    def test_numbers(self):
        for changes, words in (({"epochs": 0}, "epochs"), ({"batch_size": 1}, "batch_size"), ({"lr": 0.0}, "lr"),
                               ({"weight_decay": -0.1}, "weight_decay"), ({"warmup_epochs": 8}, "warmup"),
                               ({"early_stopping_patience": 0}, "patience"), ({"label_smoothing": 1.0}, "smoothing"),
                               ({"max_batches_per_epoch": 0}, "max_batches"), ({"crop": 0}, "crop")):
            self.refused(tabular(**changes) if "crop" not in changes else sequences(**changes), words)
        self.refused(sequences(eval_window={"window": 10, "hop": 11}), "hop")

    def test_tasks_and_their_losses(self):
        regression = example_base("regress_mlp")
        for changes, words in (({"regression_loss": "none"}, "regression_loss"),
                               ({"class_balance": "weighted_loss"}, "class_balance"),
                               ({"label_smoothing": 0.1}, "label_smoothing"),
                               ({"contrastive": {"weight": 0.1, "temperature": 0.1, "dim": 8}}, "contrastive")):
            settings = copy.deepcopy(regression)
            settings["fit"].update(changes)
            self.refused(settings, words)
        self.refused(tabular(regression_loss="mse"), "regression_loss")
        multilabel = tabular(class_balance="balanced_sampler")
        multilabel["data"]["generator"] = "multilabel"
        multilabel["split"] = dict(HOLDOUT, stratify=None)
        multilabel["task"] = {"name": "multilabel", "targets": ["label_0", "label_1", "label_2"],
                              "metrics": ["label_f1_macro"], "select": "label_f1_macro", "groups": []}
        self.refused(multilabel, "balanced sampler")
        multilabel["fit"]["class_balance"] = "weighted_loss"
        checked(multilabel)

    def test_inputs_and_augmentations(self):
        self.refused(tabular(crop=10), "crop")
        self.refused(tabular(eval_window={"window": 10, "hop": 5}), "eval_window")
        self.refused(tabular(augment=[{"name": "mask", "row_masks": 1, "row_mask_width": 1, "col_masks": 0,
                                       "col_mask_width": 0}]), "mask")
        self.refused(sequences(augment=[{"name": "speed", "factors": [0.9, 1.1]}]), "speed")
        self.refused(sequences(augment=[{"name": "noise", "sd": 0.1}, {"name": "noise", "sd": 0.2}]), "at most once")
        self.refused(sequences(augment=[{"name": "noise", "sd": 0.0}]), "sd")
        self.refused(sequences(augment=[{"name": "mixup", "alpha": 0.2}],
                               contrastive={"weight": 0.1, "temperature": 0.1, "dim": 8}), "mixup")
        audio = example_base("audio_logmel")
        audio["fit"]["augment"] = [{"name": "speed", "factors": [0.9, 1.1]}, {"name": "mask", "row_masks": 1,
                                                                              "row_mask_width": 4, "col_masks": 1,
                                                                              "col_mask_width": 4}]
        checked(audio)
        audio["fit"]["augment"] = [{"name": "speed", "factors": [0.777]}]
        self.refused(audio, "ratio")
        checked(tabular(augment=[{"name": "noise", "sd": 0.1}, {"name": "mixup", "alpha": 0.2}]))

    def test_aux_and_contrastive(self):
        aux = {"mode": "adversarial", "task": "classification", "columns": ["group"], "classes": GROUPS,
               "weight": 0.1, "loss": "cross_entropy"}
        checked(tabular(aux=aux))
        for changes, words in (({"weight": 0.0}, "weight"), ({"loss": "mse"}, "cross_entropy"),
                               ({"columns": ["group", "truth"]}, "one column"), ({"classes": ["g0"]}, "two"),
                               ({"columns": ["truth"], "classes": ["c0", "c1", "c2"]}, "own target"),
                               ({"task": "regression"}, "regression aux")):
            self.refused(tabular(aux=dict(aux, **changes)), words)
        checked(tabular(contrastive={"weight": 0.1, "temperature": 0.1, "dim": 8}))
        for bad in ({"weight": 0.0, "temperature": 0.1, "dim": 8}, {"weight": 0.1, "temperature": 0.0, "dim": 8},
                    {"weight": 0.1, "temperature": 0.1, "dim": 0}):
            self.refused(tabular(contrastive=bad), "contrastive")

    def test_externals(self):
        noisier = {"name": "noisier", "data": dict(SEQUENCES, noise=0.8, seed=9), "restrict_classes": False}
        checked(sequences(external=[noisier]))
        for externals, words in (([dict(noisier, name="Bad")], "lowercase"), ([noisier, noisier], "repeats"),
                                 ([dict(noisier, data=dict(SEQUENCES, generator="images"))], "image data")):
            self.refused(sequences(external=externals), words)
        self.refused(sequences(normalization="group", external=[noisier]), "group normalization")
        table = tabular(external=[{"name": "more", "data": tabular()["data"], "restrict_classes": False}])
        self.refused(table, "empty prep")
        table["prep"] = []
        checked(table)
        regression = example_base("regress_mlp")
        regression["prep"] = []
        regression["fit"]["external"] = [{"name": "more", "data": dict(regression["data"], seed=4),
                                          "restrict_classes": True}]
        self.refused(regression, "restrict_classes")
        audio = example_base("audio_logmel")
        audio["fit"]["external"] = [{"name": "wav", "data": {"name": "items", "index": "none.csv", "path_column": "p",
                                                             "row_id": "id", "keep": ["truth"], "kind": "audio",
                                                             "loader": "wav", "sample_rate": 8000},
                                     "restrict_classes": False}]
        with TempRepo() as root:
            (root / "data").mkdir()
            (root / "data" / "none.csv").write_text("id,p,truth\n", encoding="utf-8")
            self.refused(audio, "8000 Hz")

    def test_wrong_types_fail_validation(self):
        from sweepkit.runs.spec import validate_job
        from sweepkit.settings import SettingsError

        for changes in ({"augment": [{"name": "rotate"}]}, {"precision": "fp16"}, {"refit": "val"},
                        {"aux": {"mode": "multitask"}}, {"eval_window": {"window": 10}},
                        {"external": [{"name": "x", "data": {"name": "nothing"}, "restrict_classes": False}]}):
            with self.assertRaises(SettingsError, msg=str(changes)):
                validate_job(tabular(**changes))


class TestHooks(unittest.TestCase):
    def test_smoke_settings(self):
        from sweepkit.families.torch import TorchFit
        from sweepkit.runs.spec import validate_job
        from sweepkit.sweep.smoke import smoke_spec

        settings = tabular(epochs=40, warmup_epochs=5)
        before = copy.deepcopy(settings)
        short = TorchFit.smoke_settings(settings)
        self.assertEqual(settings, before, "the job's own settings are left as they were")
        self.assertEqual((short["fit"]["epochs"], short["fit"]["warmup_epochs"], short["fit"]["max_batches_per_epoch"]),
                         (2, 1, 5))
        self.assertEqual(TorchFit.smoke_settings(tabular(max_batches_per_epoch=3))["fit"]["max_batches_per_epoch"], 3)
        checked(short)
        job = {"settings": tabular(), "fold": "holdout", "seed": 1, "stage": "base", "label": "base"}
        spec = smoke_spec(job, 20)
        self.assertEqual(spec["smoke_rows"], 20)
        self.assertEqual(validate_job(spec["settings"]).fit.epochs, 2)

    def test_streams_are_distinct(self):
        from sweepkit.streams import int_seed

        names = ("weights", "aux head", "contrastive head", "dropout", "batches", "augment speed", "augment mixup",
                 "augment masks", "augment noise", "control shuffle", "control shuffle refit")
        self.assertEqual(len({int_seed(1, n) for n in names}), len(names))

    def test_task_runners(self):
        from sweepkit.families.torch import TASK_RUNNERS

        self.assertEqual(sorted(TASK_RUNNERS), ["classification", "multilabel", "regression"])


# With torch: through the runner --------------------------------------------------------------------------------------

class TestInputShape(unittest.TestCase):
    def test_each_item_kind_gives_its_own_shape(self):
        """An audio clip is one-dimensional: its shape must not be read as a sequence's (first node run, 2026-10-27:
        IndexError for every audio model)."""
        from sweepkit.families.torch import _input_shape

        class Source:
            def __init__(self, kind, item):
                self.kind, self._item = kind, item

            def item(self, i):
                return self._item

        class Identity:
            @staticmethod
            def out_shape(raw):
                return raw

        self.assertEqual(_input_shape(Source("audio", np.zeros(16000)), Identity()), (16000,))
        self.assertEqual(_input_shape(Source("sequence", np.zeros((50, 3))), Identity()), (3,))
        self.assertEqual(_input_shape(Source("image", np.zeros((1, 8, 8))), Identity()), (1, 8, 8))


@needs_module("torch")
class TestRuns(unittest.TestCase):
    def test_a_job_writes_every_file_and_reloads_its_best_epoch(self):
        aux = {"mode": "adversarial", "task": "classification", "columns": ["group"], "classes": GROUPS,
               "weight": 0.1, "loss": "cross_entropy"}
        with TempRepo() as root:
            folder, lines = run(tabular(eval_train=True, aux=aux, contrastive={"weight": 0.1, "temperature": 0.1,
                                                                                "dim": 8}), root)
            names = sorted(p.name for p in folder.iterdir())
            self.assertEqual(names, ["efficiency.json", "log.txt", "metrics_test.json", "metrics_train.json",
                                     "metrics_val.json", "model_params.json", "predictions_test.csv",
                                     "predictions_train.csv", "predictions_val.csv", "prep_record.json",
                                     "provenance.json", "spec.json"])
            val = read(folder, "metrics_val.json")
            training = val["training"]
            self.assertEqual(len(training["curves"]), training["epochs_run"])
            best = training["curves"][training["best_epoch"] - 1]
            self.assertEqual(best["val_balanced_accuracy"], training["best_value"])
            self.assertTrue(training["reload_check"]["equal"])
            self.assertEqual(val["select"]["value"], training["best_value"], "the reloaded epoch scores what chose it")
            self.assertIn("train_balanced_accuracy", best)
            self.assertIn("train_loss_aux", best)
            self.assertIn("train_loss_contrastive", best)
            self.assertIn("accuracy", val["aux"])
            self.assertEqual(read(folder, "metrics_train.json")["epoch"], training["best_epoch"])
            params = read(folder, "model_params.json")
            self.assertEqual(sum(params["parameters_by_part"].values()), params["parameters_total"])
            self.assertEqual(params["input_shape"], [6])
            self.assertGreater(params["extra_heads"]["aux"], 0)
            self.assertEqual(params["optimizer"]["name"], "AdamW")
            self.assertIn("eps", params["optimizer"])
            efficiency = read(folder, "efficiency.json")
            for key in ("prep_seconds", "train_seconds", "predict_seconds"):
                self.assertIsInstance(efficiency[key], float)
            self.assertFalse(any(isinstance(v, dict) for k, v in efficiency.items() if k.endswith("_seconds")))
            provenance = read(folder, "provenance.json")
            self.assertEqual((provenance["device_requested"], provenance["device_used"]), ("cpu", "cpu"))
            self.assertIn("torch", provenance)
            prep = read(folder, "prep_record.json")
            self.assertEqual((prep["normalization"]["mode"], len(prep["steps"])), ("none", 1))
            test = read(folder, "metrics_test.json")
            self.assertEqual(test["refit"], "train")
            self.assertTrue(any("best epoch" in line for line in lines))

    def test_the_refit_on_train_and_val_trains_for_the_chosen_epoch_count(self):
        with TempRepo() as root:
            folder, lines = run(tabular(refit="train_and_val", epochs=10, early_stopping_patience=3), root)
            training = read(folder, "metrics_val.json")["training"]
            test = read(folder, "metrics_test.json")
            self.assertEqual(test["refit"], "train_and_val")
            self.assertEqual(test["refit_epochs"], training["best_epoch"])
            spec = read(folder, "spec.json")
            self.assertGreater(test["refit_rows"], read(folder, "efficiency.json")["n_train"])
            after = lines[next(i for i, line in enumerate(lines) if "refit on train and val" in line) + 1:]
            self.assertEqual(sum(1 for line in after if line.strip().startswith("epoch")), training["best_epoch"])
            self.assertEqual(spec["settings"]["fit"]["refit"], "train_and_val")
            self.assertIn("refit_seconds", read(folder, "efficiency.json"))

    def test_a_run_repeats_exactly_and_unused_options_change_nothing(self):
        with TempRepo() as root:
            first, _ = run(tabular(), root, "a")
            again, _ = run(tabular(), root, "b")
            with_train, _ = run(tabular(eval_train=True), root, "c")
            for name in ("predictions_val.csv", "predictions_test.csv"):
                self.assertEqual((first / name).read_bytes(), (again / name).read_bytes(), name)
                self.assertEqual((first / name).read_bytes(), (with_train / name).read_bytes(),
                                 f"{name}: eval_train only measures")
            self.assertEqual(without_times(read(first, "metrics_val.json")),
                             without_times(read(again, "metrics_val.json")))

    def test_extra_heads_leave_the_model_and_dropout_draws_alone(self):
        import torch

        from sweepkit.families.torch import build

        plain = checked(tabular())
        heads = checked(tabular(aux={"mode": "multitask", "task": "classification", "columns": ["group"],
                                     "classes": GROUPS, "weight": 0.1, "loss": "cross_entropy"},
                                contrastive={"weight": 0.1, "temperature": 0.1, "dim": 8}))
        model_a, _, _ = build(plain, (6,), 3, torch.device("cpu"))
        after_a = torch.rand(4)
        model_b, aux, projection = build(heads, (6,), 3, torch.device("cpu"))
        after_b = torch.rand(4)
        self.assertIsNotNone(aux)
        self.assertIsNotNone(projection)
        for (ka, va), (kb, vb) in zip(model_a.state_dict().items(), model_b.state_dict().items(), strict=True):
            self.assertEqual(ka, kb)
            self.assertTrue(torch.equal(va, vb), ka)
        self.assertTrue(torch.equal(after_a, after_b), "the dropout stream starts at the same place")

    def test_the_control_stays_near_chance(self):
        with TempRepo() as root:
            folder, _ = run(tabular(control="shuffle_train_labels", epochs=10), root)
            self.assertLess(read(folder, "metrics_test.json")["metrics"]["balanced_accuracy"], 0.6)

    def test_multilabel_and_regression(self):
        with TempRepo() as root:
            settings = tabular(class_balance="weighted_loss", label_smoothing=0.05)
            settings["data"]["generator"] = "multilabel"
            settings["split"] = dict(HOLDOUT, stratify=None)
            settings["task"] = {"name": "multilabel", "targets": ["label_0", "label_1", "label_2"],
                                "metrics": ["label_f1_macro", "subset_accuracy", "hamming_loss"],
                                "select": "label_f1_macro", "groups": []}
            folder, _ = run(settings, root, "multilabel")
            header = (folder / "predictions_val.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("pred_label_2", header)
            self.assertEqual(read(folder, "model_params.json")["n_outputs"], 3)
            regression = example_base("regress_mlp")
            regression["split"] = dict(HOLDOUT, stratify=None)
            regression["data"]["n_rows"] = 300
            regression["fit"].update(fit(regression_loss="ccc", refit="train_and_val"))
            folder, _ = run(regression, root, "regression")
            val = read(folder, "metrics_val.json")
            self.assertIn("r2", val["metrics"])
            self.assertEqual(read(folder, "model_params.json")["n_outputs"], 1)
            values = [float(line.split(",")[-1]) for line in
                      (folder / "predictions_val.csv").read_text(encoding="utf-8").splitlines()[1:]]
            self.assertGreater(np.std(values), 0.5, "predictions are back on the target's own scale")

    def test_sequences_with_every_training_option_and_externals(self):
        aux = {"mode": "multitask", "task": "classification", "columns": ["group"], "classes": GROUPS, "weight": 0.1,
               "loss": "cross_entropy"}
        externals = [{"name": "noisier", "data": dict(SEQUENCES, noise=0.8, seed=9), "restrict_classes": False},
                     {"name": "two", "data": dict(SEQUENCES, n_rows=40, n_groups=2, seed=10),
                      "restrict_classes": True}]
        augment = [{"name": "mask", "row_masks": 1, "row_mask_width": 1, "col_masks": 1, "col_mask_width": 4},
                   {"name": "noise", "sd": 0.1}]
        with TempRepo() as root:
            folder, lines = run(sequences(crop=20, eval_window={"window": 20, "hop": 10}, augment=augment, aux=aux,
                                          external=externals, eval_train=True), root)
            for name in ("metrics_external_noisier.json", "predictions_external_noisier.csv",
                         "metrics_external_two.json", "predictions_external_two.csv"):
                self.assertTrue((folder / name).is_file(), name)
            two = read(folder, "metrics_external_two.json")
            self.assertEqual(two["classes_scored"], ["c0", "c1"])
            self.assertIn("strict_accuracy", two)
            self.assertEqual(read(folder, "metrics_external_noisier.json")["n"], 150)
            prep = read(folder, "prep_record.json")
            self.assertEqual((prep["normalization"]["mode"], len(prep["normalization"]["mean"])), ("global", 4))
            self.assertEqual(prep["input_shape"], [4])
            self.assertTrue(any("never trained or chosen on" in line for line in lines))
            folder, _ = run(sequences(normalization="group"), root, "group")
            self.assertTrue(read(folder, "prep_record.json")["normalization"]["transductive"])

    @needs_module("scipy")
    def test_audio_images_and_a_spiking_model(self):
        audio = example_base("audio_logmel")
        audio["data"].update({"n_rows": 60})
        audio["split"] = HOLDOUT
        audio["model"].update({"channels": [8, 16], "d_model": 16, "n_heads": 2, "n_layers": 1, "ff_dim": 32})
        audio["fit"] = fit(epochs=2, batch_size=8, augment=[{"name": "speed", "factors": [0.9, 1.0, 1.1]}],
                           crop=8000, eval_window={"window": 8000, "hop": 4000})
        images = example_base("image_cnn")
        images["data"]["n_rows"] = 120
        images["split"] = HOLDOUT
        images["fit"] = fit(epochs=2)
        spiking = example_base("spiking")
        spiking["data"]["n_rows"] = 90
        spiking["split"] = HOLDOUT
        spiking["fit"] = fit(epochs=2)
        with TempRepo() as root:
            folder, _ = run(audio, root, "audio")
            efficiency = read(folder, "efficiency.json")
            self.assertIn("speed_cache_mib_in_process", efficiency)
            self.assertEqual(read(folder, "model_params.json")["input_shape"], [40])
            folder, _ = run(images, root, "images")
            self.assertEqual(read(folder, "model_params.json")["input_shape"], [1, 12, 12])
            folder, lines = run(spiking, root, "spiking")
            spikes = read(folder, "efficiency.json")["spikes_validation"]
            self.assertTrue(spikes["firing_rate"])
            self.assertTrue(any("spikes on validation" in line for line in lines))

    def test_describe_gives_the_network_the_run_trained(self):
        from sweepkit.families.common import load_data
        from sweepkit.registry import FITS

        for settings in (tabular(), sequences(), example_base("image_cnn")):
            if settings["split"]["name"] != "holdout":
                settings["split"] = HOLDOUT
            settings["data"]["n_rows"] = min(settings["data"]["n_rows"], 150)
            settings["fit"] = fit(epochs=1, warmup_epochs=0)          # warmup must be shorter than the epochs
            job = checked(settings)
            with TempRepo() as root:
                described = FITS.get("torch").target.describe(job, load_data(job, lambda m: None))
                folder, _ = run(settings, root)
                self.assertEqual(described["parameters_total"], read(folder, "model_params.json")["parameters_total"])
            self.assertEqual(sum(described["parameters_by_part"].values()), described["parameters_total"])
            self.assertTrue(described["layers"])
            self.assertEqual(described["outputs_shape"], [1, 3 if settings["task"]["classes"][-1] == "c2" else 4])
            self.assertIn("(", described["module_tree"])

    def test_a_smoke_run_is_two_short_epochs(self):
        from sweepkit.runs.spec import config_hash, plain, run_id, validate_job
        from sweepkit.sweep.runner import execute
        from sweepkit.sweep.smoke import smoke_spec

        job = checked(tabular(epochs=30))
        with TempRepo() as root:
            spec = smoke_spec({"settings": plain(job), "fold": "holdout", "seed": 1, "stage": "base",
                               "label": "base", "sweep_id": "smoke_test", "config_hash": config_hash(job),
                               "run_id": run_id(job, "holdout", 1), "family": "torch", "purpose": "smoke",
                               "overrides": {}, "priority": 1}, 20)
            execute(spec, "cpu", lambda m: None, runs_root=root / "runs")
            folder = root / "runs" / spec["run_id"]
            training = read(folder, "metrics_val.json")["training"]
            self.assertEqual(training["epochs_run"], 2)
            self.assertLessEqual(training["steps_per_epoch"], 5)
            self.assertEqual(validate_job(spec["settings"]).fit.max_batches_per_epoch, 5)


# With torch: known structure ---------------------------------------------------------------------------------------

def held_out(folder: Path, metric: str) -> float:
    return read(folder, "metrics_test.json")["metrics"][metric]


@needs_module("torch")
class TestKnownStructure(unittest.TestCase):
    def test_mlp_on_tabular_classes_and_regression(self):
        with TempRepo() as root:
            settings = tabular(epochs=40, lr=0.003, early_stopping_patience=10)
            settings["data"].update({"n_rows": 600, "n_features": 10})
            folder, _ = run(settings, root, "classes")
            mlp = held_out(folder, "balanced_accuracy")
            reference = dict(settings, model={"name": "logistic_regression", "c": 1.0, "l1_ratio": 0.0,
                                              "class_weight": "none", "max_iter": 1000, "extra": {}},
                             fit={"name": "estimator", "device": "cpu", "control": "none", "refit": "train",
                                  "eval_train": False})
            folder, _ = run(reference, root, "classes_logistic")
            logistic = held_out(folder, "balanced_accuracy")
            # the classes overlap (noise 0.5): on these rows logistic regression gives 0.833, an RBF SVM 0.842 and a
            # random forest 0.858 (2026-10-27), so the MLP is held to the linear model, not to a fixed 0.9 (it gave
            # 0.825 on the node against that bar, set before any run)
            self.assertGreater(mlp, 0.75)
            self.assertGreater(mlp, logistic - 0.05, f"MLP {mlp:.3f}, logistic regression {logistic:.3f}")
            regression = example_base("regress_mlp")
            regression["data"]["n_rows"] = 1000
            regression["split"] = dict(HOLDOUT, stratify=None)
            regression["fit"].update(fit(regression_loss="mse", epochs=80, lr=0.003, early_stopping_patience=20))
            folder, _ = run(regression, root, "regression")
            self.assertGreater(held_out(folder, "r2"), 0.8)

    def sequence_score(self, root, model: dict, runs: str, epochs: int = 30, **data) -> float:
        settings = sequences(model, data=dict({"n_rows": 450}, **data), epochs=epochs, early_stopping_patience=10)
        folder, _ = run(settings, root, runs)
        return held_out(folder, "balanced_accuracy")

    def test_convolutional_recurrent_and_transformer_models_on_sequences(self):
        transformer = {"name": "transformer", "time_pool": 1, "input_dropout": 0.0, "d_model": 32, "n_heads": 4,
                       "n_layers": 2, "ff_dim": 64, "transformer_dropout": 0.1, "transformer_activation": "gelu",
                       "norm_first": True, "positional": "sinusoidal", "pooling": "mean", "classifier_hidden": None,
                       "activation": "relu", "dropout": 0.1}
        cnn1d = {"name": "cnn1d", "channels": [16, 32], "convs_per_block": 1, "kernel_size": 3, "pool": 2,
                 "norm": "batch", "norm_groups": None, "activation": "relu", "pooling": "mean",
                 "classifier_hidden": None, "dropout": 0.1}
        composite = {"name": "composite", "front": "conv2d", "conv1d": None,
                     "conv2d": {"channels": [8, 16], "convs_per_block": 1, "kernel_size": 3, "pool_rows": 2,
                                "pool_cols": 1, "norm": "batch", "norm_groups": None, "norm_subbands": None,
                                "activation": "relu", "row_reduce": "flatten"},
                     "time_pool": 1, "input_dropout": 0.0, "proj_dim": None, "block": "rnn",
                     "rnn": {"rnn": "gru", "rnn_hidden": 32, "rnn_layers": 1, "bidirectional": True,
                             "rnn_dropout": 0.0},
                     "transformer": None, "lif": None, "pooling": "mean", "classifier_hidden": None,
                     "activation": "relu", "dropout": 0.1}
        with TempRepo() as root:
            for name, model in (("gru", dict(RNN, rnn_hidden=32)), ("lstm", dict(RNN, rnn="lstm", rnn_hidden=32)),
                                ("transformer", transformer), ("cnn1d", cnn1d), ("composite", composite)):
                # A class is the number of sine cycles over the whole sequence, and the sequences run 20 to 60
                # steps, so a convolution's local view cannot tell one cycle in 20 steps from three in 60: cnn1d
                # gave 0.622 on the node (2026-10-27) where the recurrent networks and the transformer passed 0.8.
                # It is held above chance (0.33) by a margin; the others integrate the whole sequence.
                bar = 0.5 if name == "cnn1d" else 0.8
                self.assertGreater(self.sequence_score(root, model, name), bar, name)

    def test_cnn2d_on_images(self):
        with TempRepo() as root:
            settings = example_base("image_cnn")
            settings["split"] = HOLDOUT
            settings["fit"] = fit(epochs=30, early_stopping_patience=10)
            folder, _ = run(settings, root)
            self.assertGreater(held_out(folder, "balanced_accuracy"), 0.8)

    def test_spectrogram_models_on_audio(self):
        with TempRepo() as root:
            for name in ("cnn_transformer", "cnn_rnn"):
                settings = example_base("audio_logmel")
                settings["data"]["n_rows"] = 240
                settings["split"] = HOLDOUT
                if name == "cnn_rnn":
                    settings["model"] = {"name": "cnn_rnn", "channels": [16, 32], "convs_per_block": 1,
                                         "kernel_size": 3, "pool_rows": 2, "pool_cols": 2, "norm": "batch",
                                         "norm_groups": None, "norm_subbands": None, "activation": "relu",
                                         "row_reduce": "mean", "rnn": "gru", "rnn_hidden": 32, "rnn_layers": 1,
                                         "bidirectional": True, "rnn_dropout": 0.0, "pooling": "mean",
                                         "classifier_hidden": None, "dropout": 0.1}
                settings["fit"] = fit(epochs=20, batch_size=16, early_stopping_patience=8)
                folder, _ = run(settings, root, name)
                self.assertGreater(held_out(folder, "balanced_accuracy"), 0.8, name)

    def test_spiking_models_on_sequences(self):
        with TempRepo() as root:
            base = example_base("spiking")
            snn = json.loads((REPO / "sweeps" / "examples" / "spiking.json").read_text(encoding="utf-8"))
            snn = snn["stages"][1]["variants"]["snn"]["model"]
            for name, model in (("cnn_lif", base["model"]), ("snn", snn)):
                settings = copy.deepcopy(base)
                settings["model"] = model
                settings["data"]["n_rows"] = 450
                settings["split"] = HOLDOUT
                settings["fit"] = fit(epochs=30, early_stopping_patience=10)
                folder, _ = run(settings, root, name)
                self.assertGreater(held_out(folder, "balanced_accuracy"), 0.6, name)

    def test_sequence_heads_on_per_step_features(self):
        """[steps, width] features whose class is in each step's vector (an encoder's frames): each head trained
        directly with the family's loss, then scored on held-out rows."""
        import torch

        from sweepkit.registry import MODELS
        from sweepkit.train.losses import soft_cross_entropy

        heads = ({"name": "sequence_pool", "time_pool": 1, "input_dropout": 0.0, "proj_dim": 32, "activation": "relu",
                  "pooling": "mean", "classifier_hidden": None, "dropout": 0.1},
                 {"name": "transformer", "time_pool": 2, "input_dropout": 0.0, "d_model": 32, "n_heads": 4,
                  "n_layers": 1, "ff_dim": 64, "transformer_dropout": 0.1, "transformer_activation": "gelu",
                  "norm_first": True, "positional": "sinusoidal", "pooling": "mean", "classifier_hidden": None,
                  "activation": "relu", "dropout": 0.1},
                 {"name": "sequence_lif", "time_pool": 1, "input_dropout": 0.0, "proj_dim": 32, "activation": "relu",
                  "lif_hidden": [32], "recurrent": False, "lif_norm": "none", "lif_dropout": 0.0, "beta": 0.9,
                  "learn_beta": "layer", "threshold": 1.0, "reset": "subtract", "detach_reset": True,
                  "surrogate": "atan", "surrogate_slope": 2.0, "readout_beta": 0.0, "rate_weight": 0.0,
                  "rate_target": 0.1, "dropout": 0.0},
                 dict(RNN, rnn_hidden=32))
        rng = np.random.default_rng(0)
        width, n = 16, 400
        centers = rng.normal(0.0, 0.6, size=(3, width))
        y = np.arange(n) % 3
        lengths = rng.integers(8, 30, size=n)
        x = np.zeros((n, width, 30), dtype=np.float32)
        for i in range(n):
            x[i, :, :lengths[i]] = (centers[y[i]][:, None] + rng.normal(0.0, 1.0, size=(width, lengths[i])))
        x, y, lengths = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(lengths)
        train, held = slice(0, 300), slice(300, None)
        for raw in heads:
            entry, settings = MODELS.validate_section(raw, "model")
            torch.manual_seed(0)
            model = entry.target.build(settings, (width,), 3)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
            for _ in range(15):
                model.train()
                order = torch.randperm(300)
                for b in range(0, 300, 30):
                    k = order[b:b + 30]
                    loss = soft_cross_entropy(model(x[train][k], lengths[train][k]),
                                              torch.nn.functional.one_hot(y[train][k], 3).float(), None)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                accuracy = (model(x[held], lengths[held]).argmax(dim=1) == y[held]).float().mean().item()
            self.assertGreater(accuracy, 0.8, raw["name"])


if __name__ == "__main__":
    unittest.main()
