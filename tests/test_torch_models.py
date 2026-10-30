"""The torch family's parts: models, features, batches, augmentations, losses and the spiking layers.

Without torch (runs everywhere): every torch component's declaration and its example sweep, that the catalog loads
without importing torch, the models' and features' checks, the mel filterbank, the batches and windows, the numpy
parts of the augmentations, and the references the torch tests compare with. With torch (the node, where
SWEEPKIT_REQUIRE names torch): every model's forward and backward pass, that padding changes no row's output, the LIF
neuron against the numpy reference forward and backward, the losses against references, the log-mel against librosa,
normalization, masks, mixup, prediction on averaged windows, and one training step through sweepkit.train.loop.Step.
"""
import json
import math
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np

from tests.support import REPO, load_catalog, needs_module

load_catalog()

# Model settings used below, as a sweep file gives them
CNN2D = {"name": "cnn2d", "channels": [8, 16, 32], "convs_per_block": 1, "kernel_size": 3, "pool_rows": 2,
         "pool_cols": 2, "pooling": "mean", "classifier_hidden": None, "dropout": 0.1, "norm": "batch",
         "norm_groups": None, "norm_subbands": None, "activation": "relu"}
CNN1D = {"name": "cnn1d", "channels": [16, 32], "convs_per_block": 1, "kernel_size": 3, "pool": 2, "norm": "batch",
         "norm_groups": None, "activation": "relu", "pooling": "mean", "classifier_hidden": None, "dropout": 0.1}
MLP = {"name": "mlp", "hidden": [32, 16], "norm": "batch", "activation": "relu", "dropout": 0.1}
POOL = {"name": "sequence_pool", "time_pool": 1, "input_dropout": 0.1, "proj_dim": 64, "activation": "gelu",
        "pooling": "attention", "classifier_hidden": None, "dropout": 0.2}
TRANSFORMER = {"name": "transformer", "time_pool": 2, "input_dropout": 0.1, "d_model": 32, "n_heads": 4,
               "n_layers": 2, "ff_dim": 64, "transformer_dropout": 0.1, "transformer_activation": "gelu",
               "norm_first": True, "positional": "sinusoidal", "pooling": "stats", "classifier_hidden": 16,
               "activation": "relu", "dropout": 0.2}
SEQ_LIF = {"name": "sequence_lif", "time_pool": 2, "input_dropout": 0.0, "proj_dim": 64, "activation": "relu",
           "lif_hidden": [48, 32], "recurrent": True, "lif_norm": "layer", "lif_dropout": 0.1, "beta": 0.9,
           "learn_beta": "layer", "threshold": 1.0, "reset": "subtract", "detach_reset": False, "surrogate": "atan",
           "surrogate_slope": 2.0, "readout_beta": 0.5, "rate_weight": 0.5, "rate_target": 0.1, "dropout": 0.2}
RNN = {"name": "rnn", "time_pool": 1, "input_dropout": 0.1, "proj_dim": None, "activation": "relu", "rnn": "gru",
       "rnn_hidden": 24, "rnn_layers": 2, "bidirectional": True, "rnn_dropout": 0.1, "pooling": "attention",
       "classifier_hidden": None, "dropout": 0.2}
CNN_RNN = {"name": "cnn_rnn", "channels": [16, 32], "convs_per_block": 1, "kernel_size": 3, "pool_rows": 2,
           "pool_cols": 2, "norm": "batch", "norm_groups": None, "norm_subbands": None, "activation": "relu",
           "row_reduce": "mean", "rnn": "gru", "rnn_hidden": 24, "rnn_layers": 2, "bidirectional": True,
           "rnn_dropout": 0.1, "pooling": "attention", "classifier_hidden": None, "dropout": 0.2}
CNN_TRANSFORMER = {"name": "cnn_transformer", "channels": [8, 16], "convs_per_block": 1, "kernel_size": 3,
                   "pool_rows": 2, "pool_cols": 2, "norm": "batch", "norm_groups": None, "norm_subbands": None,
                   "activation": "relu", "row_reduce": "mean", "d_model": 32, "n_heads": 4, "n_layers": 2,
                   "ff_dim": 64, "transformer_dropout": 0.1, "transformer_activation": "gelu", "norm_first": True,
                   "positional": "sinusoidal", "pooling": "mean", "classifier_hidden": None, "dropout": 0.1}
NEURON = {"lif_hidden": [32], "recurrent": False, "lif_norm": "none", "lif_dropout": 0.1, "beta": 0.9,
          "learn_beta": "none", "threshold": 1.0, "reset": "subtract", "detach_reset": True, "surrogate": "atan",
          "surrogate_slope": 2.0, "readout_beta": 0.8, "rate_weight": 0.0, "rate_target": 0.1}
COMPOSITE = {"name": "composite", "front": "none", "conv1d": None, "conv2d": None, "time_pool": 1,
             "input_dropout": 0.0, "proj_dim": None, "block": "none", "rnn": None, "transformer": None, "lif": None,
             "pooling": "mean", "classifier_hidden": None, "activation": "relu", "dropout": 0.1}
CONV1D_BLOCK = {"channels": [8, 16], "convs_per_block": 1, "kernel_size": 3, "pool": 2, "norm": "batch",
                "norm_groups": None, "activation": "relu"}
CONV2D_BLOCK = {"channels": [8, 16], "convs_per_block": 1, "kernel_size": 3, "pool_rows": 2, "pool_cols": 2,
                "norm": "batch", "norm_groups": None, "norm_subbands": None, "activation": "relu",
                "row_reduce": "flatten"}
RNN_BLOCK = {"rnn": "lstm", "rnn_hidden": 16, "rnn_layers": 1, "bidirectional": True, "rnn_dropout": 0.0}
TRANSFORMER_BLOCK = {"d_model": 32, "n_heads": 4, "n_layers": 1, "ff_dim": 64, "transformer_dropout": 0.1,
                     "transformer_activation": "relu", "norm_first": False, "positional": "sinusoidal"}
COMPOSITES = (dict(COMPOSITE), dict(COMPOSITE, pooling="stats", proj_dim=16),
              dict(COMPOSITE, front="conv1d", conv1d=CONV1D_BLOCK, block="rnn", rnn=RNN_BLOCK, pooling="last"),
              dict(COMPOSITE, front="conv2d", conv2d=CONV2D_BLOCK, block="transformer", transformer=TRANSFORMER_BLOCK),
              dict(COMPOSITE, front="conv1d", conv1d=CONV1D_BLOCK, block="transformer", transformer=TRANSFORMER_BLOCK,
                   pooling="attention"),
              dict(COMPOSITE, front="conv2d", conv2d=dict(CONV2D_BLOCK, row_reduce="mean"), block="rnn",
                   rnn=dict(RNN_BLOCK, rnn="gru"), pooling="max"),
              dict(COMPOSITE, proj_dim=32, block="lif", lif=dict(NEURON, recurrent=True, lif_norm="batch"),
                   pooling="readout"))


def settings_of(raw: dict):
    from sweepkit.registry import MODELS

    return MODELS.validate_section(raw, "model")


def check_model(raw: dict, input_shape) -> None:
    entry, settings = settings_of(raw)
    entry.target.check(settings, input_shape)


def build(raw: dict, input_shape, n_outputs: int = 4, seed: int = 0):
    import torch

    entry, settings = settings_of(raw)
    torch.manual_seed(seed)
    return entry.target.build(settings, input_shape, n_outputs)


def items_of(arrays: list, kind: str):
    from sweepkit.data.base import ArrayItems
    from sweepkit.train.batches import ItemCache

    return ItemCache(ArrayItems(list(arrays)), kind)


def reference_contrastive(z, labels, temperature):
    """L_out of Khosla et al. (2020), equation 2, with one view, written as loops: for each anchor i, A(i) is every
    other row and P(i) the rows of A(i) with i's label; anchors with an empty P(i) are left out, and the result is the
    mean over the other anchors (0 when there is none). Rows are L2-normalized first; float64."""
    z = np.asarray(z, dtype=np.float64)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    n = z.shape[0]
    losses = []
    for i in range(n):
        others = [a for a in range(n) if a != i]
        positives = [p for p in others if labels[p] == labels[i]]
        if not positives:
            continue
        denominator = sum(math.exp(z[i] @ z[a] / temperature) for a in others)
        losses.append(-sum(math.log(math.exp(z[i] @ z[p] / temperature) / denominator) for p in positives)
                      / len(positives))
    return float(np.mean(losses)) if losses else 0.0


# Without torch --------------------------------------------------------------------------------------------------------

class TestDeclarations(unittest.TestCase):
    def test_every_torch_component_is_declared_and_in_its_example(self):
        from sweepkit.registry import FEATURES, MODELS, from_template
        from sweepkit.sweep.design import expand

        named = {}
        for name in MODELS.names():
            meta = MODELS.get(name).meta
            # the template's own torch models; a project's (src/project/) have their examples in sweeps/project/
            if meta["family"] != "torch" or not from_template(MODELS.get(name).target):
                continue
            # every torch model needs torch alone, except fine-tuning a pretrained model (the pretrained profile)
            expected = (("torch", "pretrained", ("torch", "transformers", "peft")) if name == "hf_finetune"
                        else ("torch", "torch", ("torch",)))
            self.assertEqual((meta["gpu"], meta["profile"], meta["packages"]), expected, name)
            self.assertEqual(meta["tasks"], ("classification", "multilabel", "regression"), name)
            named.setdefault(meta["example"], set()).add(("model", name))
        for name in FEATURES.names():
            meta = FEATURES.get(name).meta
            if "torch" in meta["packages"] and from_template(FEATURES.get(name).target):
                named.setdefault(meta["example"], set()).add(("features", name))
        self.assertGreaterEqual(len(named), 5)
        for example, parts in sorted(named.items()):
            path = REPO / "sweeps" / "examples" / f"{example}.json"
            self.assertTrue(path.is_file(), f"no example sweep {path.name}")
            sweep = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((sweep["sweep_id"], sweep["purpose"]), (f"example_{example}", "learning"))
            jobs, _ = expand(sweep)
            used = ({("model", j["settings"]["model"]["name"]) for j in jobs}
                    | {("features", j["settings"]["features"]["name"]) for j in jobs})
            self.assertTrue(parts <= used, f"{path.name} leaves out {sorted(parts - used)}")
            for job in jobs:
                self.assertEqual(job["settings"]["data"]["name"], "synthetic")
                self.assertTrue(all(e["data"]["name"] == "synthetic" for e in job["settings"]["fit"]["external"]))
            self.assertEqual(sorted({j["seed"] for j in jobs}), [1, 2, 3])

    def test_the_catalog_loads_without_importing_torch(self):
        code = "from sweepkit import catalog; catalog.load_all(); import sys; print('torch' in sys.modules)"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]))
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO,
                              timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "False")


class TestModelChecks(unittest.TestCase):
    def test_cnn2d_shapes_norms_and_heads(self):
        check_model(CNN2D, (40,))
        check_model(CNN2D, (1, 12, 12))
        check_model(CNN2D, (3, 16, 16))
        check_model(CNN2D, None)
        for raw, shape in ((dict(CNN2D, channels=[8, 8, 8, 8]), (1, 12, 12)),       # 12 rows halved four times
                           (dict(CNN2D, channels=[8] * 6), (40,)), (CNN2D, (1, 12)), (dict(CNN2D, kernel_size=4), None),
                           (dict(CNN2D, dropout=1.0), None), (dict(CNN2D, classifier_hidden=0), None),
                           (dict(CNN2D, pool_rows=0), None), (dict(CNN2D, channels=[]), None)):
            with self.assertRaises(ValueError, msg=str((raw, shape))):
                check_model(raw, shape)
        bad = dict(CNN2D, channels=[8, 12])
        for norm, groups, subbands in (("group", 8, None), ("group", None, None), ("subspectral", None, 1),
                                       ("subspectral", None, 32), ("batch", 4, None), ("none", None, 2)):
            with self.assertRaises(ValueError, msg=str((norm, groups, subbands))):
                check_model(dict(bad, norm=norm, norm_groups=groups, norm_subbands=subbands), (40,))
        check_model(dict(CNN2D, norm="group", norm_groups=8), (40,))
        check_model(dict(CNN2D, norm="subspectral", norm_subbands=4), (40,))

    def test_sequence_models(self):
        for raw in (CNN1D, POOL, TRANSFORMER, SEQ_LIF, RNN, CNN_RNN, CNN_TRANSFORMER) + COMPOSITES:
            check_model(raw, (40,))
            check_model(raw, None)
            with self.assertRaises(ValueError, msg=raw["name"]):
                check_model(raw, (1, 12, 12))
        for raw in (dict(CNN1D, kernel_size=2), dict(CNN1D, norm="group", norm_groups=3), dict(CNN1D, norm_groups=4),
                    dict(CNN1D, pool=0), dict(POOL, proj_dim=0), dict(POOL, time_pool=0), dict(POOL, input_dropout=1.0),
                    dict(TRANSFORMER, n_heads=5), dict(TRANSFORMER, d_model=33, n_heads=3), dict(TRANSFORMER, n_layers=0),
                    dict(TRANSFORMER, transformer_dropout=1.0), dict(RNN, rnn_layers=1), dict(RNN, rnn_layers=5),
                    dict(RNN, rnn_hidden=0), dict(SEQ_LIF, lif_hidden=[]), dict(SEQ_LIF, beta=1.0),
                    dict(CNN_RNN, rnn_layers=1), dict(CNN_RNN, channels=[16, 32, 64, 128, 256, 512]),
                    dict(CNN_TRANSFORMER, n_heads=5)):
            with self.assertRaises(ValueError, msg=str(raw)):
                check_model(raw, (40,))

    def test_composite_takes_exactly_the_blocks_it_names(self):
        for raw in (dict(COMPOSITE, front="conv1d"), dict(COMPOSITE, conv1d=CONV1D_BLOCK),
                    dict(COMPOSITE, block="rnn"), dict(COMPOSITE, rnn=RNN_BLOCK),
                    dict(COMPOSITE, pooling="last"),
                    dict(COMPOSITE, block="transformer", transformer=TRANSFORMER_BLOCK, pooling="last"),
                    dict(COMPOSITE, pooling="readout"), dict(COMPOSITE, block="lif", lif=NEURON),
                    dict(COMPOSITE, block="lif", lif=dict(NEURON, lif_hidden=[]), pooling="readout"),
                    dict(COMPOSITE, front="conv2d", conv2d=dict(CONV2D_BLOCK, channels=[8] * 6))):
            with self.assertRaises(ValueError, msg=str(raw)):
                check_model(raw, (40,))

    def test_mlp(self):
        check_model(MLP, (10,))
        check_model(dict(MLP, hidden=[]), (10,))
        for raw, shape in ((dict(MLP, hidden=[0]), (10,)), (dict(MLP, dropout=1.0), (10,)), (MLP, (4, 3))):
            with self.assertRaises(ValueError):
                check_model(raw, shape)

    def test_declared_input_shapes(self):
        from sweepkit.runs.spec import validate_job
        from sweepkit.train.inputs import declared_input_shape

        examples = REPO / "sweeps" / "examples"
        expected = {"sequence_models": (4,), "image_cnn": (1, 12, 12), "audio_logmel": (40,), "spiking": (8,),
                    "classify_mlp": None}
        for name, shape in expected.items():
            base = json.loads((examples / f"{name}.json").read_text(encoding="utf-8"))["base"]
            self.assertEqual(declared_input_shape(validate_job(base)), shape, name)

    def test_a_model_refuses_its_dataset_when_the_sweep_is_expanded(self):
        from sweepkit.runs.spec import check_job, validate_job

        base = json.loads((REPO / "sweeps" / "examples" / "image_cnn.json").read_text(encoding="utf-8"))["base"]
        base["model"]["channels"] = [8, 8, 8, 8]          # 12 by 12 images halved four times leave nothing
        with self.assertRaisesRegex(ValueError, "leave none"):
            check_job(validate_job(base))


class TestFeatureChecks(unittest.TestCase):
    def logmel(self, **changes):
        from sweepkit.features.audio import LogMelSettings

        base = dict(sample_rate=16000, win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                    log_offset=1e-6, normalization="item", group_column=None)
        return LogMelSettings(**{**base, **changes})

    def test_filterbank(self):
        from sweepkit.features.audio import hz_to_mel, mel_filterbank, mel_to_hz

        fb = mel_filterbank(16000, 512, 40, 0.0, 8000.0)
        self.assertEqual(fb.shape, (40, 257))
        self.assertTrue((fb >= 0).all())
        self.assertTrue((fb.max(axis=1) > 0).all(), "every band holds a bin")
        peaks = fb.argmax(axis=1)
        self.assertTrue((np.diff(peaks) >= 0).all(), "bands go up in frequency")
        f = np.array([0.0, 500.0, 1000.0, 4000.0, 8000.0])
        self.assertTrue(np.allclose(mel_to_hz(hz_to_mel(f)), f))
        self.assertAlmostEqual(float(hz_to_mel(1000.0)), 15.0)

    def test_checks(self):
        from sweepkit.features.audio import check, window_samples

        check(self.logmel())
        self.assertEqual(window_samples(self.logmel()), (400, 160))
        for bad in ({"n_fft": 256}, {"f_max": 9000.0}, {"f_min": 8000.0}, {"n_mels": 0}, {"log_offset": 0.0},
                    {"n_mels": 256}, {"hop_ms": 0.01}, {"normalization": "group"},
                    {"normalization": "item", "group_column": "group"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check(self.logmel(**bad))

    def test_jobs_check_the_sample_rate(self):
        from sweepkit.runs.spec import check_job, validate_job

        base = json.loads((REPO / "sweeps" / "examples" / "audio_logmel.json").read_text(encoding="utf-8"))["base"]
        self.assertEqual(base["features"]["name"], "logmel")
        check_job(validate_job(base))
        base["features"].update({"sample_rate": 22050, "n_fft": 1024})
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            check_job(validate_job(base))

    def test_group_normalization_is_declared_transductive(self):
        from sweepkit.analysis.transductive import parts

        for name, feature in (("sequence_models", "sequence"), ("image_cnn", "image"), ("audio_logmel", "logmel")):
            base = json.loads((REPO / "sweeps" / "examples" / f"{name}.json").read_text(encoding="utf-8"))["base"]
            self.assertEqual(parts(base), [], name)
            base["features"].update({"normalization": "group", "group_column": "group"})
            self.assertEqual(parts(base), [f"features {feature}"], name)

    @needs_module("librosa")
    def test_filterbank_matches_librosa(self):
        import librosa

        from sweepkit.features.audio import mel_filterbank

        for sr, n_fft, n_mels, f_max in ((16000, 512, 64, 8000.0), (16000, 1024, 128, 4000.0),
                                         (16000, 512, 40, 8000.0), (22050, 1024, 80, 11025.0)):
            ours = mel_filterbank(sr, n_fft, n_mels, 0.0, f_max)
            ref = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=f_max, htk=False,
                                      norm="slaney")
            self.assertEqual(ours.shape, ref.shape)
            self.assertLess(np.abs(ours - ref).max(), 1e-6 * max(1.0, np.abs(ref).max()))


class TestBatches(unittest.TestCase):
    def test_crops_keep_values_and_draw_once_per_long_row(self):
        from sweepkit.train.batches import crop_arrays

        rng = np.random.default_rng(0)
        steps = [rng.standard_normal((n, 6)).astype(np.float32) for n in (30, 12, 50)]
        x, lengths = crop_arrays(steps, 20, np.random.default_rng(5))
        self.assertEqual((x.shape, x.dtype, lengths.tolist()), ((3, 20, 6), np.float32, [20, 12, 20]))
        draws = np.random.default_rng(5)
        start0 = int(draws.integers(0, 30 - 20 + 1))
        start2 = int(draws.integers(0, 50 - 20 + 1))
        self.assertTrue(np.array_equal(x[0], steps[0][start0:start0 + 20]))
        self.assertTrue(np.array_equal(x[1, :12], steps[1]))
        self.assertTrue((x[1, 12:] == 0).all())
        self.assertTrue(np.array_equal(x[2], steps[2][start2:start2 + 20]))
        whole, lengths = crop_arrays(steps, None, None)
        self.assertEqual((whole.shape, lengths.tolist()), ((3, 50, 6), [30, 12, 50]))

    def test_eval_batches_cover_every_row_once_longest_first(self):
        from sweepkit.train.batches import Source, eval_batches

        arrays = [np.full((n, 2), float(n), dtype=np.float32) for n in (5, 9, 3, 7, 4)]
        source = Source("sequence", np.array([4, 0, 3, 1]), items=items_of(arrays, "sequence"))
        seen = []
        for ks, x, lengths in eval_batches(source, 3):
            self.assertEqual(x.shape, (len(ks), 2, int(lengths.max())), "sequences arrive as [batch, channels, steps]")
            for j, k in enumerate(ks):
                n = arrays[source.positions[k]].shape[0]
                self.assertEqual(int(lengths[j]), n)
                self.assertTrue(np.all(x[j, :, :n] == n))
                self.assertTrue((x[j, :, n:] == 0).all())
            seen.extend(ks.tolist())
        self.assertEqual(sorted(seen), [0, 1, 2, 3])
        self.assertEqual([arrays[source.positions[k]].shape[0] for k in seen], [9, 7, 5, 4])
        table = Source("tabular", np.array([2, 0]), table_x=np.arange(6, dtype=np.float32).reshape(2, 3))
        (ks, x, lengths), = list(eval_batches(table, 8))
        self.assertEqual((ks.tolist(), lengths), ([0, 1], None))
        images = Source("image", np.array([1, 0]), items=items_of([np.zeros((1, 4, 4)), np.ones((1, 4, 4))], "image"))
        (ks, x, lengths), = list(eval_batches(images, 8))
        self.assertEqual((x.shape, float(x[0].mean()), lengths), ((2, 1, 4, 4), 1.0, None))

    def test_window_starts(self):
        from sweepkit.train.batches import window_starts

        self.assertEqual(window_starts(500, 1000, 500), [0])
        self.assertEqual(window_starts(1000, 1000, 500), [0])
        self.assertEqual(window_starts(2000, 1000, 500), [0, 500, 1000])
        self.assertEqual(window_starts(2100, 1000, 500), [0, 500, 1000, 1100])
        self.assertEqual(window_starts(1001, 1000, 1000), [0, 1])
        with self.assertRaises(ValueError):
            window_starts(100, 0, 1)

    def test_window_batches_cover_every_window_once(self):
        from sweepkit.train.batches import Source, window_batches, window_starts

        arrays = [np.arange(n, dtype=np.float32) for n in (300, 2100, 1000, 50)]
        source = Source("audio", np.array([1, 3, 0, 2]), items=items_of(arrays, "audio"))
        seen = []
        for ks, x, lengths in window_batches(source, 3, 1000, 500):
            self.assertLessEqual(x.shape[0], 3)
            self.assertEqual(x.shape[1], lengths.max())
            for j, k in enumerate(ks):
                piece = x[j, :lengths[j]]
                whole = arrays[source.positions[k]]
                start = int(piece[0])
                self.assertTrue(np.array_equal(whole[start:start + lengths[j]], piece))
                self.assertTrue(np.all(x[j, lengths[j]:] == 0))
                seen.append((int(k), start))
        expected = [(k, s) for k in range(4) for s in window_starts(arrays[source.positions[k]].shape[0], 1000, 500)]
        self.assertEqual(sorted(seen), sorted(expected))

    def test_class_weights_positive_weights_and_orders(self):
        from sweepkit.train.batches import class_weights, positive_weights, train_order

        labels = np.array([0] * 60 + [1] * 30 + [2] * 10)
        self.assertTrue(np.allclose(class_weights(labels, 3), [100 / 180, 100 / 90, 100 / 30]))
        order = train_order(labels, "balanced_sampler", np.random.default_rng(0), 100)
        share = np.bincount(labels[order], minlength=3) / order.shape[0]
        self.assertLess(np.abs(share - 1 / 3).max(), 0.12)
        self.assertEqual(sorted(train_order(labels, "none", np.random.default_rng(0), 100).tolist()), list(range(100)))
        self.assertEqual(sorted(train_order(None, "weighted_loss", np.random.default_rng(0), 7).tolist()),
                         list(range(7)))
        with self.assertRaises(ValueError):
            class_weights(np.array([0, 0, 1]), 3)
        targets = np.array([[1, 0], [0, 0], [1, 1], [0, 0]])
        self.assertTrue(np.allclose(positive_weights(targets), [1.0, 3.0]))
        with self.assertRaises(ValueError):
            positive_weights(np.array([[1, 0], [1, 0]]))

    def test_item_cache_reads_once_and_checks_shapes(self):
        from sweepkit.data.base import ItemSource
        from sweepkit.train.batches import ItemCache, as_float_item

        class Counting(ItemSource):
            def __init__(self, arrays):
                self.arrays, self.reads = arrays, 0

            def __len__(self):
                return len(self.arrays)

            def get(self, position):
                self.reads += 1
                return self.arrays[position]

        source = Counting([np.ones((5, 3)), np.ones((7, 3)), np.ones((4, 2))])
        cache = ItemCache(source, "sequence")
        self.assertEqual(cache[1].dtype, np.float32)
        cache[1]
        cache[0]
        self.assertEqual(source.reads, 2)
        with self.assertRaisesRegex(ValueError, "channels"):
            cache[2]
        with self.assertRaises(ValueError):
            ItemCache(source, "text")
        ints = as_float_item(np.array([16384, -32768], dtype=np.int16), "audio")
        self.assertEqual(ints.tolist(), [0.5, -1.0])
        with self.assertRaises(ValueError):
            as_float_item(np.zeros((3, 4)), "audio")

    def test_training_batches_draw_crops_and_speeds_from_their_own_streams(self):
        from sweepkit.train.augment import SpeedSettings
        from sweepkit.train.batches import Source, concat_sources, training_batch

        arrays = [np.full(n, 0.25, dtype=np.float32) for n in (900, 2000, 3500, 400)]
        source = Source("audio", np.arange(4), items=items_of(arrays, "audio"))
        picked = np.array([2, 0, 3, 1, 2])
        a, b = np.random.default_rng(5), np.random.default_rng(5)
        spare = np.random.default_rng(9)
        x1, l1 = training_batch(source, picked, 1000, a)
        x2, l2 = training_batch(source, picked, 1000, b, SpeedSettings(factors=(1.0,)), spare)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(l1, l2)
        self.assertEqual(a.integers(0, 10 ** 9), b.integers(0, 10 ** 9), "the crop stream is used alike")
        self.assertEqual(spare.integers(0, 10 ** 9), np.random.default_rng(9).integers(0, 10 ** 9),
                         "a single speed draws nothing")
        table = Source("tabular", np.array([3, 5]), table_x=np.ones((2, 4), dtype=np.float32))
        x, lengths = training_batch(table, np.array([1, 0]), None, None)
        self.assertEqual((x.shape, lengths), ((2, 4), None))
        joined = concat_sources(table, Source("tabular", np.array([1]), table_x=np.zeros((1, 4), dtype=np.float32)))
        self.assertEqual((joined.positions.tolist(), joined.table_x.shape), ([3, 5, 1], (3, 4)))
        with self.assertRaises(ValueError):
            concat_sources(source, table)


class TestAugmentations(unittest.TestCase):
    def test_settings_checks(self):
        from sweepkit.train.augment import (Mask, MaskSettings, Mixup, MixupSettings, Noise, NoiseSettings, Speed,
                                            SpeedSettings, speed_ratio)

        self.assertEqual(speed_ratio(0.9), (10, 9))
        self.assertEqual(speed_ratio(1.1), (10, 11))
        self.assertEqual(speed_ratio(1.0), (1, 1))
        self.assertEqual(speed_ratio(1.05), (20, 21))
        for bad in (0.0, -1.0, 1 / 3 + 1e-4):
            with self.assertRaises(ValueError):
                speed_ratio(bad)
        Speed.check(SpeedSettings(factors=(0.9, 1.0, 1.1)))
        for bad in ((), (1.0, 1.0), (3.0,), (1.0 / 3,)):
            with self.assertRaises(ValueError, msg=str(bad)):
                Speed.check(SpeedSettings(factors=bad))
        Mask.check(MaskSettings(row_masks=2, row_mask_width=5, col_masks=0, col_mask_width=0))
        for bad in ((0, 0, 0, 0), (-1, 2, 1, 2)):
            with self.assertRaises(ValueError):
                Mask.check(MaskSettings(*bad))
        with self.assertRaises(ValueError):
            Noise.check(NoiseSettings(sd=0.0))
        with self.assertRaises(ValueError):
            Mixup.check(MixupSettings(alpha=0.0))
        self.assertEqual([c.order for c in (Speed, Mask, Noise, Mixup)], [0, 1, 2, 3])

    def test_speed_draws(self):
        from sweepkit.train.augment import draw_speeds

        rng = np.random.default_rng(4)
        np.testing.assert_array_equal(draw_speeds((0.9,), 3, rng), [0.9, 0.9, 0.9])
        self.assertEqual(rng.integers(0, 10 ** 9), np.random.default_rng(4).integers(0, 10 ** 9))
        draws = draw_speeds((0.9, 1.0, 1.1), 30000, np.random.default_rng(0))
        for f in (0.9, 1.0, 1.1):
            self.assertAlmostEqual(np.mean(draws == f), 1 / 3, delta=0.01)

    @needs_module("scipy")
    def test_speed_changes_length_and_pitch(self):
        from sweepkit.train.augment import at_speed, change_speed

        rate = 16000
        t = np.arange(2 * rate) / rate
        x = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.assertIs(change_speed(x, 1.0), x)
        for speed in (0.9, 1.1):
            y = change_speed(x, speed)
            self.assertEqual(y.dtype, np.float32)
            self.assertEqual(y.shape[0], int(np.ceil(x.shape[0] / speed)))
            spectrum = np.abs(np.fft.rfft(y.astype(float) * np.hanning(y.shape[0])))
            peak = np.fft.rfftfreq(y.shape[0], 1 / rate)[spectrum.argmax()]
            self.assertAlmostEqual(peak, 440 * speed, delta=1.0)
            self.assertLess(np.abs(y).max(), 0.25 * 8100 / 8000)
        a = [np.arange(1000, dtype=np.float32), np.arange(500, dtype=np.float32)]
        first = at_speed(a, 0, 1.1)
        self.assertIs(at_speed(a, 0, 1.1), first)
        self.assertIs(at_speed(a, 1, 1.0), a[1])
        b = [np.zeros(1000, dtype=np.float32)]
        self.assertEqual(float(np.abs(at_speed(b, 0, 1.1)).max()), 0.0, "another list gets its own copies")

    @needs_module("scipy")
    def test_training_batch_with_speeds(self):
        from sweepkit.train.augment import SpeedSettings, draw_speeds
        from sweepkit.train.batches import Source, training_batch

        arrays = [np.full(n, 0.5, dtype=np.float32) for n in (900, 2000, 3500)]
        source = Source("audio", np.arange(3), items=items_of(arrays, "audio"))
        picked = np.array([0, 1, 2, 0])
        x, lengths = training_batch(source, picked, 1000, np.random.default_rng(0), SpeedSettings(factors=(0.9, 1.1)),
                                    np.random.default_rng(1))
        self.assertEqual(x.shape, (4, 1000))
        speeds = draw_speeds((0.9, 1.1), 4, np.random.default_rng(1))
        for j, k in enumerate(picked):
            self.assertEqual(lengths[j], min(1000, int(np.ceil(arrays[k].shape[0] / speeds[j]))))
            self.assertTrue(np.all(x[j, lengths[j]:] == 0))

    def test_mixup_draws(self):
        from sweepkit.train.augment import mixup_draw

        weights, partner = mixup_draw(0.4, 1000, np.random.default_rng(0))
        self.assertEqual(weights.dtype, np.float32)
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())
        self.assertEqual(sorted(partner.tolist()), list(range(1000)))
        self.assertAlmostEqual(float(weights.mean()), 0.5, delta=0.05)
        again = mixup_draw(0.4, 1000, np.random.default_rng(0))
        np.testing.assert_array_equal(again[0], weights)
        with self.assertRaises(ValueError):
            mixup_draw(0.0, 3, np.random.default_rng(0))


class TestReferences(unittest.TestCase):
    def test_contrastive_reference_by_hand(self):
        # anchors 0 and 1 have each other as their positive and row 2 at -1; row 2 has no positive
        self.assertAlmostEqual(reference_contrastive([[1, 0], [2, 0], [-3, 0]], [0, 0, 1], 0.5),
                               math.log(1 + math.exp(-4)), places=12)
        for t in (0.07, 1.0):
            self.assertAlmostEqual(reference_contrastive(np.eye(3), [0, 0, 0], t), math.log(2), places=12)
        self.assertEqual(reference_contrastive(np.random.default_rng(0).standard_normal((3, 4)), [0, 1, 2], 0.1), 0.0)
        rng = np.random.default_rng(1)
        z, labels = rng.standard_normal((8, 5)), [0, 1, 0, 1, 2, 2, 0, 3]
        self.assertAlmostEqual(reference_contrastive(z, labels, 0.1),
                               reference_contrastive(z * rng.uniform(0.1, 10, size=(8, 1)), labels, 0.1), places=10)

    def test_concordance(self):
        from sweepkit.train.losses import concordance_numpy

        a = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.assertAlmostEqual(float(concordance_numpy(a, a)[0]), 1.0)
        self.assertAlmostEqual(float(concordance_numpy(5.0 - a, a)[0]), -1.0, msg="mirrored about the shared mean")
        # a shift lowers it: cov 1.25, variances 1.25 and 1.25, squared mean difference 1 -> 2.5 / 3.5
        self.assertAlmostEqual(float(concordance_numpy(a + 1.0, a)[0]), 2.5 / 3.5)
        b = np.random.default_rng(0).standard_normal((50, 2))
        c = b + np.random.default_rng(1).standard_normal((50, 2))
        self.assertTrue(np.allclose(concordance_numpy(3 * b + 2, 3 * c + 2), concordance_numpy(b, c)))


# With torch -------------------------------------------------------------------------------------------------------

@needs_module("torch")
@needs_module("librosa")
class TestLogMelAgainstLibrosa(unittest.TestCase):
    def test_logmel_matches_librosa(self):
        import librosa
        import torch

        from sweepkit.features.audio import LogMel, LogMelSettings

        rng = np.random.default_rng(0)
        t = np.arange(16000 * 2) / 16000
        y = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.shape)).astype(np.float32)
        for win_ms, hop_ms, n_fft, n_mels in ((25.0, 10.0, 512, 64), (50.0, 12.5, 1024, 80)):
            s = LogMelSettings(sample_rate=16000, win_ms=win_ms, hop_ms=hop_ms, n_fft=n_fft, n_mels=n_mels, f_min=0.0,
                               f_max=8000.0, log_offset=1e-6, normalization="none", group_column=None)
            feature = LogMel(s, torch.device("cpu"))
            ours = feature(torch.from_numpy(y)[None, :])[0].numpy()
            win, hop = int(round(win_ms * 16)), int(round(hop_ms * 16))
            mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=n_fft, hop_length=hop, win_length=win,
                                                 window="hann", center=True, pad_mode="constant", power=2.0,
                                                 n_mels=n_mels, fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
            ref = np.log(mel + 1e-6)
            self.assertEqual(ours.shape, ref.shape)
            self.assertLess(np.abs(ours - ref).max(), 1e-2)
            self.assertEqual(ours.shape[1], 1 + y.shape[0] // hop)
            self.assertEqual(int(feature.lengths(torch.tensor([y.shape[0]]))[0]), ours.shape[1])
            self.assertEqual(feature.out_shape((y.shape[0],)), (n_mels,))


@needs_module("torch")
class TestConvolutional(unittest.TestCase):
    def test_every_pooling_gives_finite_outputs_on_sequences_and_images(self):
        import torch

        x = torch.randn(3, 40, 101)
        lengths = torch.tensor([101, 60, 9])
        images = torch.randn(3, 2, 16, 16)
        for pooling in ("mean", "max", "meanmax", "stats", "attention"):
            for hidden in (None, 16):
                raw = dict(CNN2D, pooling=pooling, classifier_hidden=hidden)
                for model, inputs in ((build(raw, (40,)), (x, lengths)), (build(raw, (2, 16, 16)), (images, None))):
                    out = model(*inputs)
                    self.assertEqual(tuple(out.shape), (3, 4))
                    self.assertTrue(torch.isfinite(out).all(), pooling)
                    out.sum().backward()
                    self.assertEqual(model.embed(*inputs).shape[1], model.embedding_dim)
        model = build(dict(CNN1D, pooling="stats"), (40,))
        out = model(x, lengths)
        self.assertEqual(tuple(out.shape), (3, 4))
        out.sum().backward()

    def test_steps_after_blocks_match_the_network(self):
        import torch

        model = build(CNN2D, (40,)).eval()
        x = torch.randn(1, 40, 101)
        h = model.blocks(x.unsqueeze(1))
        self.assertEqual(int(model.steps_after_blocks(torch.tensor([101]))[0]), h.shape[-1])
        conv1 = build(CNN1D, (40,)).eval()
        self.assertEqual(int(conv1.steps_after_blocks(torch.tensor([101]))[0]), conv1.blocks(x).shape[-1])

    def test_masked_pool_ignores_padding(self):
        import torch

        from sweepkit.models.torch.heads import masked_pool

        h = torch.randn(2, 5, 10)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, :4] = True
        mask[1, :] = True
        padded = h.clone()
        padded[0, :, 4:] = 1e6
        for kind in ("mean", "max", "meanmax", "stats"):
            self.assertTrue(torch.allclose(masked_pool(h, mask, kind, None), masked_pool(padded, mask, kind, None)), kind)
        self.assertTrue(torch.allclose(masked_pool(h, mask, "mean", None)[0], h[0, :, :4].mean(dim=-1)))
        self.assertTrue(torch.allclose(masked_pool(h, mask, "max", None)[0], h[0, :, :4].amax(dim=-1)))

    def test_every_normalization_and_activation(self):
        import torch
        from torch import nn

        cases = [("batch", None, None), ("subspectral", None, 2), ("subspectral", None, 4), ("group", 8, None),
                 ("group", 1, None), ("none", None, None)]
        x = torch.randn(4, 40, 50)
        lengths = torch.tensor([50, 30, 20, 50])
        for norm, groups, subbands in cases:
            for activation in ("relu", "leaky_relu", "elu", "gelu", "silu"):
                raw = dict(CNN2D, channels=[8, 16, 16], pooling="stats", classifier_hidden=8, norm=norm,
                           norm_groups=groups, norm_subbands=subbands, activation=activation)
                model = build(raw, (40,))
                out = model(x, lengths)
                self.assertEqual(tuple(out.shape), (4, 4))
                self.assertTrue(torch.isfinite(out).all(), (norm, activation))
                out.sum().backward()
                convs = [m for m in model.blocks if isinstance(m, nn.Conv2d)]
                self.assertTrue(all((c.bias is not None) == (norm == "none") for c in convs), norm)
                kind = {"batch": nn.BatchNorm2d, "group": nn.GroupNorm}.get(norm)
                if kind is not None:
                    self.assertEqual(sum(isinstance(m, kind) for m in model.blocks), 3)

    def test_subspectral_norm_normalizes_each_band_apart(self):
        import torch

        from sweepkit.models.torch.cnn2d import _subspectral_class

        norm = _subspectral_class()(3, 2).train()
        x = torch.randn(8, 3, 10, 20)
        x[:, :, :5] = x[:, :, :5] * 4 + 9
        out = norm(x)
        for band in (slice(0, 5), slice(5, 10)):
            part = out[:, :, band]
            self.assertLess(part.mean(dim=(0, 2, 3)).abs().max().item(), 1e-4)
            self.assertLess((part.var(dim=(0, 2, 3), unbiased=False) - 1).abs().max().item(), 1e-2)
        odd = _subspectral_class()(3, 4)(torch.randn(4, 3, 10, 6))
        self.assertEqual(tuple(odd.shape), (4, 3, 10, 6), "bands of 3, 3, 2, 2 rows")

    def test_cnn_transformer_and_its_padding_mask(self):
        import torch

        from sweepkit.models.torch.heads import sinusoidal

        x = torch.randn(3, 40, 80)
        lengths = torch.tensor([80, 41, 12])
        for changes in ({}, {"row_reduce": "flatten"}, {"norm_first": False}, {"positional": "none"},
                        {"pooling": "attention"}, {"pooling": "stats", "classifier_hidden": 16},
                        {"norm": "group", "norm_groups": 4, "activation": "gelu", "transformer_activation": "relu"}):
            model = build(dict(CNN_TRANSFORMER, **changes), (40,))
            out = model(x, lengths)
            self.assertEqual(tuple(out.shape), (3, 4), changes)
            self.assertTrue(torch.isfinite(out).all(), changes)
            out.sum().backward()
            self.assertTrue(torch.isfinite(model.project.weight.grad).all())
        model = build(CNN_TRANSFORMER, (40,)).eval()
        h = torch.randn(2, 10, 32)
        pad = torch.arange(10)[None, :] >= torch.tensor([10, 6])[:, None]
        with torch.no_grad():
            a = model.encoder(h, src_key_padding_mask=pad)
            changed = h.clone()
            changed[1, 6:] = 100.0
            b = model.encoder(changed, src_key_padding_mask=pad)
        self.assertTrue(torch.allclose(a[1, :6], b[1, :6], atol=1e-5))
        self.assertTrue(torch.allclose(a[0], b[0], atol=1e-6))
        code = sinusoidal(5, 8, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.allclose(code[:, 0], torch.sin(torch.arange(5.0))))
        self.assertTrue(torch.allclose(code[:, 1], torch.cos(torch.arange(5.0))))


@needs_module("torch")
class TestSequenceModels(unittest.TestCase):
    def test_padding_changes_nothing(self):
        """Every row of a padded batch gives what it gives alone, whatever the padding holds (models that mask their
        input), in eval mode."""
        import torch

        rng = np.random.default_rng(1)
        x = torch.from_numpy(rng.standard_normal((3, 24, 23)).astype(np.float32))
        lengths = torch.tensor([23, 9, 2])
        cases = (POOL, dict(POOL, pooling="stats", proj_dim=None), TRANSFORMER,
                 dict(TRANSFORMER, positional="none", norm_first=False, pooling="mean"), SEQ_LIF,
                 dict(SEQ_LIF, lif_norm="batch", recurrent=False, time_pool=1), RNN,
                 dict(RNN, rnn="lstm", pooling="last", bidirectional=False), dict(RNN, time_pool=3, proj_dim=8),
                 COMPOSITES[0], COMPOSITES[1], COMPOSITES[6])
        for raw in cases:
            net = build(raw, (24,)).eval()
            padded = x.clone()
            for k, n in enumerate(lengths.tolist()):
                padded[k, :, n:] = 7.0
            with torch.no_grad():
                together = net(padded, lengths)
                self.assertEqual(tuple(together.shape), (3, 4), raw["name"])
                for k, n in enumerate(lengths.tolist()):
                    alone = net(x[k:k + 1, :, :n], lengths[k:k + 1])
                    self.assertTrue(torch.allclose(together[k], alone[0], atol=1e-4), (raw["name"], k))
            self.assertEqual(net.embed(x, lengths).shape[1], net.embedding_dim, raw["name"])

    def test_widths(self):
        self.assertEqual(build(POOL, (24,)).embedding_dim, 64)
        self.assertEqual(build(dict(POOL, proj_dim=None, pooling="meanmax"), (24,)).embedding_dim, 48)
        self.assertEqual(build(TRANSFORMER, (24,)).embedding_dim, 64)
        self.assertEqual(build(SEQ_LIF, (24,)).embedding_dim, 32)
        self.assertEqual(build(RNN, (24,)).embedding_dim, 48)
        self.assertEqual(build(dict(RNN, pooling="last", bidirectional=False), (24,)).embedding_dim, 24)

    def test_cnn_rnn_packing_keeps_padding_out(self):
        import torch

        cases = ({}, {"rnn": "lstm"}, {"bidirectional": False}, {"pooling": "last"},
                 {"rnn": "lstm", "pooling": "last"}, {"pooling": "last", "bidirectional": False},
                 {"pooling": "mean", "rnn_layers": 1, "rnn_dropout": 0.0}, {"pooling": "max"},
                 {"pooling": "meanmax"}, {"pooling": "stats"}, {"rnn": "lstm", "rnn_layers": 3})
        for case in cases:
            model = build(dict(CNN_RNN, **case), (40,)).eval()
            rng = np.random.default_rng(1)
            h = torch.from_numpy(rng.standard_normal((3, 9, model.rnn.input_size)).astype(np.float32))
            keep = torch.tensor([9, 4, 1])
            with torch.no_grad():
                together = model.sequence(h, keep)
                self.assertEqual(tuple(together.shape), (3, model.embedding_dim), case)
                for k in range(3):
                    alone = model.sequence(h[k:k + 1, :int(keep[k])], keep[k:k + 1])
                    self.assertTrue(torch.allclose(together[k], alone[0], atol=1e-5), (case, k))
                noisy = h.clone()
                noisy[1, 4:] = 100.0
                noisy[2, 1:] = -50.0
                self.assertTrue(torch.allclose(model.sequence(noisy, keep), together, atol=1e-5), case)
        for case, width in (({}, 48), ({"bidirectional": False}, 24), ({"pooling": "stats"}, 96),
                            ({"pooling": "last"}, 48), ({"pooling": "last", "bidirectional": False, "rnn": "lstm"}, 24),
                            ({"row_reduce": "flatten"}, 48), ({"classifier_hidden": 16}, 48)):
            model = build(dict(CNN_RNN, **case), (40,)).eval()
            self.assertEqual(model.embedding_dim, width, case)
            with torch.no_grad():
                logits = model(torch.randn(2, 40, 120), torch.tensor([120, 37]))
            self.assertEqual(tuple(logits.shape), (2, 4), case)
            self.assertTrue(bool(torch.isfinite(logits).all()), case)
        self.assertEqual(build(dict(CNN_RNN, row_reduce="flatten"), (40,)).rnn.input_size, 32 * 10)

    def test_composite_models(self):
        import torch

        x = torch.randn(3, 40, 60)
        lengths = torch.tensor([60, 33, 7])
        for raw in COMPOSITES:
            model = build(raw, (40,)).train()
            out = model(x, lengths)
            self.assertEqual(tuple(out.shape), (3, 4), raw)
            self.assertTrue(bool(torch.isfinite(out).all()), raw)
            out.sum().backward()
            self.assertEqual(model.embed(x, lengths).shape[1], model.embedding_dim)
            self.assertEqual(model.is_spiking, raw["block"] == "lif")
            if raw["block"] == "lif":
                model.eval()
                model.record(True)
                with torch.no_grad():
                    model(x, lengths)
                model.record(False)
                self.assertEqual(model.spike_report()["sequences"], 3)
            else:
                self.assertIsNone(model.activity_penalty())
                self.assertEqual(model.spike_report(), {})

    def test_mlp(self):
        import torch
        from torch import nn

        x = torch.randn(8, 10)
        for raw in (MLP, dict(MLP, norm="layer"), dict(MLP, norm="none", activation="gelu"), dict(MLP, hidden=[])):
            model = build(raw, (10,), 3)
            out = model(x, None)
            self.assertEqual(tuple(out.shape), (8, 3))
            out.sum().backward()
            self.assertEqual(model.embed(x, None).shape[1], model.embedding_dim)
        linear = build(dict(MLP, hidden=[], dropout=0.0), (10,), 3)
        self.assertEqual([type(m) for m in linear.modules() if isinstance(m, nn.Linear)], [nn.Linear])
        self.assertEqual(linear.embedding_dim, 10)


def neuron_settings(**changes):
    """The fields the spiking layers read, without the rest of a model's settings."""
    base = dict(beta=0.875, learn_beta="none", threshold=1.0, reset="subtract", detach_reset=False,
                surrogate="atan", surrogate_slope=2.0, rate_weight=0.0, rate_target=0.1, lif_dropout=0.0,
                lif_norm="none", recurrent=False)
    base.update(changes)
    return SimpleNamespace(**base)


@needs_module("torch")
class TestNeuron(unittest.TestCase):
    def test_forward_matches_the_reference(self):
        import torch

        from sweepkit.models.torch import lif
        from sweepkit.models.torch import lif_reference as ref

        Neuron = lif.classes()[0]
        rng = np.random.default_rng(0)
        current = rng.normal(0.5, 0.8, size=(3, 20, 5))
        for reset in ("subtract", "zero"):
            for beta in (0.0, 0.5, 0.875):
                n = Neuron((5,), neuron_settings(beta=beta, reset=reset), "n", lif.Recorder())
                got = n(torch.tensor(current), torch.ones(3, 20, 1)).numpy()
                for b in range(3):
                    want, _ = ref.lif_forward(current[b], beta, 1.0, reset)
                    self.assertTrue(np.array_equal(got[b], want), (reset, beta, b))
                got32 = n(torch.tensor(current, dtype=torch.float32), torch.ones(3, 20, 1))
                self.assertEqual(got32.dtype, torch.float32)
                self.assertTrue(np.array_equal(got32.numpy(), got), "float32 gives the same spikes here")

    def test_recurrent_forward_matches_the_reference(self):
        import torch

        from sweepkit.models.torch import lif
        from sweepkit.models.torch import lif_reference as ref

        Neuron = lif.classes()[0]
        rng = np.random.default_rng(1)
        current = rng.normal(0.3, 0.6, size=(2, 15, 4))
        weights = rng.normal(0.0, 0.8, size=(4, 4))
        rec = torch.nn.Linear(4, 4, bias=False).double()
        with torch.no_grad():
            rec.weight.copy_(torch.tensor(weights))
        for reset in ("subtract", "zero"):
            n = Neuron((4,), neuron_settings(beta=0.5, reset=reset), "n", lif.Recorder())
            got = n(torch.tensor(current), torch.ones(2, 15, 1), recurrent=rec).detach().numpy()
            for b in range(2):
                want, _ = ref.lif_forward(current[b], 0.5, 1.0, reset, recurrent=weights)
                self.assertTrue(np.array_equal(got[b], want), (reset, b))

    def test_gradients_match_the_reference(self):
        import torch

        from sweepkit.models.torch import lif
        from sweepkit.models.torch import lif_reference as ref

        Neuron = lif.classes()[0]
        rng = np.random.default_rng(2)
        current = rng.normal(0.6, 0.9, size=(12, 6))
        g = rng.normal(size=current.shape)
        for surrogate, slope in (("atan", 2.0), ("fast_sigmoid", 25.0), ("sigmoid", 5.0), ("boxcar", 1.0)):
            for reset in ("subtract", "zero"):
                for detach in (False, True):
                    s = neuron_settings(learn_beta="layer", reset=reset, detach_reset=detach, surrogate=surrogate,
                                        surrogate_slope=slope)
                    n = Neuron((6,), s, "n", lif.Recorder())
                    x = torch.tensor(current[None], requires_grad=True)
                    spikes = n(x, torch.ones(1, 12, 1))
                    (spikes[0] * torch.tensor(g)).sum().backward()
                    want_i, want_b = ref.lif_backward(current, 0.875, 1.0, reset, surrogate, slope, detach, g)
                    key = (surrogate, reset, detach)
                    self.assertLess(np.abs(x.grad[0].numpy() - want_i).max(), 1e-9, key)
                    self.assertLess(abs(n.beta_value.grad.item() - want_b), 1e-5 * max(1.0, abs(want_b)), key)

    def test_learned_leak_per_neuron_and_its_clamp(self):
        import torch

        from sweepkit.models.torch import lif

        Neuron = lif.classes()[0]
        n = Neuron((3, 2), neuron_settings(learn_beta="neuron", beta=0.5), "n", lif.Recorder())
        self.assertEqual(tuple(n.beta_value.shape), (3, 2))
        with torch.no_grad():
            n.beta_value[0, 0] = 1.7
            n.beta_value[1, 1] = -0.2
        self.assertEqual(n.beta()[0, 0].item(), 1.0)
        self.assertEqual(n.beta()[1, 1].item(), 0.0)
        out = n(torch.randn(2, 7, 3, 2), torch.ones(2, 7, 1, 1))
        self.assertEqual(tuple(out.shape), (2, 7, 3, 2))
        fixed = Neuron((4,), neuron_settings(learn_beta="none"), "n", lif.Recorder())
        self.assertEqual(list(fixed.parameters()), [], "a fixed leak is a buffer, not a parameter")

    def test_activity_penalty(self):
        import torch

        from sweepkit.models.torch import lif

        Neuron = lif.classes()[0]
        torch.manual_seed(0)
        n = Neuron((5,), neuron_settings(rate_weight=0.5, rate_target=0.2), "n", lif.Recorder())
        x = torch.rand(3, 10, 5) * 1.5
        real = torch.ones(3, 10, 1)
        real[1, 6:] = 0.0
        spikes = n(x, real)
        rates = (spikes * real).sum(dim=(0, 1)) / real.expand_as(spikes).sum(dim=(0, 1))
        self.assertAlmostEqual(n.penalty.item(), ((rates - 0.2) ** 2).mean().item(), places=6)
        n.eval()
        n(x, real)
        self.assertIsNone(n.penalty, "no penalty outside training")


@needs_module("torch")
class TestSpikingParts(unittest.TestCase):
    def test_pool_time(self):
        import torch

        from sweepkit.models.torch import lif

        h = torch.arange(1.0, 6.0).reshape(1, 5, 1)
        pooled, mask = lif.pool_time(h, torch.ones(1, 5, dtype=torch.bool), 2)
        self.assertEqual(pooled[0, :, 0].tolist(), [1.5, 3.5, 5.0])
        self.assertEqual(mask[0].tolist(), [True, True, True])
        pooled, mask = lif.pool_time(h, torch.tensor([[True, True, True, False, False]]), 2)
        self.assertEqual(pooled[0, :, 0].tolist(), [1.5, 3.0, 0.0])
        self.assertEqual(mask[0].tolist(), [True, True, False])
        same, _ = lif.pool_time(h, torch.ones(1, 5, dtype=torch.bool), 1)
        self.assertIs(same, h)

    def test_readout_matches_the_reference(self):
        import torch

        from sweepkit.models.torch import lif
        from sweepkit.models.torch import lif_reference as ref

        rng = np.random.default_rng(3)
        spikes = (rng.random((2, 9, 4)) < 0.3).astype(np.float64)
        valid = [9, 5]
        mask = torch.arange(9)[None, :] < torch.tensor(valid)[:, None]
        for beta in (0.0, 0.5, 0.9):
            got = lif.readout(torch.tensor(spikes), mask, beta).numpy()
            for b in range(2):
                self.assertTrue(np.allclose(got[b], ref.readout_embedding(spikes[b], valid[b], beta), atol=1e-6))

    def test_step_batch_norm_uses_real_steps_only(self):
        import torch

        from sweepkit.models.torch import lif

        _, StepBatchNorm, _ = lif.classes()
        torch.manual_seed(0)
        x = torch.randn(2, 6, 3)
        mask = torch.arange(6)[None, :] < torch.tensor([6, 4])[:, None]
        a, b = StepBatchNorm(3), StepBatchNorm(3)
        out_a = a(x, mask)
        changed = x.clone()
        changed[1, 4:] = 50.0
        out_b = b(changed, mask)
        self.assertTrue(torch.allclose(out_a, out_b))
        self.assertTrue(torch.allclose(a.norm.running_mean, b.norm.running_mean))
        self.assertTrue((out_a[1, 4:] == 0).all())
        real = x[mask]
        self.assertTrue(torch.allclose(out_a[mask], (real - real.mean(0)) / torch.sqrt(real.var(0, unbiased=False) + 1e-5),
                                       atol=1e-5))

    def test_spiking_linear_ignores_padded_steps(self):
        import torch

        from sweepkit.models.torch import lif

        _, _, SpikingLinear = lif.classes()
        mask = torch.arange(7)[None, :] < torch.tensor([7, 4])[:, None]
        for norm in ("none", "batch", "layer"):
            for recurrent in (False, True):
                torch.manual_seed(1)
                layer = SpikingLinear(4, 6, neuron_settings(lif_norm=norm, recurrent=recurrent), "l", lif.Recorder(),
                                      real_input=True)
                x = torch.randn(2, 7, 4) * 2
                a = layer(x, mask)
                changed = x.clone()
                changed[1, 4:] = -30.0
                b = layer(changed, mask)
                self.assertTrue(torch.equal(a[:, :4], b[:, :4]), (norm, recurrent))
                self.assertTrue(torch.equal(a[0], b[0]), (norm, recurrent))
                self.assertTrue(torch.equal(lif.readout(a, mask, 0.8), lif.readout(b, mask, 0.8)))

    def test_time_dropout_keeps_one_mask_per_sequence(self):
        import torch

        from sweepkit.models.torch import lif

        torch.manual_seed(0)
        x = torch.ones(4, 9, 16)
        out = lif.time_dropout(x, 0.5, True)
        self.assertTrue(torch.equal(out, out[:, :1].expand_as(out)), "the same units dropped at every step")
        self.assertTrue(set(out.unique().tolist()) <= {0.0, 2.0})
        self.assertIs(lif.time_dropout(x, 0.5, False), x)

    def test_counts_and_report(self):
        import torch

        from sweepkit.models.torch import lif

        _, _, SpikingLinear = lif.classes()
        rec = lif.Recorder()
        rec.on = True
        first = SpikingLinear(3, 5, neuron_settings(), "lif1", rec, real_input=True)
        second = SpikingLinear(5, 2, neuron_settings(recurrent=True), "lif2", rec, real_input=False)
        mask = torch.arange(6)[None, :] < torch.tensor([6, 3])[:, None]
        rec.sequences += 2
        s1 = first(torch.randn(2, 6, 3) * 3, mask)
        s2 = second(s1, mask)
        m = mask[..., None].float()
        self.assertEqual(rec.sums["lif1.mac"], 9 * 3 * 5, "real steps times fan-in times fan-out")
        self.assertEqual(rec.sums["lif2.ac"], (s1 * m).sum().item() * 2 + (s2 * m).sum().item() * 2)
        self.assertEqual(rec.sums["lif1.neuron_steps"], 9 * 5)
        report = lif.report(rec)
        self.assertEqual(report["sequences"], 2)
        self.assertAlmostEqual(report["firing_rate"]["lif1"], (s1 * m).sum().item() / 45)
        self.assertAlmostEqual(report["mac_per_sequence"], 9 * 3 * 5 / 2)
        self.assertAlmostEqual(report["energy_uj_per_sequence_estimate"],
                               (report["mac_per_sequence"] * 4.6 + report["ac_per_sequence"] * 0.9) / 1e6)


HYBRID = dict({"name": "cnn_lif", "channels": [8, 16], "convs_per_block": 1, "kernel_size": 3, "pool_rows": 2,
               "pool_cols": 2, "norm": "batch", "norm_groups": None, "norm_subbands": None, "activation": "relu",
               "row_reduce": "mean", "time_pool": 1}, **NEURON, dropout=0.1)
SNN = dict({"name": "snn", "encoding": "frames", "static_steps": None, "channels": [4, 8], "convs_per_block": 1,
            "kernel_size": 3, "pool_rows": 2, "pool_cols": 2, "conv_norm": "batch"}, **dict(NEURON, lif_hidden=[16]),
           dropout=0.1)


@needs_module("torch")
class TestSpikingModels(unittest.TestCase):
    def run_model(self, model, x, lengths):
        import torch

        out = model(x, lengths)
        self.assertEqual(tuple(out.shape), (x.shape[0], 4))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        return out

    def test_hybrid(self):
        import torch

        x = torch.randn(3, 40, 80) * 2
        lengths = torch.tensor([80, 41, 12])
        for changes in ({}, {"row_reduce": "flatten"}, {"time_pool": 3}, {"lif_hidden": [32, 16, 8]},
                        {"recurrent": True}, {"lif_norm": "batch"}, {"lif_norm": "layer"}, {"learn_beta": "layer"},
                        {"learn_beta": "neuron", "lif_hidden": [32, 16]}, {"reset": "zero", "detach_reset": False},
                        {"surrogate": "boxcar", "surrogate_slope": 1.0}, {"readout_beta": 0.0},
                        {"rate_weight": 0.5}, {"norm": "group", "norm_groups": 4, "activation": "gelu"}):
            model = build(dict(HYBRID, **changes), (40,)).train()
            self.run_model(model, x, lengths)
            if changes.get("rate_weight"):
                self.assertIsNotNone(model.activity_penalty())
            else:
                self.assertIsNone(model.activity_penalty())
        model = build(HYBRID, (40,))
        self.assertEqual(model.steps_after_blocks(lengths).tolist(), [20, 10, 3])
        self.assertEqual(model.embedding_dim, 32)

    def test_hybrid_readout_ignores_padding_past_the_front_end(self):
        import torch

        model = build(dict(HYBRID, recurrent=True, lif_norm="batch", time_pool=2), (40,)).eval()
        x = torch.randn(2, 40, 64)
        lengths = torch.tensor([64, 24])
        with torch.no_grad():
            a = model(x, lengths)
            changed = x.clone()
            changed[1, :, 40:] = 25.0          # steps the front end's receptive field cannot reach
            b = model(changed, lengths)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_fully_spiking_both_encodings(self):
        import torch

        x = torch.randn(3, 16, 24) * 2
        lengths = torch.tensor([24, 13, 5])
        for changes in ({}, {"lif_hidden": []}, {"lif_hidden": [16, 8], "recurrent": True, "lif_norm": "batch"},
                        {"conv_norm": "none"}, {"learn_beta": "neuron"}, {"convs_per_block": 2},
                        {"encoding": "repeat", "static_steps": 3},
                        {"encoding": "repeat", "static_steps": 2, "lif_hidden": []},
                        {"encoding": "repeat", "static_steps": 2, "learn_beta": "neuron", "rate_weight": 0.3}):
            model = build(dict(SNN, **changes), (16,)).train()
            self.run_model(model, x, lengths)

    def test_recording_through_a_model(self):
        import torch

        x = torch.randn(2, 16, 24) * 2
        lengths = torch.tensor([24, 10])
        for model in (build(dict(HYBRID, lif_hidden=[8, 8]), (16,)), build(SNN, (16,)),
                      build(dict(SNN, encoding="repeat", static_steps=3), (16,)), build(SEQ_LIF, (16,))):
            model.eval()
            model.record(True)
            with torch.no_grad():
                model(x, lengths)
                model(x, lengths)
            model.record(False)
            report = model.spike_report()
            self.assertEqual(report["sequences"], 4)
            self.assertTrue(report["firing_rate"])
            self.assertTrue(all(0.0 <= r <= 1.0 for r in report["firing_rate"].values() if r is not None))
            self.assertGreater(report["mac_per_sequence"], 0)
            with torch.no_grad():
                model(x, lengths)
            self.assertEqual(model.spike_report()["sequences"], 4, "nothing is added while recording is off")


@needs_module("torch")
class TestHeadsAndLosses(unittest.TestCase):
    LABELS = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 0, 1, 4, 2, 0, 1]     # labels 3 and 4 have one row each

    def test_aux_head_multitask_and_adversarial(self):
        import torch

        from sweepkit.models.torch.heads import build_aux_head

        torch.manual_seed(0)
        plain = build_aux_head("multitask", 6, 2, 0.0)
        reverse = build_aux_head("adversarial", 6, 2, 0.0)
        reverse.load_state_dict(plain.state_dict())
        e1 = torch.randn(4, 6, requires_grad=True)
        e2 = e1.detach().clone().requires_grad_(True)
        out1, out2 = plain(e1), reverse(e2)
        self.assertTrue(torch.equal(out1, out2), "the reversal changes nothing going forward")
        out1.pow(2).sum().backward()
        out2.pow(2).sum().backward()
        self.assertTrue(torch.allclose(e1.grad, -e2.grad), "and negates the gradient going back")
        self.assertTrue(torch.allclose(plain.linear.weight.grad, reverse.linear.weight.grad),
                        "the head itself still learns its target")
        with self.assertRaises(ValueError):
            build_aux_head("none", 6, 2, 0.0)

    def test_projection_head(self):
        import torch

        from sweepkit.models.torch.heads import build_projection_head

        head = build_projection_head(12, 5)
        self.assertEqual(head(torch.randn(3, 12)).shape, (3, 5))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 12 * 12 + 12 + 12 * 5 + 5)
        for width, dim in ((12, 0), (0, 5)):
            with self.assertRaises(ValueError):
                build_projection_head(width, dim)

    def test_soft_cross_entropy_and_multilabel_loss(self):
        import torch

        from sweepkit.train.losses import multilabel_loss, soft_cross_entropy

        logits = torch.randn(6, 4)
        y = torch.tensor([0, 1, 2, 3, 1, 0])
        ours = soft_cross_entropy(logits, torch.nn.functional.one_hot(y, 4).float(), None)
        self.assertAlmostEqual(ours.item(), torch.nn.functional.cross_entropy(logits, y).item(), places=5)
        weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
        weighted = soft_cross_entropy(logits, torch.nn.functional.one_hot(y, 4).float(), weights)
        per_row = torch.nn.functional.cross_entropy(logits, y, reduction="none") * weights[y]
        self.assertAlmostEqual(weighted.item(), per_row.mean().item(), places=5)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        z = torch.randn(3, 2)
        pos = torch.tensor([2.0, 0.5])
        manual = -(pos * targets * torch.log(torch.sigmoid(z)) + (1 - targets) * torch.log(1 - torch.sigmoid(z))).mean()
        self.assertAlmostEqual(multilabel_loss(z, targets, pos).item(), manual.item(), places=5)

    def test_regression_losses_and_concordance(self):
        import torch

        from sweepkit.train.losses import concordance, concordance_numpy, regression_loss

        rng = np.random.default_rng(0)
        p = torch.from_numpy(rng.standard_normal((20, 2)))
        t = torch.from_numpy(rng.standard_normal((20, 2)))
        self.assertAlmostEqual(regression_loss(p, t, "mse").item(), float(((p - t) ** 2).mean()), places=5)
        self.assertAlmostEqual(regression_loss(p, t, "l1").item(), float((p - t).abs().mean()), places=5)
        self.assertAlmostEqual(regression_loss(p, t, "ccc").item(),
                               1.0 - float(concordance_numpy(p.numpy(), t.numpy()).mean()), places=5)
        self.assertTrue(np.allclose(concordance(p, t).numpy(), concordance_numpy(p.numpy(), t.numpy())))
        self.assertEqual(regression_loss(p.to(torch.bfloat16), t, "mse").dtype, torch.float32)
        with self.assertRaises(ValueError):
            regression_loss(p, t, "huber")

    def test_contrastive_loss_matches_the_reference(self):
        import torch

        from sweepkit.train.losses import supervised_contrastive_loss

        rng = np.random.default_rng(2)
        labels = torch.tensor(self.LABELS)
        for t in (0.07, 0.1, 0.3):
            for scale in (0.01, 1.0, 30.0):
                z = rng.standard_normal((16, 8)) * scale
                ref = reference_contrastive(z, self.LABELS, t)
                got64 = supervised_contrastive_loss(torch.from_numpy(z), labels, t)
                got32 = supervised_contrastive_loss(torch.from_numpy(z).float(), labels, t)
                self.assertEqual((got64.dtype, got32.dtype), (torch.float64, torch.float32))
                self.assertAlmostEqual(got64.item(), ref, places=10)
                self.assertLess(abs(got32.item() - ref), 1e-4 * max(1.0, abs(ref)))
        z = torch.from_numpy(rng.standard_normal((16, 8))).to(torch.bfloat16)
        got = supervised_contrastive_loss(z, labels, 0.1)
        self.assertEqual(got.dtype, torch.float32)
        self.assertAlmostEqual(got.item(), reference_contrastive(z.float().numpy(), self.LABELS, 0.1), places=4)

    def test_contrastive_gradient_and_no_positive_pair(self):
        import torch

        from sweepkit.train.losses import supervised_contrastive_loss

        z = torch.from_numpy(np.random.default_rng(4).standard_normal((10, 4))).requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1, 1, 2, 0, 3, 2, 1])
        for t in (0.07, 0.3):
            self.assertTrue(torch.autograd.gradcheck(lambda x, t=t: supervised_contrastive_loss(x, labels, t), (z,)))
        free = torch.randn(5, 3, requires_grad=True)
        loss = supervised_contrastive_loss(free, torch.arange(5), 0.1)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(free.grad, torch.zeros_like(free)))


@needs_module("torch")
class TestInputs(unittest.TestCase):
    def test_masks(self):
        import torch

        from sweepkit.train.augment import MaskSettings, axis_masks

        x = torch.ones(4, 40, 100)
        lengths = torch.tensor([100, 50, 20, 100])
        s = MaskSettings(row_masks=2, row_mask_width=5, col_masks=2, col_mask_width=10)
        out = axis_masks(x, lengths, s, torch.Generator().manual_seed(0))
        self.assertTrue(((out == 0).all(dim=2).sum(dim=1) <= 10).all())
        zero_steps = (out == 0).all(dim=1)
        for b in range(4):
            idx = torch.nonzero(zero_steps[b]).flatten()
            if idx.numel():
                self.assertLess(int(idx.max()), int(lengths[b]), "column masks fall inside the row's own steps")
        again = axis_masks(x, lengths, s, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(out, again))
        images = axis_masks(torch.ones(3, 2, 12, 12), None, MaskSettings(1, 3, 1, 3), torch.Generator().manual_seed(1))
        self.assertEqual(tuple(images.shape), (3, 2, 12, 12))
        self.assertTrue(torch.equal(images[:, 0], images[:, 1]), "an image's channels share its masks")

    def test_noise_and_mixup(self):
        import torch

        from sweepkit.train.augment import NoiseSettings, add_noise, mixup_mix

        x = torch.zeros(2, 3, 6)
        noisy = add_noise(x, torch.tensor([6, 2]), NoiseSettings(sd=1.0), torch.Generator().manual_seed(0))
        self.assertTrue((noisy[1, :, 2:] == 0).all(), "no noise on padded steps")
        self.assertGreater(noisy[0].abs().sum().item(), 0.0)
        x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        lengths = torch.tensor([4, 2])
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights, partner = np.array([0.75, 1.0], dtype=np.float32), np.array([1, 0])
        mixed, mixed_lengths, (mt, none) = mixup_mix(x, lengths, [t, None], weights, partner)
        self.assertTrue(torch.allclose(mixed[0], 0.75 * x[0] + 0.25 * x[1]))
        self.assertTrue(torch.allclose(mixed[1], x[1]))
        self.assertEqual(mixed_lengths.tolist(), [4, 4])
        self.assertTrue(torch.allclose(mt[0], torch.tensor([0.75, 0.25])))
        self.assertTrue(torch.allclose(mt[1], torch.tensor([0.0, 1.0])))
        self.assertIsNone(none)
        table, no_lengths, _ = mixup_mix(torch.eye(2), None, [], weights, partner)
        self.assertIsNone(no_lengths)
        self.assertTrue(torch.allclose(table[0], torch.tensor([0.75, 0.25])))

    def test_item_and_group_normalization(self):
        import torch

        from sweepkit.features.normalize import Normalizer

        x = torch.randn(2, 8, 50) * 3 + 7
        out = Normalizer("item")(x, torch.tensor([50, 30]))
        self.assertLess(out[1, :, :30].mean(dim=-1).abs().max().item(), 1e-4)
        self.assertLess((out[1, :, :30].std(dim=-1, unbiased=False) - 1).abs().max().item(), 1e-2)
        self.assertEqual(out[1, :, 30:].abs().max().item(), 0.0)
        image = Normalizer("item")(torch.randn(2, 3, 5, 5) * 4 + 1, None)
        self.assertLess(image.mean(dim=(2, 3)).abs().max().item(), 1e-4)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 8, 50, generator=g)
        x[:2] = x[:2] * 2 + 5
        x[2:] = x[2:] * 0.5 - 3
        lengths = torch.tensor([50, 50, 50, 50])
        group_of = np.array([0, 0, 1, 1])
        mean = torch.stack([x[:2].mean(dim=(0, 2)), x[2:].mean(dim=(0, 2))])
        std = torch.stack([x[:2].std(dim=(0, 2), unbiased=False), x[2:].std(dim=(0, 2), unbiased=False)])
        norm = Normalizer("group", mean, std, group_of)
        out = norm(x, lengths, np.array([0, 1, 2, 3]))
        for part in (slice(0, 2), slice(2, 4)):
            self.assertLess(out[part].mean(dim=(0, 2)).abs().max().item(), 1e-5)
            self.assertLess((out[part].std(dim=(0, 2), unbiased=False) - 1).abs().max().item(), 1e-4)
        self.assertTrue(torch.allclose(norm(x[[3, 0]], lengths[:2], np.array([3, 0])), out[[3, 0]]))
        short = norm(x, torch.tensor([50, 20, 50, 10]), np.array([0, 1, 2, 3]))
        self.assertEqual(short[1, :, 20:].abs().max().item(), 0.0)
        self.assertTrue(norm.record()["transductive"])
        with self.assertRaises(ValueError):
            norm(x, lengths)
        with self.assertRaises(ValueError):
            Normalizer("group", mean, std)

    def test_group_statistics_match_a_direct_computation(self):
        import torch

        from sweepkit.features.audio import LogMel, LogMelSettings
        from sweepkit.features.normalize import statistics
        from sweepkit.train.batches import Source

        rng = np.random.default_rng(0)
        audio = [(rng.integers(-3000, 3000, size=n) / 32768.0).astype(np.float32) for n in (16000, 12000, 20000, 8000,
                                                                                             16000)]
        group_of = np.array([0, 1, 0, 1, 1, 2])          # group 2 has no row among these
        s = LogMelSettings(sample_rate=16000, win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                           log_offset=1e-6, normalization="group", group_column="group")
        feature = LogMel(s, torch.device("cpu"))
        source = Source("audio", np.arange(5), items=items_of(audio, "audio"))
        mean, std = statistics(source, feature, 2, torch.device("cpu"), group_of, 3)
        self.assertEqual(tuple(mean.shape), (3, 40))
        for g in (0, 1):
            frames = torch.cat([feature(torch.from_numpy(audio[r])[None, :])[0] for r in np.flatnonzero(group_of[:5] == g)],
                               dim=1).double()
            self.assertTrue(torch.allclose(mean[g].double(), frames.mean(dim=1), atol=1e-4))
            self.assertTrue(torch.allclose(std[g].double(), frames.std(dim=1, unbiased=False), atol=1e-3))
        self.assertTrue(torch.equal(mean[2], torch.zeros(40)) and torch.equal(std[2], torch.ones(40)))
        global_mean, _ = statistics(source, feature, 2, torch.device("cpu"))
        everything = torch.cat([feature(torch.from_numpy(a)[None, :])[0] for a in audio], dim=1).double()
        self.assertTrue(torch.allclose(global_mean.double(), everything.mean(dim=1), atol=1e-4))

    def test_predict_averages_window_probabilities(self):
        import torch

        from sweepkit.features.audio import LogMel, LogMelSettings
        from sweepkit.features.normalize import Normalizer
        from sweepkit.train.batches import Source, window_starts
        from sweepkit.train.loop import Network

        model = build(dict(CNN2D, pooling="mean"), (40,)).eval()
        feature = LogMel(LogMelSettings(sample_rate=16000, win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0,
                                        f_max=8000.0, log_offset=1e-6, normalization="item", group_column=None),
                         torch.device("cpu"))
        norm = Normalizer("item")
        rng = np.random.default_rng(0)
        audio = [(rng.integers(-3000, 3000, size=n) / 32768.0).astype(np.float32) for n in (40000, 12000, 25000)]
        source = Source("audio", np.arange(3), items=items_of(audio, "audio"))
        net = Network(model, None, None, feature, norm, torch.device("cpu"), "fp32", "classification", None)
        window = SimpleNamespace(window=16000, hop=8000)
        got = net.predict(source, 1, window)
        whole = net.predict(source, 1)
        self.assertIsNone(got.aux)
        for k in range(3):
            probs = []
            for start in window_starts(audio[k].shape[0], 16000, 8000):
                piece = audio[k][start:start + 16000]
                with torch.no_grad():
                    x = feature(torch.from_numpy(piece)[None, :])
                    frames = feature.lengths(torch.tensor([piece.shape[0]])).clamp_max(x.shape[-1])
                    probs.append(torch.softmax(model(norm(x, frames), frames), dim=-1)[0].numpy())
            self.assertTrue(np.allclose(got.main[k], np.mean(probs, axis=0), atol=1e-5), k)
        self.assertTrue(np.allclose(got.main[1], whole.main[1], atol=1e-5), "a short row is one window")
        self.assertTrue(np.allclose(got.main.sum(axis=1), 1.0))


@needs_module("torch")
class TestStep(unittest.TestCase):
    def sequences(self, n=4):
        rng = np.random.default_rng(1)
        arrays = [rng.standard_normal((s, 24)).astype(np.float32) for s in (30, 22, 30, 15)][:n]
        from sweepkit.train.batches import Source

        return Source("sequence", np.arange(n), items=items_of(arrays, "sequence"))

    def network(self, model, aux=None, contrastive=None, aux_task=None, task="classification"):
        import torch

        from sweepkit.features.normalize import Normalizer
        from sweepkit.features.sequence import Identity
        from sweepkit.train.loop import Network

        return Network(model, aux, contrastive, Identity(), Normalizer("item"), torch.device("cpu"), "fp32", task,
                       aux_task)

    def losses(self, task="classification", aux_task=None, aux_loss=None, aux_weight=0.0, c_weight=0.0, c_t=None,
               weights=None, regression="none"):
        from sweepkit.train.loop import Losses

        return Losses(task, weights, None, regression, aux_task, aux_loss, aux_weight, c_weight, c_t)

    def test_the_step_is_its_parts(self):
        """Masks, the normalization, the model and the loss, from the same streams, give exactly Step.loss."""
        import torch

        from sweepkit.features.normalize import Normalizer
        from sweepkit.train.augment import MaskSettings, axis_masks
        from sweepkit.train.batches import training_batch
        from sweepkit.train.loop import Step, Streams, Targets
        from sweepkit.train.losses import soft_cross_entropy

        source = self.sequences()
        model = build(RNN, (24,), 4).train()
        weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
        mask = MaskSettings(row_masks=2, row_mask_width=3, col_masks=2, col_mask_width=4)
        step = Step(self.network(model), {"mask": mask}, self.losses(weights=weights))
        targets = Targets(np.eye(4, dtype=np.float32), np.arange(4), None)
        picked = np.array([2, 0, 3, 1])
        torch.manual_seed(11)
        loss, parts = step.loss(source, picked, targets, 20, Streams(5))
        self.assertEqual(list(parts), ["main"])
        torch.manual_seed(11)
        streams = Streams(5)
        x_np, lengths_np = training_batch(source, picked, 20, streams.batches)
        x, lengths = torch.from_numpy(x_np), torch.from_numpy(lengths_np)
        x = axis_masks(Normalizer("item")(x, lengths), lengths, mask, streams.masks)
        reference = soft_cross_entropy(model(x, lengths), torch.from_numpy(targets.main[picked]), weights)
        self.assertEqual(loss.item(), reference.item())

    def test_extra_heads_and_the_penalty_add_their_weighted_losses(self):
        from sweepkit.models.torch.heads import build_aux_head, build_projection_head
        from sweepkit.train.augment import MixupSettings
        from sweepkit.train.loop import Step, Streams, Targets

        source = self.sequences()
        picked = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        for kind in ("multitask", "adversarial"):
            model = build(dict(SEQ_LIF, rate_weight=0.3), (24,), 4).train()
            aux = build_aux_head(kind, model.embedding_dim, 2, model.head_dropout)
            step = Step(self.network(model, aux=aux, aux_task="classification"), {"mixup": MixupSettings(alpha=0.4)},
                        self.losses(aux_task="classification", aux_loss="cross_entropy", aux_weight=0.5))
            targets = Targets(np.eye(4, dtype=np.float32), labels, np.eye(2, dtype=np.float32)[[0, 1, 0, 1]])
            loss, parts = step.loss(source, picked, targets, None, Streams(2))
            self.assertAlmostEqual(loss.item(), parts["main"].item() + parts["penalty"].item()
                                   + 0.5 * parts["aux"].item(), places=5)
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in step.network.parameters() if p.requires_grad))
        model = build(RNN, (24,), 2).train()
        head = build_projection_head(model.embedding_dim, 16)
        step = Step(self.network(model, contrastive=head), {}, self.losses(c_weight=0.5, c_t=0.1))
        targets = Targets(np.eye(2, dtype=np.float32)[labels], labels, None)
        loss, parts = step.loss(source, picked, targets, None, Streams(3))
        self.assertAlmostEqual(loss.item(), parts["main"].item() + 0.5 * parts["contrastive"].item(), places=5)
        loss.backward()
        self.assertGreater(head.out.weight.grad.abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "mixup"):
            Step(self.network(model, contrastive=head), {"mixup": MixupSettings(alpha=0.2)},
                 self.losses(c_weight=0.5, c_t=0.1))
        with self.assertRaisesRegex(ValueError, "projection head"):
            Step(self.network(model), {}, self.losses(c_weight=0.5, c_t=0.1))

    def test_regression_and_multilabel_steps(self):
        import torch

        from sweepkit.train.loop import Step, Streams, Targets

        source = self.sequences()
        picked = np.arange(4)
        model = build(dict(RNN, pooling="mean"), (24,), 1).train()
        step = Step(self.network(model, task="regression"), {}, self.losses(task="regression", regression="ccc"))
        loss, _ = step.loss(source, picked, Targets(np.array([[1.0], [-1.0], [0.5], [0.0]], dtype=np.float32), None,
                                                    None), None, Streams(1))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        model = build(dict(RNN, pooling="mean"), (24,), 3).train()
        step = Step(self.network(model, task="multilabel"), {}, self.losses(task="multilabel"))
        loss, _ = step.loss(source, picked, Targets(np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]],
                                                             dtype=np.float32), None, None), None, Streams(1))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

    def test_a_tiny_model_learns(self):
        import torch

        from sweepkit.train.losses import soft_cross_entropy

        for raw, epochs, share in ((dict(CNN2D, channels=[8, 16]), 60, 0.5), (dict(HYBRID, lif_dropout=0.0, dropout=0.0,
                                                                                 learn_beta="layer"), 150, 0.6)):
            model = build(raw, (40,))
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
            x = torch.randn(16, 40, 64)
            y = torch.arange(16) % 4
            x[y == 1, :10] += 3.0
            x[y == 2, 20:] -= 3.0
            x[y == 3, :, :32] += 2.0
            lengths = torch.full((16,), 64)
            target = torch.nn.functional.one_hot(y, 4).float()
            first = None
            for _ in range(epochs):
                loss = soft_cross_entropy(model(x, lengths), target, None)
                first = loss.item() if first is None else first
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.assertLess(loss.item(), share * first, raw["name"])


if __name__ == "__main__":
    unittest.main()
