"""Pretrained Hugging Face models: "hf_finetune" (sweepkit.models.torch.hf_finetune), frozen-encoder feature extraction
(sweepkit.data.extract, the extract-features command) and the pinning they share (sweepkit.pretrained).

Without torch or transformers (runs everywhere): the declaration (the pretrained profile, text input, the example
text_finetune, not transductive), that every configuration of text_finetune passes its checks without a download, the
settings the checks refuse (a revision that is not a commit, a local folder outside data/, LoRA settings without LoRA,
extra arguments the source decides), the extraction's own checks and its image and audio preparation, and the folder
fingerprint. With torch, transformers and tokenizers (the node, where SWEEPKIT_REQUIRE names them), on tiny random
models written into a temporary data/ folder with a local tokenizer, so no test needs the network: the fine-tuned
model runs through the runner in each training mode (full, top layers, LoRA) and records its source and the SHA-256
of the weights it started from; only the named layers or the adapters train; a run repeats exactly on the CPU; on
synthetic texts whose class is in their words it reaches balanced accuracy 0.8; extraction writes a store the store
dataset reads, whose rows are the pooled outputs of the model itself and do not depend on the batch size; images
through a tiny ViT and audio through a tiny wav2vec 2.0 give one vector per row. With SWEEPKIT_NETWORK=1 as well, the
pinned Hub model of the example downloads, records its commit, and the example's base runs a short job. The thresholds
are first proposals, not yet set from runs: the node session sets them from what the runs give, a little below.
"""
import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np

from tests.support import REPO, TempRepo, load_catalog, needs_module

load_catalog()

HUB_ID = "sentence-transformers/all-MiniLM-L6-v2"
HUB_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LOCAL = "models/tiny_bert"
TEXT = {"name": "synthetic", "generator": "text", "n_rows": 150, "n_features": 8, "n_groups": 3, "noise": 0.0,
        "seed": 1}
FINETUNE = {"name": "hf_finetune", "source": "local", "model_id": LOCAL, "revision": None, "max_length": 16,
            "pooling": "mean", "train": "full", "top_layers": None, "lora_r": None, "lora_alpha": None,
            "lora_dropout": None, "lora_targets": [], "activation": "relu", "dropout": 0.1, "classifier_hidden": None,
            "extra": {}}
NETWORK = os.environ.get("SWEEPKIT_NETWORK") == "1"


def example() -> dict:
    return json.loads((REPO / "sweeps" / "examples" / "text_finetune.json").read_text(encoding="utf-8"))


def text_job(model=None, **fit_changes) -> dict:
    """The example's base on a local model, with fewer rows and epochs."""
    base = example()["base"]
    base["data"] = dict(TEXT)
    base["model"] = dict(model or FINETUNE)
    base["fit"].update(dict({"epochs": 4, "lr": 0.001, "warmup_epochs": 1, "device": "cpu"}, **fit_changes))
    return base


def checked(settings: dict):
    from sweepkit.runs.spec import check_job, validate_job

    job = validate_job(settings)
    check_job(job)
    return job


def run(settings: dict, root, runs: str = "runs", seed: int = 1) -> Path:
    from sweepkit.runs.spec import config_hash, plain, run_id
    from sweepkit.sweep.runner import execute

    job = checked(settings)
    spec = {"sweep_id": "pretrained_test", "purpose": "learning", "stage": "base", "label": "base", "overrides": {},
            "priority": 1, "config_hash": config_hash(job), "run_id": run_id(job, "holdout", seed),
            "fold": "holdout", "seed": seed, "family": job.family, "settings": plain(job)}
    execute(spec, "cpu", lambda message: None, runs_root=Path(root) / runs)
    return Path(root) / runs / spec["run_id"]


def read(folder: Path, name: str):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def write_tiny_bert(root: Path) -> Path:
    """A two-layer BERT with random weights and a word-level tokenizer of the synthetic texts' words, saved as
    save_pretrained saves them, in data/models/tiny_bert/."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    folder = root / "data" / LOCAL
    folder.mkdir(parents=True)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [f"w{c}_{j}" for c in range(3) for j in range(5)] + \
        [f"the{j}" for j in range(5)]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]").save_pretrained(str(folder))
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(str(folder))
    return folder


# Without torch --------------------------------------------------------------------------------------------------------

class TestDeclarationAndChecks(unittest.TestCase):
    def test_registry_entry(self):
        from sweepkit.analysis.transductive import parts
        from sweepkit.registry import MODELS

        meta = MODELS.get("hf_finetune").meta
        self.assertEqual((meta["family"], meta["kinds"], meta["gpu"], meta["profile"], meta["example"]),
                         ("torch", ("text",), "torch", "pretrained", "text_finetune"))
        self.assertEqual(meta["packages"], ("torch", "transformers", "peft"))
        self.assertEqual(parts(example()["base"]), [])

    def test_every_example_configuration_passes_its_checks_without_a_download(self):
        from sweepkit.sweep.design import expand

        jobs, summary = expand(example())
        self.assertEqual(summary["configs"], 5)
        for job in jobs:
            checked(job["settings"])
        model = example()["base"]["model"]
        self.assertEqual((model["source"], model["model_id"], model["revision"]), ("hub", HUB_ID, HUB_REVISION))

    def test_settings_the_checks_refuse(self):
        hub = dict(FINETUNE, source="hub", model_id=HUB_ID, revision=HUB_REVISION)
        lora = dict(FINETUNE, train="lora", lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_targets=["query"])
        bad = [(dict(hub, revision="main"), "commit"), (dict(hub, revision=None), "commit"),
               (dict(hub, model_id="no-owner"), "owner"), (dict(FINETUNE, revision=HUB_REVISION), "revision None"),
               (dict(FINETUNE, model_id="../outside"), "data/"), (dict(FINETUNE, max_length=1), "max_length"),
               (dict(FINETUNE, top_layers=2), "top_layers"), (dict(FINETUNE, train="top"), "top_layers"),
               (dict(FINETUNE, lora_r=4), "lora"), (dict(lora, lora_targets=[]), "lora_targets"),
               (dict(lora, lora_dropout=1.0), "lora_dropout"), (dict(FINETUNE, extra={"revision": "x"}), "extra"),
               (dict(FINETUNE, extra={"trust_remote_code": True}), "extra"),
               (dict(FINETUNE, extra={"dtype": "float16"}), "dtype")]
        for model, words in bad:
            with self.assertRaises(ValueError, msg=words) as caught:
                checked(text_job(model))
            self.assertIn(words, str(caught.exception))
        checked(text_job(hub))
        checked(text_job(lora))
        checked(text_job(dict(FINETUNE, train="top", top_layers=1)))
        tabular = text_job()
        tabular["data"] = dict(TEXT, generator="classification")
        with self.assertRaises(ValueError):
            checked(tabular)

    def test_the_folder_fingerprint(self):
        from sweepkit.pretrained import folder_sha256

        with TempRepo() as root:
            folder = root / "data" / "models" / "m"
            (folder / "sub").mkdir(parents=True)
            (folder / "a.json").write_text("{}", encoding="utf-8")
            (folder / "sub" / "b.bin").write_bytes(b"\x00\x01")
            first = folder_sha256(folder)
            self.assertEqual(first, folder_sha256(folder))
            (folder / "sub" / "b.bin").write_bytes(b"\x00\x02")
            self.assertNotEqual(first, folder_sha256(folder))


class TestExtractionChecks(unittest.TestCase):
    def test_the_arguments(self):
        from sweepkit.data.extract import check_arguments

        check_arguments("text", "mean", 32, 8, "float32", "cpu")
        check_arguments("image", "cls", None, 8, "bfloat16", "cuda")
        check_arguments("audio", "mean", None, 1, "float32", "mps")
        for args in (("tabular", "mean", None, 8, "float32", "cpu"), ("text", "mean", None, 8, "float32", "cpu"),
                     ("image", "mean", 32, 8, "float32", "cpu"), ("audio", "mean", None, 4, "float32", "cpu"),
                     ("text", "max", 32, 8, "float32", "cpu"), ("text", "mean", 32, 8, "float16", "cpu")):
            with self.assertRaises(ValueError, msg=str(args)):
                check_arguments(*args)

    def test_images_become_three_channels(self):
        from sweepkit.data.extract import image_array

        gray = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = image_array(gray)
        self.assertEqual((out.shape, out.dtype), ((3, 3, 4), np.float32))
        self.assertTrue(all(np.array_equal(out[c], gray) for c in range(3)))
        self.assertEqual(image_array(np.zeros((1, 5, 5))).shape, (3, 5, 5))
        self.assertEqual(image_array(np.zeros((3, 5, 5))).shape, (3, 5, 5))
        for shape in ((4, 5, 5), (5,), (2, 2, 2, 2)):
            with self.assertRaises(ValueError):
                image_array(np.zeros(shape))

    def test_the_audio_sample_rate_comes_from_the_dataset(self):
        from types import SimpleNamespace

        from sweepkit.data.extract import sample_rate

        self.assertEqual(sample_rate({"name": "synthetic"}, None), 16000)
        self.assertEqual(sample_rate({"name": "items"}, SimpleNamespace(sample_rate=22050)), 22050)
        with self.assertRaises(ValueError):
            sample_rate({"name": "items"}, SimpleNamespace(sample_rate=None))

    def test_refused_before_any_model_loads(self):
        from sweepkit.data.extract import extract_features

        with TempRepo() as root:
            table = root / "blobs.json"
            table.write_text(json.dumps({"name": "synthetic", "generator": "blobs", "n_rows": 20, "n_features": 3,
                                         "n_groups": 2, "noise": 0.5, "seed": 1}), encoding="utf-8")
            texts = root / "texts.json"
            texts.write_text(json.dumps(dict(TEXT, n_rows=20)), encoding="utf-8")
            cases = [((table, "local", LOCAL, None, "mean", None, 4, "float32", "cpu", (), "s"),
                      "text, image or audio"),
                     ((texts, "local", LOCAL, None, "mean", None, 4, "float32", "cpu", (), "s"), "max-length"),
                     ((texts, "local", LOCAL, None, "mean", 16, 4, "float32", "cpu", ("absent",), "s"), "absent"),
                     ((texts, "hub", HUB_ID, "main", "mean", 16, 4, "float32", "cpu", (), "s"), "commit")]
            for args, words in cases:
                with self.assertRaises(ValueError, msg=words) as caught:
                    extract_features(*args, log=lambda message: None)
                self.assertIn(words, str(caught.exception))
            self.assertFalse((root / "data" / "stores").exists(), "nothing written")


# With torch and transformers -----------------------------------------------------------------------------------------

@needs_module("torch")
class TestPooling(unittest.TestCase):
    def test_each_pooling(self):
        from types import SimpleNamespace

        import torch

        from sweepkit.data.extract import pool

        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        outputs = SimpleNamespace(last_hidden_state=hidden, pooler_output=None)
        self.assertTrue(torch.equal(pool(outputs, "cls", mask), hidden[:, 0]))
        mean = pool(outputs, "mean", mask)
        self.assertTrue(torch.allclose(mean[0], hidden[0, :2].mean(dim=0)))
        self.assertTrue(torch.allclose(mean[1], hidden[1].mean(dim=0)))
        self.assertTrue(torch.allclose(pool(outputs, "mean", None), hidden.mean(dim=1)))
        with self.assertRaises(ValueError):
            pool(outputs, "pooler", mask)
        maps = SimpleNamespace(last_hidden_state=torch.ones(2, 5, 3, 3), pooler_output=torch.ones(2, 5, 1, 1))
        self.assertEqual(tuple(pool(maps, "mean", None).shape), (2, 5))
        self.assertEqual(tuple(pool(maps, "pooler", None).shape), (2, 5))
        with self.assertRaises(ValueError):
            pool(maps, "cls", None)

    def test_the_weights_fingerprint(self):
        import torch

        from sweepkit.pretrained import weights_sha256

        torch.manual_seed(0)
        layer = torch.nn.Linear(3, 2)
        first = weights_sha256(layer)
        self.assertEqual(first, weights_sha256(layer))
        with torch.no_grad():
            layer.bias[0] += 1.0
        self.assertNotEqual(first, weights_sha256(layer))


@needs_module("torch")
@needs_module("transformers")
@needs_module("tokenizers")
@needs_module("peft")
class TestFinetune(unittest.TestCase):
    def test_each_training_mode_through_the_runner(self):
        import torch
        from transformers import AutoModel

        from sweepkit.pretrained import folder_sha256, weights_sha256

        with TempRepo() as root:
            folder = write_tiny_bert(root)
            loaded = AutoModel.from_pretrained(str(folder), local_files_only=True, dtype=torch.float32)
            weights = weights_sha256(loaded)
            modes = {"full": {}, "top": {"train": "top", "top_layers": 1},
                     "lora": {"train": "lora", "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                              "lora_targets": ["query", "value"]}}
            for mode, changes in modes.items():
                run_folder = run(text_job(dict(FINETUNE, **changes)), root, mode)
                params = read(run_folder, "model_params.json")
                record = params["model_record"]
                self.assertEqual((record["source"], record["model_id"], record["revision"], record["commit_hash"]),
                                 ("local", LOCAL, None, None), mode)
                self.assertEqual(record["weights_sha256"], weights, mode)
                self.assertEqual(record["folder_sha256"], folder_sha256(folder), mode)
                self.assertEqual((record["train"], record["max_length"], record["pooling"]), (mode, 16, "mean"))
                self.assertEqual(params["input_shape"], [])
                if mode == "full":
                    self.assertEqual(record["encoder_trainable"], record["encoder_parameters"])
                    self.assertEqual(params["parameters_trainable"], params["parameters_total"])
                elif mode == "top":
                    self.assertEqual(record["layers_trained"], ["encoder.layer.1"])
                    self.assertEqual(record["encoder_trainable"],
                                     sum(p.numel() for p in loaded.encoder.layer[1].parameters()))
                else:
                    # 2 layers x (query, value) x (A: 4 x 32 and B: 32 x 4)
                    self.assertEqual((record["adapted_layers"], record["encoder_trainable"]), (4, 1024))
                    self.assertEqual((record["lora_config"]["r"], record["lora_config"]["lora_alpha"]), (4, 8))
                    self.assertEqual(sorted(record["lora_config"]["target_modules"]), ["query", "value"])
                self.assertTrue(read(run_folder, "metrics_val.json")["training"]["reload_check"]["equal"], mode)

    def test_a_run_repeats_exactly(self):
        with TempRepo() as root:
            write_tiny_bert(root)
            first = run(text_job(), root, "a")
            again = run(text_job(), root, "b")
            for name in ("predictions_val.csv", "predictions_test.csv"):
                self.assertEqual((first / name).read_bytes(), (again / name).read_bytes(), name)

    def test_the_class_in_the_words_is_learned(self):
        with TempRepo() as root:
            write_tiny_bert(root)
            settings = text_job(epochs=20, lr=0.001, early_stopping_patience=5)
            settings["data"]["n_rows"] = 300
            folder = run(settings, root)
            self.assertGreaterEqual(read(folder, "metrics_test.json")["metrics"]["balanced_accuracy"], 0.8)


@needs_module("torch")
@needs_module("transformers")
@needs_module("tokenizers")
class TestExtraction(unittest.TestCase):
    def extract(self, root: Path, section: dict, name: str, **changes):
        from sweepkit.data.extract import extract_features

        data_file = root / f"{name}.json"
        data_file.write_text(json.dumps(section), encoding="utf-8")
        arguments = dict(source="local", model_id=LOCAL, revision=None, pooling="mean", max_length=16, batch_size=8,
                         dtype="float32", device="cpu", keep=("truth",), name=name)
        arguments.update(changes)
        return extract_features(data_file, log=lambda message: None, **arguments)

    def test_text_becomes_a_store_of_the_models_own_pooled_outputs(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from sweepkit.data.convert import read_store
        from sweepkit.datasets.store import Store, StoreSettings
        from sweepkit.data.synthetic import make

        with TempRepo() as root:
            folder = write_tiny_bert(root)
            section = dict(TEXT, n_rows=40)
            record = self.extract(root, section, "texts")
            store = read_store("texts")
            self.assertEqual(store["values"].shape, (40, 32))
            self.assertEqual(record["sha256"], hashlib.sha256((root / "data" / "stores" / "texts" /
                                                               "values.npy").read_bytes()).hexdigest())
            self.assertEqual(record["encoder"]["pooling"], "mean")
            self.assertEqual(record["source"]["data"]["generator"], "text")
            texts = make("text", 40, 8, 3, 0.0, 1)["items"]
            tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
            model = AutoModel.from_pretrained(str(folder), local_files_only=True, dtype=torch.float32).eval()
            with torch.no_grad():
                one = tokenizer([texts[0]], return_tensors="pt")
                expected = model(**one).last_hidden_state[0].mean(dim=0).numpy()
            self.assertTrue(np.allclose(store["values"][0], expected, atol=1e-5))
            self.assertEqual(store["meta"]["truth"][0], "c0")
            single = self.extract(root, section, "texts_one", batch_size=1)
            self.assertTrue(np.allclose(read_store("texts_one")["values"], store["values"], atol=1e-5),
                            "padding never enters a row's mean")
            self.assertNotEqual(single["sha256"], "")
            settings = StoreSettings(store="texts", features=("*",), keep=("truth",), filter_column=None,
                                     filter_values=())
            data = Store().load(settings)
            self.assertEqual(data.fingerprint["source_sha256"], record["sha256"])
            self.assertTrue(np.array_equal(data.x[np.arange(3)], store["values"][:3]))
            cls = self.extract(root, section, "texts_cls", pooling="cls")
            self.assertEqual(cls["features"], 32)
            with self.assertRaises(Exception):
                self.extract(root, section, "texts", pooling="cls")      # a store is written once

    @needs_module("torchvision")
    def test_images_through_a_tiny_vit(self):
        from transformers import ViTConfig, ViTImageProcessor, ViTModel

        with TempRepo() as root:
            folder = root / "data" / "models" / "tiny_vit"
            folder.mkdir(parents=True)
            config = ViTConfig(image_size=16, patch_size=8, num_channels=3, hidden_size=32, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=64)
            ViTModel(config).save_pretrained(str(folder))
            ViTImageProcessor(do_resize=True, size={"height": 16, "width": 16}, do_rescale=False, do_normalize=True,
                              image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]).save_pretrained(str(folder))
            section = {"name": "synthetic", "generator": "images", "n_rows": 12, "n_features": 16, "n_groups": 2,
                       "noise": 0.3, "seed": 1}
            record = self.extract(root, section, "images", model_id="models/tiny_vit", pooling="cls", max_length=None,
                                  batch_size=4)
            self.assertEqual((record["rows"], record["features"]), (12, 32))
            self.assertTrue(record["encoder"]["processor"]["class"].startswith("ViTImageProcessor"))
            self.assertIn("image_mean", record["encoder"]["processor"]["configuration"])

    def test_audio_through_a_tiny_wav2vec2(self):
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

        with TempRepo() as root:
            folder = root / "data" / "models" / "tiny_wav2vec2"
            folder.mkdir(parents=True)
            config = Wav2Vec2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                                    conv_dim=(32, 32), conv_stride=(5, 4), conv_kernel=(10, 8),
                                    num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
            Wav2Vec2Model(config).save_pretrained(str(folder))
            Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                     return_attention_mask=False).save_pretrained(str(folder))
            section = {"name": "synthetic", "generator": "audio", "n_rows": 4, "n_features": 1, "n_groups": 2,
                       "noise": 0.05, "seed": 1}
            record = self.extract(root, section, "sounds", model_id="models/tiny_wav2vec2", max_length=None,
                                  batch_size=1)
            self.assertEqual((record["rows"], record["features"], record["encoder"]["sample_rate"]), (4, 32, 16000))


@unittest.skipUnless(NETWORK, "SWEEPKIT_NETWORK=1 allows the download of the example's Hub model")
@needs_module("torch")
@needs_module("transformers")
@needs_module("peft")
class TestHub(unittest.TestCase):
    def test_the_pinned_example_model_downloads_and_runs(self):
        settings = example()["base"]
        settings["data"]["n_rows"] = 60
        settings["fit"].update({"epochs": 1, "warmup_epochs": 0, "max_batches_per_epoch": 2, "device": "cpu"})
        with TempRepo() as root:
            folder = run(settings, root)
            record = read(folder, "model_params.json")["model_record"]
        self.assertEqual((record["source"], record["model_id"], record["revision"]), ("hub", HUB_ID, HUB_REVISION))
        self.assertEqual(record["commit_hash"], HUB_REVISION)
        self.assertEqual(len(record["weights_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
