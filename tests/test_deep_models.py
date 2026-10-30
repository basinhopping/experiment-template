"""Deep embedders and deep clustering (sweepkit.models.embed.deep and sweepkit.models.cluster.deep) on data with known
structure.

Without torch (runs everywhere): the declarations of "autoencoder", "vae", "autoencoder_kmeans" and "dec" (the torch
profile, PyTorch's GPU path, their example sweeps), that every configuration of embed_deep and cluster_deep passes its
checks, and the settings the checks refuse. With torch (the node, where SWEEPKIT_REQUIRE names torch): on planted
blobs, the two-dimensional codes of the autoencoder and of the VAE keep the planted groups apart (k-nearest-neighbour
label agreement at least 0.9 and 0.85), a fit row's code comes back from transform, and the decoder rebuilds the
scaled rows better than their column means do; k-means on the codes and DEC find the planted groups (adjusted Rand
index at least 0.9); DEC's soft assignments are probabilities whose arg max is its label, its kernel and target
distribution match the paper's formulas on a case worked by hand; a seed gives the same codes and clusters twice on
the CPU and another seed other codes; and the example bases run through the embedder and clusterer runners. The
thresholds are first proposals, not yet set from runs: the node session sets them from what the runs give, a little
below.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from tests.support import REPO, TempRepo, load_catalog, needs_module

load_catalog()

AE = {"hidden": [32], "latent_dim": 2, "activation": "relu", "dropout": 0.0, "epochs": 60, "batch_size": 32,
      "lr": 0.003, "weight_decay": 0.0001}
VAE = dict(AE, beta=1.0)
AE_KMEANS = dict(AE, latent_dim=3, epochs=50, n_clusters=3, n_init=4)
DEC = dict(AE_KMEANS, alpha=1.0, refine_epochs=30, refine_lr=0.001, update_interval=5, tol=0.001)


def example(name: str) -> dict:
    return json.loads((REPO / "sweeps" / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def checked(settings: dict):
    from sweepkit.runs.spec import check_job, validate_job

    job = validate_job(settings)
    check_job(job)
    return job


def with_model(name: str, model: dict) -> dict:
    """The example's base with another model section."""
    base = example(name)["base"]
    base["model"] = model
    return base


def build(name: str, settings: dict, seed: int = 1):
    from sweepkit.models.base import build_clusterer, build_embedder
    from sweepkit.registry import MODELS
    from sweepkit.settings import Selected

    entry, validated = MODELS.validate_section(dict(settings, name=name), "model")
    make = build_embedder if entry.meta["family"] == "embedder" else build_clusterer
    return make(Selected(name, validated), seed, "cpu")


def planted(n_groups: int):
    """Scaled planted blobs: 600 rows, 10 columns."""
    from sweepkit.data.synthetic import make

    made = make("blobs", 600, 10, n_groups, 1.0, 1)
    x = np.asarray(made["x"], dtype=np.float64)
    return (x - x.mean(axis=0)) / x.std(axis=0), made["columns"]["truth"]


def label_agreement(e, labels, k: int = 10) -> float:
    """Leave-one-out majority vote of the k nearest rows' labels in the embedding."""
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    found = NearestNeighbors(n_neighbors=k + 1).fit(e).kneighbors(e, return_distance=False)
    correct = 0
    for i, row in enumerate(found):
        values, counts = np.unique(labels[[j for j in row if j != i][:k]], return_counts=True)
        correct += values[np.argmax(counts)] == labels[i]
    return correct / len(labels)


def ari(truth, labels) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(truth, labels))


def run(settings: dict, root, fold: str, seed: int = 1) -> Path:
    """The run folder of one job through the runner, on the CPU."""
    from sweepkit.runs.spec import config_hash, plain, run_id
    from sweepkit.sweep.runner import execute

    job = checked(settings)
    spec = {"sweep_id": "deep_test", "purpose": "learning", "stage": "base", "label": "base", "overrides": {},
            "priority": 1, "config_hash": config_hash(job), "run_id": run_id(job, fold, seed), "fold": fold,
            "seed": seed, "family": job.family, "settings": plain(job)}
    execute(spec, "cpu", lambda message: None, runs_root=Path(root) / "runs")
    return Path(root) / "runs" / spec["run_id"]


# Without torch --------------------------------------------------------------------------------------------------------

class TestDeclarations(unittest.TestCase):
    def test_registry_entries(self):
        from sweepkit.registry import MODELS

        expected = {"autoencoder": ("embedder", "embed_deep"), "vae": ("embedder", "embed_deep"),
                    "autoencoder_kmeans": ("clusterer", "cluster_deep"), "dec": ("clusterer", "cluster_deep")}
        for name, (family, sweep) in expected.items():
            meta = MODELS.get(name).meta
            self.assertEqual((meta["family"], meta["gpu"], meta["profile"], meta["example"]),
                             (family, "torch", "torch", sweep), name)
            self.assertIn("torch", meta["packages"], name)

    def test_every_example_configuration_passes_its_checks(self):
        from sweepkit.sweep.design import expand

        for name in ("embed_deep", "cluster_deep"):
            jobs, summary = expand(example(name))
            self.assertGreater(summary["configs"], 3, name)
            for job in jobs:
                checked(job["settings"])

    def test_the_gpu_path_is_pytorch(self):
        for name, model in (("embed_deep", dict(AE, name="autoencoder")), ("cluster_deep", dict(DEC, name="dec"))):
            settings = with_model(name, model)
            settings["fit"]["device"] = "cuda"
            checked(settings)

    def test_settings_the_checks_refuse(self):
        bad = [("embed_deep", dict(AE, name="autoencoder", latent_dim=0)),
               ("embed_deep", dict(AE, name="autoencoder", hidden=[0])),
               ("embed_deep", dict(AE, name="autoencoder", dropout=1.0)),
               ("embed_deep", dict(AE, name="autoencoder", epochs=0)),
               ("embed_deep", dict(AE, name="autoencoder", lr=0.0)),
               ("embed_deep", dict(VAE, name="vae", beta=0.0)),
               ("cluster_deep", dict(AE_KMEANS, name="autoencoder_kmeans", n_clusters=1)),
               ("cluster_deep", dict(AE_KMEANS, name="autoencoder_kmeans", n_init=0)),
               ("cluster_deep", dict(DEC, name="dec", alpha=0.0)),
               ("cluster_deep", dict(DEC, name="dec", tol=1.0)),
               ("cluster_deep", dict(DEC, name="dec", update_interval=0)),
               ("cluster_deep", dict(DEC, name="dec", refine_lr=0.0))]
        for name, model in bad:
            with self.assertRaises(ValueError, msg=str(model)):
                checked(with_model(name, model))
        checked(with_model("embed_deep", dict(VAE, name="vae", hidden=[])))


# With torch ----------------------------------------------------------------------------------------------------------

@needs_module("torch")
class TestDeepEmbedders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.truth = planted(4)

    def test_the_autoencoder_keeps_planted_groups_apart(self):
        model = build("autoencoder", AE)
        e = model.fit_transform(self.x)
        self.assertEqual(e.shape, (600, 2))
        self.assertGreaterEqual(label_agreement(e, self.truth), 0.9)
        self.assertTrue(model.can_transform() and model.can_inverse())
        self.assertTrue(np.allclose(model.transform(self.x[:50]), e[:50], atol=1e-5))
        rebuilt = model.inverse_transform(e)
        self.assertEqual(rebuilt.shape, self.x.shape)
        self.assertLess(float(np.mean((rebuilt - self.x) ** 2)), 1.0, "the columns' means rebuild with 1.0")
        params = model.params()
        self.assertEqual(len(params["loss_by_epoch"]), AE["epochs"])
        self.assertLess(params["loss_by_epoch"][-1], params["loss_by_epoch"][0])
        self.assertEqual(params["optimizer"]["name"], "AdamW")
        self.assertIn("eps", params["optimizer"])
        self.assertEqual(params["steps_per_epoch"], 600 // AE["batch_size"])

    def test_the_vae_keeps_planted_groups_apart(self):
        model = build("vae", VAE)
        e = model.fit_transform(self.x)
        self.assertGreaterEqual(label_agreement(e, self.truth), 0.85)
        self.assertGreater(model.info()["final_kl"], 0.0)
        self.assertTrue(model.params()["variational"])

    def test_a_seed_gives_the_same_codes_and_another_seed_others(self):
        first = build("vae", VAE, seed=3).fit_transform(self.x)
        again = build("vae", VAE, seed=3).fit_transform(self.x)
        other = build("vae", VAE, seed=4).fit_transform(self.x)
        self.assertTrue(np.array_equal(first, again))
        self.assertFalse(np.array_equal(first, other))

    def test_the_example_runs_through_the_embedder_runner(self):
        with TempRepo() as root:
            folder = run(example("embed_deep")["base"], root, "holdout")
            criteria = json.loads((folder / "criteria.json").read_text(encoding="utf-8"))
            self.assertIsInstance(criteria["criteria"]["reconstruction_error"], float)
            self.assertIsInstance(criteria["criteria"]["trustworthiness"], float)
            self.assertEqual(np.load(folder / "embedding.npy").shape[1], 2)
            params = json.loads((folder / "model_params.json").read_text(encoding="utf-8"))
            self.assertEqual(params["model"], "autoencoder")


@needs_module("torch")
class TestDeepClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.truth = planted(3)

    def fitted(self, name: str, settings: dict, seed: int = 1):
        model = build(name, settings, seed)
        model.set_columns(tuple(f"f{j}" for j in range(self.x.shape[1])), {})
        model.fit(self.x)
        return model

    def test_kmeans_on_the_codes_finds_the_planted_groups(self):
        model = self.fitted("autoencoder_kmeans", AE_KMEANS)
        self.assertGreaterEqual(ari(self.truth, model.labels()), 0.9)
        self.assertTrue(model.can_assign())
        self.assertEqual(model.assign(self.x).tolist(), model.labels().tolist())
        params = model.params()
        self.assertEqual((params["kmeans"]["n_init"], params["kmeans"]["max_iter"]), (4, 300))
        self.assertIn("latent_inertia", model.info())

    def test_dec_finds_the_planted_groups_with_probabilities(self):
        model = self.fitted("dec", DEC)
        labels = model.labels()
        self.assertGreaterEqual(ari(self.truth, labels), 0.9)
        self.assertTrue(np.allclose(model.q.sum(axis=1), 1.0))
        self.assertEqual(labels.tolist(), model.q.argmax(axis=1).tolist())
        probability = model.soft()["probability"]
        self.assertTrue(((probability > 1.0 / 3 - 1e-9) & (probability <= 1.0 + 1e-9)).all())
        self.assertEqual(model.assign(self.x).tolist(), labels.tolist())
        info = model.info()
        self.assertTrue(1 <= info["refine_epochs_run"] <= DEC["refine_epochs"])
        params = model.params()
        self.assertEqual(params["refine_optimizer"]["name"], "Adam")
        self.assertEqual(len(params["kl_by_epoch"]), info["refine_epochs_run"])

    def test_dec_kernel_and_target_by_hand(self):
        import torch

        from sweepkit.models.cluster.deep import soft_assignment, target_distribution

        z = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        centres = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
        q = soft_assignment(z, centres, 1.0)
        # alpha 1: q_ij proportional to 1 / (1 + d_ij^2); row 0: d^2 = 0 and 9; row 1: d^2 = 25 and 16
        expected = torch.tensor([[1.0, 0.1], [1.0 / 26, 1.0 / 17]])
        expected = expected / expected.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(q, expected, atol=1e-6))
        p = target_distribution(q)
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(p, weight / weight.sum(dim=1, keepdim=True), atol=1e-6))
        self.assertTrue(torch.allclose(p.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertGreater(float(p[0, 0]), float(q[0, 0]), "the target sharpens a confident row")

    def test_a_seed_gives_the_same_clusters(self):
        first = self.fitted("dec", DEC, seed=5)
        again = self.fitted("dec", DEC, seed=5)
        self.assertEqual(first.labels().tolist(), again.labels().tolist())
        self.assertTrue(np.array_equal(first.q, again.q))

    def test_the_example_runs_through_the_clusterer_runner(self):
        with TempRepo() as root:
            folder = run(example("cluster_deep")["base"], root, "sub1")
            criteria = json.loads((folder / "criteria.json").read_text(encoding="utf-8"))
            self.assertIsInstance(criteria["criteria"]["silhouette"], float)
            settings = with_model("cluster_deep", dict(DEC, name="dec"))
            header = (run(settings, root, "sub1") / "assignments.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header.split(","), ["row_id", "cluster", "status", "probability"])


if __name__ == "__main__":
    unittest.main()
