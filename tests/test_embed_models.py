"""Embedders on data with known structure: manifold methods unroll a swiss roll or a circle (trustworthiness), PCoA
on Euclidean distances is PCA, a diffusion map recovers a circle's angle, t-SNE keeps planted blobs apart (k-nearest-
neighbour label agreement), and every method that places new rows gives a fit row its own coordinates back.

Thresholds were set from runs in the container (scikit-learn 1.9.1, SciPy 1.18.1); the observed value is in a comment
beside each. Tests of optional packages (umap-learn, openTSNE, PaCMAP, TriMap, PHATE, cuML) skip where the package is
missing, and fail instead where SWEEPKIT_REQUIRE names it (the node); their thresholds are first proposals, to be
confirmed by that run.
"""
import unittest

from tests.support import load_catalog, needs_gpu, needs_module


def build(name: str, settings: dict, seed: int = 1, device: str = "cpu"):
    from sweepkit.registry import MODELS
    from sweepkit.settings import validate

    entry = MODELS.get(name)
    return entry.target(validate(entry.settings_class, settings, name), seed, device)


def data(generator: str, n_rows: int, n_features: int, n_groups: int, noise: float, seed: int):
    import numpy as np

    from sweepkit.data.synthetic import make

    made = make(generator, n_rows, n_features, n_groups, noise, seed)
    return np.asarray(made["x"], dtype=np.float64), made["columns"]


def trust(x, e, k: int = 10) -> float:
    from sklearn.manifold import trustworthiness

    return float(trustworthiness(x, e, n_neighbors=k))


def label_agreement(e, labels, k: int = 10) -> float:
    """Leave-one-out majority vote of the k nearest rows' labels in the embedding."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    found = NearestNeighbors(n_neighbors=k + 1).fit(e).kneighbors(e, return_distance=False)
    correct = 0
    for i, row in enumerate(found):
        values, counts = np.unique(labels[[j for j in row if j != i][:k]], return_counts=True)
        correct += values[np.argmax(counts)] == labels[i]
    return correct / len(labels)


def angle_recovery(e, angle) -> float:
    """1 when the first two coordinates go round the circle with the true angle (any rotation or reflection)."""
    import numpy as np

    found = np.arctan2(e[:, 1], e[:, 0])
    return float(max(abs(np.mean(np.exp(1j * (found - angle)))), abs(np.mean(np.exp(1j * (found + angle))))))


LLE = {"n_components": 2, "n_neighbors": 10, "reg": 0.001, "eigen_solver": "dense", "hessian_tol": 1e-4,
       "modified_tol": 1e-12, "extra": {}}
TSNE = {"n_components": 2, "perplexity": 30.0, "early_exaggeration": 12.0, "learning_rate": None, "max_iter": 500,
        "metric": "euclidean", "init": "pca", "method": "barnes_hut", "angle": 0.5, "extra": {}}
DIFFUSION = {"n_components": 2, "kernel": "adaptive", "bandwidth": None, "n_neighbors": 10, "alpha": 1.0, "t": 1,
             "solver": "dense"}
UMAP = {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "spread": 1.0, "metric": "euclidean",
        "n_epochs": 200, "learning_rate": 1.0, "init": "spectral", "set_op_mix_ratio": 1.0,
        "local_connectivity": 1.0, "repulsion_strength": 1.0, "negative_sample_rate": 5, "densmap": False,
        "gpu_build_algo": "brute_force_knn", "extra": {}}


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()
        cls.roll, cls.roll_columns = data("swiss_roll", 800, 3, 4, 0.05, 1)
        cls.clean_roll, _ = data("swiss_roll", 800, 3, 4, 0.0, 1)
        cls.circle, cls.circle_columns = data("circle", 400, 2, 1, 0.02, 2)

    def test_isomap_unrolls_and_places_fit_rows_exactly(self):
        import numpy as np

        model = build("isomap", {"n_components": 2, "n_neighbors": 10, "metric": "euclidean", "eigen_solver": "auto",
                                 "path_method": "auto", "extra": {}})
        e = model.fit_transform(self.roll)
        self.assertGreaterEqual(trust(self.roll, e), 0.95)                   # observed 0.986
        self.assertTrue(model.can_transform())
        self.assertFalse(model.can_inverse())
        self.assertTrue(np.allclose(model.transform(self.roll[:40]), e[:40], atol=1e-5))
        self.assertIn("kernel_reconstruction_error", model.info())

    def test_lle_variants(self):
        standard = build("lle", dict(LLE, method="standard"))
        self.assertGreaterEqual(trust(self.clean_roll, standard.fit_transform(self.clean_roll)), 0.95)   # 0.988
        # modified, Hessian and LTSA give each coordinate unit variance, which stretches the roll's short side,
        # so their trustworthiness against the original distances is lower (observed 0.84, 0.87, 0.87)
        for method in ("modified", "hessian", "ltsa"):
            model = build("lle", dict(LLE, method=method))
            self.assertGreaterEqual(trust(self.clean_roll, model.fit_transform(self.clean_roll)), 0.8, method)
            self.assertTrue(model.can_transform())

    def test_spectral_embedding_and_mds(self):
        spectral = build("spectral_embedding", {"n_components": 2, "affinity": "nearest_neighbors", "n_neighbors": 10,
                                                "gamma": None, "eigen_solver": "arpack", "extra": {}})
        self.assertGreaterEqual(trust(self.roll, spectral.fit_transform(self.roll)), 0.93)            # 0.965
        self.assertFalse(spectral.can_transform())
        for metric_mds in (True, False):
            mds = build("mds", {"n_components": 2, "metric_mds": metric_mds, "metric": "euclidean",
                                "init": "classical_mds", "n_init": 1, "max_iter": 300, "eps": 1e-6,
                                "normalized_stress": True, "extra": {}})
            self.assertGreaterEqual(trust(self.circle, mds.fit_transform(self.circle)), 0.95)     # 1.000
            self.assertFalse(mds.can_transform())
            self.assertIn("stress", mds.info())

    def test_pcoa_on_euclidean_distances_is_pca(self):
        import numpy as np
        from sklearn.decomposition import PCA

        x, new = self.roll[:600], self.roll[600:]
        model = build("pcoa", {"n_components": 2, "metric": "euclidean"})
        e = model.fit_transform(x)
        pca = PCA(n_components=2).fit(x)
        reference = pca.transform(x)
        signs = np.sign((e * reference).sum(axis=0))
        self.assertTrue(np.allclose(e * signs, reference, atol=1e-4))
        self.assertAlmostEqual(model.info()["explained_variance"], float(pca.explained_variance_ratio_.sum()), places=6)
        # Gower's add-a-point formula is exact for Euclidean distances: new rows land where PCA puts them
        self.assertTrue(np.allclose(model.transform(new) * signs, pca.transform(new), atol=1e-4))

    def test_pcoa_reports_negative_eigenvalues(self):
        import numpy as np

        x = np.abs(self.roll)
        model = build("pcoa", {"n_components": 2, "metric": "braycurtis"})
        e = model.fit_transform(x)
        self.assertGreater(model.info()["negative_eigenvalue_share"], 0.05)   # observed 0.24
        self.assertTrue(np.allclose(model.transform(x[:30]), e[:30], atol=1e-5))

    def test_diffusion_map_recovers_the_angle_and_places_new_rows(self):
        import numpy as np

        angle = self.circle_columns["reference"].astype(float)
        model = build("diffusion_map", DIFFUSION)
        e = model.fit_transform(self.circle)
        self.assertGreaterEqual(angle_recovery(e, angle), 0.95)
        self.assertTrue(np.allclose(model.transform(self.circle[:50]), e[:50], atol=1e-6))
        other, other_columns = data("circle", 200, 2, 1, 0.02, 9)
        placed = model.transform(other)
        self.assertGreaterEqual(angle_recovery(placed, other_columns["reference"].astype(float)), 0.95)
        fixed = build("diffusion_map", dict(DIFFUSION, kernel="fixed", bandwidth=0.1, n_neighbors=None, alpha=0.5))
        self.assertGreaterEqual(angle_recovery(fixed.fit_transform(self.circle), angle), 0.95)

    def test_diffusion_map_solvers_agree(self):
        import numpy as np

        dense = build("diffusion_map", dict(DIFFUSION, n_components=3)).fit_transform(self.roll[:300])
        sparse = build("diffusion_map", dict(DIFFUSION, n_components=3, solver="arpack")).fit_transform(self.roll[:300])
        self.assertTrue(np.allclose(np.abs(dense), np.abs(sparse), atol=1e-4))

    def test_tsne_keeps_blobs_apart(self):
        x, columns = data("blobs", 300, 10, 4, 1.0, 3)
        model = build("tsne", dict(TSNE, max_iter=300))
        e = model.fit_transform(x)
        self.assertGreaterEqual(label_agreement(e, columns["truth"]), 0.9)   # observed 1.0
        self.assertFalse(model.can_transform())
        self.assertIn("kl_divergence", model.info())

    def test_same_seed_same_embedding(self):
        import numpy as np

        for name, settings in (("tsne", dict(TSNE, max_iter=250)), ("lle", dict(LLE, method="standard")),
                               ("diffusion_map", DIFFUSION)):
            a = build(name, settings, seed=5).fit_transform(self.roll[:150])
            b = build(name, settings, seed=5).fit_transform(self.roll[:150])
            self.assertTrue(np.array_equal(a, b), name)


class TestChecksAtExpansion(unittest.TestCase):
    """Impossible settings fail when the sweep is expanded, not when the job runs."""

    @classmethod
    def setUpClass(cls):
        load_catalog()

    def job(self, model: dict, device: str):
        from sweepkit.runs.spec import check_job, validate_job

        raw = {"data": {"name": "synthetic", "generator": "swiss_roll", "n_rows": 200, "n_features": 3, "n_groups": 4,
                        "noise": 0.05, "seed": 1},
               "split": {"name": "all"},
               "task": {"name": "embedding", "criteria": ["trustworthiness"], "select": "trustworthiness",
                        "sample_rows": None, "sample_salt": "s", "n_neighbors": 10, "reference": None,
                        "reveal_metrics": [], "criteria_options": {}},
               "prep": [], "features": {"name": "none"}, "model": model,
               "fit": {"name": "embedder", "device": device, "assign": "fit_rows"}}
        check_job(validate_job(raw))

    def test_refusals(self):
        bad = [
            ({"name": "lle", **dict(LLE, method="hessian", n_neighbors=4)}, "cpu", "hessian"),
            ({"name": "tsne", **dict(TSNE, n_components=4)}, "cpu", "at most 3"),
            ({"name": "tsne", **dict(TSNE, method="fft")}, "cpu", "cuML only"),
            ({"name": "tsne", **dict(TSNE, n_components=3)}, "cuda", "2 only"),
            ({"name": "umap", **dict(UMAP, init="pca")}, "cuda", "spectral or random"),
            ({"name": "umap", **dict(UMAP, densmap=True)}, "cuda", "densmap"),
            ({"name": "diffusion_map", **dict(DIFFUSION, kernel="fixed")}, "cpu", "bandwidth"),
            ({"name": "diffusion_map", **dict(DIFFUSION, n_neighbors=1)}, "cpu", "at least 2"),
            ({"name": "spectral_embedding", "n_components": 2, "affinity": "rbf", "n_neighbors": None, "gamma": None,
              "eigen_solver": "arpack", "extra": {}}, "cuda", "nearest_neighbors only"),
        ]
        for model, device, message in bad:
            with self.assertRaisesRegex(ValueError, message):
                self.job(model, device)
        self.job({"name": "umap", **UMAP}, "cpu")
        self.job({"name": "tsne", **TSNE}, "cpu")


class TestThroughTheRunner(unittest.TestCase):
    """Real runs through the embedder family: rows placed by transform (fit_holdout, assign predict) and a reduce
    step feeding a clusterer."""

    def test_smoke_of_a_small_sweep(self):
        import json
        from pathlib import Path

        from tests.support import TempRepo

        load_catalog()
        base = {"data": {"name": "synthetic", "generator": "swiss_roll", "n_rows": 400, "n_features": 3,
                         "n_groups": 4, "noise": 0.05, "seed": 1},
                "split": {"name": "fit_holdout", "fit_fraction": 0.75, "salt": "t"},
                "task": {"name": "embedding", "criteria": ["trustworthiness", "continuity", "knn_preservation"],
                         "select": "trustworthiness", "sample_rows": 80, "sample_salt": "s", "n_neighbors": 8,
                         "reference": "truth", "reveal_metrics": ["knn_label_agreement"], "criteria_options": {}},
                "prep": [{"name": "scale", "method": "zscore", "columns": ["*"], "per_group": None}],
                "features": {"name": "none"},
                "model": {"name": "pcoa", "n_components": 2, "metric": "euclidean"},
                "fit": {"name": "embedder", "device": "cpu", "assign": "predict"}}
        sweep = {"sweep_id": "embed_runner", "description": "embedders through the runner", "purpose": "learning",
                 "follows": [], "folds": ["holdout"], "base": base,
                 "stages": [{"name": "base", "kind": "base", "priority": 2, "seeds": [1]},
                            {"name": "v", "kind": "variants", "priority": 1, "seeds": [1], "variants": {
                                "isomap": {"model": {"name": "isomap", "n_components": 2, "n_neighbors": 10,
                                                     "metric": "euclidean", "eigen_solver": "auto",
                                                     "path_method": "auto", "extra": {}}},
                                "diffusion": {"model": {"name": "diffusion_map", **DIFFUSION}}}}]}
        with TempRepo() as root:
            path = Path(root) / "s.json"
            path.write_text(json.dumps(sweep), encoding="utf-8")
            from sweepkit.sweep.smoke import smoke

            result = smoke(path, "cpu", 100, lambda m: None)
            self.assertEqual(result["failed"], 0)


@needs_module("umap")
class TestUMAP(unittest.TestCase):
    def test_blobs_and_transform(self):
        load_catalog()
        x, columns = data("blobs", 600, 10, 4, 1.0, 3)
        model = build("umap", UMAP)
        e = model.fit_transform(x[:500])
        self.assertGreaterEqual(label_agreement(e, columns["truth"][:500]), 0.9)
        self.assertTrue(model.can_transform())
        self.assertFalse(model.can_inverse())
        self.assertEqual(model.transform(x[500:]).shape, (100, 2))


@needs_module("openTSNE")
class TestOpenTSNE(unittest.TestCase):
    def test_blobs_and_new_rows(self):
        import numpy as np

        load_catalog()
        x, columns = data("blobs", 600, 10, 4, 1.0, 3)
        settings = {"n_components": 2, "perplexity": 30.0, "learning_rate": None, "early_exaggeration": None,
                    "early_exaggeration_iter": 250, "n_iter": 500, "exaggeration": None, "dof": 1.0,
                    "initialization": "pca", "metric": "euclidean", "neighbors": "exact",
                    "negative_gradient_method": "fft", "transform_perplexity": 5.0, "transform_n_iter": 250,
                    "extra": {}}
        model = build("opentsne", settings)
        e = model.fit_transform(x[:500])
        truth = np.asarray(columns["truth"])
        self.assertGreaterEqual(label_agreement(e, truth[:500]), 0.9)
        placed = model.transform(x[500:])
        combined = np.vstack([e, placed])
        self.assertGreaterEqual(label_agreement(combined, truth), 0.9)


@needs_module("pacmap")
class TestPaCMAP(unittest.TestCase):
    def test_blobs_and_transform(self):
        load_catalog()
        x, columns = data("blobs", 600, 10, 4, 1.0, 3)
        model = build("pacmap", {"n_components": 2, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 2.0,
                                 "distance": "euclidean", "lr": 1.0, "num_iters": 450, "apply_pca": True,
                                 "knn_backend": "faiss", "extra": {}})
        e = model.fit_transform(x[:500])
        self.assertGreaterEqual(label_agreement(e, columns["truth"][:500]), 0.9)
        self.assertEqual(model.transform(x[500:]).shape, (100, 2))


@needs_module("trimap")
@needs_module("torch")
class TestTriMap(unittest.TestCase):
    def test_blobs(self):
        load_catalog()
        x, columns = data("blobs", 600, 10, 4, 1.0, 3)
        model = build("trimap", {"n_components": 2, "n_inliers": 12, "n_outliers": 4, "n_random": 3,
                                 "distance": "euclidean", "lr": 0.1, "n_iters": 400, "apply_pca": True,
                                 "opt_method": "dbd", "init": "pca", "knn_backend": "torch", "extra": {}})
        e = model.fit_transform(x)
        self.assertGreaterEqual(label_agreement(e, columns["truth"]), 0.9)
        self.assertFalse(model.can_transform())


@needs_module("phate")
class TestPHATE(unittest.TestCase):
    def test_blobs(self):
        load_catalog()
        x, columns = data("blobs", 600, 10, 4, 1.0, 3)
        model = build("phate", {"n_components": 2, "knn": 5, "decay": 40.0, "n_landmark": None, "t": None,
                                "gamma": 1.0, "n_pca": None, "mds_solver": "sgd", "mds": "metric",
                                "knn_dist": "euclidean", "mds_dist": "euclidean", "knn_max": None, "extra": {}})
        e = model.fit_transform(x)
        self.assertGreaterEqual(label_agreement(e, columns["truth"]), 0.85)
        self.assertFalse(model.can_transform())


@needs_module("cuml")
@needs_gpu()
class TestCuML(unittest.TestCase):
    """The GPU paths reach the same structure as the CPU ones."""

    def test_umap_tsne_and_spectral_on_the_gpu(self):
        load_catalog()
        x, columns = data("blobs", 2000, 10, 4, 1.0, 3)
        truth = columns["truth"]
        self.assertGreaterEqual(label_agreement(build("umap", UMAP, device="cuda").fit_transform(x), truth), 0.9)
        tsne = build("tsne", dict(TSNE, method="fft"), device="cuda")
        self.assertGreaterEqual(label_agreement(tsne.fit_transform(x), truth), 0.9)
        spectral = build("spectral_embedding", {"n_components": 2, "affinity": "nearest_neighbors", "n_neighbors": 15,
                                                "gamma": None, "eigen_solver": "arpack", "extra": {}}, device="cuda")
        self.assertGreaterEqual(label_agreement(spectral.fit_transform(x), truth), 0.9)


if __name__ == "__main__":
    unittest.main()
