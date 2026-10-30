"""Sweep expansion, enqueue and the file queue."""
import json
import tempfile
import unittest
from pathlib import Path

from sweepkit.sweep.design import SweepError, apply, expand, get_path
from tests.support import TempRepo, load_catalog
from tests.sweep_fixtures import classify_base, cluster_base, sweep

load_catalog()


def base_stage(seeds=(1, 2, 3)):
    return {"name": "base", "kind": "base", "priority": 100, "seeds": list(seeds)}


class TestHeader(unittest.TestCase):
    def test_required_keys_and_ids(self):
        good = sweep(cluster_base(), [base_stage()])
        expand(good)
        for key in ("description", "purpose", "follows"):
            bad = dict(good)
            del bad[key]
            with self.assertRaisesRegex(SweepError, "missing"):
                expand(bad)
        for bad_id in ("a b", "a__b", ""):
            with self.assertRaises(SweepError):
                expand({**good, "sweep_id": bad_id})
        with self.assertRaisesRegex(SweepError, "purpose"):
            expand({**good, "purpose": "explore"})
        with self.assertRaisesRegex(SweepError, "follow itself"):
            expand({**good, "follows": ["t1"]})

    def test_folds_must_be_the_splits(self):
        with self.assertRaisesRegex(SweepError, "not folds"):
            expand(sweep(cluster_base(), [base_stage()], folds=["fold1"]))


class TestCheckCommand(unittest.TestCase):
    def test_check_lists_each_configuration_with_its_hash(self):
        import contextlib
        import io

        from sweepkit import cli

        s = sweep(cluster_base(), [base_stage(), {"name": "k", "kind": "one_factor_at_a_time", "priority": 50,
                                                  "seeds": [1, 2, 3], "factors": {"model.n_clusters": [2, 4]}}])
        jobs, _ = expand(s)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            path.write_text(json.dumps(s), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["check", "--sweep-file", str(path)]), 0)
        printed = json.loads(out.getvalue())
        listed = printed["configurations"]
        self.assertEqual([c["label"] for c in listed], ["base", "model.n_clusters=2", "model.n_clusters=4"])
        self.assertEqual(sum(c["jobs"] for c in listed), len(jobs))
        k4 = next(j["config_hash"] for j in jobs if j["label"] == "model.n_clusters=4")
        self.assertEqual(listed[2], {"config": k4[:10], "stage": "k", "label": "model.n_clusters=4", "device": "cpu",
                                     "jobs": len(jobs) // 3})


class TestStages(unittest.TestCase):
    def test_kinds_and_counts(self):
        stages = [base_stage([1, 2]),
                  {"name": "k", "kind": "one_factor_at_a_time", "priority": 90, "seeds": [1, 2],
                   "factors": {"model.n_clusters": [2, 4], "model.init": ["random"]}},
                  {"name": "g", "kind": "grid", "priority": 80, "seeds": [1],
                   "factors": {"model.n_clusters": [3, 5], "model.n_init": [2, 4]}},
                  {"name": "v", "kind": "variants", "priority": 70, "seeds": [1],
                   "variants": {"tenth": {"model.tol": 0.1}}}]
        jobs, summary = expand(sweep(cluster_base(), stages))
        counts = {s["name"]: s["configs"] for s in summary["stages"]}
        # the grid's (3, 2) is the base itself and is left out
        self.assertEqual(counts, {"base": 1, "k": 3, "g": 3, "v": 1})
        self.assertEqual(summary["jobs"], 2 * 2 * (1 + 3) + 2 * (3 + 1))
        self.assertTrue(all(j["run_id"].startswith(j["config_hash"][:12]) for j in jobs))
        self.assertEqual({j["run_id"].split("_")[-1] for j in jobs}, {"seed1", "seed2"})

    def test_duplicates_refused_except_random_draws(self):
        stages = [base_stage([1]),
                  {"name": "a", "kind": "variants", "priority": 1, "seeds": [1], "variants": {"x": {"model.tol": 0.1}}},
                  {"name": "b", "kind": "variants", "priority": 1, "seeds": [1], "variants": {"y": {"model.tol": 0.1}}}]
        with self.assertRaisesRegex(SweepError, "identical"):
            expand(sweep(cluster_base(), stages))
        with self.assertRaisesRegex(SweepError, "identical to the base"):
            expand(sweep(cluster_base(), [{"name": "v", "kind": "variants", "priority": 1, "seeds": [1],
                                           "variants": {"same": {"model.tol": 1e-4}}}]))

    def test_random_is_deterministic_and_extends_exactly(self):
        space = {"model.n_clusters": [2, 3, 4, 5, 6], "model.n_init": [1, 2, 3], "model.tol": [0.1, 0.01, 0.001]}

        def random_stage(n, skip):
            return {"name": "r", "kind": "random", "priority": 1, "seeds": [1], "space": space, "n_configs": n,
                    "sampling_seed": 7, "skip_draws": skip}

        big = expand(sweep(cluster_base(), [random_stage(8, 0)]))[0]
        again = expand(sweep(cluster_base(), [random_stage(8, 0)]))[0]
        self.assertEqual([j["config_hash"] for j in big], [j["config_hash"] for j in again])
        head = expand(sweep(cluster_base(), [random_stage(5, 0)]))[0]
        tail = expand(sweep(cluster_base(), [random_stage(3, 5)]))[0]
        big_hashes = [j["config_hash"] for j in big if j["fold"] == "sub1"]
        self.assertEqual([j["config_hash"] for j in head if j["fold"] == "sub1"]
                         + [j["config_hash"] for j in tail if j["fold"] == "sub1"], big_hashes)
        self.assertEqual(len(set(big_hashes)), 8)

    def test_random_too_small_space_fails(self):
        stage = {"name": "r", "kind": "random", "priority": 1, "seeds": [1], "space": {"model.n_clusters": [2]},
                 "n_configs": 3, "sampling_seed": 1, "skip_draws": 0}
        with self.assertRaisesRegex(SweepError, "distinct valid"):
            expand(sweep(cluster_base(), [stage]))


class TestOverrides(unittest.TestCase):
    def test_paths_through_prep_and_whole_sections(self):
        base = cluster_base()
        out = apply(base, {"prep.scale.method": "robust", "model.n_clusters": 5})
        self.assertEqual(out["prep"][0]["method"], "robust")
        self.assertEqual(get_path(out, "prep.0.method"), "robust")
        with self.assertRaisesRegex(SweepError, "names no setting"):
            apply(base, {"model.n_cluster": 5})
        with self.assertRaisesRegex(SweepError, "named 'impute'"):
            apply(base, {"prep.impute.method": "mean"})
        replaced = apply(base, {"model": {"name": "minibatch_kmeans", "n_clusters": 3, "batch_size": 64,
                                          "n_init": 1, "max_iter": 50, "extra": {}}})
        self.assertEqual(replaced["model"]["name"], "minibatch_kmeans")
        jobs, _ = expand(sweep(base, [base_stage([1]), {"name": "v", "kind": "variants", "priority": 1, "seeds": [1],
                                                         "variants": {"robust": {"prep.scale.method": "robust"}}}]))
        self.assertEqual(len({j["config_hash"] for j in jobs}), 2)

    def test_one_protocol_per_sweep(self):
        stages = [base_stage([1]), {"name": "v", "kind": "variants", "priority": 1, "seeds": [1],
                                    "variants": {"other data": {"data.seed": 4}}}]
        with self.assertRaisesRegex(SweepError, "one evaluation protocol"):
            expand(sweep(cluster_base(), stages))

    def test_invalid_settings_fail_at_expansion(self):
        stages = [{"name": "v", "kind": "variants", "priority": 1, "seeds": [1],
                   "variants": {"typo": {"model.init": "kmeans++"}}}]
        with self.assertRaisesRegex(SweepError, "model.init"):
            expand(sweep(cluster_base(), stages))

    def test_family_and_task_checks(self):
        base = cluster_base()
        base["fit"] = {"name": "estimator", "device": "cpu", "control": "none", "refit": "train", "eval_train": False}
        with self.assertRaisesRegex(SweepError, "fits the estimator family"):
            expand(sweep(base, [base_stage([1])]))
        base = classify_base()
        base["split"] = {"name": "all"}
        with self.assertRaisesRegex(SweepError, "roles"):
            expand(sweep(base, [base_stage([1])], folds=["all"]))


class TestEnqueueAndQueue(unittest.TestCase):
    def write(self, folder: Path, sweep_dict: dict) -> Path:
        path = folder / f"{sweep_dict['sweep_id']}.json"
        path.write_text(json.dumps(sweep_dict), encoding="utf-8")
        return path

    def test_enqueue_records_once_and_routes_by_device(self):
        from sweepkit.sweep.enqueue import enqueue
        from sweepkit.sweep.queue import Queue

        with TempRepo() as root, tempfile.TemporaryDirectory() as d:
            path = self.write(Path(d), sweep(cluster_base(), [base_stage([1, 2, 3])]))
            summary = enqueue(path, lambda m: None)
            self.assertEqual(summary["jobs"], 6)
            self.assertTrue((root / "artifacts" / "sweeps" / "t1" / "jobs.csv").is_file())
            with self.assertRaises(FileExistsError):
                enqueue(path, lambda m: None)
            queue = Queue(root / "queue")
            self.assertIsNone(queue.take("gpu0_1", ("cuda",)))
            claimed = queue.take("cpu_1", ("cpu", "any"))
            self.assertIsNotNone(claimed)
            self.assertEqual(queue.counts("t1")["running"], 1)
            self.assertEqual(queue.requeue_running(), 1)
            self.assertEqual(queue.park("t1", "later"), 6)
            self.assertIsNone(queue.take("cpu_1", ("cpu", "any")))
            self.assertEqual(queue.unpark("later"), 6)
            (root / "queue" / "STOP_gpu2").write_text("")
            self.assertTrue(queue.stop_requested("gpu2_1"))
            self.assertFalse(queue.stop_requested("gpu1_1"))

    def test_confirmation_may_not_reuse_followed_seeds(self):
        from sweepkit.sweep.design import SweepError as Err
        from sweepkit.sweep.enqueue import enqueue

        with TempRepo(), tempfile.TemporaryDirectory() as d:
            enqueue(self.write(Path(d), sweep(cluster_base(), [base_stage([1, 2, 3])], sweep_id="screen1")),
                    lambda m: None)
            confirm = sweep(cluster_base(), [base_stage([3, 4, 5])], sweep_id="confirm1", purpose="confirm",
                            follows=["screen1"])
            with self.assertRaisesRegex(Err, "reuses seeds \\[3\\]"):
                enqueue(self.write(Path(d), confirm), lambda m: None)
            confirm["stages"] = [base_stage([6, 7, 8, 9, 10])]
            enqueue(self.write(Path(d), confirm), lambda m: None)

    def test_several_sweeps_are_enqueued_together_or_not_at_all(self):
        """A round's sweeps go in with one enqueue: each is checked before any is recorded or queued."""
        from sweepkit.sweep.design import SweepError as Err
        from sweepkit.sweep.enqueue import enqueue_all

        with TempRepo() as root, tempfile.TemporaryDirectory() as d:
            screen = self.write(Path(d), sweep(cluster_base(), [base_stage([1, 2, 3])], sweep_id="screen1"))
            lost = self.write(Path(d), sweep(cluster_base(), [base_stage([6, 7, 8, 9, 10])], sweep_id="confirm0",
                                             purpose="confirm", follows=["nowhere"]))
            with self.assertRaisesRegex(Err, "follows 'nowhere'"):
                enqueue_all([screen, lost], lambda m: None)
            (Path(d) / "copy").mkdir()
            twice = self.write(Path(d) / "copy", sweep(cluster_base(), [base_stage([4])], sweep_id="screen1"))
            with self.assertRaisesRegex(Err, "given more than once"):
                enqueue_all([screen, twice], lambda m: None)
            self.assertFalse((root / "artifacts" / "sweeps" / "screen1").exists())
            self.assertEqual(list((root / "queue" / "pending").glob("*")) if (root / "queue" / "pending").is_dir()
                             else [], [])
            # a sweep may follow one enqueued with it; its seeds are checked against that one's
            reuse = self.write(Path(d), sweep(cluster_base(), [base_stage([3, 4, 5])], sweep_id="confirm1",
                                              purpose="confirm", follows=["screen1"]))
            with self.assertRaisesRegex(Err, "reuses seeds \\[3\\]"):
                enqueue_all([screen, reuse], lambda m: None)
            confirm = self.write(Path(d), sweep(cluster_base(), [base_stage([6, 7, 8, 9, 10])], sweep_id="confirm2",
                                                purpose="confirm", follows=["screen1"]))
            summaries = enqueue_all([screen, confirm], lambda m: None)
            self.assertEqual([s["sweep_id"] for s in summaries], ["screen1", "confirm2"])
            self.assertEqual(len(list((root / "queue" / "pending").glob("*.json"))), 6 + 10)
            # one already enqueued: nothing more goes in
            other = self.write(Path(d), sweep(cluster_base(), [base_stage([1])], sweep_id="other1"))
            with self.assertRaisesRegex(FileExistsError, "already enqueued"):
                enqueue_all([other, screen], lambda m: None)
            self.assertFalse((root / "artifacts" / "sweeps" / "other1").exists())

    def test_one_seed_screen_warns(self):
        from sweepkit.sweep.enqueue import enqueue

        with TempRepo(), tempfile.TemporaryDirectory() as d:
            notes = []
            enqueue(self.write(Path(d), sweep(cluster_base(), [base_stage([1])])), notes.append)
            self.assertTrue(any("one seed" in n for n in notes))
            # a clustering sweep: one seed is enough only for a method without randomness
            self.assertTrue(any("same result for every seed" in n for n in notes))

    def test_one_seed_warning_by_family(self):
        from sweepkit.sweep.design import warnings_for

        stages = [{"name": "s", "seeds": [1]}, {"name": "t", "seeds": [1, 2, 3]}]
        clusters = warnings_for({"purpose": "screen"}, {"family": "clusterer", "stages": stages})
        supervised = warnings_for({"purpose": "screen"}, {"family": "estimator", "stages": stages})
        self.assertEqual(len(clusters), 1)
        self.assertIn("compared over two or more subsamples", clusters[0])
        self.assertEqual(len(supervised), 1)
        self.assertIn("one seed's ranking does not predict another's", supervised[0])
        self.assertEqual(warnings_for({"purpose": "learning"}, {"family": "estimator", "stages": stages}), [])


if __name__ == "__main__":
    unittest.main()
