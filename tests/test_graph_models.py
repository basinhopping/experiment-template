"""Graph networks ("gcn" and "gat", sweepkit.models.torch.gnn) and the torch family's graph input.

Without torch (runs everywhere): their declarations (the torch profile, the graph kind, the example graph_networks)
and the transductive marking that sweepkit.analysis.transductive reads, so every report on such a configuration says
so; every configuration of graph_networks and of data_graph (the graph dataset's example, read from files) passes its
checks, and the checks refuse an external dataset on a graph job, input
that is not a graph, and impossible settings; preparation steps are allowed on a graph's node features; the graph
batches: GraphInput keeps each directed edge once and drops self loops, a training batch is the whole graph with the
batch's nodes, evaluation is one batch of every node of a role, and the refit's joined source keeps the graph. With
torch (the node, where SWEEPKIT_REQUIRE names torch): one GCN layer against the dense propagation rules (D^-1/2 (A + I)
D^-1/2 and the mean), one GAT layer against a dense softmax over each node's incoming edges, a node's output does not
move when a node it has no edge from changes, and through the runner on a planted graph (three communities, features
weakly informative): GCN and GAT reach balanced accuracy 0.8 on the test nodes, the shuffled-label control stays
below 0.6, a run repeats exactly on the CPU, the refit on training plus validation nodes runs, model_params.json and
describe-model give the network, and data_graph's base runs on the files make-example-data writes. The thresholds
are first proposals, not yet set from runs: the node session sets them from what the runs give, a little below.
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from tests.support import REPO, TempRepo, load_catalog, needs_module

load_catalog()

GCN = {"name": "gcn", "hidden": [16], "normalize": "symmetric", "self_loops": True, "activation": "relu",
       "input_dropout": 0.5, "dropout": 0.5, "classifier_hidden": None}
GAT = {"name": "gat", "hidden": [8], "heads": 4, "negative_slope": 0.2, "attention_dropout": 0.3, "self_loops": True,
       "activation": "elu", "input_dropout": 0.3, "dropout": 0.3, "classifier_hidden": None}
GRAPH = {"name": "synthetic", "generator": "graph", "n_rows": 150, "n_features": 4, "n_groups": 3, "noise": 0.5,
         "seed": 1}


def example(name: str = "graph_networks") -> dict:
    return json.loads((REPO / "sweeps" / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def graph_job(model=None, **fit_changes) -> dict:
    """The example's base on a smaller planted graph, with a holdout split."""
    base = example()["base"]
    base["data"] = dict(GRAPH)
    base["split"] = {"name": "holdout", "val_fraction": 0.2, "test_fraction": 0.2, "stratify": "truth", "salt": "t",
                     "identity_fold": False}
    base["model"] = copy.deepcopy(model or GCN)
    base["fit"].update(fit_changes)
    return base


def checked(settings: dict):
    from sweepkit.runs.spec import check_job, validate_job

    job = validate_job(settings)
    check_job(job)
    return job


def run(settings: dict, root, runs: str = "runs", fold: str = "holdout", seed: int = 1) -> Path:
    from sweepkit.runs.spec import config_hash, plain, run_id
    from sweepkit.sweep.runner import execute

    job = checked(settings)
    spec = {"sweep_id": "graph_test", "purpose": "learning", "stage": "base", "label": "base", "overrides": {},
            "priority": 1, "config_hash": config_hash(job), "run_id": run_id(job, fold, seed), "fold": fold,
            "seed": seed, "family": job.family, "settings": plain(job)}
    execute(spec, "cpu", lambda message: None, runs_root=Path(root) / runs)
    return Path(root) / runs / spec["run_id"]


def read(folder: Path, name: str):
    return json.loads((folder / name).read_text(encoding="utf-8"))


# Without torch --------------------------------------------------------------------------------------------------------

class TestDeclarations(unittest.TestCase):
    def test_registry_entries_and_the_transductive_marking(self):
        from sweepkit.analysis.transductive import parts
        from sweepkit.registry import MODELS

        for name in ("gcn", "gat"):
            meta = MODELS.get(name).meta
            self.assertEqual((meta["family"], meta["kinds"], meta["gpu"], meta["profile"], meta["example"]),
                             ("torch", ("graph",), "torch", "torch", "graph_networks"), name)
        base = example()["base"]
        self.assertEqual(parts(base), ["model gcn"])
        self.assertEqual(parts(dict(base, model=GAT)), ["model gat"])
        self.assertEqual(parts(example("data_graph")["base"]), ["model gcn"])
        self.assertEqual(parts(example("classify_mlp")["base"]), [])

    def test_every_example_configuration_passes_its_checks(self):
        from sweepkit.sweep.design import expand

        for name, configs in (("graph_networks", 5), ("data_graph", 2)):
            jobs, summary = expand(example(name))
            self.assertEqual(summary["configs"], configs, name)
            for job in jobs:
                checked(job["settings"])
        checked(graph_job(GAT))

    def test_what_the_checks_refuse(self):
        external = graph_job()
        external["fit"]["external"] = [{"name": "more", "data": dict(GRAPH, seed=2), "restrict_classes": False}]
        not_a_graph = graph_job()
        not_a_graph["data"] = dict(GRAPH, generator="classification")
        bad = [(external, "external"), (not_a_graph, "graph"),
               (graph_job(dict(GCN, hidden=[])), "hidden"), (graph_job(dict(GCN, input_dropout=1.0)), "dropout"),
               (graph_job(dict(GAT, heads=0)), "heads"), (graph_job(dict(GAT, attention_dropout=1.0)), "attention")]
        for settings, words in bad:
            with self.assertRaises(ValueError, msg=words) as caught:
                checked(settings)
            self.assertIn(words, str(caught.exception))

    def test_preparation_steps_act_on_node_features(self):
        settings = graph_job()
        settings["prep"] = [{"name": "scale", "method": "robust", "columns": ["*"], "per_group": None}]
        checked(settings)


class TestGraphBatches(unittest.TestCase):
    def setUp(self):
        from sweepkit.train.batches import GraphInput

        x = np.arange(15, dtype=np.float64).reshape(5, 3)
        # 0 -> 1 twice, a self loop on 2, and both directions of 3 - 4
        edges = np.array([[0, 0, 2, 3, 4, 1], [1, 1, 2, 4, 3, 2]])
        self.graph = GraphInput(x, edges)

    def test_each_directed_edge_once_without_self_loops(self):
        from sweepkit.train.batches import GraphInput

        self.assertEqual(sorted(map(tuple, self.graph.edges.T.tolist())), [(0, 1), (1, 2), (3, 4), (4, 3)])
        self.assertEqual((self.graph.n_nodes, self.graph.x.dtype), (5, np.float32))
        with self.assertRaises(ValueError):
            GraphInput(np.zeros((3, 2)), np.array([[0], [3]]))
        with self.assertRaises(ValueError):
            GraphInput(np.zeros((3, 2)), np.array([0, 1]))

    def test_training_and_evaluation_batches(self):
        from sweepkit.train.batches import GraphBatch, Source, concat_sources, eval_batches, training_batch

        train = Source("graph", np.array([4, 0, 2]), graph=self.graph)
        val = Source("graph", np.array([1, 3]), graph=self.graph)
        x, lengths = training_batch(train, np.array([2, 0]), None, np.random.default_rng(0))
        self.assertIsInstance(x, GraphBatch)
        self.assertIs(x.graph, self.graph)
        self.assertEqual((x.nodes.tolist(), lengths), ([2, 4], None))
        batches = list(eval_batches(val, 1))
        self.assertEqual(len(batches), 1, "one batch of every node, whatever the batch size")
        ks, batch, lengths = batches[0]
        self.assertEqual((ks.tolist(), batch.nodes.tolist(), len(batch), lengths), ([0, 1], [1, 3], 2, None))
        self.assertTrue(np.array_equal(train.item(1), self.graph.x[0]))
        joined = concat_sources(train, val)
        self.assertIs(joined.graph, self.graph)
        self.assertEqual(joined.positions.tolist(), [4, 0, 2, 1, 3])
        self.assertEqual(train.subset([1]).positions.tolist(), [0])
        with self.assertRaises(ValueError):
            Source("graph", np.array([0]))


# With torch ----------------------------------------------------------------------------------------------------------

def _dense(n: int, source, target):
    """M [n, n] with M[i, j] = 1 for each edge j -> i."""
    import torch

    m = torch.zeros(n, n)
    m[target, source] = 1.0
    return m


@needs_module("torch")
class TestLayers(unittest.TestCase):
    def setUp(self):
        import torch

        from sweepkit.registry import MODELS
        from sweepkit.settings import validate
        from sweepkit.train.batches import GraphInput

        torch.manual_seed(0)
        self.n = 6
        edges = np.array([[0, 1, 1, 2, 3, 4, 2, 5], [1, 0, 2, 1, 4, 3, 0, 0]])
        self.graph = GraphInput(np.random.default_rng(0).normal(size=(self.n, 4)), edges)
        self.validate = lambda name, raw: validate(MODELS.get(name).settings_class,
                                                   {k: v for k, v in raw.items() if k != "name"}, name)

    def network(self, raw: dict):
        from sweepkit.registry import MODELS

        net = MODELS.get(raw["name"]).target.build(self.validate(raw["name"], raw), (4,), 3)
        return net.eval()

    def test_one_gcn_layer_is_the_dense_propagation_rule(self):
        import torch

        from sweepkit.models.torch.gnn import edges_with_loops, gcn_weights

        x, e = self.graph.tensors("cpu")
        for normalize in ("symmetric", "mean"):
            net = self.network(dict(GCN, hidden=[5], normalize=normalize, input_dropout=0.0, dropout=0.0))
            layer = net.layers[0]
            source, target = edges_with_loops(e, self.n, True)
            with torch.no_grad():
                got = layer(x, source, target, gcn_weights(source, target, self.n, normalize, x.dtype))
                m = _dense(self.n, source, target)
                degree = m.sum(dim=1)
                if normalize == "symmetric":
                    a_hat = m / torch.sqrt(degree[:, None] * degree[None, :])
                else:
                    a_hat = m / degree[:, None]
                expected = a_hat @ layer.linear(x) + layer.bias
            self.assertTrue(torch.allclose(got, expected, atol=1e-5), normalize)

    def test_one_gat_layer_is_a_softmax_over_incoming_edges(self):
        import torch
        from torch import nn

        from sweepkit.models.torch.gnn import edges_with_loops

        net = self.network(dict(GAT, hidden=[3], heads=2, input_dropout=0.0, dropout=0.0, attention_dropout=0.0))
        layer = net.layers[0]
        x, e = self.graph.tensors("cpu")
        source, target = edges_with_loops(e, self.n, True)
        with torch.no_grad():
            got = layer(x, source, target)
            z = layer.linear(x).view(self.n, 2, 3)
            expected = torch.zeros(self.n, 2, 3)
            for i in range(self.n):
                incoming = source[target == i]
                for h in range(2):
                    scores = torch.stack([nn.functional.leaky_relu(
                        (z[j, h] * layer.att_source[0, h]).sum() + (z[i, h] * layer.att_target[0, h]).sum(), 0.2)
                        for j in incoming.tolist()])
                    alpha = torch.softmax(scores, dim=0)
                    expected[i, h] = (alpha[:, None] * z[incoming, h]).sum(dim=0)
            expected = expected.reshape(self.n, 6) + layer.bias
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_a_node_reads_only_the_nodes_with_an_edge_to_it(self):
        import torch

        from sweepkit.train.batches import GraphBatch, GraphInput

        # elu is strictly increasing, so a change before the activation is a change after it
        for raw in (dict(GCN, hidden=[5], activation="elu"), dict(GAT, hidden=[3], heads=2)):
            net = self.network(raw)
            nodes = np.arange(self.n)
            with torch.no_grad():
                before = net.embed(GraphBatch(self.graph, nodes), None)
                far = self.graph.x.copy()
                far[3] += 5.0                   # node 0 receives edges from 1, 2 and 5 only
                after = net.embed(GraphBatch(GraphInput(far, self.graph.edges), nodes), None)
                near = self.graph.x.copy()
                near[2] += 5.0
                moved = net.embed(GraphBatch(GraphInput(near, self.graph.edges), nodes), None)
            self.assertTrue(torch.allclose(before[0], after[0]), raw["name"])
            self.assertFalse(torch.allclose(before[0], moved[0]), raw["name"])
            self.assertEqual(tuple(before.shape), (self.n, net.embedding_dim))


@needs_module("torch")
class TestRuns(unittest.TestCase):
    def test_gcn_and_gat_find_the_planted_communities_and_the_control_does_not(self):
        with TempRepo() as root:
            for model, runs in ((GCN, "gcn"), (GAT, "gat")):
                folder = run(graph_job(model, lr=0.01 if model is GCN else 0.005), root, runs)
                self.assertGreaterEqual(read(folder, "metrics_test.json")["metrics"]["balanced_accuracy"], 0.8,
                                        runs)
            # With shuffled training labels each community's prediction is a random class, so one seed can land on
            # the right one by chance (two or three communities of three in about a quarter of seeds); the mean over
            # five seeds stays near chance (a third).
            control = [read(run(graph_job(control="shuffle_train_labels"), root, "control", seed=seed),
                            "metrics_test.json")["metrics"]["balanced_accuracy"] for seed in range(1, 6)]
            self.assertLess(float(np.mean(control)), 0.6, control)

    def test_a_run_repeats_exactly_and_writes_the_graph_network(self):
        with TempRepo() as root:
            first = run(graph_job(GAT), root, "a")
            again = run(graph_job(GAT), root, "b")
            for name in ("predictions_val.csv", "predictions_test.csv"):
                self.assertEqual((first / name).read_bytes(), (again / name).read_bytes(), name)
            params = read(first, "model_params.json")
            self.assertEqual(params["input_shape"], [4])
            self.assertEqual(sum(params["parameters_by_part"].values()), params["parameters_total"])
            prep = read(first, "prep_record.json")
            self.assertEqual(len(prep["steps"]), 1)
            training = read(first, "metrics_val.json")["training"]
            self.assertEqual(training["steps_per_epoch"], 1, "one full-graph step per epoch")
            self.assertTrue(training["reload_check"]["equal"])

    def test_a_smoke_run_cuts_the_roles_but_keeps_the_whole_graph(self):
        from sweepkit.runs.spec import config_hash, plain, run_id, validate_job
        from sweepkit.sweep.runner import execute
        from sweepkit.sweep.smoke import smoke_spec

        job = checked(graph_job())
        spec = smoke_spec({"settings": plain(job), "fold": "holdout", "seed": 1, "stage": "base", "label": "base",
                           "sweep_id": "smoke_test", "config_hash": config_hash(job),
                           "run_id": run_id(job, "holdout", 1), "family": "torch", "purpose": "smoke",
                           "overrides": {}, "priority": 1}, 20)
        self.assertEqual(validate_job(spec["settings"]).fit.epochs, 2)
        with TempRepo() as root:
            execute(spec, "cpu", lambda message: None, runs_root=root / "runs")
            folder = root / "runs" / spec["run_id"]
            training = read(folder, "metrics_val.json")["training"]
            self.assertEqual((training["epochs_run"], training["steps_per_epoch"]), (2, 1))
            self.assertLessEqual(read(folder, "efficiency.json")["n_train"], 20)
            self.assertLessEqual(read(folder, "metrics_test.json")["n"], 20)

    def test_the_refit_on_train_and_val_nodes(self):
        with TempRepo() as root:
            folder = run(graph_job(refit="train_and_val", epochs=40, early_stopping_patience=10), root)
            test = read(folder, "metrics_test.json")
            self.assertEqual(test["refit"], "train_and_val")
            self.assertEqual(test["refit_epochs"], read(folder, "metrics_val.json")["training"]["best_epoch"])

    def test_describe_gives_the_network(self):
        from sweepkit.families.torch import describe
        from sweepkit.families.common import load_data

        job = checked(graph_job(GAT))
        with TempRepo():
            data = load_data(job, lambda message: None)
            report = describe(job, data)
        self.assertEqual(report["input_shape"], [4])
        self.assertEqual(report["outputs_shape"], [1, 3])
        self.assertIn("graph of 150 nodes", report["input"])

    def test_the_graph_datasets_example_runs_on_the_example_files(self):
        from sweepkit.data.examples import make_example_data

        with TempRepo() as root:
            make_example_data(lambda message: None)
            settings = example("data_graph")["base"]
            settings["fit"].update({"epochs": 30, "early_stopping_patience": None})
            folder = run(settings, root, fold="fold1")
            self.assertGreaterEqual(read(folder, "metrics_val.json")["select"]["value"], 0.6)


if __name__ == "__main__":
    unittest.main()
