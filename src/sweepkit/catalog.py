"""Every component module, imported once so that its registry entry exists.

Adding a component means adding its module and one import line here, in the tuple of its kind (sweepkit.cli scaffold
writes both). A project's own components live in src/project/ and are listed in src/project/catalog.py, which
load_all imports when it exists, so the template's files stay untouched and later template fixes merge cleanly.

Optional packages (torch, cuML, hdbscan, umap, ...) are imported inside functions, never here, so load_all and every
settings check work with the core profile alone.
"""
import importlib.util

DATA = ("sweepkit.datasets.synthetic", "sweepkit.datasets.table", "sweepkit.datasets.store", "sweepkit.datasets.items")
SPLITS_AND_TASKS = ("sweepkit.splits.schemes", "sweepkit.tasks.supervised", "sweepkit.tasks.unsupervised")
METRICS = ("sweepkit.metrics.classification", "sweepkit.metrics.regression", "sweepkit.metrics.multilabel",
           "sweepkit.metrics.clustering", "sweepkit.metrics.embedding", "sweepkit.metrics.outlier")
PREPARATION = ("sweepkit.prep.clean", "sweepkit.prep.transform", "sweepkit.prep.scale", "sweepkit.prep.select",
               "sweepkit.prep.batch", "sweepkit.prep.reduce")
# per-batch input transforms and training-batch augmentations of the torch family
FEATURES = ("sweepkit.features.none", "sweepkit.features.sequence", "sweepkit.features.audio")
# fit procedures, one per family
FIT_PROCEDURES = ("sweepkit.families.estimator", "sweepkit.families.unsupervised")
ESTIMATORS = ("sweepkit.models.estimators",)
CLUSTERERS = ("sweepkit.models.cluster.centroid", "sweepkit.models.cluster.density", "sweepkit.models.cluster.mixed",
              "sweepkit.models.cluster.hierarchical", "sweepkit.models.cluster.graph",
              "sweepkit.models.cluster.other", "sweepkit.models.cluster.consensus",
              "sweepkit.models.cluster.newer", "sweepkit.models.cluster.singlecell", "sweepkit.models.cluster.deep")
EMBEDDERS = ("sweepkit.models.embed.linear", "sweepkit.models.embed.manifold", "sweepkit.models.embed.neighbors",
             "sweepkit.models.embed.diffusion", "sweepkit.models.embed.deep")
DETECTORS = ("sweepkit.models.outlier.detectors",)
# the torch family: its fit procedure, models, features and augmentations
TORCH = ("sweepkit.families.torch", "sweepkit.train.augment", "sweepkit.models.torch.mlp",
         "sweepkit.models.torch.cnn1d", "sweepkit.models.torch.cnn2d", "sweepkit.models.torch.rnn",
         "sweepkit.models.torch.transformer", "sweepkit.models.torch.sequence_heads",
         "sweepkit.models.torch.cnn_transformer", "sweepkit.models.torch.cnn_rnn", "sweepkit.models.torch.cnn_lif",
         "sweepkit.models.torch.snn", "sweepkit.models.torch.composite", "sweepkit.models.torch.gnn",
         "sweepkit.models.torch.hf_finetune")

MODULES = (DATA + SPLITS_AND_TASKS + METRICS + PREPARATION + FEATURES + FIT_PROCEDURES + ESTIMATORS + CLUSTERERS
           + EMBEDDERS + DETECTORS + TORCH)

_LOADED = False


def load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    for name in MODULES:
        importlib.import_module(name)
    if importlib.util.find_spec("project") is not None and importlib.util.find_spec("project.catalog") is not None:
        importlib.import_module("project.catalog").load_all()
    _LOADED = True
