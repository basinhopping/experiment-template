"""The interface every torch model implements, and the decorator that registers one.

A torch model is registered with `torch_model`, which wraps a build function

    build(settings, input_shape, n_outputs) -> torch.nn.Module

input_shape is one row's input as the model sees it, after the job's feature (sweepkit.features):
    tabular   (columns,)               a batch is x [batch, columns]
    sequence  (channels,)              a batch is x [batch, channels, steps] with lengths [batch], the real steps of
                                       each row (padding comes after them and holds zeros)
    image     (channels, height, width)  a batch is x [batch, channels, height, width]; lengths is None
n_outputs is the number of classes (classification), labels (multilabel) or 1 (regression).

The module must have:
    embed(x, lengths)      the pooled summary [batch, embedding_dim] (lengths is None for tabular and image input)
    classify(embedding)    the outputs [batch, n_outputs]: logits, or the standardized value for regression
    forward(x, lengths)    classify(embed(x, lengths))
    embedding_dim          the width of embed's output (extra heads read it)
    head_dropout           the dropout of the classifier (the auxiliary head uses the same)
and may have (the spiking models do):
    activity_penalty()     a loss term added in training, or None
    record(on)             start or stop counting spikes and operations
    spike_report()         the counts as a dict, written to efficiency.json
    is_spiking             False when a model has record and spike_report but counts nothing in this configuration
                           (the composite model without its LIF block); a model with spike_report and without
                           is_spiking counts spikes

The training loop (sweepkit.train.loop) calls only these, so a new model, a model that wraps a library (Hugging Face
encoders, graph networks), or an autoencoder's encoder used as a classifier needs nothing else from the family.

torch is imported inside functions only: registering a model, validating its settings and expanding a sweep work
without torch.

Adding to the family (where each piece plugs in):
    a model           a module in this folder: a frozen settings class, check(settings, input_shape or None), and a
                      build function decorated with torch_model(...); its module is listed in sweepkit.catalog.TORCH
                      and an example sweep in sweeps/examples/ uses it. The registered target exposes build, check and
                      check_job. A model for other tasks passes tasks= (an autoencoder: ("embedding",)).
    another task      sweepkit.families.torch.TASK_RUNNERS[task] = run(job, spec, offered, log) -> the run folder's
                      files, written as that task's family writes them (sweepkit.families.unsupervised for embeddings
                      and clusters); the fit's checks, smoke_settings and describe apply unchanged.
    another input     sweepkit.train.batches (Source, training_batch, eval_batches) and sweepkit.families.torch._sources
                      gain a branch for the kind (graph, text). sweepkit.train.loop.to_device hands anything that is
                      not a numpy array to the feature and the model as it is (a list of strings, for a model that
                      tokenizes inside embed).
    a feature         register_feature(name, in_kinds=..., out_kind=...); the target is built with (settings, device)
                      and has normalization and group_column, out_shape(input_shape), lengths(lengths) and __call__(x),
                      and may have a static declared_shape(settings, item_shape) and check_job(job)
                      (sweepkit.features.sequence and audio are the examples).
    an augmentation   register_augment(name, kinds=...); the target has an order (0: on the dataset's rows before the
                      feature; above 0: on the model's input after it) and a static check(settings), and
                      sweepkit.train.loop.Step applies it from its own named random stream.
"""
import dataclasses

from sweepkit.registry import register_model

TASKS = ("classification", "multilabel", "regression")


def torch_model(name: str, settings: type, kinds: tuple, description: str, example: str, check, tasks: tuple = TASKS):
    """Register a torch model. check(settings, input_shape or None) raises ValueError for settings that cannot
    work; it runs when a sweep is expanded (input_shape when the settings alone tell it, else None) and again when
    the model is built (with the real input shape). tasks: the supervised ones unless the model says otherwise."""
    def decorator(build):
        class TorchModel:
            @staticmethod
            def build(s, input_shape, n_outputs: int):
                shape = tuple(int(v) for v in input_shape)
                check(s, shape)
                return build(s, shape, int(n_outputs))

            @staticmethod
            def check_job(job) -> None:
                from sweepkit.train.inputs import declared_input_shape

                check(job.model, declared_input_shape(job))

        # check(settings, input_shape or None): the model's own checks, which build and check_job also run
        TorchModel.check = staticmethod(check)
        TorchModel.__name__ = f"TorchModel_{name}"
        TorchModel.__qualname__ = TorchModel.__name__
        TorchModel.__module__ = build.__module__    # where the model was written (template or project)
        register_model(name, settings=settings, family="torch", tasks=tasks, kinds=kinds, gpu="torch",
                       profile="torch", packages=("torch",), description=description, example=example)(TorchModel)
        return build
    return decorator


def deterministic_cudnn(device: str = "cuda") -> dict:
    """cuDNN set to choose deterministic algorithms, the same on every run (PyTorch's notes on reproducibility):
    cudnn.deterministic on, cudnn.benchmark off. Other CUDA kernels (index_add_, some backward passes) can still differ
    in the last bits between two runs on a GPU, so a GPU run repeats to rounding, not to the bit.

    On the CPU a run repeats exactly only with PyTorch's deterministic algorithms: the backward of indexing a tensor
    by an index tensor (index_put_ with accumulate, which the graph networks' z[source] uses) adds in parallel in an
    order that varies, and the first node run (2026-10-27) saw a GAT's predictions differ in the sixth decimal between
    two identical runs. So CPU runs switch them on, and GPU runs off (warn_only: an operation without a deterministic
    version warns instead of stopping the run). Returns what was set, for provenance.json."""
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    on_cpu = device == "cpu"
    torch.use_deterministic_algorithms(on_cpu, warn_only=True)
    return {"cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms": on_cpu}


def parameter_counts(module) -> dict:
    """{"total": every parameter, "trainable": those with requires_grad, "by_part": {top-level child: count}}."""
    by_part = {name: int(sum(p.numel() for p in child.parameters())) for name, child in module.named_children()}
    own = int(sum(p.numel() for p in module.parameters(recurse=False)))
    if own:
        by_part["(own)"] = own
    return {"total": int(sum(p.numel() for p in module.parameters())),
            "trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)), "by_part": by_part}


def settings_dict(settings) -> dict:
    return dataclasses.asdict(settings)
