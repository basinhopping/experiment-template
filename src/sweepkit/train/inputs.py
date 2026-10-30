"""What a torch job's model reads, worked out from the settings alone (no data, no torch), for the checks that run when
a sweep is expanded.

    dataset_kind(job)          the dataset's kind (tabular, sequence, image, audio, text, graph)
    declared_input_shape(job)  one row's input shape as the model sees it, after the job's feature, when the settings
                               alone tell it: the synthetic sequence and image generators (sweepkit.data.synthetic
                               .item_shape) and the log-mel feature (its n_mels). Otherwise None: the shape is then
                               checked when the model is built, from the data.
"""
from sweepkit.registry import DATASETS


def dataset_kind(job) -> str:
    kinds = DATASETS.get(job.data_name).meta["kinds"]
    if len(kinds) == 1:
        return kinds[0]
    kind = getattr(job.data, "kind", None)
    if kind not in kinds:
        raise ValueError(f"dataset {job.data_name!r}: settings kind {kind!r} is not one of {kinds}")
    return kind


def _dataset_item_shape(job):
    if job.data_name == "synthetic":
        from sweepkit.data.synthetic import item_shape

        return item_shape(job.data.generator, job.data.n_features)
    return None


def declared_input_shape(job):
    """A tuple, or None when the settings alone do not tell it."""
    if job.features_name == "none":
        return _dataset_item_shape(job)
    from sweepkit.registry import FEATURES

    target = FEATURES.get(job.features_name).target
    declared = getattr(target, "declared_shape", None)
    if declared is None:
        return None
    return declared(job.features, _dataset_item_shape(job))
