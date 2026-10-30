"""features "sequence" and "image": a torch job's sequences or images as the dataset gives them, with the input's
normalization (sweepkit.features.normalize: none, item, global, or group with its column). A torch job with features
"none" reads its inputs without any normalization.

A feature is called on a batch on the job's device: lengths(lengths) gives the steps after the feature (unchanged
here), __call__(x) the transformed batch (unchanged here), out_shape(input_shape) one row's shape after it. Features
that change the input (sweepkit.features.audio's log-mel) implement the same methods.
"""
import dataclasses
from typing import Literal

from sweepkit.features.normalize import check_mode
from sweepkit.registry import register_feature


@dataclasses.dataclass(frozen=True)
class SequenceFeatureSettings:
    normalization: Literal["none", "item", "global", "group"]
    group_column: str | None       # the dataset column whose groups get their own statistics (group); None otherwise


@dataclasses.dataclass(frozen=True)
class ImageFeatureSettings:
    normalization: Literal["none", "item", "global", "group"]
    group_column: str | None


class _PassThrough:
    def __init__(self, settings, device):
        self.settings = settings
        self.normalization = settings.normalization
        self.group_column = settings.group_column

    @staticmethod
    def declared_shape(settings, input_shape):
        return input_shape

    @staticmethod
    def is_transductive(section: dict) -> bool:
        """Group normalization takes its statistics from every row of the job (sweepkit.analysis.transductive)."""
        return section.get("normalization") == "group"

    def out_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def lengths(self, lengths):
        return lengths

    def __call__(self, x):
        return x


@register_feature("sequence", settings=SequenceFeatureSettings, in_kinds=("sequence",), out_kind="sequence",
                  description="sequences as given, with per-channel normalization", profile="torch",
                  packages=("torch",), example="sequence_models")
class SequenceFeature(_PassThrough):
    @staticmethod
    def check_job(job) -> None:
        check_mode(job.features.normalization, job.features.group_column, "features sequence")


@register_feature("image", settings=ImageFeatureSettings, in_kinds=("image",), out_kind="image",
                  description="images as given, with per-channel normalization", profile="torch", packages=("torch",),
                  example="image_cnn")
class ImageFeature(_PassThrough):
    @staticmethod
    def check_job(job) -> None:
        check_mode(job.features.normalization, job.features.group_column, "features image")


class Identity(_PassThrough):
    """What a torch job with features "none" uses: no transform and no normalization."""

    def __init__(self):
        self.settings = None
        self.normalization = "none"
        self.group_column = None
