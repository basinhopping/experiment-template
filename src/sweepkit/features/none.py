"""features "none": the model reads the dataset's inputs as they are (every family except torch, and torch models
that read tables, sequences or graphs directly)."""
import dataclasses

from sweepkit.registry import register_feature


@dataclasses.dataclass(frozen=True)
class NoFeatureSettings:
    pass


@register_feature("none", settings=NoFeatureSettings, in_kinds=("tabular", "sequence", "image", "audio", "graph", "text"),
                  out_kind="tabular", description="the inputs as the dataset gives them", profile="core", packages=())
class NoFeatures:
    pass
