"""Registries: a dataset, split, task, preparation step, feature, augmentation, model or fit procedure is a module
plus one registry entry.

A component registers itself with a decorator that names it, declares its settings class, and declares what the
rest of the package needs to know about it:

    @register_model("hdbscan", settings=HDBSCANSettings, family="clusterer", tasks=("clustering", "outlier"),
                    kinds=("tabular",), profile="core", packages=("sklearn",), gpu="cuml",
                    description="HDBSCAN, scikit-learn's implementation", example="cluster_hdbscan")
    class HDBSCAN(Clusterer): ...

The settings class is checked when the module is imported (frozen dataclass, no defaults, supported types), so a bad
declaration fails at import rather than in the middle of a run. The declarations feed smoke-all, check-additions and
docs/PROFILES.md. Components are imported by sweepkit.catalog.load_all(), which lists every module once.
"""
import dataclasses

from sweepkit import settings as settings_module
from sweepkit.settings import Selected, SettingsError, check_settings_class, validate

FAMILIES = ("torch", "estimator", "clusterer", "embedder", "detector")
TASK_NAMES = ("classification", "multilabel", "regression", "clustering", "embedding", "outlier")
KINDS = ("tabular", "sequence", "image", "audio", "graph", "text")


@dataclasses.dataclass(frozen=True)
class Entry:
    kind: str
    name: str
    settings_class: type
    target: object
    meta: dict


class Registry:
    def __init__(self, kind: str, required_meta: tuple, optional_meta: tuple):
        self.kind = kind
        self.required_meta = required_meta
        self.optional_meta = optional_meta
        self._entries: dict = {}

    def register(self, name: str, settings: type, **meta):
        if not name or not name.replace("_", "").isalnum() or name != name.lower() or name[0].isdigit():
            raise ValueError(f"{self.kind} name {name!r}: use lowercase letters, digits and _, starting with a letter")
        check_settings_class(settings)
        missing = [k for k in self.required_meta if k not in meta]
        unknown = sorted(k for k in meta if k not in self.required_meta + self.optional_meta)
        if missing or unknown:
            raise TypeError(f"{self.kind} {name!r}: declaration missing {missing}, unknown {unknown}")
        _check_meta(self.kind, name, meta)

        def decorator(target):
            if name in self._entries:
                raise ValueError(f"{self.kind} {name!r} is registered twice")
            self._entries[name] = Entry(self.kind, name, settings, target, dict(meta))
            return target

        return decorator

    def get(self, name: str) -> Entry:
        if name not in self._entries:
            raise KeyError(f"no {self.kind} named {name!r}; registered: {sorted(self._entries)}")
        return self._entries[name]

    def names(self) -> list:
        return sorted(self._entries)

    def entries(self) -> list:
        return [self._entries[n] for n in sorted(self._entries)]

    def validate_section(self, raw, where: str, families: tuple = ()) -> tuple:
        """Validate a settings section of the form {"name": <registered name>, **fields}.

        Returns (entry, validated settings). The name selects the settings class; every other key must be a field
        of that class. `families` limits a model section to components of those families.
        """
        if type(raw) is not dict:
            raise SettingsError(f"{where}: expected a mapping with a 'name', got {type(raw).__name__}")
        if "name" not in raw:
            raise SettingsError(f"{where}: missing 'name' (one of {self.names()})")
        name = raw["name"]
        if type(name) is not str:
            raise SettingsError(f"{where}.name: expected text, got {name!r}")
        try:
            entry = self.get(name)
        except KeyError as err:
            raise SettingsError(f"{where}.name: {err.args[0]}") from None
        if families and entry.meta.get("family") not in families:
            raise SettingsError(f"{where}.name: {name!r} is a {entry.meta.get('family')} {self.kind}; this place "
                                f"takes {', '.join(families)}")
        fields = {k: v for k, v in raw.items() if k != "name"}
        return entry, validate(entry.settings_class, fields, where)


def from_template(target) -> bool:
    """Whether a registered component (its class or function) was written in the template's own package, rather
    than in a project's src/project/. The generated docs list the template's components only."""
    return getattr(target, "__module__", "").startswith("sweepkit.")


def _check_meta(kind: str, name: str, meta: dict) -> None:
    where = f"{kind} {name!r}"
    if "family" in meta and meta["family"] not in FAMILIES:
        raise TypeError(f"{where}: family {meta['family']!r} is not one of {FAMILIES}")
    for key in ("tasks",):
        if key in meta:
            bad = [t for t in meta[key] if t not in TASK_NAMES]
            if type(meta[key]) is not tuple or bad:
                raise TypeError(f"{where}: {key} must be a tuple of {TASK_NAMES}, got {meta[key]!r}")
    for key in ("kinds", "in_kinds"):
        if key in meta:
            bad = [k for k in meta[key] if k not in KINDS]
            if type(meta[key]) is not tuple or bad:
                raise TypeError(f"{where}: {key} must be a tuple of {KINDS}, got {meta[key]!r}")
    if "out_kind" in meta and meta["out_kind"] not in KINDS:
        raise TypeError(f"{where}: out_kind {meta['out_kind']!r} is not one of {KINDS}")
    if "packages" in meta and (type(meta["packages"]) is not tuple
                               or not all(type(p) is str for p in meta["packages"])):
        raise TypeError(f"{where}: packages must be a tuple of import names")
    for key in ("profile", "description", "gpu", "example"):
        if key in meta and type(meta[key]) is not str:
            raise TypeError(f"{where}: {key} must be text")


# Every component that needs a package outside the standard library and the core profile names it in `packages`
# (import names) and its environment profile in `profile`, so that docs/PROFILES.md, smoke-all and a missing
# package's error message all come from one place.
COMPONENT_META = ("description", "profile", "packages")

DATASETS = Registry("dataset", ("kinds",) + COMPONENT_META, ("example",))
SPLITS = Registry("split", ("roles", "description"), ())
TASKS = Registry("task", ("roles", "description"), ())
PREP = Registry("prep step", ("kinds",) + COMPONENT_META,
                ("removes_rows", "transductive", "example", "gpu"))
FEATURES = Registry("feature", ("in_kinds", "out_kind") + COMPONENT_META, ("example",))
AUGMENTS = Registry("augmentation", ("kinds",) + COMPONENT_META, ())
MODELS = Registry("model", ("family", "tasks", "kinds", "gpu", "example") + COMPONENT_META, ("smoke", "cpu_packages"))
FITS = Registry("fit procedure", ("family", "description"), ())

register_dataset = DATASETS.register
register_split = SPLITS.register
register_task = TASKS.register
register_prep = PREP.register
register_feature = FEATURES.register
register_augment = AUGMENTS.register
register_model = MODELS.register
register_fit = FITS.register

REGISTRIES = {r.kind: r for r in (DATASETS, SPLITS, TASKS, PREP, FEATURES, AUGMENTS, MODELS, FITS)}
# short names used by settings.section_type
BY_SHORT_NAME = {"dataset": DATASETS, "split": SPLITS, "task": TASKS, "prep": PREP, "feature": FEATURES,
                 "augment": AUGMENTS, "model": MODELS, "fit": FITS}


def _resolve_section(registry_kind: str, families: tuple, raw, where: str) -> Selected:
    if registry_kind not in BY_SHORT_NAME:
        raise SettingsError(f"{where}: no registry {registry_kind!r}")
    entry, validated = BY_SHORT_NAME[registry_kind].validate_section(raw, where, families)
    return Selected(entry.name, validated)


settings_module.set_section_resolver(_resolve_section)


class MetricRegistry:
    """Metrics are plain functions, so their registry holds no settings class. Each declares:
    tasks      the tasks whose runs it can score
    direction  "max" or "min": which way is better
    needs      what it reads: any of "y_true", "y_pred", "scores" (probabilities or decision values), "x"
               (the evaluated data), "labels" (cluster labels), "reference" (known labels of clustered rows)
    weighted   True when it accepts sample_weight, so a bootstrap draw can be scored by row weights
    label_free True for unsupervised criteria that read no reference labels
    options    criterion parameters it reads from the task's criteria_options, as (name, type) pairs; a task that
               names the criterion must give each one (no hidden constants)
    descriptive True for a number that describes a result but cannot choose one (cluster count, noise share)
    """

    def __init__(self):
        self._entries = {}

    def register(self, name: str, tasks: tuple, direction: str, needs: tuple, weighted: bool, label_free: bool,
                 description: str, options: tuple = (), descriptive: bool = False):
        if direction not in ("max", "min"):
            raise TypeError(f"metric {name!r}: direction must be max or min")
        bad = [t for t in tasks if t not in TASK_NAMES]
        if bad:
            raise TypeError(f"metric {name!r}: unknown tasks {bad}")

        def decorator(fn):
            if name in self._entries:
                raise ValueError(f"metric {name!r} is registered twice")
            self._entries[name] = {"name": name, "fn": fn, "tasks": tasks, "direction": direction, "needs": needs,
                                   "weighted": weighted, "label_free": label_free, "description": description,
                                   "options": tuple(options), "descriptive": descriptive, "batched": None}
            return fn

        return decorator

    def add_batched(self, name: str, fn) -> None:
        """A vectorized form of a registered weighted metric: fn(fold, weights) -> values [draws, seeds], where fold
        holds y_true [rows], y_pred [seeds, rows], scores [seeds, rows, classes] or None, scores_kind and names, and
        weights [draws, rows]. The paired bootstrap uses it instead of calling the metric once per draw and seed.
        tests/test_metrics.py checks every batched form against its metric under random weights."""
        entry = self.get(name)
        if not entry["weighted"]:
            raise TypeError(f"metric {name!r} takes no row weights, so it has no batched form")
        if entry.get("batched") is not None:
            raise ValueError(f"metric {name!r} has a batched form already")
        entry["batched"] = fn

    def get(self, name: str) -> dict:
        if name not in self._entries:
            raise KeyError(f"no metric named {name!r}; registered: {sorted(self._entries)}")
        return self._entries[name]

    def names(self, task: str | None = None) -> list:
        return sorted(n for n, e in self._entries.items() if task is None or task in e["tasks"])


METRICS = MetricRegistry()
register_metric = METRICS.register
