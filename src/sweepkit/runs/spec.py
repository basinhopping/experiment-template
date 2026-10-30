"""A job's settings: seven sections, each validated against its declaration.

    data      {"name": <registered dataset>, ...its settings}       which rows and inputs
    split     {"name": <registered split scheme>, ...}              which rows play which role in each fold
    task      {"name": <registered task>, ...}                      what is predicted or found, and how it is judged
    prep      [{"name": <registered prep step>, ...}, ...]          preparation steps, in order, fitted on the
                                                                    training (or fit) rows only
    features  {"name": <registered feature>, ...}                   per-batch input transforms of the torch family
                                                                    ("none" for every other family)
    model     {"name": <registered model>, ...}                     the algorithm and its hyperparameters
    fit       {"name": <family>, ...}                               how the model is fitted (the family's settings)

A missing or unknown section raises, and so does anything the sections' own validation refuses. `check_job` then
applies the checks that need several sections at once (the model's family matches the fit section, the model reads
what the dataset and features give, the task is one the model does, the split gives the roles the task needs), and
each component's own cross-checks, so a bad value fails when the sweep is expanded, not at 3 a.m. The validated
settings are hashed; the hash, the fold and the seed name the run.

`plain` gives every setting, including fields added after runs existed (settings.ADDED_FIELDS); `canonical`, which
is what gets hashed, leaves an added field out while it holds its value from before it existed, so a configuration
that does what an older one did keeps the older hash.
"""
import dataclasses

from sweepkit.hashing import sha256_json
from sweepkit.registry import DATASETS, FEATURES, FITS, MODELS, PREP, SPLITS, TASKS
from sweepkit.settings import SettingsError, canonical_plain, to_plain

SECTIONS = ("data", "split", "task", "prep", "features", "model", "fit")
# The sections that define the evaluation protocol. They must be the same in every configuration of a sweep, so
# every configuration is compared on the same validation (or evaluation) rows.
PROTOCOL_SECTIONS = ("data", "split", "task")


@dataclasses.dataclass(frozen=True)
class Job:
    data_name: str
    data: object
    split_name: str
    split: object
    task_name: str
    task: object
    prep: tuple            # ((name, settings), ...)
    features_name: str
    features: object
    model_name: str
    model: object
    fit_name: str
    fit: object

    @property
    def family(self) -> str:
        return MODELS.get(self.model_name).meta["family"]


def validate_job(raw) -> Job:
    if type(raw) is not dict:
        raise SettingsError(f"job settings must be a mapping, got {type(raw).__name__}")
    missing = [s for s in SECTIONS if s not in raw]
    unknown = sorted(k for k in raw if k not in SECTIONS)
    if missing or unknown:
        raise SettingsError(f"job settings: missing sections {missing}, unknown sections {unknown}")
    data_entry, data = DATASETS.validate_section(raw["data"], "data")
    split_entry, split = SPLITS.validate_section(raw["split"], "split")
    task_entry, task = TASKS.validate_section(raw["task"], "task")
    if type(raw["prep"]) is not list:
        raise SettingsError(f"prep: expected a list of steps (possibly empty), got {type(raw['prep']).__name__}")
    prep = []
    for i, step in enumerate(raw["prep"]):
        entry, validated = PREP.validate_section(step, f"prep[{i}]")
        prep.append((entry.name, validated))
    feature_entry, features = FEATURES.validate_section(raw["features"], "features")
    model_entry, model = MODELS.validate_section(raw["model"], "model")
    fit_entry, fit = FITS.validate_section(raw["fit"], "fit")
    return Job(data_entry.name, data, split_entry.name, split, task_entry.name, task, tuple(prep),
               feature_entry.name, features, model_entry.name, model, fit_entry.name, fit)


def _section(name: str, settings, convert) -> dict:
    return {"name": name, **convert(settings)}


def plain(job: Job) -> dict:
    return {
        "data": _section(job.data_name, job.data, to_plain),
        "split": _section(job.split_name, job.split, to_plain),
        "task": _section(job.task_name, job.task, to_plain),
        "prep": [_section(n, s, to_plain) for n, s in job.prep],
        "features": _section(job.features_name, job.features, to_plain),
        "model": _section(job.model_name, job.model, to_plain),
        "fit": _section(job.fit_name, job.fit, to_plain),
    }


def canonical(job: Job) -> dict:
    """plain(job) without the added fields that hold their earlier values; this is hashed."""
    return {
        "data": _section(job.data_name, job.data, canonical_plain),
        "split": _section(job.split_name, job.split, canonical_plain),
        "task": _section(job.task_name, job.task, canonical_plain),
        "prep": [_section(n, s, canonical_plain) for n, s in job.prep],
        "features": _section(job.features_name, job.features, canonical_plain),
        "model": _section(job.model_name, job.model, canonical_plain),
        "fit": _section(job.fit_name, job.fit, canonical_plain),
    }


def config_hash(job: Job) -> str:
    return sha256_json(canonical(job))


def protocol_hash(job: Job) -> str:
    """The hash of the evaluation protocol alone (data, split, task)."""
    c = canonical(job)
    return sha256_json({k: c[k] for k in PROTOCOL_SECTIONS})


def run_id(job: Job, fold: str, seed: int) -> str:
    return f"{config_hash(job)[:12]}_{fold}_seed{seed}"


def control_of(settings: dict) -> str:
    """The leak-test control recorded in a run's settings ("none" for families without one)."""
    return settings["fit"].get("control", "none")


def input_kind(job: Job) -> str:
    """The kind of input the model reads: the dataset's kind, or what the features turn it into."""
    if job.features_name == "none":
        return _dataset_kind(job)
    feature = FEATURES.get(job.features_name).meta
    kind = _dataset_kind(job)
    if kind not in feature["in_kinds"]:
        raise ValueError(f"features {job.features_name!r} read {feature['in_kinds']}, but dataset {job.data_name!r} "
                         f"gives {kind}")
    return feature["out_kind"]


def _dataset_kind(job: Job) -> str:
    kinds = DATASETS.get(job.data_name).meta["kinds"]
    if len(kinds) == 1:
        return kinds[0]
    kind = getattr(job.data, "kind", None)
    if kind not in kinds:
        raise ValueError(f"dataset {job.data_name!r}: settings kind {kind!r} is not one of {kinds}")
    return kind


def check_job(job: Job) -> None:
    """Checks across sections. Needs no torch and reads no data, so it runs when a sweep is expanded."""
    model_meta = MODELS.get(job.model_name).meta
    family = model_meta["family"]
    if FITS.get(job.fit_name).meta["family"] != family:
        raise ValueError(f"fit {job.fit_name!r} fits the {FITS.get(job.fit_name).meta['family']} family, but model "
                         f"{job.model_name!r} is a {family}; use fit {{\"name\": \"{family}\", ...}}")
    if job.task_name not in model_meta["tasks"]:
        raise ValueError(f"model {job.model_name!r} does {list(model_meta['tasks'])}, not task {job.task_name!r}")
    if family != "torch" and job.features_name != "none":
        raise ValueError(f"features {job.features_name!r} are per-batch transforms of the torch family; the "
                         f"{family} family reads features {{\"name\": \"none\"}}")
    kind = input_kind(job)
    if kind not in model_meta["kinds"]:
        raise ValueError(f"model {job.model_name!r} reads {list(model_meta['kinds'])}, but gets {kind} input")
    task_roles = TASKS.get(job.task_name).meta["roles"]
    split_roles = SPLITS.get(job.split_name).meta["roles"]
    if task_roles != split_roles:
        raise ValueError(f"task {job.task_name!r} needs folds with roles {list(task_roles)}, but split "
                         f"{job.split_name!r} gives {list(split_roles)}")
    # preparation steps act on the dataset's columns: a table's, or a graph's node features (as a table)
    prep_kind = "tabular" if _dataset_kind(job) == "graph" else kind
    for i, (name, settings) in enumerate(job.prep):
        meta = PREP.get(name).meta
        if prep_kind not in meta["kinds"]:
            raise ValueError(f"prep[{i}] {name!r} works on {list(meta['kinds'])}, but the data are {kind}")
    names = [n for n, _ in job.prep]
    # every component may check itself against the whole job
    for registry, name in ((DATASETS, job.data_name), (SPLITS, job.split_name), (TASKS, job.task_name),
                           (FEATURES, job.features_name), (MODELS, job.model_name), (FITS, job.fit_name)):
        hook = getattr(registry.get(name).target, "check_job", None)
        if hook is not None:
            hook(job)
    for name in dict.fromkeys(names):
        hook = getattr(PREP.get(name).target, "check_job", None)
        if hook is not None:
            hook(job)
