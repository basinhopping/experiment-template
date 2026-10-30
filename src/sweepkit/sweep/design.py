"""Expand a sweep file into jobs: one job per configuration, fold and seed.

A sweep file is JSON with a complete base configuration and a list of stages:

    sweep_id      letters, digits, _ and -; names the sweep's records and run folders
    description   one or two sentences: what the sweep asks
    purpose       learning, smoke, screen, confirm, final, control, protocol or regression
    follows       the sweep IDs this one builds on (possibly none); a confirm or final sweep may not reuse their seeds
    folds         the folds every configuration runs on (names from the split scheme)
    base          every setting, in the seven sections of sweepkit.runs.spec
    stages        each with a name, a priority (higher runs first), the seeds, and one kind:
      "base"                  the base configuration alone (for example with 5 seeds, to measure seed noise)
      "variants"              named configurations, each a set of overrides of the base (controls, alternatives)
      "one_factor_at_a_time"  each level of each factor changed alone from the base
      "random"                n_configs configurations drawing every factor's level at random (sampling_seed;
                              skip_draws > 0 skips that many accepted draws, so an extension draws exactly what one
                              larger stage would have)
      "grid"                  every combination of the factors' levels; the combination that equals the base is left
                              out, since it is the base itself

An override names a setting by its path, for example {"model.min_cluster_size": 50}. A path goes through mappings by
key and through the prep list by step name (when that step occurs once) or by position: "prep.scale.method" or
"prep.1.method". A whole section can be replaced, for example {"model": {"name": "kmeans", ...}}. A level may change
several settings together. A factor given as {"model.min_samples": [5, 10]} is short for levels
[{"model.min_samples": 5}, {"model.min_samples": 10}]. Every override must name a setting that exists in the base,
so a typo fails instead of adding a setting nothing reads.

One evaluation protocol per sweep: data, split and task must be the same in every configuration, so every
configuration is compared on the same validation (or evaluation) rows. Every configuration is validated and checked
before anything is queued, so a typo or an impossible value stops the enqueue instead of failing jobs overnight.

Stage kinds are pluggable: register_stage_kind adds one (docs/EXTENDING.md).
"""
import copy
import itertools
import json
import random

from sweepkit.hashing import canonical_json
from sweepkit.registry import SPLITS
from sweepkit.runs.spec import PROTOCOL_SECTIONS, canonical, check_job, config_hash, plain, run_id, validate_job

TOP_KEYS = ("sweep_id", "description", "purpose", "follows", "folds", "base", "stages")
PURPOSES = ("learning", "smoke", "screen", "confirm", "final", "control", "protocol", "regression")
STAGE_COMMON = ("name", "kind", "priority", "seeds")


class SweepError(ValueError):
    pass


# Paths into settings --------------------------------------------------------------------------------------------

def _list_index(items: list, part: str, key: str, upto: str) -> int:
    if part.isdigit():
        index = int(part)
        if index >= len(items):
            raise SweepError(f"override {key!r}: {upto} has {len(items)} entries, no position {index}")
        return index
    matches = [i for i, item in enumerate(items) if isinstance(item, dict) and item.get("name") == part]
    if len(matches) != 1:
        raise SweepError(f"override {key!r}: {len(matches)} entries of {upto} are named {part!r}; name one that "
                         "occurs once, or give its position")
    return matches[0]


def _walk(settings, key: str):
    """(container, final key or index) for a path; raises SweepError when the path names no setting."""
    parts = key.split(".")
    if not key or any(not p for p in parts):
        raise SweepError(f"override {key!r}: not a path")
    node = settings
    for depth, part in enumerate(parts):
        upto = ".".join(parts[:depth]) or "the configuration"
        if isinstance(node, dict):
            if part not in node:
                raise SweepError(f"override {key!r}: {'.'.join(parts[:depth + 1])} names no setting of the base "
                                 "configuration")
            target = part
        elif isinstance(node, list):
            target = _list_index(node, part, key, upto)
        else:
            raise SweepError(f"override {key!r}: {upto} is a value, not a section")
        if depth == len(parts) - 1:
            return node, target
        node = node[target]
    raise AssertionError("unreachable")


def get_path(settings: dict, key: str):
    container, target = _walk(settings, key)
    return container[target]


def set_path(settings: dict, key: str, value) -> None:
    container, target = _walk(settings, key)
    container[target] = copy.deepcopy(value)


def apply(base: dict, overrides: dict) -> dict:
    if type(overrides) is not dict:
        raise SweepError(f"overrides must be a mapping of setting paths to values, got {overrides!r}")
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        set_path(out, key, value)
    return out


def levels_of(factor: str, levels) -> list:
    """A factor's levels as override dicts, expanding the {"path": [values]} shorthand."""
    if type(levels) is not list or not levels:
        raise SweepError(f"factor {factor!r}: levels must be a non-empty list")
    if all(type(level) is dict for level in levels) and "." not in factor:
        return levels
    if "." in factor and all(type(level) is dict for level in levels) and all(
            any("." in k for k in level) for level in levels):
        return levels
    if "." in factor:
        return [{factor: level} for level in levels]
    raise SweepError(f"factor {factor!r}: give every level as overrides, or name one setting by its path and list "
                     "values")


def label_of(overrides: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v, sort_keys=True)}" for k, v in overrides.items()) or "base"


def changed_only(base: dict, overrides: dict) -> dict:
    return {k: v for k, v in overrides.items() if canonical_json(get_path(base, k)) != canonical_json(v)}


# Stage kinds ----------------------------------------------------------------------------------------------------

STAGE_KINDS = {}


def register_stage_kind(kind: str, keys: tuple):
    """A stage kind: keys it takes beyond name, kind, priority and seeds, and a function
    (stage, base settings, context) -> [(label, overrides), ...]. The context offers context.config(overrides) ->
    (job, config hash), which validates and checks a configuration and raises on an invalid one."""
    def decorator(fn):
        if kind in STAGE_KINDS:
            raise ValueError(f"stage kind {kind!r} is registered twice")
        STAGE_KINDS[kind] = (tuple(keys), fn)
        return fn
    return decorator


@register_stage_kind("base", ())
def _base_stage(stage, base, context):
    return [("base", {})]


@register_stage_kind("variants", ("variants",))
def _variants_stage(stage, base, context):
    if type(stage["variants"]) is not dict or not stage["variants"]:
        raise SweepError(f"stage {stage['name']}: variants must be a non-empty mapping of name to overrides")
    return list(stage["variants"].items())


@register_stage_kind("one_factor_at_a_time", ("factors",))
def _ofat_stage(stage, base, context):
    out = []
    for factor, levels in stage["factors"].items():
        for level in levels_of(factor, levels):
            label = label_of(level) if list(level) == [factor] else f"{factor}: {label_of(level)}"
            out.append((label, level))
    return out


@register_stage_kind("grid", ("factors",))
def _grid_stage(stage, base, context):
    lists = [levels_of(factor, levels) for factor, levels in stage["factors"].items()]
    if not lists:
        raise SweepError(f"stage {stage['name']}: a grid needs at least one factor")
    out = []
    for combination in itertools.product(*lists):
        overrides = {}
        for level in combination:
            overlap = set(overrides) & set(level)
            if overlap:
                raise SweepError(f"stage {stage['name']}: two factors set {sorted(overlap)}")
            overrides.update(level)
        changed = changed_only(base, overrides)
        if changed:
            out.append((f"grid: {label_of(changed)}", overrides))
    return out


@register_stage_kind("random", ("space", "n_configs", "sampling_seed", "skip_draws"))
def _random_stage(stage, base, context):
    name = stage["name"]
    for key in ("n_configs", "sampling_seed", "skip_draws"):
        if type(stage[key]) is not int or stage[key] < 0:
            raise SweepError(f"stage {name}: {key} must be a whole number of at least 0")
    if stage["n_configs"] < 1:
        raise SweepError(f"stage {name}: n_configs must be at least 1")
    space = {f: levels_of(f, levels) for f, levels in stage["space"].items()}
    if not space:
        raise SweepError(f"stage {name}: the space needs at least one factor")
    for factor, levels in space.items():
        for level in levels:
            try:
                apply(base, level)
            except SweepError as err:
                raise SweepError(f"stage {name}, factor {factor}, level {level}: {err}") from None
    rng = random.Random(stage["sampling_seed"])
    target = stage["skip_draws"] + stage["n_configs"]
    attempts, accepted, invalid = 0, [], 0
    accepted_hashes = {context.base_hash}
    while len(accepted) < target:
        attempts += 1
        if attempts > 50 * target:
            raise SweepError(f"stage {name}: only {len(accepted)} distinct valid configurations in {attempts - 1} "
                             f"draws; the space is too small for skip_draws + n_configs = {target}")
        overrides = {}
        for levels in space.values():
            overrides.update(rng.choice(levels))
        try:
            _, h = context.config(apply(base, overrides))
        except (ValueError, TypeError):
            invalid += 1
            continue
        if h in accepted_hashes:
            continue
        accepted_hashes.add(h)
        accepted.append(overrides)
    context.note(name, "invalid_random_draws_skipped", invalid)
    out = []
    for k, overrides in enumerate(accepted[stage["skip_draws"]:], stage["skip_draws"] + 1):
        out.append((f"random {k}: {label_of(changed_only(base, overrides))}", overrides))
    return out


class _Context:
    def __init__(self, base_hash: str):
        self.base_hash = base_hash
        self.notes = {}
        self._cache = {}

    def config(self, settings: dict):
        key = canonical_json(settings)
        if key not in self._cache:
            job = validate_job(settings)
            check_job(job)
            self._cache[key] = (job, config_hash(job))
        return self._cache[key]

    def note(self, stage: str, key: str, value) -> None:
        self.notes.setdefault(stage, {})[key] = value


# Expansion ------------------------------------------------------------------------------------------------------

def check_header(sweep: dict) -> None:
    if type(sweep) is not dict:
        raise SweepError("a sweep file holds one JSON object")
    missing = [k for k in TOP_KEYS if k not in sweep]
    unknown = sorted(k for k in sweep if k not in TOP_KEYS)
    if missing or unknown:
        raise SweepError(f"sweep file: missing {missing}, unknown {unknown}")
    sweep_id = sweep["sweep_id"]
    if type(sweep_id) is not str or not sweep_id or not all(c.isalnum() or c in "_-" for c in sweep_id) \
            or "__" in sweep_id:
        raise SweepError(f"sweep_id {sweep_id!r}: letters, digits, _ and - only, without a double underscore")
    if type(sweep["description"]) is not str or not sweep["description"].strip():
        raise SweepError("description: say in a sentence what the sweep asks")
    if sweep["purpose"] not in PURPOSES:
        raise SweepError(f"purpose {sweep['purpose']!r} is not one of {PURPOSES}")
    follows = sweep["follows"]
    if type(follows) is not list or any(type(f) is not str or not f for f in follows) or len(set(follows)) != len(follows):
        raise SweepError("follows: a list of distinct sweep IDs (empty when the sweep builds on none)")
    if sweep_id in follows:
        raise SweepError("follows: a sweep cannot follow itself")


def expand(sweep: dict) -> tuple:
    """(jobs, summary). Raises SweepError or a settings error for any problem."""
    check_header(sweep)
    sweep_id = sweep["sweep_id"]
    try:
        base_job = validate_job(sweep["base"])
        check_job(base_job)
    except (ValueError, TypeError) as err:
        raise SweepError(f"base: {err}") from None
    base_hash = config_hash(base_job)
    base_settings = plain(base_job)
    context = _Context(base_hash)
    folds = sweep["folds"]
    if type(folds) is not list or not folds or len(set(folds)) != len(folds):
        raise SweepError("folds must be a non-empty list without repeats")
    known_folds = SPLITS.get(base_job.split_name).target.fold_names(base_job.split)
    if any(type(f) is not str or "__" in f for f in known_folds):
        raise SweepError(f"split {base_job.split_name!r} names a fold with a double underscore: {list(known_folds)}")
    strangers = [f for f in folds if f not in known_folds]
    if strangers:
        raise SweepError(f"folds {strangers} are not folds of the base configuration's split "
                         f"{base_settings['split']}; its folds are {list(known_folds)}")
    if type(sweep["stages"]) is not list or not sweep["stages"]:
        raise SweepError("stages: a non-empty list")
    names = [stage.get("name") if type(stage) is dict else None for stage in sweep["stages"]]
    if None in names or len(set(names)) != len(names):
        raise SweepError(f"every stage needs a name, and names may not repeat: {names}")
    configs = []           # (stage, label, overrides, job, hash, priority, seeds)
    where_seen = {}
    summary = {"sweep_id": sweep_id, "purpose": sweep["purpose"], "follows": sweep["follows"], "stages": []}
    n_base_stages = 0
    for stage in sweep["stages"]:
        for key in STAGE_COMMON:
            if key not in stage:
                raise SweepError(f"stage {stage['name']}: missing {key!r}")
        kind, name, priority, seeds = stage["kind"], stage["name"], stage["priority"], stage["seeds"]
        if kind not in STAGE_KINDS:
            raise SweepError(f"stage {name}: kind {kind!r} is not one of {sorted(STAGE_KINDS)}")
        if type(priority) is not int or not 0 <= priority <= 999:
            raise SweepError(f"stage {name}: priority must be a whole number from 0 to 999")
        if type(seeds) is not list or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
                or len(set(seeds)) != len(seeds):
            raise SweepError(f"stage {name}: seeds must be a list of distinct whole numbers of at least 0")
        keys, fn = STAGE_KINDS[kind]
        allowed = set(STAGE_COMMON) | set(keys)
        extra, missing_keys = sorted(set(stage) - allowed), sorted(allowed - set(stage))
        if extra or missing_keys:
            raise SweepError(f"stage {name}: unknown keys {extra}, missing keys {missing_keys}")
        if kind == "base":
            n_base_stages += 1
        stage_configs = fn(stage, base_settings, context)
        dropped = 0
        for label, overrides in stage_configs:
            try:
                job, h = context.config(apply(base_settings, overrides))
            except SweepError:
                raise
            except (ValueError, TypeError) as err:
                raise SweepError(f"stage {name}, {label}: {err}") from None
            if kind != "base" and h == base_hash:
                raise SweepError(f"stage {name}, {label}: identical to the base configuration")
            if h in where_seen:
                if kind == "random":
                    dropped += 1          # a random draw that another stage already holds: dropped, counted
                    continue
                if kind != "base":
                    raise SweepError(f"stage {name}, {label}: identical to {where_seen[h]}")
            where_seen.setdefault(h, f"{name}/{label}")
            configs.append((name, label, overrides, job, h, priority, seeds))
        entry = {"name": name, "kind": kind, "priority": priority, "seeds": seeds,
                 "configs": sum(1 for c in configs if c[0] == name), **context.notes.get(name, {})}
        if kind == "random":
            entry["duplicates_of_other_stages_dropped"] = dropped
        summary["stages"].append(entry)
    if n_base_stages > 1:
        raise SweepError("a sweep holds at most one stage of kind base (the baseline every configuration is "
                         "compared with)")
    base_protocol = {k: canonical(base_job)[k] for k in PROTOCOL_SECTIONS}
    for stage, label, overrides, job, h, priority, seeds in configs:
        protocol = {k: canonical(job)[k] for k in PROTOCOL_SECTIONS}
        if protocol != base_protocol:
            changed = [k for k in PROTOCOL_SECTIONS if protocol[k] != base_protocol[k]]
            raise SweepError(f"stage {stage}, {label}: changes {changed}; a sweep holds one evaluation protocol "
                             "(data, split and task), so put another protocol in its own sweep file")
    jobs, run_ids = [], set()
    for stage, label, overrides, job, h, priority, seeds in configs:
        settings = plain(job)
        for fold in folds:
            for seed in seeds:
                rid = run_id(job, fold, seed)
                if rid in run_ids:
                    raise SweepError(f"run {rid} appears twice (stage {stage}, {label})")
                run_ids.add(rid)
                jobs.append({"sweep_id": sweep_id, "purpose": sweep["purpose"], "stage": stage, "label": label,
                             "overrides": overrides, "priority": priority, "config_hash": h, "run_id": rid,
                             "fold": fold, "seed": seed, "family": job.family, "settings": settings})
    summary["configs"] = len(configs)
    summary["jobs"] = len(jobs)
    summary["base_config_hash"] = base_hash
    summary["seeds"] = sorted({s for c in configs for s in c[6]})
    summary["family"] = base_job.family
    return jobs, summary


def warnings_for(sweep: dict, summary: dict) -> list:
    """Things enqueue prints before queueing: a one-seed stage in a sweep meant to rank or report (screen with at
    least three seeds; report with five), with the exception for unsupervised methods that give the same result for
    every seed (compared over subsamples instead)."""
    out = []
    unsupervised = summary.get("family") in ("clusterer", "embedder", "detector")
    if sweep["purpose"] not in ("learning", "smoke", "regression"):
        for stage in summary["stages"]:
            if len(stage["seeds"]) == 1 and unsupervised:
                out.append(f"stage {stage['name']} runs one seed in a {sweep['purpose']} sweep: enough only for a "
                           "method that gives the same result for every seed (HDBSCAN, DBSCAN, agglomerative), "
                           "compared over two or more subsamples; a method with randomness needs at least 3 seeds")
            elif len(stage["seeds"]) == 1:
                out.append(f"stage {stage['name']} runs one seed in a {sweep['purpose']} sweep: one seed's ranking "
                           "does not predict another's (screen with at least 3 seeds, report with 5)")
    return out
