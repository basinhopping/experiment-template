"""propose: the next configurations to try, chosen by Bayesian search (Optuna's TPE sampler) from the finished
configurations of a sweep, written as a new sweep file. Every round stays a recorded, fixed sweep, so grid, random
and Bayesian search can be compared on the same footing.

The search space is one stage of the finished sweep (a random, grid or one_factor_at_a_time stage): its factors and
their levels. Every configuration of the sweep whose settings match one level of every factor becomes a finished
trial, valued by its choosing metric (the fold mean per seed, then the mean over complete seeds, from the choosing
files only). The sampler then proposes n configurations that the sweep has not tried, and propose writes them with
the sweep's base as a new sweep file:
    artifacts/proposals/<name>_<utc>/sweep.json    stages base and proposed, the seeds given, follows the old sweep
    artifacts/proposals/<name>_<utc>/record.json   the trials the sampler saw, its settings, and what it proposed
Copy sweep.json into sweeps/ to run it. Needs the search profile (Optuna); the API calls follow Optuna 5.0.0
(create_trial, Study.add_trial, Study.ask with fixed_distributions, TPESampler with seed and n_startup_trials).
"""
import numpy as np

from sweepkit.analysis.records import (choosing_value, complete_runs, complete_seeds, configurations, result_folds,
                                       sweep_record, task_of)
from sweepkit.analysis.validation import select_of
from sweepkit.hashing import canonical_json
from sweepkit.sweep.design import apply, get_path, levels_of
from sweepkit.tasks.unsupervised import STABILITY


def _space(sweep_file: dict, stage_name: str) -> dict:
    stage = next((s for s in sweep_file["stages"] if s["name"] == stage_name), None)
    if stage is None:
        raise ValueError(f"no stage {stage_name!r}; stages: {[s['name'] for s in sweep_file['stages']]}")
    raw = stage.get("space", stage.get("factors"))
    if raw is None:
        raise ValueError(f"stage {stage_name} ({stage['kind']}) has no factors to search over")
    return {factor: levels_of(factor, levels) for factor, levels in raw.items()}


def _level_index(settings: dict, levels: list):
    for k, level in enumerate(levels):
        if all(canonical_json(get_path(settings, key)) == canonical_json(value) for key, value in level.items()):
            return k
    return None


def finished_trials(sweep_id: str, space: dict) -> list:
    record = sweep_record(sweep_id)
    task_name, task = task_of(record["sweep_file"])
    select, direction = select_of(task)
    if select == STABILITY:
        raise ValueError("propose needs a choosing metric or criterion; stability is computed by analyze across runs")
    folds = result_folds(record["sweep_file"]["folds"])
    out = []
    for config in configurations(record, complete_runs(sweep_id)).values():
        if config["control"] != "none":
            continue
        params = {f: _level_index(config["settings"], levels) for f, levels in space.items()}
        if any(v is None for v in params.values()):
            continue
        per_seed = []
        for seed in complete_seeds(config, folds):
            values = [choosing_value(config["runs"][seed][f], task_name, select) for f in folds]
            if None in values:
                per_seed = []
                break
            per_seed.append(float(np.mean(values)))
        if per_seed:
            out.append({"config_hash": config["config_hash"], "label": config["label"], "params": params,
                        "value": float(np.mean(per_seed)), "seeds": len(per_seed)})
    return out


def propose(sweep_id: str, stage: str, n_configs: int, seed: int, n_startup: int, name: str, seeds: list,
            priority: int) -> dict:
    import optuna

    from sweepkit.sweep.design import check_header, expand

    if n_configs < 1 or n_startup < 0 or not seeds:
        raise ValueError("--n-configs at least 1, --n-startup at least 0, and the seeds of the new sweep")
    record = sweep_record(sweep_id)
    sweep_file = record["sweep_file"]
    space = _space(sweep_file, stage)
    trials = finished_trials(sweep_id, space)
    if not trials:
        raise ValueError(f"no finished configuration of {sweep_id} sits on a level of every factor of stage {stage}")
    _, direction = select_of(task_of(sweep_file)[1])
    distributions = {f: optuna.distributions.CategoricalDistribution(list(range(len(levels))))
                     for f, levels in space.items()}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize" if direction == "max" else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup))
    for t in trials:
        study.add_trial(optuna.trial.create_trial(params=t["params"], distributions=distributions, value=t["value"]))
    tried = {canonical_json(t["params"]) for t in trials}
    proposals, asked = [], 0
    while len(proposals) < n_configs:
        asked += 1
        if asked > 50 * n_configs:
            raise ValueError(f"the sampler kept proposing configurations already tried; {len(proposals)} new ones in "
                             f"{asked - 1} proposals (the space may be nearly exhausted)")
        trial = study.ask(distributions)
        key = canonical_json(trial.params)
        if key in tried:
            continue
        tried.add(key)
        overrides = {}
        for factor, k in trial.params.items():
            overrides.update(space[factor][k])
        proposals.append({"params": dict(trial.params), "overrides": overrides})
    base = sweep_file["base"]
    new = {"sweep_id": name, "description": f"{n_configs} configurations proposed by Bayesian search (TPE, seed {seed}) "
                                            f"from the finished runs of {sweep_id}, stage {stage}, against its base.",
           "purpose": "screen", "follows": [sweep_id], "folds": list(sweep_file["folds"]), "base": base,
           "stages": [{"name": "base", "kind": "base", "priority": priority, "seeds": list(seeds)},
                      {"name": "proposed", "kind": "variants", "priority": priority, "seeds": list(seeds),
                       "variants": {f"proposed_{k}": p["overrides"] for k, p in enumerate(proposals, 1)}}]}
    check_header(new)
    for p in proposals:
        apply(base, p["overrides"])
    expand(new)
    return {"sweep_file": new, "record": {"from_sweep": sweep_id, "stage": stage, "sampler": "TPESampler",
                                          "seed": seed, "n_startup_trials": n_startup, "direction": direction,
                                          "trials": trials, "proposals": proposals,
                                          "optuna": optuna.__version__}}
