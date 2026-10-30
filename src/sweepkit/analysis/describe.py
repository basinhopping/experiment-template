"""describe-model: a configuration exactly as a job builds it, before or after it runs.

The configuration comes from a sweep file (its base, or a configuration by the first 8 or more characters of its
hash) or from a recorded selection. The description holds:
  - every setting of the seven sections, as validated (fields added later shown with their values);
  - the preparation steps in order, with what each will do;
  - for the torch family, the network as training builds it: the module tree, every leaf layer's input and output
    shapes on the dataset's declared input, and parameters per part (the fit procedure's describe hook);
  - for the other families, every parameter of the library object the model builds, its own defaults included (the
    same list a run writes to model_params.json), so a library default can never hide; its random seed is shown as
    the rule that gives each run its own (the run's seed through the "model" stream, sweepkit.streams.int_seed);
  - when runs of the configuration exist, their recorded parameters compared with these.
Writes artifacts/architectures/<name>_<utc>/description.json and description.txt.
"""
import json

from sweepkit import paths
from sweepkit.registry import FITS, MODELS
from sweepkit.runs.spec import config_hash, plain, validate_job
from sweepkit.settings import Selected

SEED_KEYS = ("random_state", "seed", "random_seed")
PER_RUN = 'per run: sweepkit.streams.int_seed(run seed, "model")'


def _job_from(source: str, config: str) -> tuple:
    """(name, settings dict, config hash) of the configuration asked for."""
    if source.startswith("selection:"):
        from sweepkit.analysis.records import find_selection

        if config != "selected":
            raise ValueError("with a selection, give --config selected")
        s = find_selection(source[len("selection:"):])
        return s["name"], s["settings"], s["config_hash"]
    from sweepkit.sweep.design import expand
    from sweepkit.sweep.enqueue import load_sweep_file

    jobs, summary = expand(load_sweep_file(paths.repo_root() / source if not source.startswith("/") else source))
    wanted = summary["base_config_hash"] if config == "base" else config
    if config != "base" and len(config) < 8:
        raise ValueError("--config: base, or 8 or more characters of a configuration hash")
    matches = sorted({j["config_hash"] for j in jobs if j["config_hash"].startswith(wanted)})
    if len(matches) != 1:
        raise ValueError(f"--config {config!r} matches {len(matches)} configurations of {summary['sweep_id']}")
    job = next(j for j in jobs if j["config_hash"] == matches[0])
    return f"{summary['sweep_id']}_{matches[0][:12]}", job["settings"], matches[0]


def _is_seed(key: str) -> bool:
    return key.split("__")[-1] in SEED_KEYS


def per_run_seeds(params: dict, probe: int) -> dict:
    """The parameters, with a seed parameter that holds `probe` (the seed the description built with) shown as each
    run's own."""
    return {k: (PER_RUN if _is_seed(k) and v == probe else v) for k, v in params.items()}


def library_object_params(job) -> dict:
    """Every parameter of the library object the model builds, on the CPU, unfitted. A run builds it with its own
    seed (sweepkit.streams.int_seed(the run's seed, "model")); this one is built with a probe seed, and a seed
    parameter that holds it is shown as each run's own."""
    from sweepkit.models.base import library_params

    probe = 1234567                  # not a setting: the seed this description builds with, recognized afterwards
    entry = MODELS.get(job.model_name)
    family = entry.meta["family"]
    if family == "estimator":
        return per_run_seeds(library_params(entry.target.make(job.model, job.task_name, probe, "cpu")), probe)
    model = entry.target(job.model, probe, "cpu")
    make = getattr(model, "make", None)
    if make is not None:
        try:
            return per_run_seeds(library_params(make(job.model, probe, "cpu")), probe)
        except ImportError as err:
            return {"not_built": f"its package is not installed here: {err}"}
    return {"settings": json.loads(json.dumps(plain(job)["model"]))}


def describe(source: str, config: str, log) -> dict:
    name, settings, h = _job_from(source, config)
    job = validate_job(settings)
    if config_hash(job) != h:
        raise ValueError("the settings do not hash to the configuration hash; the code changed since they were written")
    out = {"name": name, "config_hash": h, "family": job.family, "settings": plain(job),
           "prep": [{"step": n, **plain(job)["prep"][i]} for i, (n, _) in enumerate(job.prep)]}
    hook = getattr(FITS.get(job.fit_name).target, "describe", None)
    if hook is not None:
        from sweepkit.families.common import load_data

        out["network"] = hook(job, load_data(job, log))
    else:
        out["library_parameters"] = library_object_params(job)
    out["runs"] = _compare_with_runs(h, out)
    return out


def _compare_with_runs(h: str, out: dict) -> list:
    """Parameter counts or library parameters recorded by runs of this configuration, against the description."""
    from sweepkit.analysis.records import is_check_run, read_json

    found = []
    root = paths.all_runs_dir()
    if not root.is_dir():
        return found
    for folder in sorted(root.glob(f"*/{h[:12]}_*")):
        if not folder.is_dir() or ".failed-" in folder.name or folder.name.endswith(".tmp") or is_check_run(folder):
            continue
        recorded = read_json(folder, "model_params.json", "choose")
        entry = {"run": f"{folder.parent.name}/{folder.name}"}
        if "network" in out and "parameters_total" in recorded:
            entry["parameters_total"] = recorded["parameters_total"]
            entry["same"] = recorded["parameters_total"] == out["network"].get("parameters_total")
        elif "library_parameters" in out:
            params = recorded.get("params", {})
            shared = set(params) & set(out["library_parameters"])
            entry["differ"] = sorted(k for k in shared if params[k] != out["library_parameters"][k]
                                     and not _is_seed(k))
            entry["same"] = not entry["differ"]
        found.append(entry)
        if len(found) >= 5:
            break
    return found


def render(d: dict) -> str:
    lines = [f"{d['name']}: configuration {d['config_hash'][:12]}, family {d['family']}"]
    for section in ("data", "split", "task", "features", "model", "fit"):
        lines.append(f"  {section}: {json.dumps(d['settings'][section], sort_keys=True)}")
    for i, step in enumerate(d["prep"]):
        lines.append(f"  prep {i}: {json.dumps(step, sort_keys=True)}")
    if "network" in d:
        net = d["network"]
        lines.append(f"  network: {net.get('parameters_total')} parameters")
        for row in net.get("layers", []):
            lines.append(f"    {row}")
    else:
        lines.append("  library parameters, defaults included:")
        for k, v in sorted(d["library_parameters"].items()):
            lines.append(f"    {k} = {v!r}")
    for r in d["runs"]:
        lines.append(f"  run {r['run']}: " + ("matches" if r.get("same") else f"DIFFERS {r.get('differ', '')}"))
    return "\n".join(lines)
