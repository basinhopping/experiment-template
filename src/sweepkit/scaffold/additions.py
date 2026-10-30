"""check-additions and smoke-all: every registered component has what makes it trustworthy, and every one runs.

check-additions (also a unit test, tests/test_additions.py) refuses a component without:
  - an example sweep: the file its declaration names (sweeps/examples/<example>.json, or sweeps/project/ for a
    project's own components) exists, expands with the current code, and uses the component in at least one
    configuration; a component that names no example must be used by some example sweep;
  - a test: some file under tests/ names it in quotes ("<name>");
  - for metrics, which have no example: a test that names it;
  - for stage kinds: an example sweep with a stage of that kind, and a test that names it.
smoke-all then runs, for every component of a profile (or all), one configuration that uses it (smoke_job: from its
declared example first, then the other examples in order of name, the first whose packages are all installed here),
through the real runner on a few rows (as smoke does), and writes one table:
    artifacts/smoke/smoke_all_<utc>/table.csv
        kind, name, profile, example, status, seconds, device (the device that ran), fallbacks, error
    status is ok, FAILED, or skipped (a package the configuration needs is not installed here; with SWEEPKIT_REQUIRE
    naming that package the skip becomes a failure, as in the tests)
make-example-data runs first, so the examples that read files find them.
"""
import importlib.util
import json
import os
import time
import traceback

from sweepkit import paths

COMPONENT_KINDS = ("dataset", "split", "task", "prep step", "feature", "augmentation", "model", "fit procedure")
# the job section that names a component of each kind, as (section, key) in a job's settings
SECTION_OF = {"dataset": "data", "split": "split", "task": "task", "feature": "features", "model": "model",
              "fit procedure": "fit"}


def example_folders() -> list:
    root = paths.repo_root() / "sweeps"
    return [root / "examples", root / "project"]


def example_path(example: str):
    for folder in example_folders():
        path = folder / f"{example}.json"
        if path.is_file():
            return path
    return None


def _uses(settings: dict) -> set:
    """(kind, name) pairs a job's settings use."""
    out = set()
    for kind, section in SECTION_OF.items():
        out.add((kind, settings[section]["name"]))
    for step in settings["prep"]:
        out.add(("prep step", step["name"]))
        for value in step.values():
            if isinstance(value, dict) and "name" in value:
                out.add(("model", value["name"]))           # a nested section, such as reduce's embedder
    for value in settings["model"].values():
        if isinstance(value, dict) and "name" in value:
            out.add(("model", value["name"]))
    for aug in settings["fit"].get("augment", []) if isinstance(settings["fit"].get("augment"), list) else []:
        if isinstance(aug, dict) and "name" in aug:
            out.add(("augmentation", aug["name"]))
    return out


def expanded_examples() -> dict:
    """example name -> {"path", "jobs" or "error", "uses"}."""
    from sweepkit.sweep.design import expand

    out = {}
    for folder in example_folders():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            entry = {"path": path}
            try:
                jobs, _ = expand(json.loads(path.read_text(encoding="utf-8")))
                entry["jobs"] = jobs
                entry["uses"] = set().union(*(_uses(j["settings"]) for j in jobs))
            except Exception as err:          # every problem of an example is reported, not raised
                entry["error"] = f"{type(err).__name__}: {err}"
                entry["uses"] = set()
            out[path.stem] = entry
    return out


def missing_packages(job: dict, offered: str = "cpu") -> list:
    """Import names that the components a job uses declare and this machine lacks, in order of first use. A model's
    cpu_packages count only when the job would run on the CPU: its GPU path needs none of them (scanpy's Leiden needs
    igraph on the CPU, rapids-singlecell's on the GPU does not)."""
    from sweepkit import devices
    from sweepkit.registry import REGISTRIES

    requested = job["settings"].get("fit", {}).get("device", "cpu")
    missing = []
    for kind, name in sorted(_uses(job["settings"])):
        try:
            meta = REGISTRIES[kind].get(name).meta
        except KeyError:
            continue
        modules = list(meta.get("packages", ()))
        gpu = meta.get("gpu", "none")
        on_gpu = (offered == "cuda" and requested in ("cuda", "auto") and gpu != "none"
                  and devices.gpu_path_ready(gpu))
        if not on_gpu:
            modules += list(meta.get("cpu_packages", ()))
        for module in modules:
            if module not in missing and importlib.util.find_spec(module) is None:
                missing.append(module)
    return missing


def _needs_more_than_core(example: dict) -> bool:
    from sweepkit.registry import REGISTRIES

    for kind, name in example.get("uses", ()):
        try:
            if REGISTRIES[kind].get(name).meta.get("profile", "core") != "core":
                return True
        except KeyError:
            continue
    return False


def smoke_job(kind: str, name: str, meta: dict, examples: dict, method=None, ready=None):
    """The configuration smoke-all runs for a component, as (example name, job): the first job that uses it, from
    its declared example first, then the examples that need only the core profile, then the others, each group in
    order of name; with `ready` (a function of a job), the first such job for which it is true. For a preparation
    step, `method` asks for a step with that method. None when no example uses it."""
    def uses(job) -> bool:
        if method is None:
            return (kind, name) in _uses(job["settings"])
        return any(step["name"] == name and step.get("method") == method for step in job["settings"]["prep"])

    preferred = meta.get("example")
    order = ([preferred] if preferred in examples else []) + sorted(
        (e for e in examples if e != preferred), key=lambda e: (_needs_more_than_core(examples[e]), e))
    for example in order:
        for job in examples[example].get("jobs", []):
            if uses(job) and (ready is None or ready(job)):
                return example, job
    return None


def _test_texts() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((paths.repo_root() / "tests").glob("*.py")))


def check_additions() -> dict:
    from sweepkit import catalog
    from sweepkit.registry import METRICS, REGISTRIES
    from sweepkit.sweep.design import STAGE_KINDS

    catalog.load_all()
    examples = expanded_examples()
    tests = _test_texts()
    used_anywhere = set().union(*(e["uses"] for e in examples.values())) if examples else set()
    problems, checked = [], 0
    for name, e in examples.items():
        if "error" in e:
            problems.append(f"example {e['path'].relative_to(paths.repo_root()).as_posix()} does not expand: "
                            f"{e['error']}")
    for kind in COMPONENT_KINDS:
        registry = REGISTRIES[kind]
        for entry in registry.entries():
            checked += 1
            where = f"{kind} {entry.name}"
            example = entry.meta.get("example")
            if example is not None:
                e = examples.get(example)
                if e is None:
                    problems.append(f"{where}: its example sweeps/examples/{example}.json is missing")
                elif "error" not in e and (kind, entry.name) not in e["uses"]:
                    problems.append(f"{where}: its example {example} has no configuration that uses it")
            elif (kind, entry.name) not in used_anywhere:
                problems.append(f"{where}: no example sweep uses it (name one in its declaration)")
            if f'"{entry.name}"' not in tests and f"'{entry.name}'" not in tests:
                problems.append(f"{where}: no test under tests/ names it")
    for name in METRICS.names():
        checked += 1
        if f'"{name}"' not in tests and f"'{name}'" not in tests:
            problems.append(f"metric {name}: no test under tests/ names it")
    stage_kinds_used = {stage.get("kind") for e in examples.values() if "error" not in e
                        for stage in json.loads(e["path"].read_text(encoding="utf-8")).get("stages", [])}
    for name in sorted(STAGE_KINDS):
        checked += 1
        if name not in stage_kinds_used:
            problems.append(f"stage kind {name}: no example sweep has a stage of this kind")
        if f'"{name}"' not in tests and f"'{name}'" not in tests:
            problems.append(f"stage kind {name}: no test under tests/ names it")
    return {"checked": checked, "examples": len(examples), "problems": problems, "ok": not problems}


def _required() -> set:
    return {x.strip() for x in os.environ.get("SWEEPKIT_REQUIRE", "").split(",") if x.strip()}


def smoke_all(profile: str, offered: str, rows: int, log) -> dict:
    from sweepkit import catalog
    from sweepkit.data.examples import make_example_data
    from sweepkit.families.common import csv_text
    from sweepkit.registry import REGISTRIES
    from sweepkit.runs.store import OutputFolder, stamp
    from sweepkit.sweep.runner import execute
    from sweepkit.sweep.smoke import smoke_spec

    catalog.load_all()
    make_example_data(log)
    examples = expanded_examples()
    required = _required()
    folder = OutputFolder(paths.smoke_dir(), f"smoke_all_{stamp()}").begin()
    table, counts = [], {"ok": 0, "FAILED": 0, "skipped": 0}
    try:
        for kind in COMPONENT_KINDS:
            for entry in REGISTRIES[kind].entries():
                meta = entry.meta
                p = meta.get("profile", "core")
                if profile != "all" and p != profile:
                    continue
                chosen = smoke_job(kind, entry.name, meta, examples, ready=lambda j: not missing_packages(j, offered))
                chosen = chosen or smoke_job(kind, entry.name, meta, examples)
                # kind, name, profile, example, status, seconds, device, fallbacks, error
                row = [kind, entry.name, p, chosen[0] if chosen else "", "", "", "", "", ""]
                if chosen is None:
                    row[4], row[8] = "FAILED", "no example configuration uses it"
                    counts["FAILED"] += 1
                    table.append(row)
                    log(f"  FAILED   {kind} {entry.name}: no example configuration uses it")
                    continue
                job = chosen[1]
                missing = missing_packages(job, offered)
                if missing and not (set(missing) & required):
                    row[4], row[8] = "skipped", f"needs {', '.join(missing)}"
                    counts["skipped"] += 1
                    table.append(row)
                    log(f"  skipped  {kind} {entry.name}: needs {', '.join(missing)}")
                    continue
                started = time.perf_counter()
                runs_root = folder.tmp / "runs" / f"{kind.replace(' ', '_')}_{entry.name}"
                try:
                    spec = smoke_spec(job, rows)
                    execute(spec, offered, lambda m: None, runs_root=runs_root)
                    row[4] = "ok"
                    counts["ok"] += 1
                    provenance = json.loads((runs_root / spec["run_id"] / "provenance.json").read_text(
                        encoding="utf-8"))
                    row[6] = str(provenance.get("device_used", ""))
                    row[7] = "; ".join(str(f) for f in provenance.get("fallbacks") or [])
                except Exception as err:
                    row[4], row[8] = "FAILED", f"{type(err).__name__}: {err}"
                    counts["FAILED"] += 1
                    log(traceback.format_exc())
                row[5] = f"{time.perf_counter() - started:.2f}"
                table.append(row)
                where = f" on {row[6]}" if row[6] else ""
                log(f"  {row[4]:7s}  {kind} {entry.name} ({row[5]}s{where}){': ' + row[8] if row[8] else ''}")
        folder.write_text("table.csv", csv_text(["kind", "name", "profile", "example", "status", "seconds", "device",
                                                 "fallbacks", "error"], table))
        folder.write_json("summary.json", {"profile": profile, "device": offered, "rows": rows, **counts})
    except BaseException as err:
        folder.fail(err)
        raise
    saved = folder.complete()
    log(f"smoke-all ({profile}, {offered}): {counts['ok']} ok, {counts['FAILED']} failed, {counts['skipped']} skipped; "
        f"saved {saved}")
    log("SMOKE-ALL OK" if counts["FAILED"] == 0 else "SMOKE-ALL FAILED")
    return {"counts": counts, "folder": str(saved), "table": table}
