"""Command line: python -m sweepkit.cli <command> --argument value ...

Every argument is required: nothing here has a default, so every command line says everything it does. Exit codes:
0 passed, 1 failed (or refused), 3 not finished (run the same command again). Results go to fixed places under
artifacts/ (sweepkit.paths); no command takes an output path. `python -m sweepkit.cli <command> --help` lists a
command's arguments, and docs/COMMANDS.md describes each command with an example.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import sweepkit  # noqa: F401  applies the CUDA MPS fix (when asked) before anything imports torch or CuPy
from sweepkit import catalog, paths
from sweepkit.runs.provenance import collect
from sweepkit.runs.store import OutputFolder, stamp

COMMANDS = []
SECTION_OF = {}             # command name -> the section docs/COMMANDS.md lists it under
_SECTION = ["Commands"]


def section(title: str) -> None:
    """The section the commands registered after this call belong to (docs/COMMANDS.md)."""
    _SECTION[0] = title


def command(name: str, help_text: str, *arguments):
    """Register a command. Each argument is (flag, options); every one is required."""
    def decorator(fn):
        COMMANDS.append((name, help_text, arguments, fn))
        SECTION_OF[name] = _SECTION[0]
        return fn
    return decorator


def arg(flag: str, help_text: str, **options) -> tuple:
    return flag, dict(options, help=help_text)


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def _save(parent: Path, folder_id: str, files: dict, entry: str) -> Path:
    """Write a report folder once: every file, then provenance.json."""
    folder = OutputFolder(parent, folder_id).begin()
    try:
        for name, content in files.items():
            if name.endswith(".json"):
                folder.write_json(name, content)
            elif isinstance(content, bytes):
                folder.write_bytes(name, content)
            else:
                folder.write_text(name, content)
        folder.write_json("provenance.json", collect(f"sweepkit.cli {entry}"))
    except BaseException as err:
        folder.fail(err)
        raise
    return folder.complete()


def _log(message: str) -> None:
    print(message, flush=True)


# Sweeps and the queue ----------------------------------------------------------------------------------------------

section("Sweeps and the queue")

@command("check", "expand a sweep file and print what it would run, with each configuration's stage, label, hash "
                  "and fit.device (nothing is queued or recorded)",
         arg("--sweep-file", "the sweep file"))
def cmd_check(args) -> int:
    from sweepkit.sweep.design import expand, warnings_for
    from sweepkit.sweep.enqueue import load_sweep_file

    sweep = load_sweep_file(Path(args.sweep_file))
    jobs, summary = expand(sweep)
    for warning in warnings_for(sweep, summary):
        print(f"WARNING {warning}")
    # each configuration with the first 10 characters of its hash, which select, a families file for rule-fix and
    # describe-model take, known before anything runs
    configurations = {}
    for job in jobs:
        entry = configurations.setdefault(job["config_hash"], {"config": job["config_hash"][:10], "stage": job["stage"],
                                                               "label": job["label"],
                                                               "device": job["settings"]["fit"].get("device", "cpu"),
                                                               "jobs": 0})
        entry["jobs"] += 1
    _print({**summary, "configurations": list(configurations.values())})
    return 0


@command("smoke", "run every configuration of a sweep file once on a few rows, before enqueueing it",
         arg("--sweep-file", "the sweep file"),
         arg("--device", "what this machine offers: cpu, cuda or mps"),
         arg("--rows", "rows per role in each smoke run, for example 200", type=int))
def cmd_smoke(args) -> int:
    from sweepkit.sweep.smoke import smoke

    result = smoke(Path(args.sweep_file), args.device, args.rows, _log)
    return 0 if result["failed"] == 0 else 1


@command("enqueue", "expand sweep files, record them, and put their jobs in the queue: all of them, or none when "
                    "any is refused",
         arg("--sweep-file", "the sweep file, or several", nargs="+"))
def cmd_enqueue(args) -> int:
    from sweepkit.sweep.enqueue import enqueue_all

    for summary in enqueue_all([Path(f) for f in args.sweep_file], _log):
        _print({k: summary[k] for k in ("sweep_id", "purpose", "configs", "jobs", "seeds", "notes")})
    return 0


@command("run", "run every job of a sweep file in this process (local mode; resumable)",
         arg("--sweep-file", "the sweep file"),
         arg("--device", "what this machine offers: cpu, cuda or mps"),
         arg("--max-consecutive-failures", "stop after this many failed jobs in a row", type=int))
def cmd_run(args) -> int:
    from sweepkit.sweep.local import run_local

    result = run_local(Path(args.sweep_file), args.device, args.max_consecutive_failures, _log)
    return 0 if result["failed"] == 0 else 1


@command("worker", "take jobs from the queue until nothing is left that this worker can run",
         arg("--worker-id", "<group>_<n>, for example gpu0_1 or cpu_1 (STOP_<group> stops a group)"),
         arg("--device", "what this worker offers: cpu, cuda or mps"),
         arg("--min-free-gb", "stop when the disk has less free space than this", type=float),
         arg("--max-consecutive-failures", "stop after this many failed jobs in a row", type=int))
def cmd_worker(args) -> int:
    from sweepkit.sweep.worker import run

    return run(args.worker_id, args.device, args.min_free_gb, args.max_consecutive_failures)


@command("local-workers", "start N worker processes on this machine and wait for them (no tmux, any system)",
         arg("--n", "how many workers", type=int),
         arg("--device", "what they offer: cpu, cuda or mps"),
         arg("--min-free-gb", "each stops below this much free disk", type=float),
         arg("--max-consecutive-failures", "each stops after this many failed jobs in a row", type=int))
def cmd_local_workers(args) -> int:
    from sweepkit.sweep.local import local_workers

    return local_workers(args.n, args.device, args.min_free_gb, args.max_consecutive_failures, _log)


@command("status", "queue, failures and the ranking of complete runs of a sweep (choosing files only)",
         arg("--sweep", "the sweep ID"),
         arg("--top", "how many (configuration, seed) pairs to list", type=int),
         arg("--since", "a UTC time such as 2026-10-27T18:00:00, or all"))
def cmd_status(args) -> int:
    from sweepkit.sweep.status import render, summarise

    print(render(summarise(args.sweep, None if args.since == "all" else args.since), args.top))
    return 0


@command("park", "move a sweep's pending jobs aside (queue/parked/<name>/), where no worker looks",
         arg("--sweep", "the sweep ID"), arg("--name", "a name for the parked set"))
def cmd_park(args) -> int:
    from sweepkit.sweep.queue import Queue

    print(f"parked {Queue(paths.queue_dir()).park(args.sweep, args.name)} jobs of {args.sweep} as {args.name}")
    return 0


@command("unpark", "put a parked set of jobs back in the queue", arg("--name", "the parked set"))
def cmd_unpark(args) -> int:
    from sweepkit.sweep.queue import Queue

    print(f"moved {Queue(paths.queue_dir()).unpark(args.name)} jobs back to pending")
    return 0


@command("requeue-running", "move jobs left in queue/running back to pending, after every worker stopped",
         arg("--all-workers-stopped", "say yes once no worker runs (tmux ls shows none; local workers exited)",
             choices=["yes"]))
def cmd_requeue_running(args) -> int:
    from sweepkit.sweep.queue import Queue

    print(f"moved {Queue(paths.queue_dir()).requeue_running()} jobs from running back to pending")
    return 0


@command("requeue-failed", "move a sweep's failed jobs back to pending (after fixing what failed them)",
         arg("--sweep", "the sweep ID"))
def cmd_requeue_failed(args) -> int:
    from sweepkit.sweep.queue import Queue

    print(f"moved {Queue(paths.queue_dir()).requeue_failed(args.sweep)} failed jobs of {args.sweep} back to pending")
    return 0


# Choosing, then judging --------------------------------------------------------------------------------------------

section("Choosing, then judging")

@command("analyze", "compare a sweep's configurations with its baseline on the choosing files only",
         arg("--sweep", "the sweep ID"),
         arg("--confidence", "for example 0.95", type=float),
         arg("--n-boot", "bootstrap draws, for example 2000", type=int),
         arg("--seed", "seed of the bootstrap's random generator", type=int),
         arg("--meaningful", "P(better) threshold whose interval's upper end counts as meaningful, for example 0.75",
             type=float),
         arg("--top", "how many configurations to print", type=int))
def cmd_analyze(args) -> int:
    from sweepkit.analysis.validation import analyze, render

    report = analyze(args.sweep, args.confidence, args.n_boot, args.seed, args.meaningful)
    text = render(report, args.top)
    print(text)
    saved = _save(paths.analysis_dir(), f"{args.sweep}_{stamp()}", {"report.json": report, "report.txt": text + "\n"},
                  "analyze")
    print(f"saved {saved}")
    return 0


@command("select", "record the chosen configuration of an analyzed sweep, before any test or reveal file is read",
         arg("--sweep", "the sweep ID"),
         arg("--config", "the configuration hash, or its first 8 or more characters"),
         arg("--name", "a name for this selection, for example screen1_best"),
         arg("--reason", "why this configuration, in one sentence"))
def cmd_select(args) -> int:
    from sweepkit.analysis.selection import selection_record

    record = selection_record(args.sweep, args.config, args.name, args.reason)
    saved = _save(paths.selections_dir(), args.name, {"selection.json": record}, "select")
    value = record["choosing"].get("value")
    print(f"selected {record['config_hash'][:12]} of {record['sweep_id']} ({record['label']}), "
          f"{record['select']['metric']} {value:.4f}, rank {record['rank']} of {record['n_ranked']}, "
          f"from {record['analysis_report']['folder']}")
    print(f"saved {saved}")
    return 0


@command("rule-fix", "store a RULE file with its UTC time and SHA-256, before the results it governs exist",
         arg("--name", "a name for the RULE, for example round3"),
         arg("--rule-file", "the RULE text (templates/RULE.txt is a starting point)"),
         arg("--sweeps", "the sweep IDs it covers", nargs="+"),
         arg("--families", "a JSON file of comparison families for test-compare, or none"),
         arg("--late-reason", "why the RULE is fixed after runs of its sweeps exist (recorded, and printed with every "
                              "report that uses it), or none"))
def cmd_rule_fix(args) -> int:
    from sweepkit.analysis.rules import late_note, rule_fix

    families = None if args.families == "none" else Path(args.families)
    reason = None if args.late_reason == "none" else args.late_reason
    folder = rule_fix(args.name, Path(args.rule_file), list(args.sweeps), families, reason)
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    print(f"RULE {record['name']} fixed at {record['fixed_utc']}, SHA-256 {record['rule_sha256']}, covering "
          f"{', '.join(record['sweeps'])}; {len(record['families'])} comparison families")
    for line in filter(None, (record.get("warning"), late_note(record))):
        print(line)
    if reason is not None and record["late_reason"] is None:
        print("no run of these sweeps exists yet, so the late reason was not needed and is not recorded")
    print(f"saved {folder}")
    return 0


def _judging_report(kind: str, selection: str, confidence: float) -> int:
    from sweepkit.analysis.rules import now_utc, reading_record
    from sweepkit.analysis.selection import render, reveal_report, test_report

    started = now_utc()
    report = (test_report if kind == "test" else reveal_report)(selection, confidence)
    text = render(report, kind)
    print(text)
    sel = report["selection"]
    reading = reading_record(kind, [{"sweep_id": sel["sweep_id"], "config_hash": sel["config_hash"]}],
                             report["rule"], sel["name"], started)
    parent = paths.test_reports_dir() if kind == "test" else paths.reveal_reports_dir()
    saved = _save(parent, f"{selection}_{stamp()}", {"report.json": report, "report.txt": text + "\n",
                                                    "reading.json": reading}, f"{kind}-report")
    print(f"saved {saved}")
    check = report.get("identity_check")
    return 1 if check and check["compared"] and not check["agree"] else 0


@command("test-report", "the test results of a supervised selection (reads only its configuration's test files)",
         arg("--selection", "the selection name"), arg("--confidence", "for example 0.95", type=float))
def cmd_test_report(args) -> int:
    return _judging_report("test", args.selection, args.confidence)


@command("reveal-report", "agreement with the reference labels of an unsupervised selection (reads only its "
                          "configuration's reveal files)",
         arg("--selection", "the selection name"), arg("--confidence", "for example 0.95", type=float))
def cmd_reveal_report(args) -> int:
    return _judging_report("reveal", args.selection, args.confidence)


@command("test-compare", "run exactly the comparison families a RULE fixed, on test or reveal files",
         arg("--rule", "the RULE folder name rule-fix printed (under artifacts/rules/)"),
         arg("--confidence", "for example 0.95", type=float),
         arg("--n-boot", "bootstrap draws, for example 2000", type=int),
         arg("--seed", "seed of the bootstrap's random generator", type=int),
         arg("--meaningful", "P(better) threshold, for example 0.75", type=float))
def cmd_test_compare(args) -> int:
    from sweepkit.analysis.comparisons import render, test_compare
    from sweepkit.analysis.rules import now_utc, reading_record

    started = now_utc()
    report = test_compare(args.rule, args.confidence, args.n_boot, args.seed, args.meaningful)
    text = render(report)
    print(text)
    kinds = sorted({f["files"] for f in report["families"]})
    reading = reading_record("+".join(kinds), report["configs"], report["rule"], None, started)
    saved = _save(paths.comparisons_dir(), f"{args.rule}_{stamp()}",
                  {"report.json": report, "report.txt": text + "\n", "reading.json": reading}, "test-compare")
    print(f"saved {saved}")
    return 0


@command("compare-runs", "a sweep's runs against every copy of the same runs in other sweeps (exact reproduction)",
         arg("--sweep", "the sweep ID"),
         arg("--differences-shown", "how many differences to list per copy", type=int))
def cmd_compare_runs(args) -> int:
    from sweepkit.analysis.compare import compare_sweep, render

    report = compare_sweep(args.sweep, args.differences_shown)
    print(render(report))
    if report["verdict"] == "INCOMPLETE":
        return 3
    saved = _save(paths.checks_dir(), f"compare_runs_{args.sweep}_{stamp()}", {"report.json": report}, "compare-runs")
    print(f"saved {saved}")
    return 0 if report["verdict"] in ("SAME", "NO COPIES") else 1


@command("atlas", "every sweep, configuration, setting and value tried, as tables",
         arg("--format", "csv, or xlsx for a workbook as well (needs openpyxl)", choices=["csv", "xlsx"]))
def cmd_atlas(args) -> int:
    from sweepkit.analysis.atlas import build, write

    atlas = build()
    folder = OutputFolder(paths.atlas_dir(), f"atlas_{stamp()}").begin()
    try:
        write(folder, atlas, args.format)
        folder.write_json("provenance.json", collect("sweepkit.cli atlas"))
    except BaseException as err:
        folder.fail(err)
        raise
    print(f"{len(atlas['rows'])} configurations, {len(atlas['keys'])} setting paths; saved {folder.complete()}")
    return 0


# Data --------------------------------------------------------------------------------------------------------------

section("Data")

def _none(value: str):
    return None if value == "none" else value


@command("make-synthetic", "write a synthetic table with planted structure to data/synthetic/<name>.csv (and a sample)",
         arg("--generator", "for example blobs, moons, mixed, classification, regression_linear, outliers"),
         arg("--n-rows", "rows", type=int), arg("--n-features", "columns", type=int),
         arg("--n-groups", "clusters, classes or topics, as the generator documents", type=int),
         arg("--noise", "the generator's spread", type=float), arg("--seed", "the data's seed", type=int),
         arg("--name", "the file's name under data/synthetic/"),
         arg("--chunk-rows", "rows written at a time", type=int),
         arg("--sample-rows", "rows in the sample file beside it", type=int))
def cmd_make_synthetic(args) -> int:
    from sweepkit.data.make import write_synthetic

    record = write_synthetic(args.generator, args.n_rows, args.n_features, args.n_groups, args.noise, args.seed,
                             args.name, args.chunk_rows, args.sample_rows, _log)
    _print({"csv": record["csv"], "sample": record["sample"], "columns": len(record["columns"])})
    return 0


@command("convert", "turn a large CSV under data/ into a memory-mapped store (or Parquet), once, in chunks",
         arg("--source", "the CSV or TSV, relative to data/"),
         arg("--name", "the store's name: data/stores/<name>/"),
         arg("--sep", "the column separator: comma or tab", choices=["comma", "tab"]),
         arg("--row-id", "the row ID column, or none (row numbers)"),
         arg("--features", "input columns: names, re:<regex>, or * for every other column", nargs="+"),
         arg("--exclude", "columns that are never inputs, or none", nargs="+"),
         arg("--keep", "columns carried along as text (targets, groups, labels), or none", nargs="+"),
         arg("--chunk-rows", "rows read at a time, for example 100000", type=int),
         arg("--to", "store (memory-mapped float32) or parquet", choices=["store", "parquet"]))
def cmd_convert(args) -> int:
    from sweepkit.data.convert import convert

    def listed(values):
        return () if values == ["none"] else tuple(values)

    record = convert(args.source, args.name, "," if args.sep == "comma" else "\t", _none(args.row_id),
                     tuple(args.features), listed(args.exclude), listed(args.keep), args.chunk_rows, args.to, _log)
    _print({k: record[k] for k in ("rows", "features", "kept", "sha256", "seconds")})
    return 0


@command("extract-features", "turn every row of text, images or audio into one vector with a frozen pretrained "
         "encoder, written once as a store (data/stores/<name>/) for the store dataset",
         arg("--data-file", "a JSON file holding one dataset section (its kind chooses text, image or audio)"),
         arg("--source", "hub (a Hub model pinned to a commit) or local (a folder under data/)",
             choices=["hub", "local"]),
         arg("--model-id", "hub: <owner>/<name>; local: the folder, relative to data/"),
         arg("--revision", "hub: the commit's 40 hexadecimal characters; local: none"),
         arg("--pooling", "cls (the first position), mean (over the real positions) or pooler (the model's pooler "
             "output)", choices=["cls", "mean", "pooler"]),
         arg("--max-length", "text: tokens per row, longer rows cut; images and audio: none"),
         arg("--batch-size", "rows per forward pass (audio: 1)", type=int),
         arg("--dtype", "the encoder's weights: float32 or bfloat16 (values are stored as float32)",
             choices=["float32", "bfloat16"]),
         arg("--device", "where the encoder runs: cpu, cuda or mps", choices=["cpu", "cuda", "mps"]),
         arg("--keep", "dataset columns carried along as text (targets, groups), or none", nargs="+"),
         arg("--name", "the store's name: data/stores/<name>/"))
def cmd_extract_features(args) -> int:
    from sweepkit.data.extract import extract_features

    max_length = None if args.max_length == "none" else int(args.max_length)
    record = extract_features(Path(args.data_file), args.source, args.model_id, _none(args.revision), args.pooling,
                              max_length, args.batch_size, args.dtype, args.device,
                              () if args.keep == ["none"] else tuple(args.keep), args.name, _log)
    _print({k: record[k] for k in ("rows", "features", "sha256", "seconds")})
    return 0


@command("ingest", "copy a received folder into data/received/<name>/ with a manifest; run again until finalized",
         arg("--name", "the copy's name under data/received/"),
         arg("--source", "the received folder (it is only read)"),
         arg("--max-seconds", "stop cleanly after this long; the next call continues", type=float))
def cmd_ingest(args) -> int:
    from sweepkit.data.ingest import ingest

    result = ingest(args.name, Path(args.source), args.max_seconds)
    _print(result)
    return 0 if result["phase"] in ("finalized", "already_ingested") else 3


@command("verify-data", "re-hash an ingested copy against its manifest", arg("--name", "the copy's name"))
def cmd_verify_data(args) -> int:
    from sweepkit.data.ingest import verify

    report = verify(args.name)
    saved = _save(paths.checks_dir(), f"verify_data_{args.name}_{stamp()}", {"report.json": report}, "verify-data")
    _print({k: report[k] for k in ("ok", "n_manifest_rows", "n_files_present", "n_rehashed", "missing", "extra",
                                   "size_mismatch", "hash_mismatch")})
    print(f"saved {saved}")
    return 0 if report["ok"] else 1


@command("profile-data", "look at a data file or folder before any model runs (sizes, types, missing values, joins)",
         arg("--target", "a file or folder, relative to data/"),
         arg("--row-id", "the row ID column of a table, or none"),
         arg("--labels", "a label table to join on the row IDs (relative to data/), or none"),
         arg("--label-key", "the label table's key column, or none"),
         arg("--max-rows", "rows to read, or all"),
         arg("--top-values", "how many most frequent values to list per text column", type=int),
         arg("--hide", "columns whose values must not be shown, comma-separated (reference labels and any other "
                       "column that judges a choice), or none"))
def cmd_profile_data(args) -> int:
    from sweepkit.data.profile import profile, render

    max_rows = None if args.max_rows == "all" else int(args.max_rows)
    hide = () if args.hide == "none" else tuple(c.strip() for c in args.hide.split(",") if c.strip())
    report = profile(args.target, _none(args.row_id), _none(args.labels), _none(args.label_key), max_rows,
                     args.top_values, hide)
    text = render(report)
    print(text)
    slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in args.target)[-60:]
    saved = _save(paths.checks_dir(), f"profile_{slug}_{stamp()}", {"report.json": report, "report.txt": text + "\n"},
                  "profile-data")
    print(f"saved {saved}")
    return 0


@command("estimate", "measure one configuration on growing parts of the data and predict the full run's cost",
         arg("--sweep-file", "the sweep file"),
         arg("--config", "base, or 8 or more characters of a configuration hash"),
         arg("--device", "what this machine offers: cpu, cuda or mps"),
         arg("--fractions", "increasing fractions of the fold's fitting rows, for example 0.01 0.02 0.05",
             nargs="+", type=float),
         arg("--workers", "how many jobs will run at once (memory check)", type=int))
def cmd_estimate(args) -> int:
    from sweepkit.data.estimate import estimate, render

    result = estimate(Path(args.sweep_file), args.config, args.device, list(args.fractions), args.workers, _log)
    print(render(result))
    print(f"saved {result['folder']}")
    return 0 if not result["problems"] else 1


# Results to read and take home -------------------------------------------------------------------------------------

section("Results to read and take home")

def _reading(result: dict, started: str, kind: str) -> dict:
    from sweepkit.analysis.rules import reading_record

    return reading_record(kind, [{"sweep_id": result["sweep_id"], "config_hash": result["config_hash"]}],
                          result["rule"], result["selection"], started)


@command("export", "the results of a selection to take home: rows with IDs, cluster summaries, plots, settings",
         arg("--selection", "the selection name"),
         arg("--runs", "all, or one run ID of the selected configuration"),
         arg("--reveal", "yes to include the files that judge the choice (reference labels, test predictions; needs "
                         "the RULE discipline), no otherwise", choices=["yes", "no"]),
         arg("--plots", "yes or no", choices=["yes", "no"]),
         arg("--top-features", "how many distinguishing columns to list per cluster", type=int),
         arg("--formats", "csv, or csv+parquet (needs pyarrow)", choices=["csv", "csv+parquet"]))
def cmd_export(args) -> int:
    from sweepkit.analysis.export import collect, files, report_json
    from sweepkit.analysis.rules import now_utc

    started = now_utc()
    result = collect(args.selection, args.runs, args.reveal == "yes", args.top_features, args.plots == "yes")
    out = files(result, args.formats)
    out["report.json"] = report_json(result)
    if result["reveal"]:
        out["reading.json"] = _reading(result, started, "reveal" if result["family"] != "estimator" else "test")
    for warning in result["warnings"]:
        print(warning)
    saved = _save(paths.exports_dir(), f"{args.selection}_{stamp()}", out, "export")
    print(f"{len(result['rows'])} rows from {len(result['runs'])} runs; saved {saved}")
    return 0


@command("cluster-report", "print what a selected clustering found: sizes, distinguishing columns, agreement tables",
         arg("--selection", "the selection name"),
         arg("--runs", "all, or one run ID of the selected configuration"),
         arg("--reveal", "yes to add the agreement with the reference labels (needs the RULE discipline), or no",
             choices=["yes", "no"]),
         arg("--top-features", "how many distinguishing columns to print per cluster", type=int))
def cmd_cluster_report(args) -> int:
    from sweepkit.analysis.cluster_report import render
    from sweepkit.analysis.export import collect, report_json
    from sweepkit.analysis.rules import now_utc

    started = now_utc()
    result = collect(args.selection, args.runs, args.reveal == "yes", args.top_features, True)
    if result["family"] != "clusterer":
        raise SystemExit(f"{args.selection} is a {result['family']} selection; cluster-report reads clusterings")
    text = render(result, args.top_features)
    print(text)
    out = {"report.txt": text + "\n", "report.json": report_json(result)}
    out.update({f"plots/{k}": v for k, v in result["pictures"].items()})
    if result["reveal"]:
        out["reading.json"] = _reading(result, started, "reveal")
    saved = _save(paths.exports_dir(), f"{args.selection}_report_{stamp()}", out, "cluster-report")
    print(f"saved {saved}")
    return 0


@command("describe-model", "a configuration exactly as a job builds it: settings, preparation, the network or the "
                           "library object's every parameter",
         arg("--source", "a sweep file, or selection:<name>"),
         arg("--config", "base or 8 or more characters of a configuration hash (sweep file), or selected"))
def cmd_describe_model(args) -> int:
    from sweepkit.analysis.describe import describe, render

    d = describe(args.source, args.config, _log)
    text = render(d)
    print(text)
    saved = _save(paths.architectures_dir(), f"{d['name']}_{stamp()}",
                  {"description.json": d, "description.txt": text + "\n"}, "describe-model")
    print(f"saved {saved}")
    return 0


@command("propose", "the next configurations from Bayesian search over a finished sweep, as a new sweep file "
                    "(needs the search profile)",
         arg("--sweep", "the finished sweep"),
         arg("--stage", "the stage whose factors and levels are the search space"),
         arg("--n-configs", "how many new configurations", type=int),
         arg("--seed", "the sampler's seed", type=int),
         arg("--n-startup", "finished trials before TPE replaces random sampling (Optuna's default is 10)", type=int),
         arg("--name", "the new sweep's sweep_id"),
         arg("--seeds", "the new sweep's seeds", nargs="+", type=int),
         arg("--priority", "the new sweep's priority", type=int))
def cmd_propose(args) -> int:
    from sweepkit.analysis.propose import propose

    result = propose(args.sweep, args.stage, args.n_configs, args.seed, args.n_startup, args.name, list(args.seeds),
                     args.priority)
    saved = _save(paths.proposals_dir(), f"{args.name}_{stamp()}",
                  {"sweep.json": result["sweep_file"], "record.json": result["record"]}, "propose")
    print(f"{len(result['record']['trials'])} finished configurations seen, {len(result['record']['proposals'])} "
          f"proposed; copy {saved}/sweep.json into sweeps/ to run it")
    return 0


# Rounds ------------------------------------------------------------------------------------------------------------

section("Rounds")

@command("round-new", "make a round's preparation folder (README, RULE, plan, lines) from the templates",
         arg("--round", "the round's name, for example round3"),
         arg("--date", "the round's date, YYYYMMDD"),
         arg("--question", "what the round asks, in one or two sentences"),
         arg("--sweep-files", "the round's sweep files, under sweeps/", nargs="+"),
         arg("--workers-per-gpu", "worker processes per GPU in the schedule", type=int),
         arg("--gpus", "the GPUs the schedule starts with, for example 0,1,2,3"),
         arg("--end-time", "when the schedule stops every worker: \"YYYY-MM-DD HH:MM\" in the node's local time, "
                           "for example \"2026-10-29 08:00\", on the round's date or at most 6 days after it, or "
                           "\"now + 45 minutes\" (at least 30) or \"now + 10 hours\" (at most 168) from the start"),
         arg("--environment", "the node's environment the round runs in, as built by node/setup_env.sh, for example "
                              "main or rapids"),
         arg("--profiles", "the profiles installed in that environment, for example core unsupervised (line 02 "
                           "tests the environment against them)", nargs="+"),
         arg("--min-free-gb", "the free disk in GB the round keeps in the home folder, the floor agreed for the "
                              "project: line 03 does not start below it, and the workers stop below it", type=int))
def cmd_round_new(args) -> int:
    from sweepkit.analysis.rounds import plan_path, round_new

    folder = round_new(args.round, args.date, args.question, list(args.sweep_files), args.workers_per_gpu, args.gpus,
                       args.end_time, args.environment, list(args.profiles), args.min_free_gb)
    print(f"made {folder} and the plan {plan_path(args.round, args.date)} (it travels with the code): fill in "
          f"RULE_{args.round}.txt, fix it with rule-fix, then run round-check")
    return 0


@command("round-check", "check a round's preparation folder: RULE fixed before results, lines, plan, README",
         arg("--round", "the round's name"), arg("--date", "the round's date, YYYYMMDD"))
def cmd_round_check(args) -> int:
    from sweepkit.analysis.rounds import render_check, round_check

    report = round_check(args.round, args.date)
    print(render_check(report))
    return 0 if report["ok"] else 1


# Extending ---------------------------------------------------------------------------------------------------------

section("Extending")

@command("scaffold", "write a new component: its module, catalog line, example sweep and test",
         arg("--kind", "clusterer, embedder, detector, estimator, torch_model, prep, metric, split, dataset or "
                       "stage_kind"),
         arg("--name", "its registered name: lowercase letters, digits and _"),
         arg("--where", "project (src/project/, the usual place) or template (the template's own code)",
             choices=["project", "template"]))
def cmd_scaffold(args) -> int:
    from sweepkit.scaffold.scaffold import scaffold

    written = scaffold(args.kind, args.name, args.where)
    for key, path in written.items():
        if key != "module_name":
            print(f"{key}: {path}")
    print("next: edit the module (its docstring says what to change), run its test, then check-additions")
    return 0


@command("check-additions", "every registered component and stage kind has an example sweep that uses it, and each "
                            "(metrics included) a test that names it",
         arg("--show", "how many problems to print", type=int))
def cmd_check_additions(args) -> int:
    from sweepkit.scaffold.additions import check_additions

    report = check_additions()
    for problem in report["problems"][:args.show]:
        print(f"PROBLEM {problem}")
    if len(report["problems"]) > args.show:
        print(f"... and {len(report['problems']) - args.show} more")
    print(f"{report['checked']} components and {report['examples']} example sweeps checked: "
          + ("ADDITIONS OK" if report["ok"] else f"{len(report['problems'])} PROBLEMS"))
    return 0 if report["ok"] else 1


@command("smoke-all", "run every component of a profile once, from its example sweep, on a few rows",
         arg("--profile", "a profile name (core, torch, unsupervised, ...) or all"),
         arg("--device", "what this machine offers: cpu, cuda or mps"),
         arg("--rows", "rows per role, for example 100", type=int))
def cmd_smoke_all(args) -> int:
    from sweepkit.scaffold.additions import smoke_all

    result = smoke_all(args.profile, args.device, args.rows, _log)
    return 0 if result["counts"]["FAILED"] == 0 else 1


@command("make-example-data", "write the small files the example sweeps of file datasets read (data/synthetic/)",
         arg("--confirm", "yes", choices=["yes"]))
def cmd_make_example_data(args) -> int:
    from sweepkit.data.examples import make_example_data

    _print(make_example_data(_log))
    return 0


# Environments and documentation ------------------------------------------------------------------------------------

section("Environments and documentation")

@command("env-check", "installed package versions against the profiles' pins, and the devices this process sees",
         arg("--profiles", "the profiles this environment was built from, for example core torch unsupervised",
             nargs="+"))
def cmd_env_check(args) -> int:
    from sweepkit.env import env_check, render_check

    report = env_check(list(args.profiles))
    print(render_check(report))
    saved = _save(paths.checks_dir(), f"env_{'_'.join(args.profiles)}_{stamp()}", {"report.json": report}, "env-check")
    print(f"saved {saved}")
    return 0 if report["ok"] else 1


@command("required-modules", "the import names an environment built from these profiles must have (for "
                             "SWEEPKIT_REQUIRE)", arg("--profiles", "profiles", nargs="+"))
def cmd_required_modules(args) -> int:
    from sweepkit.env import required_imports

    print(",".join(required_imports(list(args.profiles))))
    return 0


@command("profiles-doc", "write docs/PROFILES.md from the profile table and the registry's declarations",
         arg("--write", "yes to write the file, no to print it", choices=["yes", "no"]))
def cmd_profiles_doc(args) -> int:
    from sweepkit.env import profiles_doc

    text = profiles_doc()
    if args.write == "no":
        print(text)
        return 0
    target = paths.repo_root() / "docs" / "PROFILES.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


@command("commands-doc", "write docs/COMMANDS.md from these commands' own definitions and their examples",
         arg("--write", "yes to write the file, no to print it", choices=["yes", "no"]))
def cmd_commands_doc(args) -> int:
    from sweepkit.commands_doc import render

    text = render()
    if args.write == "no":
        print(text)
        return 0
    target = paths.repo_root() / "docs" / "COMMANDS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


@command("test-matrix", "write the testing matrix, the last section of TESTING.md, from the components, the example "
         "sweeps and the tests",
         arg("--write", "yes to write the section into TESTING.md, no to print it", choices=["yes", "no"]))
def cmd_test_matrix(args) -> int:
    from sweepkit.test_matrix import render, testing_md_with

    text = render()
    if args.write == "no":
        print(text)
        return 0
    target = paths.repo_root() / "TESTING.md"
    target.write_text(testing_md_with(text), encoding="utf-8", newline="\n")
    print(f"wrote the testing matrix into {target}")
    return 0


@command("example-round", "write docs/EXAMPLE_ROUND.md by running two complete rounds in a temporary copy",
         arg("--write", "yes to write the file, no to print it", choices=["yes", "no"]))
def cmd_example_round(args) -> int:
    from sweepkit.example_round import render

    text = render(_log if args.write == "yes" else (lambda message: None))
    if args.write == "no":
        print(text)
        return 0
    target = paths.repo_root() / "docs" / "EXAMPLE_ROUND.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


# Code --------------------------------------------------------------------------------------------------------------

section("Code")

@command("tree-digest", "one SHA-256 over the code tree, the same number node/tree_digest.sh prints",
         arg("--root", "the repository folder, for example ."))
def cmd_tree_digest(args) -> int:
    from sweepkit.hashing import tree_digest

    print(tree_digest(Path(args.root)))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sweepkit.cli", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text, arguments, fn in COMMANDS:
        c = sub.add_parser(name, help=help_text, description=help_text)
        for flag, options in arguments:
            c.add_argument(flag, required=True, **options)
        c.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    from sweepkit.analysis.rounds import RoundError
    from sweepkit.analysis.rules import DisciplineError
    from sweepkit.runs.store import AlreadyComplete
    from sweepkit.settings import SettingsError
    from sweepkit.sweep.design import SweepError

    catalog.load_all()
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except DisciplineError as err:
        print(f"REFUSED {err}", file=sys.stderr)
        return 1
    except AlreadyComplete as err:
        print(f"ERROR {err} exists already: results are written once; give another name", file=sys.stderr)
        return 1
    except (SettingsError, SweepError, RoundError, FileNotFoundError, FileExistsError) as err:
        # a setting, a sweep file or a path to correct: one line rather than a traceback (SWEEPKIT_TRACEBACK=1 shows
        # where it came from)
        if os.environ.get("SWEEPKIT_TRACEBACK") == "1":
            raise
        print(f"ERROR {type(err).__name__}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
