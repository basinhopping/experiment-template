"""docs/COMMANDS.md, written from the command line's own definitions (sweepkit.cli.COMMANDS) and one example per
command, so the reference cannot drift from the code: `python -m sweepkit.cli commands-doc --write yes` rewrites it,
and tests/test_docs.py fails when the file differs from what this writes or when an example does not parse.
"""

INTRO = """# Commands

Written by `python -m sweepkit.cli commands-doc --write yes` from the commands' own definitions; edit
`src/sweepkit/cli.py` or `src/sweepkit/commands_doc.py`, not this file.

Every command is `python -m sweepkit.cli <command> --argument value ...`, run from the repository folder with an
environment's Python (START_HERE.md; it finds the package by itself), or any Python with `PYTHONPATH=src` set. On a
node, `bash node/run.sh <command> ...` runs it with the project's environment and keeps its output in `logs/`. Every
argument is required: no command has a default, so a command line says everything it does. Results go to fixed places under `artifacts/`; no command takes an
output path. Exit codes: 0 passed, 1 failed or refused, 3 not finished (run the same command again).
`python -m sweepkit.cli <command> --help` prints the same arguments.

The order below is the order of the work: sweeps and the queue, choosing then judging, data, results, rounds,
extending, environments, code. docs/CONCEPTS.md explains the ideas; docs/WALKTHROUGH.md runs one result end to end.
"""

# one example per command; tests/test_docs.py parses each with the real parser
EXAMPLES = {
    "check": "check --sweep-file sweeps/examples/cluster_kmeans.json",
    "smoke": "smoke --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --rows 200",
    "enqueue": "enqueue --sweep-file sweeps/examples/cluster_kmeans.json",
    "run": "run --sweep-file sweeps/examples/cluster_kmeans.json --device cpu --max-consecutive-failures 3",
    "worker": "worker --worker-id gpu0_1 --device cuda --min-free-gb 20 --max-consecutive-failures 5",
    "local-workers": "local-workers --n 4 --device cpu --min-free-gb 5 --max-consecutive-failures 5",
    "status": "status --sweep example_cluster_kmeans --top 10 --since all",
    "park": "park --sweep example_cluster_kmeans --name kmeans_later",
    "unpark": "unpark --name kmeans_later",
    "requeue-running": "requeue-running --all-workers-stopped yes",
    "requeue-failed": "requeue-failed --sweep example_cluster_kmeans",
    "analyze": ("analyze --sweep example_cluster_kmeans --confidence 0.95 --n-boot 2000 --seed 1 --meaningful 0.75 "
                "--top 10"),
    "select": ("select --sweep example_cluster_kmeans --config 1a2b3c4d5e --name kmeans_best "
               "--reason \"highest silhouette and stable across seeds\""),
    "rule-fix": ("rule-fix --name round1 --rule-file artifacts/analyses/round1_prep_20261101/RULE_round1.txt --sweeps "
                 "example_cluster_kmeans --families none --late-reason none"),
    "test-report": "test-report --selection logistic_best --confidence 0.95",
    "reveal-report": "reveal-report --selection kmeans_best --confidence 0.95",
    "test-compare": ("test-compare --rule round1_20261028T020000_000000Z --confidence 0.95 --n-boot 2000 --seed 1 "
                     "--meaningful 0.75"),
    "compare-runs": "compare-runs --sweep example_cluster_kmeans --differences-shown 5",
    "atlas": "atlas --format csv",
    "make-synthetic": ("make-synthetic --generator blobs --n-rows 1000000 --n-features 10 --n-groups 5 --noise 1.0 "
                       "--seed 1 --name blobs_1m --chunk-rows 100000 --sample-rows 1000"),
    "convert": ("convert --source cells/cells.csv --name cells --sep comma --row-id cell_id --features \"*\" "
                "--exclude none --keep cell_type batch --chunk-rows 100000 --to store"),
    "extract-features": ("extract-features --data-file sweeps/project/notes_data.json --source hub "
                         "--model-id sentence-transformers/all-MiniLM-L6-v2 "
                         "--revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --pooling mean --max-length 128 "
                         "--batch-size 64 --dtype float32 --device cuda --keep label group --name notes_minilm"),
    "ingest": "ingest --name cells_v1 --source /path/to/the/received/folder --max-seconds 3600",
    "verify-data": "verify-data --name cells_v1",
    "profile-data": ("profile-data --target cells/cells.csv --row-id cell_id --labels none --label-key none --hide "
                     "cell_type "
                     "--max-rows all --top-values 10"),
    "estimate": ("estimate --sweep-file sweeps/project/full_run.json --config base --device cuda --fractions 0.01 "
                 "0.02 0.05 --workers 3"),
    "export": ("export --selection kmeans_best --runs all --reveal no --plots yes --top-features 10 "
               "--formats csv+parquet"),
    "cluster-report": "cluster-report --selection kmeans_best --runs all --reveal no --top-features 5",
    "describe-model": "describe-model --source selection:kmeans_best --config selected",
    "propose": ("propose --sweep round1_screen --stage search --n-configs 20 --seed 1 --n-startup 10 "
                "--name round2_proposed --seeds 1 2 3 --priority 5"),
    "round-new": ("round-new --round round1 --date 20261028 --question \"Which number of clusters is most stable?\" "
                  "--sweep-files sweeps/project/round1_screen.json --workers-per-gpu 3 --gpus 0,1 "
                  "--end-time \"2026-10-29 08:00\" --environment main --profiles core unsupervised --min-free-gb 15"),
    "round-check": "round-check --round round1 --date 20261028",
    "scaffold": "scaffold --kind clusterer --name my_method --where project",
    "check-additions": "check-additions --show 50",
    "smoke-all": "smoke-all --profile core --device cpu --rows 100",
    "make-example-data": "make-example-data --confirm yes",
    "env-check": "env-check --profiles core torch unsupervised",
    "required-modules": "required-modules --profiles core torch unsupervised",
    "profiles-doc": "profiles-doc --write yes",
    "commands-doc": "commands-doc --write yes",
    "test-matrix": "test-matrix --write yes",
    "example-round": "example-round --write yes",
    "tree-digest": "tree-digest --root .",
}


def _what(options: dict) -> str:
    parts = [options.get("help", "")]
    if options.get("choices"):
        parts.append("one of " + ", ".join(f"`{c}`" for c in options["choices"]))
    if options.get("nargs"):
        parts.append("one or more values")
    kind = options.get("type")
    if kind is int:
        parts.append("a whole number")
    elif kind is float:
        parts.append("a number")
    return "; ".join(p for p in parts if p)


def render() -> str:
    from sweepkit import cli

    lines = [INTRO.rstrip("\n"), ""]
    section = None
    for name, help_text, arguments, _ in cli.COMMANDS:
        here = cli.SECTION_OF[name]
        if here != section:
            section = here
            lines += [f"## {section}", ""]
        lines += [f"### {name}", "", help_text[0].upper() + help_text[1:] + ".", ""]
        if arguments:
            lines += ["| Argument | What it is |", "|---|---|"]
            lines += [f"| `{flag}` | {_what(options)} |" for flag, options in arguments]
            lines.append("")
        lines += ["```", f"python -m sweepkit.cli {EXAMPLES[name]}", "```", ""]
    return "\n".join(lines).rstrip("\n") + "\n"
