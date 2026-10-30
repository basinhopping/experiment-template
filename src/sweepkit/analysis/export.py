"""export and cluster-report: the results of a recorded selection, to take home and to read.

export writes artifacts/exports/<selection>_<utc>/:
    rows.csv (and rows.parquet with --formats csv+parquet)
                         one line per evaluated row of every exported run: run, fold, seed, row ID and the run's
                         output for it (cluster and its probability and outlier score where the method gives them;
                         embedding coordinates; anomaly score and flag; supervised: the validation prediction)
    sizes.csv            cluster sizes per run (clustering)
    cluster_summary.csv  each cluster's mean, SD, median, minimum and maximum of every input column, in the data's
                         own units (clustering)
    distinguishing.csv   per cluster, the columns that most distinguish it (standardized mean difference), ranked
    contingency.csv      clusters against the reference labels, with --reveal yes only
    plots/*.png          with --plots yes: principal components coloured by cluster, sizes, a heatmap of cluster
                         means, the dendrogram when there is one, the score histogram for outlier detection
    settings.json        the selection record (settings, reason, choosing values)
    packages.json        the package versions of the first exported run
    README.txt           what each file holds
--reveal yes also reads the files that judge the choice (reveal_reference.csv for unsupervised work; the test
predictions for supervised work), so it follows the RULE discipline exactly as reveal-report and test-report do, and
writes reading.json.

cluster-report prints the sizes, the most distinguishing columns and (with --reveal yes) the agreement table, and
saves the same text beside its plots under artifacts/exports/<selection>_report_<utc>/.
"""
import io
import json

import numpy as np

from sweepkit import paths
from sweepkit.analysis import cluster_report as cr
from sweepkit.analysis.records import (complete_runs, configurations, find_selection, is_check_run, is_supervised,
                                       read_json, read_npy, read_rows, sweep_record)
from sweepkit.families.common import csv_text, load_data
from sweepkit.runs.spec import validate_job


def _runs(selection: dict, which: str) -> list:
    record = sweep_record(selection["sweep_id"])
    config = configurations(record, complete_runs(selection["sweep_id"]))[selection["config_hash"]]
    runs = [r for by_fold in config["runs"].values() for r in by_fold.values() if not is_check_run(r["folder"])]
    runs.sort(key=lambda r: (r["spec"]["seed"], r["spec"]["fold"]))
    if which != "all":
        runs = [r for r in runs if r["spec"]["run_id"] == which]
        if not runs:
            raise ValueError(f"no complete run {which!r} of the selected configuration")
    if not runs:
        raise ValueError("the selected configuration has no complete run")
    return runs


def _discipline(selection: dict, reveal: bool) -> tuple:
    if not reveal:
        return [], None
    from sweepkit.analysis.rules import covering_rule, enforce, late_note

    folder, rule = covering_rule(selection["sweep_id"], selection["selected_utc"])
    problems = [] if rule else [f"no RULE covering sweep {selection['sweep_id']} was fixed before selection "
                                f"{selection['name']} was recorded; a RULE covers only choices recorded after it: "
                                "fix one with rule-fix, then record the choice again with select under a new name; "
                                "rule-fix asks for a reason (--late-reason) when runs of the sweep exist already"]
    notes = [late_note(rule)] if rule and late_note(rule) else []
    return enforce(problems, "export --reveal yes") + notes, (folder.name if folder else None)


def _positions(data, ids: list) -> np.ndarray:
    index = {rid: k for k, rid in enumerate(data.table.ids.tolist())}
    return np.array([index[i] for i in ids], dtype=np.int64)


def collect(selection_name: str, which: str, reveal: bool, top: int, plots: bool) -> dict:
    selection = find_selection(selection_name)
    warnings, rule = _discipline(selection, reveal)
    runs = _runs(selection, which)
    job = validate_job(selection["settings"])
    family = job.family
    data = load_data(job, lambda m: None) if job.data_name else None
    names = list(data.feature_names) if data is not None and data.x is not None else []
    rows, header, per_run, pictures = [], None, [], {}
    task = selection["task"]
    for run in runs:
        spec, folder = run["spec"], run["folder"]
        base = [spec["run_id"], spec["fold"], spec["seed"]]
        entry = {"run_id": spec["run_id"], "fold": spec["fold"], "seed": spec["seed"]}
        if family == "clusterer":
            got = read_rows(folder, "assignments.csv", "choose")
            extra = [k for k in got[0] if k not in ("row_id", "cluster", "status")]
            header = header or ["run_id", "fold", "seed", "row_id", "cluster", "status"] + extra
            rows += [base + [g["row_id"], g["cluster"], g["status"]] + [g[k] for k in extra] for g in got]
            kept = [g for g in got if g["status"] != "removed_by_prep"]
            labels = np.array([int(g["cluster"]) for g in kept])
            x = np.asarray(data.x[_positions(data, [g["row_id"] for g in kept])], dtype=np.float64)
            entry.update(cr.summaries(x, names, labels, top))
            if reveal:
                ref = {r["row_id"]: r["reference"] for r in read_rows(folder, "reveal_reference.csv", "judge")}
                reference = np.array([ref.get(g["row_id"], "") for g in kept])
                entry["contingency"] = cr.contingency(labels, reference)
            if plots and run is runs[0]:
                title = f"{selection_name}: {spec['run_id']}"
                pictures["clusters_pca.png"] = cr.plot_pca(x, labels, title)
                pictures["cluster_sizes.png"] = cr.plot_sizes(entry["sizes"], title)
                columns = []
                for d in entry["distinguishing"]:
                    if d["column"] not in columns:
                        columns.append(d["column"])
                if columns and len({lab for lab in labels.tolist() if lab >= 0}) > 0:
                    pictures["cluster_means.png"] = cr.plot_heatmap(x, names, labels, columns[:max(top, 1) * 2], title)
                if (folder / "linkage.npy").is_file():
                    pictures["dendrogram.png"] = cr.plot_dendrogram(read_npy(folder, "linkage.npy", "choose"), title)
        elif family == "embedder":
            ids = [r["row_id"] for r in read_rows(folder, "embedding_rows.csv", "choose")]
            coords = read_npy(folder, "embedding.npy", "choose")
            header = header or ["run_id", "fold", "seed", "row_id"] + [f"c{j}" for j in range(coords.shape[1])]
            rows += [base + [rid] + [f"{v:.6g}" for v in coords[k]] for k, rid in enumerate(ids)]
            entry["rows"] = len(ids)
        elif family == "detector":
            got = read_rows(folder, "scores.csv", "choose")
            header = header or ["run_id", "fold", "seed", "row_id", "score", "flagged", "status"]
            rows += [base + [g["row_id"], g["score"], g["flagged"], g["status"]] for g in got]
            entry["rows"] = len(got)
            if plots and run is runs[0]:
                kept = [g for g in got if g["status"] != "removed_by_prep"]
                pictures["scores.png"] = cr.plot_scores(np.array([float(g["score"]) for g in kept]),
                                                        np.array([g["flagged"] == "1" for g in kept]),
                                                        f"{selection_name}: {spec['run_id']}")
        else:
            got = read_rows(folder, "predictions_val.csv", "choose")
            cols = list(got[0])
            header = header or ["run_id", "fold", "seed", "role"] + cols
            rows += [base + ["val"] + [g[c] for c in cols] for g in got]
            if reveal and is_supervised(task):
                rows += [base + ["test"] + [g[c] for c in cols]
                         for g in read_rows(folder, "predictions_test.csv", "judge")]
            entry["rows"] = len(got)
        per_run.append(entry)
    provenance = read_json(runs[0]["folder"], "provenance.json", "choose")
    return {"selection": selection_name, "sweep_id": selection["sweep_id"], "config_hash": selection["config_hash"],
            "label": selection["label"], "family": family, "warnings": warnings, "rule": rule, "reveal": reveal,
            "runs": per_run, "header": header, "rows": rows, "pictures": pictures, "selection_record": selection,
            "packages": provenance.get("packages", {})}


def files(result: dict, formats: str) -> dict:
    out = {"rows.csv": csv_text(result["header"], result["rows"])}
    if formats == "csv+parquet":
        import pandas as pd

        buffer = io.BytesIO()
        pd.DataFrame(result["rows"], columns=result["header"]).to_parquet(buffer, index=False)
        out["rows.parquet"] = buffer.getvalue()
    if result["family"] == "clusterer":
        for name, key, columns in (("sizes.csv", "sizes", ["cluster", "rows", "share"]),
                                   ("cluster_summary.csv", "summary", ["cluster", "column", "mean", "sd", "median",
                                                                       "min", "max"]),
                                   ("distinguishing.csv", "distinguishing", ["cluster", "rank", "column",
                                                                             "effect_size_d", "mean_in_cluster",
                                                                             "mean_elsewhere"])):
            out[name] = csv_text(["run_id"] + columns, [[r["run_id"]] + [row[c] for c in columns]
                                                        for r in result["runs"] for row in r[key]])
        if result["reveal"]:
            out["contingency.csv"] = csv_text(["run_id", "cluster", "reference", "rows"],
                                              [[r["run_id"], c["cluster"], c["reference"], c["rows"]]
                                               for r in result["runs"] for c in r.get("contingency", [])])
    for name, png in result["pictures"].items():
        out[f"plots/{name}"] = png
    out["settings.json"] = result["selection_record"]
    out["packages.json"] = result["packages"]
    listing = "\n".join(f"  {name}: {_what(name, result['header'])}" for name in sorted(out))
    out["README.txt"] = (f"Export of selection {result['selection']} ({result['sweep_id']} / "
                         f"{result['config_hash'][:12]}, {result['label']}), {len(result['runs'])} runs, "
                         f"{'with' if result['reveal'] else 'without'} the files that judge the choice.\n"
                         + "".join(w + "\n" for w in result["warnings"])
                         + _transductive_line(result["selection_record"])
                         + "Files:\n" + listing + "\n")
    return out


def _what(name: str, header=None) -> str:
    """One line on what an export file holds, for its README (rows.csv with its own columns)."""
    columns = ", ".join(header or [])
    known = {
        "rows.csv": f"one line per evaluated row of every exported run, with the columns {columns}: the run, the row "
                    "and what the run gave it (status says whether a preparation step removed the row)"
                    if "status" in (header or []) else
                    f"one line per evaluated row of every exported run, with the columns {columns}: the run, the row "
                    "and what the run gave it",
        "rows.parquet": "rows.csv as Parquet",
        "sizes.csv": "the rows and share of each cluster, per run",
        "cluster_summary.csv": "mean, SD, median, minimum and maximum of every column within each cluster, per run",
        "distinguishing.csv": "the columns that most set each cluster apart from the other rows (effect size d), per "
                              "run",
        "contingency.csv": "rows of each cluster by reference label (the reveal files; only with --reveal yes)",
        "settings.json": "the selection: the configuration's settings, the reason it was chosen, when",
        "packages.json": "the package versions the runs used",
        "plots/clusters_pca.png": "the first run's rows on their first two principal components, coloured by "
                                  "cluster",
        "plots/cluster_sizes.png": "the size of each cluster in the first run",
        "plots/cluster_means.png": "each cluster's mean of its most distinguishing columns, in standard deviations, "
                                   "in the first run",
        "plots/dendrogram.png": "the tree of the hierarchical clustering of the first run",
        "plots/scores.png": "the outlier scores of the first run, flagged rows marked",
    }
    if name in known:
        return known[name]
    return "a plot" if name.startswith("plots/") else "see the export command in docs/COMMANDS.md"


def _transductive_line(selection: dict) -> str:
    from sweepkit.analysis.transductive import note, parts

    found = parts(selection["settings"])
    return f"NOTE the selected configuration is {note(found)}\n" if found else ""


def report_json(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("rows", "pictures", "header")}


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
