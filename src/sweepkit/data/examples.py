"""make-example-data: the small files the example sweeps of the file-based datasets read, made by code under
data/synthetic/ (the template ships no data). Run once; what exists is left alone, so running it again is harmless.

    data/synthetic/example_table.csv        mixed numeric and categorical columns, planted clusters (truth), groups
    data/stores/example_store/              the convert store of data/synthetic/example_blobs.csv
    data/synthetic/example_items/           one .npy sequence per row and index.csv (row_id, path, truth)
    data/synthetic/example_graph/           nodes.csv (row_id, features, truth) and edges.csv (source, target)
    data/synthetic/example_distances.csv    a precomputed distance matrix (Bray-Curtis, one column per row ID), truth
    data/synthetic/example_cells.h5ad       planted cell types as counts, when anndata is installed
    each registered dataset's EXAMPLE_FILE  written by its write_example (scaffold's dataset template has both)

smoke-all calls it before the examples that need these files; docs/WALKTHROUGH.md uses them too.
"""
import csv

import numpy as np

from sweepkit import paths
from sweepkit.data.synthetic import make


def _table(name: str, generator: str, n_rows: int, n_features: int, n_groups: int, log) -> None:
    from sweepkit.data.make import write_synthetic

    if not (paths.synthetic_dir() / f"{name}.csv").is_file():
        write_synthetic(generator, n_rows, n_features, n_groups, 0.8, 1, name, n_rows, max(10, n_rows // 10), log)


def make_example_data(log) -> dict:
    made = []
    _table("example_table", "mixed", 400, 6, 3, log)
    _table("example_blobs", "blobs", 2000, 5, 3, log)
    made += ["synthetic/example_table.csv", "synthetic/example_blobs.csv"]
    if not (paths.data_dir() / "stores" / "example_store" / "record.json").is_file():
        from sweepkit.data.convert import convert

        convert("synthetic/example_blobs.csv", "example_store", ",", "row_id", ("*",), (),
                ("truth", "group", "published_split"), 1000, "store", log)
    made.append("stores/example_store")
    items = paths.synthetic_dir() / "example_items"
    if not (items / "index.csv").is_file():
        items.mkdir(parents=True, exist_ok=True)
        out = make("sequences", 120, 3, 3, 0.3, 1)
        rows = []
        for i, seq in enumerate(out["items"]):
            path = items / f"r{i:03d}.npy"
            np.save(path, np.asarray(seq, dtype=np.float32), allow_pickle=False)
            rows.append([f"r{i:03d}", f"synthetic/example_items/r{i:03d}.npy", out["columns"]["truth"][i]])
        with open(items / "index.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_id", "path", "truth"])
            writer.writerows(rows)
    made.append("synthetic/example_items")
    graph = paths.synthetic_dir() / "example_graph"
    if not (graph / "edges.csv").is_file():
        graph.mkdir(parents=True, exist_ok=True)
        out = make("graph", 150, 4, 3, 0.5, 1)
        x, truth, edges = out["x"], out["columns"]["truth"], out["edges"]
        with open(graph / "nodes.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_id"] + [f"f{j}" for j in range(x.shape[1])] + ["truth"])
            for i in range(x.shape[0]):
                writer.writerow([f"n{i:03d}"] + [f"{v:.7g}" for v in x[i]] + [truth[i]])
        with open(graph / "edges.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source", "target"])
            writer.writerows([f"n{a:03d}", f"n{b:03d}"] for a, b in zip(edges[0], edges[1]))
    made.append("synthetic/example_graph")
    distances = paths.synthetic_dir() / "example_distances.csv"
    if not distances.is_file():
        # a precomputed distance matrix: Bray-Curtis between 200 rows of counts with 3 planted groups, one column per
        # row ID, and each row's group in truth
        from scipy.spatial.distance import cdist

        out = make("counts", 200, 30, 3, 0.5, 1)
        d = cdist(out["x"], out["x"], metric="braycurtis")
        ids = [f"s{i:03d}" for i in range(d.shape[0])]
        with open(distances, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_id"] + ids + ["truth"])
            for i, rid in enumerate(ids):
                writer.writerow([rid] + [f"{v:.8g}" for v in d[i]] + [out["columns"]["truth"][i]])
    made.append("synthetic/example_distances.csv")
    cells = paths.synthetic_dir() / "example_cells.h5ad"
    try:
        import anndata
    except ImportError:
        log("anndata is not installed: data/synthetic/example_cells.h5ad not made (the singlecell profile has it)")
    else:
        if not cells.is_file():
            out = make("counts", 600, 200, 4, 1.0, 1)
            import pandas as pd

            obs = pd.DataFrame({"truth": out["columns"]["truth"]}, index=[f"c{i:04d}" for i in range(600)])
            var = pd.DataFrame(index=[f"g{j:03d}" for j in range(out["x"].shape[1])])
            anndata.AnnData(X=np.asarray(out["x"], dtype=np.float32), obs=obs, var=var).write_h5ad(cells)
        made.append("synthetic/example_cells.h5ad")
    made += _dataset_examples(log)
    return {"made_or_present": made}


def _dataset_examples(log) -> list:
    """Files that registered datasets ask for (a class attribute EXAMPLE_FILE, relative to data/, and a staticmethod
    write_example(path)), as scaffold's dataset template does: written when missing."""
    from sweepkit import catalog
    from sweepkit.registry import DATASETS

    catalog.load_all()
    made = []
    for entry in DATASETS.entries():
        name = getattr(entry.target, "EXAMPLE_FILE", None)
        write = getattr(entry.target, "write_example", None)
        if name is None or write is None:
            continue
        path = paths.data_path(name)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path)
            log(f"dataset {entry.name}: wrote its example file data/{name}")
        made.append(name)
    return made
