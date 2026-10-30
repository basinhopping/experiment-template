"""A stand-in for a person's own script, for TESTING.md's laptop-to-GPU check. Not a test.

It does what such scripts usually do: read a CSV with pandas, take the f columns, run the hdbscan package's HDBSCAN
with a few settings written out, and write row_id,label. The check runs it unchanged on the CPU, then on the GPU
through node/rapids_run.sh (cuML's zero-code-change mode), and compares the labels with the template's version
(sweeps/checks/laptop_hdbscan.json):

    python tests/laptop_standin.py <rows.csv> <labels.csv>
    python tests/laptop_standin.py --compare <labels_a.csv> <labels_b.csv> [<rows.csv with a truth column>]

--compare prints the adjusted Rand index between two label files on the rows they share, and each file's against the
truth column when a rows file is given. A label file may also be a run's assignments.csv (row_id and cluster).
"""
import sys
import time

import pandas as pd


def cluster(rows: str, out: str) -> None:
    import hdbscan

    frame = pd.read_csv(rows)
    x = frame[[c for c in frame.columns if c.startswith("f")]].to_numpy()
    start = time.time()
    labels = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=10, metric="euclidean",
                             cluster_selection_method="eom").fit_predict(x)
    seconds = time.time() - start
    labels = pd.Series(labels).to_numpy()
    found = len(set(labels.tolist()) - {-1})
    print(f"{x.shape[0]} rows, {x.shape[1]} columns: {found} clusters, noise fraction {float((labels == -1).mean()):.4f}, "
          f"HDBSCAN {seconds:.1f} s")
    pd.DataFrame({"row_id": frame["row_id"].to_numpy(), "label": labels}).to_csv(out, index=False)
    print(f"wrote {out}")


def labels_of(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    column = "label" if "label" in frame.columns else "cluster"
    return frame[["row_id", column]].rename(columns={column: "label"})


def compare(a: str, b: str, rows=None) -> None:
    from sklearn.metrics import adjusted_rand_score

    left, right = labels_of(a), labels_of(b)
    both = left.merge(right, on="row_id", suffixes=("_a", "_b"))
    print(f"{len(both)} rows in both: adjusted Rand index {adjusted_rand_score(both.label_a, both.label_b):.6f}, "
          f"identical labels {bool((both.label_a == both.label_b).all())}")
    if rows:
        truth = pd.read_csv(rows, usecols=["row_id", "truth"])
        for name, labels in ((a, left), (b, right)):
            with_truth = labels.merge(truth, on="row_id")
            print(f"{name} against the truth: adjusted Rand index "
                  f"{adjusted_rand_score(with_truth.truth, with_truth.label):.6f} on {len(with_truth)} rows")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        compare(*sys.argv[2:5])
    elif len(sys.argv) == 3:
        cluster(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
