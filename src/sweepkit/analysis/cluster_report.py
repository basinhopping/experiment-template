"""What a clustering found, in the data's own units: sizes, each cluster's summary, the features that most
distinguish it, agreement tables with known labels (only after a selection, under the RULE discipline), and plots.

Used by export (the files to take home) and cluster-report (the printed summary). Every number comes from the
choosing files of the selected runs (assignments.csv) and the dataset's input columns as the dataset gives them,
before any preparation step, so a summary reads in the units the data were measured in.

Distinguishing features: for each cluster and column, the standardized mean difference between the cluster and the
other evaluated rows (the difference of means over the pooled standard deviation, Cohen's d), ranked by its size.
Noise rows (cluster -1) are summarized as their own group and are not a cluster.

Plots (matplotlib, written as PNG): the evaluated rows on their first two principal components (of the standardized
columns), coloured by cluster, which is a picture for looking and not a measure of anything; cluster sizes; a
heatmap of the standardized cluster means over the most distinguishing columns; the dendrogram when the method wrote
a linkage matrix; and, for outlier detection, the histogram of scores.
"""
import io

import numpy as np


def summaries(x: np.ndarray, names: list, labels: np.ndarray, top: int) -> dict:
    """{"sizes": rows, "summary": rows, "distinguishing": rows} for one run."""
    x = np.asarray(x, dtype=np.float64)
    groups = sorted(set(labels.tolist()))
    sizes, summary, distinguishing = [], [], []
    n = labels.size
    for g in groups:
        member = labels == g
        sizes.append({"cluster": int(g), "rows": int(member.sum()), "share": float(member.mean())})
        inside, outside = x[member], x[~member]
        for j, name in enumerate(names):
            col = inside[:, j]
            summary.append({"cluster": int(g), "column": name, "mean": float(np.mean(col)),
                            "sd": float(np.std(col, ddof=1)) if col.size > 1 else None,
                            "median": float(np.median(col)), "min": float(np.min(col)), "max": float(np.max(col))})
        if g < 0 or outside.shape[0] < 2 or inside.shape[0] < 2:
            continue
        d = []
        for j, name in enumerate(names):
            a, b = inside[:, j], outside[:, j]
            pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
            d.append((float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0, name, float(a.mean()),
                      float(b.mean())))
        d.sort(key=lambda t: -abs(t[0]))
        for rank, (effect, name, mean_in, mean_out) in enumerate(d[:top], 1):
            distinguishing.append({"cluster": int(g), "rank": rank, "column": name, "effect_size_d": effect,
                                   "mean_in_cluster": mean_in, "mean_elsewhere": mean_out})
    return {"sizes": sizes, "summary": summary, "distinguishing": distinguishing, "rows": int(n)}


def contingency(labels: np.ndarray, reference: np.ndarray) -> list:
    """Rows of the cluster by reference-label table (counts), for the rows that have a reference label."""
    keep = reference != ""
    out = []
    for g in sorted(set(labels[keep].tolist())):
        for r in sorted(set(reference[keep].tolist())):
            count = int(((labels == g) & (reference == r) & keep).sum())
            if count:
                out.append({"cluster": int(g), "reference": r, "rows": count})
    return out


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    return buffer.getvalue()


def plot_pca(x: np.ndarray, labels: np.ndarray, title: str) -> bytes:
    plt = _figure()
    z = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    u, s, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    coords = u[:, :2] * s[:2] if z.shape[1] >= 2 else np.column_stack([u[:, 0] * s[0], np.zeros(z.shape[0])])
    fig, ax = plt.subplots(figsize=(6, 5))
    for g in sorted(set(labels.tolist())):
        member = labels == g
        ax.scatter(coords[member, 0], coords[member, 1], s=6, alpha=0.7,
                   label="noise" if g < 0 else f"cluster {g}", color="lightgrey" if g < 0 else None)
    ax.set_xlabel("first principal component (standardized columns)")
    ax.set_ylabel("second principal component")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=7, loc="best")
    data = _png(fig)
    plt.close(fig)
    return data


def plot_sizes(sizes: list, title: str) -> bytes:
    plt = _figure()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ["noise" if r["cluster"] < 0 else str(r["cluster"]) for r in sizes]
    ax.bar(names, [r["rows"] for r in sizes], color="steelblue")
    ax.set_xlabel("cluster")
    ax.set_ylabel("rows")
    ax.set_title(title)
    data = _png(fig)
    plt.close(fig)
    return data


def plot_heatmap(x: np.ndarray, names: list, labels: np.ndarray, columns: list, title: str) -> bytes:
    plt = _figure()
    z = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    idx = [names.index(c) for c in columns]
    groups = [g for g in sorted(set(labels.tolist())) if g >= 0]
    means = np.array([z[labels == g][:, idx].mean(axis=0) for g in groups])
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(idx) + 2), max(3, 0.4 * len(groups) + 1.5)))
    image = ax.imshow(means, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(idx)), columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(groups)), [f"cluster {g}" for g in groups], fontsize=7)
    fig.colorbar(image, ax=ax, label="mean in standard deviations")
    ax.set_title(title)
    data = _png(fig)
    plt.close(fig)
    return data


def plot_dendrogram(linkage: np.ndarray, title: str) -> bytes:
    plt = _figure()
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(7, 4))
    dendrogram(linkage, ax=ax, truncate_mode="lastp", p=30, no_labels=True, color_threshold=None)
    ax.set_ylabel("merge distance")
    ax.set_title(title)
    data = _png(fig)
    plt.close(fig)
    return data


def plot_scores(scores: np.ndarray, flagged: np.ndarray, title: str) -> bytes:
    plt = _figure()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores[~flagged], bins=50, alpha=0.7, label="not flagged")
    if flagged.any():
        ax.hist(scores[flagged], bins=50, alpha=0.7, label="flagged", color="firebrick")
    ax.set_xlabel("anomaly score (higher is more unusual)")
    ax.set_ylabel("rows")
    ax.legend(fontsize=7)
    ax.set_title(title)
    data = _png(fig)
    plt.close(fig)
    return data


def render(report: dict, top: int) -> str:
    lines = [f"cluster report of selection {report['selection']} ({report['sweep_id']} / {report['config_hash'][:12]}, "
             f"{report['label']})"]
    if report.get("warnings"):
        lines = list(report["warnings"]) + lines
    for run in report["runs"]:
        lines.append(f"run {run['run_id']}: {run['rows']} evaluated rows")
        lines.append("  sizes: " + ", ".join(("noise" if s["cluster"] < 0 else f"cluster {s['cluster']}")
                                             + f" {s['rows']} ({s['share']:.1%})" for s in run["sizes"]))
        for g in sorted({d["cluster"] for d in run["distinguishing"]}):
            best = [d for d in run["distinguishing"] if d["cluster"] == g][:top]
            lines.append(f"  cluster {g}: " + "; ".join(f"{d['column']} d {d['effect_size_d']:+.2f} "
                                                        f"({d['mean_in_cluster']:.3g} against {d['mean_elsewhere']:.3g})"
                                                        for d in best))
        if run.get("contingency"):
            lines.append("  against the reference labels (rows): " + "; ".join(
                f"{c['cluster']}/{c['reference']} {c['rows']}" for c in run["contingency"][:20]))
    return "\n".join(lines)
