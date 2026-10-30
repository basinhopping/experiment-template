"""analyze: compare the configurations of a finished sweep on the files that choose, never on a test or reveal file.

Reads the sweep's record (artifacts/sweeps/<sweep_id>/) and the choosing files of its complete runs
(sweepkit.analysis.records lists them). The sweep's task fixes what chooses: a supervised task's select metric
(from metrics_val.json), an unsupervised task's label-free criterion (from criteria.json), or stability (computed
here from the assignment, embedding or score files).

For every configuration:
  - the fold mean of the choosing value for each seed with every fold complete, and a t interval over seeds;
  - the mean over seeds of every other recorded metric or criterion (None where some run has none);
  - supervised: the choosing metric within each group of the task's group columns;
  - unsupervised: stability across seeds and across folds (sweepkit.analysis.stability).
For every configuration that can be compared with the sweep's baseline (the stage of kind base):
  - supervised: on the seeds both have (at least two), the per-seed differences with a t interval, and a
    hierarchical paired bootstrap of the difference in the fold-mean metric (seeds resampled per model, rows
    resampled within each fold and shared by both); a metric without row weights falls back to the seed bootstrap;
  - label-free criterion: the folds (paired) and each configuration's seeds resampled (folds_and_seeds_bootstrap):
    two or more folds with one seed each, or one fold with two or more seeds, so a deterministic method (the same
    result for every seed) is compared on one seed over its subsamples;
  - stability: a bootstrap over the folds both have (at least two), each fold's mean over its seed pairs paired;
  - P(configuration beats baseline) with a bootstrap interval: over the seed values (supervised), fold by fold
    (label-free criteria), or over the folds (stability);
  - Holm's adjustment of the bootstrap p-values across every compared configuration, and a verdict in the
    metric's direction: better or worse when the adjusted p is below 1 - confidence, else "no detectable
    difference".
Also: for random stages, every factor level's mean choosing value over the configurations drawn with it; and
groups of configurations whose output files are byte-identical (they built the same model).
Controls (fit.control other than none) are listed apart and never ranked or compared.
"""
import numpy as np

from sweepkit.analysis.records import (OUTPUT_FILE, choosing_body, choosing_value, complete_runs, complete_seeds,
                                       configurations, file_path, fold_means, is_supervised, result_folds,
                                       sweep_record, task_of)
from sweepkit.analysis.stability import stability
from sweepkit.analysis.transductive import note as transductive_note
from sweepkit.analysis.transductive import parts as transductive_parts
from sweepkit.eval.stats import (NoVariation, UndefinedInDraw, folds_and_seeds_bootstrap, holm, paired_bootstrap,
                                 paired_units_bootstrap, probability_of_outperforming_by_fold,
                                 probability_of_outperforming_interval, seed_bootstrap, t_interval, verdict)
from sweepkit.hashing import canonical_json, sha256_file
from sweepkit.registry import METRICS
from sweepkit.splits.base import CHECK_FOLD
from sweepkit.sweep.design import get_path, label_of, levels_of
from sweepkit.tasks.unsupervised import STABILITY


def metric_names(task_name: str, task: dict) -> list:
    return list(task["metrics"]) if is_supervised(task_name) else list(task["criteria"])


def select_of(task: dict) -> tuple:
    """(choosing metric, its direction)."""
    name = task["select"]
    return name, ("max" if name == STABILITY else METRICS.get(name)["direction"])


def _mean_or_none(values: dict):
    v = list(values.values())
    return None if not v or any(x is None for x in v) else float(np.mean(v))


def _group_values(config: dict, folds: list, task_name: str, select: str, columns: list) -> dict:
    """{column: {group: mean over seeds of the fold-mean metric within the group}}; None where undefined."""
    out = {}
    seeds = complete_seeds(config, folds)
    for column in columns:
        per_group = {}
        for seed in seeds:
            for fold in folds:
                body = choosing_body(config["runs"][seed][fold], task_name)
                for group, entry in body["by_group"].get(column, {}).items():
                    value = None if entry["metrics"] is None else entry["metrics"][select]
                    per_group.setdefault(group, {}).setdefault(seed, []).append(value)
        out[column] = {}
        for group, by_seed in sorted(per_group.items()):
            means = []
            for seed in seeds:
                values = by_seed.get(seed, [])
                means.append(None if len(values) != len(folds) or None in values else float(np.mean(values)))
            out[column][group] = None if not means or None in means else float(np.mean(means))
    return out


def summarise(config: dict, folds: list, task_name: str, task: dict, confidence: float) -> dict:
    select, direction = select_of(task)
    seeds = complete_seeds(config, folds)
    by_metric = {m: fold_means(config, folds, lambda r, m=m: choosing_value(r, task_name, m))
                 for m in metric_names(task_name, task)}
    out = {"config_hash": config["config_hash"], "stage": config["stage"], "kind": config["kind"],
           "label": config["label"], "control": config["control"], "family": config["family"], "seeds": seeds,
           "transductive": transductive_parts(config["settings"]),
           "n_runs": sum(len(v) for v in config["runs"].values()),
           "metrics": {m: _mean_or_none(v) for m, v in by_metric.items()}}
    if not is_supervised(task_name) and seeds:
        out["stability"] = stability(config, folds, task_name, task)
    if select == STABILITY:
        out["value_by_seed"] = None
        out["value"] = out.get("stability", {}).get("value")
        out["interval"] = None
    else:
        out["value_by_seed"] = by_metric[select]
        out["value"] = _mean_or_none(by_metric[select]) if seeds else None
        defined = [v for v in by_metric[select].values() if v is not None]
        out["interval"] = (t_interval(defined, confidence)
                           if len(defined) >= 2 and len(defined) == len(by_metric[select]) else None)
    if is_supervised(task_name) and task.get("groups") and seeds:
        out["by_group"] = _group_values(config, folds, task_name, select, list(task["groups"]))
    out["select"] = {"metric": select, "direction": direction}
    return out


def _fold_arrays(config: dict, seeds: list, folds: list, task_name: str) -> list:
    """Per fold: the validation rows, their truth, and every seed's predictions and scores (paired bootstrap)."""
    from sweepkit.families.supervised_io import read_predictions

    out = []
    for fold in folds:
        reads = [read_predictions(file_path(config["runs"][s][fold]["folder"], "predictions_val.csv", "choose"),
                                  task_name) for s in seeds]
        first = reads[0]
        for r in reads[1:]:
            if r["row_ids"] != first["row_ids"] or not np.array_equal(r["y_true"], first["y_true"]):
                raise ValueError(f"{config['config_hash'][:12]} {fold}: seeds disagree on the validation rows")
        scores = None
        if all(r["scores"] is not None for r in reads):
            scores = np.stack([r["scores"] for r in reads])
        out.append({"row_ids": first["row_ids"], "y_true": first["y_true"],
                    "y_pred": np.stack([r["y_pred"] for r in reads]), "scores": scores,
                    "scores_kind": first["scores_kind"], "names": first["names"]})
    return out


def _fold_seed_values(config: dict, folds: list, task_name: str, select: str) -> np.ndarray:
    """[folds, complete seeds] of the choosing value (label-free criteria)."""
    seeds = complete_seeds(config, folds)
    values = [[choosing_value(config["runs"][s][f], task_name, select) for s in seeds] for f in folds]
    if any(v is None for row in values for v in row):
        raise NoVariation(f"{select} is undefined in some run")
    return np.asarray(values, dtype=float)


def versus(config: dict, base: dict, summaries: dict, folds: list, task_name: str, task: dict, confidence: float,
           n_boot: int, meaningful: float, rng) -> dict | None:
    """The comparison with the baseline, or None (with the reason in the summary) when it cannot be made."""
    select, direction = select_of(task)
    a, b = summaries[config["config_hash"]], summaries[base["config_hash"]]
    if select == STABILITY:
        fa = a.get("stability", {}).get("across_seeds", {}).get("by_fold", {})
        fb = b.get("stability", {}).get("across_seeds", {}).get("by_fold", {})
        shared = sorted(set(fa) & set(fb))
        if len(shared) < 2:
            a["not_compared"] = ("stability is compared fold by fold (each fold's mean over its seed pairs), and "
                                 f"this configuration and the baseline share {len(shared)} such folds; it needs 2")
            return None
        boot = paired_units_bootstrap(fa, fb, n_boot, confidence, rng, direction, "fold")
        poo = probability_of_outperforming_interval([fa[f] for f in shared], [fb[f] for f in shared], n_boot,
                                                    confidence, rng, meaningful, direction)
        return {"units": "folds", "folds": shared, "bootstrap": boot, "p_better": poo}
    if not is_supervised(task_name):
        return _versus_label_free(config, base, a, folds, task_name, select, direction, confidence, n_boot,
                                  meaningful, rng)
    va, vb = a["value_by_seed"], b["value_by_seed"]
    seeds = sorted(s for s in set(va) & set(vb) if va[s] is not None and vb[s] is not None)
    if len(seeds) < 2:
        a["not_compared"] = f"shares {len(seeds)} seeds with defined values with the baseline; a comparison needs 2"
        return None
    diffs = [va[s] - vb[s] for s in seeds]
    metric = METRICS.get(select)
    if metric["weighted"]:
        # 250 draws are scored at once (memory); the chunk size fixes the order of the random draws, so it stays 250
        try:
            boot = paired_bootstrap(_fold_arrays(config, seeds, folds, task_name),
                                    _fold_arrays(base, seeds, folds, task_name), metric, n_boot, confidence, rng, 250)
        except UndefinedInDraw as err:
            a["not_compared"] = str(err)
            return None
    else:
        boot = seed_bootstrap({s: va[s] for s in seeds}, {s: vb[s] for s in seeds}, n_boot, confidence, rng, direction)
    poo = probability_of_outperforming_interval([va[s] for s in seeds], [vb[s] for s in seeds], n_boot, confidence,
                                                rng, meaningful, direction)
    return {"units": "seeds", "seeds": seeds, "difference_by_seed": dict(zip(seeds, diffs)),
            "seed_interval": t_interval(diffs, confidence), "bootstrap": boot, "p_better": poo}


def _versus_label_free(config: dict, base: dict, a: dict, folds: list, task_name: str, select: str, direction: str,
                       confidence: float, n_boot: int, meaningful: float, rng) -> dict | None:
    """A label-free criterion against the baseline: the folds (paired) and each configuration's seeds resampled. Two
    folds with one seed each are enough, so a deterministic method needs no second seed; one fold needs two seeds."""
    try:
        fa = _fold_seed_values(config, folds, task_name, select)
        fb = _fold_seed_values(base, folds, task_name, select)
    except NoVariation as err:
        a["not_compared"] = str(err)
        return None
    if len(folds) * min(fa.shape[1], fb.shape[1]) < 2:
        a["not_compared"] = ("one fold and one seed: a label-free comparison needs two or more folds (subsamples), or "
                             "two seeds of one fold")
        return None
    try:
        boot = folds_and_seeds_bootstrap(fa, fb, n_boot, confidence, rng, direction)
    except NoVariation as err:
        a["not_compared"] = str(err)
        return None
    poo = probability_of_outperforming_by_fold(fa, fb, n_boot, confidence, rng, meaningful, direction)
    by_fold = fa.mean(axis=1) - fb.mean(axis=1)
    return {"units": "folds and seeds", "seeds": complete_seeds(config, folds),
            "seeds_baseline": complete_seeds(base, folds),
            "difference_by_fold": dict(zip(folds, by_fold.tolist())), "bootstrap": boot, "p_better": poo}


def factor_levels(sweep_file: dict, configs: dict, summaries: dict, direction: str) -> list:
    """For each random stage: every factor's levels with the mean choosing value of the configurations drawn with it."""
    out = []
    for stage in sweep_file["stages"]:
        if stage["kind"] != "random":
            continue
        members = [c for c in configs.values() if c["stage"] == stage["name"]]
        for factor, raw_levels in stage["space"].items():
            rows = []
            for level in levels_of(factor, raw_levels):
                def matches(c, level=level):
                    return all(canonical_json(get_path(c["settings"], k)) == canonical_json(v) for k, v in level.items())
                values = [summaries[c["config_hash"]]["value"] for c in members if matches(c)]
                values = [v for v in values if v is not None]
                rows.append({"level": label_of(level), "n_configs": len(values),
                             "mean": float(np.mean(values)) if values else None})
            means = [r["mean"] for r in rows if r["mean"] is not None]
            best = (max if direction == "max" else min)(means) if means else None
            out.append({"stage": stage["name"], "factor": factor, "levels": rows, "best_level_mean": best,
                        "spread": float(max(means) - min(means)) if len(means) > 1 else 0.0})
    out.sort(key=lambda e: -e["spread"])
    return out


def identical_outputs(configs: dict, folds: list) -> list:
    """Groups of configurations whose output files are byte-identical on every (fold, seed) they share in full."""
    digests = {}
    for h, c in configs.items():
        seeds = complete_seeds(c, folds)
        if not seeds:
            continue
        name = OUTPUT_FILE[c["family"]]
        parts = []
        for seed in seeds:
            for fold in folds:
                _, digest = sha256_file(file_path(c["runs"][seed][fold]["folder"], name, "choose"))
                parts.append(f"{fold}|{seed}|{digest}")
        digests.setdefault("\n".join(parts), []).append(h)
    return [sorted(group) for group in digests.values() if len(group) > 1]


def analyze(sweep_id: str, confidence: float, n_boot: int, seed: int, meaningful: float) -> dict:
    if not 0 < confidence < 1 or n_boot < 100 or not 0.5 < meaningful < 1:
        raise ValueError("confidence in (0, 1), n_boot at least 100, meaningful in (0.5, 1)")
    record = sweep_record(sweep_id)
    sweep_file = record["sweep_file"]
    folds = result_folds(sweep_file["folds"])
    if not folds:
        raise ValueError(f"{sweep_id} runs only the {CHECK_FOLD} check fold, which no choosing step reads")
    task_name, task = task_of(sweep_file)
    select, direction = select_of(task)
    runs = complete_runs(sweep_id)
    configs = configurations(record, runs)
    base_hashes = [h for h, c in configs.items() if c["kind"] == "base"]
    if len(base_hashes) > 1:
        raise ValueError(f"{sweep_id}: expected at most one baseline configuration, found {len(base_hashes)}")
    base = configs[base_hashes[0]] if base_hashes else None
    rng = np.random.default_rng(seed)
    summaries = {h: summarise(c, folds, task_name, task, confidence) for h, c in configs.items()}
    compared = []
    for h, c in configs.items():
        if base is None or h == base["config_hash"] or c["control"] != "none" or summaries[h]["value"] is None:
            continue
        if summaries[base["config_hash"]]["value"] is None:
            continue
        result = versus(c, base, summaries, folds, task_name, task, confidence, n_boot, meaningful, rng)
        if result is not None:
            compared.append((h, result))
    if compared:
        adjusted = holm([v["bootstrap"]["p_two_sided"] for _, v in compared])
        for (h, v), p in zip(compared, adjusted):
            v["holm_p"] = p
            v["verdict"] = verdict(v["bootstrap"]["difference"], p, 1 - confidence, direction)
            summaries[h]["versus_baseline"] = v
    candidates = [s for s in summaries.values() if s["control"] == "none" and s["seeds"]]
    sign = -1 if direction == "max" else 1
    ranked = sorted((s for s in candidates if s["value"] is not None), key=lambda s: sign * s["value"])
    undefined = [s for s in candidates if s["value"] is None]
    expected = {row["config_hash"] for row in record["jobs"]}
    return {"sweep_id": sweep_id, "task": task_name, "supervised": is_supervised(task_name),
            "select": {"metric": select, "direction": direction}, "folds": folds,
            "check_folds": [f for f in sweep_file["folds"] if f == CHECK_FOLD], "confidence": confidence,
            "n_boot": n_boot, "seed": seed, "meaningful": meaningful, "n_runs": len(runs),
            "n_expected_runs": len(record["jobs"]), "n_configs": len(configs), "n_expected_configs": len(expected),
            "baseline": base["config_hash"] if base else None, "n_compared": len(compared), "ranked": ranked,
            "undefined": undefined, "incomplete": [h for h, s in summaries.items() if not s["seeds"]],
            "controls": [s for s in summaries.values() if s["control"] != "none"],
            "factors": factor_levels(sweep_file, configs, summaries, direction),
            "identical_outputs": identical_outputs(configs, folds)}


def _value(v, digits: int) -> str:
    return "undefined" if v is None else f"{v:.{digits}f}"


def render(report: dict, top: int) -> str:
    sel = report["select"]
    better = "larger" if sel["direction"] == "max" else "smaller"
    lines = [f"sweep {report['sweep_id']} ({report['task']}): {report['n_configs']} of {report['n_expected_configs']} "
             f"configurations, {report['n_runs']} of {report['n_expected_runs']} runs complete; chosen by "
             f"{sel['metric']} ({better} is better); {'validation files' if report['supervised'] else 'label-free files'}"
             f" only; {round(report['confidence'] * 100)}% intervals, {report['n_boot']} bootstrap draws"]
    if report["n_runs"] < report["n_expected_runs"]:
        lines.append(f"NOTE {report['n_expected_runs'] - report['n_runs']} runs are not complete (failed or not run "
                     "yet); seeds missing a fold are left out")
    if not report["supervised"] and len(report["folds"]) >= 2:
        bound = 2 / 2 ** len(report["folds"])
        lines.append(f"NOTE label-free comparisons take the larger of the bootstrap's p and a sign-flip test's over the "
                     f"{len(report['folds'])} folds, so no p here falls below {bound:.4f}"
                     + ("; a difference needs about 8 or more subsamples to be detectable" if bound >= 0.01 else ""))
    for fold in report.get("check_folds", []):
        lines.append(f"NOTE the {fold} fold validates on test rows: its runs are a check of the evaluation paths, "
                     "never read here; test-report shows whether its validation and test numbers agree")
    for group in report["identical_outputs"]:
        lines.append("NOTE byte-identical outputs (the same model was built): " + ", ".join(h[:12] for h in group))
    ranked = report["ranked"]
    base = next((s for s in ranked + report["undefined"] if s["config_hash"] == report["baseline"]), None)
    if base is None:
        lines.append("no baseline in this sweep (no stage of kind base): configurations are ranked, not compared")
    else:
        bi = base["interval"]
        lines.append(f"baseline {base['config_hash'][:12]}: seeds {base['seeds']}, {sel['metric']} "
                     f"{_value(base['value'], 4)}" + (f" [{bi['low']:.4f}, {bi['high']:.4f}], SD {bi['sd']:.4f}"
                                                      if bi else ""))
    for s in report["controls"]:
        lines.append(f"control {s['label']}: {sel['metric']} {_value(s['value'], 4)} over seeds {s['seeds']}")
    lines.append(f"top {min(top, len(ranked))} of {len(ranked)} by {sel['metric']}; difference = configuration minus "
                 "baseline:")
    for s in ranked[:top]:
        v = s.get("versus_baseline")
        head = f"  {s['value']:.4f}"
        if s["interval"]:
            head += f" [{s['interval']['low']:.4f}, {s['interval']['high']:.4f}]"
        if v:
            b, pb = v["bootstrap"], v["p_better"]
            head += f"  diff {b['difference']:+.4f} bootstrap [{b['low']:+.4f}, {b['high']:+.4f}]"
            if "seed_interval" in v:
                head += f" seeds [{v['seed_interval']['low']:+.4f}, {v['seed_interval']['high']:+.4f}]"
            if "difference_by_fold" in v:
                by_fold = list(v["difference_by_fold"].values())
                head += f" fold range [{min(by_fold):+.4f}, {max(by_fold):+.4f}]"
            head += f"  P(better) {pb['p_a_beats_b']:.2f}  Holm p {v['holm_p']:.3f}  {v['verdict']}"
        elif s["config_hash"] == report["baseline"]:
            head += "  (baseline)"
        elif s.get("not_compared"):
            head += f"  (not compared: {s['not_compared']})"
        lines.append(head + f"  | {s['config_hash'][:12]} {s['stage']}: {s['label']}")
        if s.get("transductive"):
            lines.append("      " + transductive_note(s["transductive"]))
        others = {m: x for m, x in s["metrics"].items() if m != sel["metric"]}
        if others:
            lines.append("      " + "  ".join(f"{m} {_value(x, 4)}" for m, x in others.items()))
        if s.get("stability") and sel["metric"] != STABILITY:
            st = s["stability"]
            lines.append(f"      stability ({st['measure']}) {_value(st['value'], 3)} over "
                         f"{st['across_seeds']['n_pairs'] + st['across_folds']['n_pairs']} pairs")
        for column, groups in (s.get("by_group") or {}).items():
            lines.append(f"      by {column}: " + "  ".join(f"{g} {_value(x, 4)}" for g, x in groups.items()))
    if report["undefined"]:
        lines.append(f"{len(report['undefined'])} configurations have no defined {sel['metric']} (ranked last): "
                     + ", ".join(f"{s['config_hash'][:12]} {s['label']}" for s in report["undefined"][:5]))
    if report["factors"]:
        lines.append(f"factor levels in random stages (mean {sel['metric']} of the configurations drawn with each):")
        for f in report["factors"]:
            levels = "  ".join(f"{r['level']}: {r['mean']:.4f} (n={r['n_configs']})" for r in f["levels"]
                               if r["mean"] is not None)
            lines.append(f"  {f['factor']} (spread {f['spread']:.4f}): {levels}")
    return "\n".join(lines)
