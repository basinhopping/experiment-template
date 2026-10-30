"""test-compare: run exactly the comparison families a RULE fixed, on test or reveal files, with Holm's correction
within each family.

For each comparison (A against B):
  test families (supervised): every complete seed of each configuration; a hierarchical paired bootstrap of the
      difference in the fold-mean metric (seeds resampled per model, test rows resampled within each fold and shared
      by both, so both configurations must have the same test rows: the same data, split and task); a metric without
      row weights falls back to the bootstrap over seeds. Each configuration needs two or more complete seeds;
  reveal families (unsupervised): the reveal metric per fold and seed, the folds (paired) and each configuration's
      seeds resampled (folds_and_seeds_bootstrap), as analyze does for label-free criteria: two or more folds with
      one seed each, or one fold with two or more seeds;
  P(A beats B) with a bootstrap interval: over the seed values (test families), fold by fold (reveal families).
Then Holm's adjustment within the family, and a verdict in the metric's direction.
"""
import numpy as np

from sweepkit.analysis.records import (complete_runs, complete_seeds, configurations, file_path, find_selection,
                                       is_supervised, read_json, result_folds, sweep_record, task_of)
from sweepkit.analysis.rules import earlier_readings, enforce, find_rule, late_note
from sweepkit.eval.stats import (NoVariation, UndefinedInDraw, folds_and_seeds_bootstrap, holm, paired_bootstrap,
                                 probability_of_outperforming_by_fold, probability_of_outperforming_interval,
                                 seed_bootstrap, verdict)
from sweepkit.registry import METRICS
from sweepkit.runs.spec import PROTOCOL_SECTIONS


def resolve(ref: dict) -> tuple:
    """(sweep_id, config_hash) of a side."""
    if "selection" in ref:
        s = find_selection(ref["selection"])
        return s["sweep_id"], s["config_hash"]
    record = sweep_record(ref["sweep"])
    hashes = sorted({row["config_hash"] for row in record["jobs"]})
    matches = [h for h in hashes if h.startswith(ref["config"])]
    if len(matches) != 1:
        raise ValueError(f"{ref['config']!r} matches {len(matches)} configurations of {ref['sweep']}")
    return ref["sweep"], matches[0]


def _config(sweep_id: str, config_hash: str) -> tuple:
    record = sweep_record(sweep_id)
    configs = configurations(record, complete_runs(sweep_id))
    if config_hash not in configs:
        raise ValueError(f"{sweep_id} / {config_hash[:12]} has no complete run")
    folds = result_folds(record["sweep_file"]["folds"])
    seeds = complete_seeds(configs[config_hash], folds)
    if not seeds:
        raise ValueError(f"{sweep_id} / {config_hash[:12]}: no seed has every fold complete")
    return record, configs[config_hash], folds, seeds


def _protocol(record: dict) -> dict:
    return {k: record["sweep_file"]["base"][k] for k in PROTOCOL_SECTIONS}


def _test_arrays(config: dict, seeds: list, folds: list, task_name: str) -> list:
    from sweepkit.families.supervised_io import read_predictions

    out = []
    for fold in folds:
        reads = [read_predictions(file_path(config["runs"][s][fold]["folder"], "predictions_test.csv", "judge"),
                                  task_name) for s in seeds]
        first = reads[0]
        for r in reads[1:]:
            if r["row_ids"] != first["row_ids"]:
                raise ValueError(f"{config['config_hash'][:12]} {fold}: seeds disagree on the test rows")
        scores = np.stack([r["scores"] for r in reads]) if all(r["scores"] is not None for r in reads) else None
        out.append({"row_ids": first["row_ids"], "y_true": first["y_true"],
                    "y_pred": np.stack([r["y_pred"] for r in reads]), "scores": scores,
                    "scores_kind": first["scores_kind"], "names": first["names"]})
    return out


def _fold_seed_values(config: dict, seeds: list, folds: list, name: str, metric: str) -> np.ndarray:
    """[folds, seeds] of a judging file's metric (reveal metrics: agreement with the reference labels)."""
    return np.asarray([[read_json(config["runs"][s][f]["folder"], name, "judge")["metrics"][metric] for s in seeds]
                       for f in folds], dtype=float)


def _seed_values(config: dict, seeds: list, folds: list, name: str, metric: str) -> dict:
    out = {}
    for s in seeds:
        values = [read_json(config["runs"][s][f]["folder"], name, "judge")["metrics"][metric] for f in folds]
        out[s] = None if None in values else float(np.mean(values))
    return out


def compare_pair(family: dict, a_ref: dict, b_ref: dict, confidence: float, n_boot: int, meaningful: float,
                 rng) -> dict:
    (sa, ha), (sb, hb) = resolve(a_ref), resolve(b_ref)
    ra, ca, folds_a, seeds_a = _config(sa, ha)
    rb, cb, folds_b, seeds_b = _config(sb, hb)
    if _protocol(ra) != _protocol(rb) or folds_a != folds_b:
        raise ValueError(f"{sa}/{ha[:12]} and {sb}/{hb[:12]} differ in data, split, task or folds; a comparison "
                         "needs the same evaluation rows")
    task_name, task = task_of(ra["sweep_file"])
    metric = METRICS.get(family["metric"])
    if task_name not in metric["tasks"]:
        raise ValueError(f"family {family['name']}: {family['metric']} does not score {task_name}")
    supervised = is_supervised(task_name)
    if (family["files"] == "test") != supervised:
        raise ValueError(f"family {family['name']}: files {family['files']} do not fit the {task_name} task")
    fewest = min(len(seeds_a), len(seeds_b))
    if supervised and fewest < 2:
        raise ValueError(f"family {family['name']}: {sa}/{ha[:12]} and {sb}/{hb[:12]} have {len(seeds_a)} and "
                         f"{len(seeds_b)} complete seeds; a test comparison needs 2 each")
    if not supervised and len(folds_a) * fewest < 2:
        raise ValueError(f"family {family['name']}: one fold and one seed; a reveal comparison needs two or more "
                         "folds, or two seeds of one fold")
    name = "metrics_test.json" if supervised else "reveal.json"
    if not supervised and family["metric"] not in task["reveal_metrics"]:
        raise ValueError(f"family {family['name']}: the runs' reveal files hold {task['reveal_metrics']}, not "
                         f"{family['metric']}")
    va = _seed_values(ca, seeds_a, folds_a, name, family["metric"])
    vb = _seed_values(cb, seeds_b, folds_b, name, family["metric"])
    if None in va.values() or None in vb.values():
        raise ValueError(f"family {family['name']}: {family['metric']} is undefined in some run")
    out = {"a": {"sweep_id": sa, "config_hash": ha, "label": ca["label"], "seeds": seeds_a, "by_seed": va},
           "b": {"sweep_id": sb, "config_hash": hb, "label": cb["label"], "seeds": seeds_b, "by_seed": vb}}
    try:
        if supervised and metric["weighted"]:
            boot = paired_bootstrap(_test_arrays(ca, seeds_a, folds_a, task_name),
                                    _test_arrays(cb, seeds_b, folds_b, task_name), metric, n_boot, confidence, rng, 250)
        elif not supervised:
            # reveal metrics: folds (paired) and seeds resampled, as analyze does for label-free criteria
            boot = folds_and_seeds_bootstrap(_fold_seed_values(ca, seeds_a, folds_a, name, family["metric"]),
                                             _fold_seed_values(cb, seeds_b, folds_b, name, family["metric"]),
                                             n_boot, confidence, rng, metric["direction"])
        else:
            boot = seed_bootstrap(va, vb, n_boot, confidence, rng, metric["direction"])
    except (UndefinedInDraw, NoVariation) as err:
        # reported as not made; Holm still counts it, with p = 1, so the family keeps the size the RULE fixed
        return {**out, "bootstrap": None, "p_better": None, "not_compared": str(err)}
    if supervised:
        poo = probability_of_outperforming_interval(list(va.values()), list(vb.values()), n_boot, confidence, rng,
                                                    meaningful, metric["direction"])
    else:
        poo = probability_of_outperforming_by_fold(_fold_seed_values(ca, seeds_a, folds_a, name, family["metric"]),
                                                   _fold_seed_values(cb, seeds_b, folds_b, name, family["metric"]),
                                                   n_boot, confidence, rng, meaningful, metric["direction"])
    return {**out, "bootstrap": boot, "p_better": poo}


def test_compare(rule_folder: str, confidence: float, n_boot: int, seed: int, meaningful: float) -> dict:
    folder, rule = find_rule(rule_folder)
    if not rule["families"]:
        raise ValueError(f"{rule_folder} lists no comparison families; it can only cover reports")
    resolved = {}
    for family in rule["families"]:
        for pair in family["comparisons"]:
            for ref in (pair["a"], pair["b"]):
                resolved[str(sorted(ref.items()))] = resolve(ref)
    configs = set(resolved.values())
    problems = []
    earlier = earlier_readings(configs, rule["fixed_utc"])
    if earlier:
        problems.append(f"judging files of configurations this RULE names were read before it was fixed "
                        f"({rule['fixed_utc']}): {', '.join(earlier[:3])}")
    warnings = enforce(problems, "test-compare") + ([late_note(rule)] if late_note(rule) else [])
    rng = np.random.default_rng(seed)
    families = []
    alpha = 1 - confidence
    for family in rule["families"]:
        rows = [compare_pair(family, p["a"], p["b"], confidence, n_boot, meaningful, rng) for p in family["comparisons"]]
        adjusted = holm([1.0 if r["bootstrap"] is None else r["bootstrap"]["p_two_sided"] for r in rows])
        direction = METRICS.get(family["metric"])["direction"]
        for r, p in zip(rows, adjusted):
            r["holm_p"] = p
            r["verdict"] = ("not compared" if r["bootstrap"] is None
                            else verdict(r["bootstrap"]["difference"], p, alpha, direction))
        families.append({"name": family["name"], "files": family["files"], "metric": family["metric"],
                         "direction": direction, "comparisons": rows})
    return {"rule": folder.name, "rule_fixed_utc": rule["fixed_utc"], "rule_sha256": rule["rule_sha256"],
            "warnings": warnings, "confidence": confidence, "n_boot": n_boot, "seed": seed,
            "meaningful": meaningful, "families": families,
            "configs": [{"sweep_id": s, "config_hash": h} for s, h in sorted(configs)]}


def render(report: dict) -> str:
    lines = list(report["warnings"])
    lines.append(f"test-compare of RULE {report['rule']} (fixed {report['rule_fixed_utc']}): "
                 f"{len(report['families'])} families, {round(report['confidence'] * 100)}% intervals, "
                 f"{report['n_boot']} bootstrap draws, Holm within each family")
    for family in report["families"]:
        better = "larger" if family["direction"] == "max" else "smaller"
        lines.append(f"family {family['name']} ({family['files']} files, {family['metric']}, {better} is better):")
        for r in family["comparisons"]:
            b, pb = r["bootstrap"], r["p_better"]
            if b is None:
                lines.append(f"  {r['a']['label']} ({r['a']['config_hash'][:12]}, {r['a']['sweep_id']}) against "
                             f"{r['b']['label']} ({r['b']['config_hash'][:12]}, {r['b']['sweep_id']}): NOT COMPARED, "
                             f"{r['not_compared']} (counted in Holm with p = 1)")
                continue
            lines.append(f"  {r['a']['label']} ({r['a']['config_hash'][:12]}, {r['a']['sweep_id']}) {b['a']:.4f} "
                         f"against {r['b']['label']} ({r['b']['config_hash'][:12]}, {r['b']['sweep_id']}) {b['b']:.4f}")
            floor = (f" (the least {b['n_boot']} draws can give)"
                     if b.get("p_floor") is not None and b["p_two_sided"] <= b["p_floor"] else "")
            lines.append(f"    diff {b['difference']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}]  P(A better) "
                         f"{pb['p_a_beats_b']:.2f} [{pb['low']:.2f}, {pb['high']:.2f}]  p {b['p_two_sided']:.4f}"
                         f"{floor}  Holm p {r['holm_p']:.4f}  {r['verdict']}")
    return "\n".join(lines)
