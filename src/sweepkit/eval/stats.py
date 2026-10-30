"""Statistics for backed results.

- t_interval: mean, SD and t confidence interval over seeds. With 5 seeds, df = 4 and the 95% critical value is
  2.776, so the half-width is 1.24 times the SD.
- paired_bootstrap: the difference in a fold-mean metric between two models (A minus B), with a percentile interval
  from a hierarchical bootstrap. Each draw resamples each model's seeds, and within every fold resamples the rows
  once, using the same row resample for both models. The shared row resample is the pairing: a wrong implementation
  that resamples rows separately per model gives much wider intervals, and the tests check that. A metric with a
  batched form (every metric computed from a confusion matrix, log loss, Brier, and the regression metrics built
  from weighted sums) scores all draws at once; any other metric that accepts row weights is recomputed per draw and
  seed, which is slower (AUROC with 2,000 draws, 5 seeds and 5 folds is 50,000 calls per model).
- folds_and_seeds_bootstrap: for label-free criteria and reveal metrics, which do not decompose over rows: the
  folds resampled (paired), then each configuration's seeds. Its p is the larger of the bootstrap's and a sign-flip
  test's over the folds (sign_flip_p), which cannot go below 2 / 2^folds: three subsamples can show stability, but
  never a difference.
- paired_units_bootstrap: a value measured once per unit (stability per fold), units resampled jointly, with the
  same sign-flip bound.
- seed_bootstrap: the difference of fold means with seeds resampled per model, for a supervised metric that takes no
  row weights.
- Every bootstrap p counts each tail as (k + 1) / (n + 1) of the n draws (Davison and Hinkley 1997), so it is
  never 0: at least 2 / (n + 1), recorded as p_floor (0.0010 for 2,000 draws).
- Two cases give no comparison rather than a misleading one: UndefinedInDraw (the metric has no value under some
  draw of the rows) and NoVariation (no spread at all to estimate from).
- probability_of_outperforming: P(A > B) over all pairs of runs, ties counted half, with a percentile bootstrap
  interval (Bouthillier et al. 2021: significant if the lower end is above 0.5, meaningful if the upper end is above
  a threshold such as 0.75). For a metric where smaller is better the comparison is flipped.
- probability_of_outperforming_by_fold: the same within each fold, averaged over the folds, with the folds
  resampled: for label-free criteria and reveal metrics, whose values move with the fold far more than with the
  seed. With one seed each, it is the share of folds where A is better.
- holm: Holm's step-down adjustment of p-values for a family of comparisons.

Every function takes an explicit numpy Generator, so results are reproducible.
"""
import numpy as np


def t_interval(values, confidence: float) -> dict:
    from scipy import stats

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("need at least two values")
    n = x.size
    sd = float(x.std(ddof=1))
    t_crit = float(stats.t.ppf((1 + confidence) / 2, n - 1))
    half = t_crit * sd / np.sqrt(n)
    mean = float(x.mean())
    return {"n": n, "mean": mean, "sd": sd, "df": n - 1, "confidence": confidence, "t_critical": t_crit,
            "half_width": float(half), "low": mean - float(half), "high": mean + float(half),
            "min": float(x.min()), "max": float(x.max())}


def _check_fold(fold: dict, name: str) -> None:
    for key in ("row_ids", "y_true", "y_pred"):
        if key not in fold:
            raise ValueError(f"{name}: fold is missing {key!r}")
    n = len(fold["row_ids"])
    if np.asarray(fold["y_true"]).shape[0] != n:
        raise ValueError(f"{name}: y_true does not match row_ids")
    pred = np.asarray(fold["y_pred"])
    if pred.shape[:2] != (pred.shape[0], n) or pred.ndim < 2:
        raise ValueError(f"{name}: y_pred must be seeds x rows, got {pred.shape}")


def paired_bootstrap(model_a: list, model_b: list, metric: dict, n_boot: int, confidence: float,
                     rng: np.random.Generator, chunk: int) -> dict:
    """Difference in the fold-mean metric, A minus B.

    model_a, model_b: one dict per fold, in the same fold order, each with
      row_ids    the evaluated rows, in a fixed order (the same in both models)
      y_true     their truth
      y_pred     seeds x rows (x labels for multi-label) predictions; the first axis lists the same seeds in the same
                 order in every fold, because a drawn seed is the same seed in every fold
      scores     seeds x rows x classes, or None
      scores_kind, names   as in sweepkit.tasks.supervised.Outputs
    metric: the metric's registry entry (sweepkit.registry.METRICS.get(name)); it must accept row weights.
    """
    from sweepkit.tasks.supervised import Outputs

    if not metric["weighted"]:
        raise ValueError(f"metric {metric['name']} does not accept row weights; it cannot be bootstrapped over rows")
    if len(model_a) != len(model_b) or not model_a:
        raise ValueError("both models need the same, nonzero number of folds")
    folds = []
    for i, (fa, fb) in enumerate(zip(model_a, model_b)):
        _check_fold(fa, f"A fold {i}")
        _check_fold(fb, f"B fold {i}")
        if list(fa["row_ids"]) != list(fb["row_ids"]):
            raise ValueError(f"fold {i}: the models' rows differ; the comparison cannot be paired")
        if not np.array_equal(np.asarray(fa["y_true"]), np.asarray(fb["y_true"])):
            raise ValueError(f"fold {i}: the models' truths differ")
        folds.append((fa, fb))
    batched = metric.get("batched")

    def value_per_seed(fold: dict, weights: np.ndarray) -> np.ndarray:
        """[B, S] metric of every seed under each draw's row weights."""
        y_pred = np.asarray(fold["y_pred"])
        if batched is not None:
            return np.asarray(batched(fold, weights), dtype=float)
        out = np.empty((weights.shape[0], y_pred.shape[0]))
        scores = fold.get("scores")
        for s in range(y_pred.shape[0]):
            outputs = Outputs(np.asarray(fold["y_true"]), y_pred[s], None if scores is None else np.asarray(scores)[s],
                              fold["scores_kind"], tuple(fold["names"]))
            for b in range(weights.shape[0]):
                out[b, s] = metric["fn"](outputs, weights[b])
        return out

    def observed(which: int) -> float:
        per_fold = []
        for fa, fb in folds:
            fold = fa if which == 0 else fb
            n = len(fold["row_ids"])
            per_fold.append(value_per_seed(fold, np.ones((1, n)))[0].mean())
        return float(np.mean(per_fold))

    obs_a, obs_b = observed(0), observed(1)
    s_a, s_b = np.asarray(folds[0][0]["y_pred"]).shape[0], np.asarray(folds[0][1]["y_pred"]).shape[0]
    for i, (fa, fb) in enumerate(folds):
        if np.asarray(fa["y_pred"]).shape[0] != s_a or np.asarray(fb["y_pred"]).shape[0] != s_b:
            raise ValueError(f"fold {i}: every fold needs the same seeds as the first")
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        seed_w_a = rng.multinomial(s_a, np.full(s_a, 1 / s_a), size=b) / s_a    # [b, S_a]
        seed_w_b = rng.multinomial(s_b, np.full(s_b, 1 / s_b), size=b) / s_b
        value_a = np.zeros(b)
        value_b = np.zeros(b)
        for fa, fb in folds:
            n = len(fa["row_ids"])
            w = rng.multinomial(n, np.full(n, 1 / n), size=b).astype(float)   # shared by both models
            try:
                value_a += (value_per_seed(fa, w) * seed_w_a).sum(axis=1)
                value_b += (value_per_seed(fb, w) * seed_w_b).sum(axis=1)
            except ValueError as err:
                raise UndefinedInDraw(f"{metric['name']} is undefined in a bootstrap draw of the rows ({err}); a "
                                      "class or label with few rows can be left out of a draw") from None
        diffs[done:done + b] = (value_a - value_b) / len(folds)
        done += b
    return _summary(obs_a, obs_b, diffs, confidence, n_boot, metric["direction"],
                    {"metric": metric["name"], "n_folds": len(folds), "n_seeds_a": s_a, "n_seeds_b": s_b,
                     "resampled": "seeds and rows", "batched": batched is not None})


class UndefinedInDraw(ValueError):
    """A metric that has no value under some bootstrap draw of the rows (a class with few rows drawn zero times):
    the comparison is reported as not made, with this reason, rather than computed from the draws that happen to
    be defined."""


def _summary(obs_a: float, obs_b: float, diffs: np.ndarray, confidence: float, n_boot: int, direction: str,
             extra: dict) -> dict:
    alpha = 1 - confidence
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # each tail counted as (k + 1) / (n + 1) (Davison and Hinkley 1997), so a p from n draws is never 0: at least
    # 2 / (n + 1), 0.0010 for 2,000 draws, which is the most such a bootstrap can show
    n = diffs.size
    below, above = int((diffs <= 0).sum()), int((diffs >= 0).sum())
    p_two_sided = min(1.0, 2 * min((below + 1) / (n + 1), (above + 1) / (n + 1)))
    # Monte Carlo error of the p-value, so a p near a threshold is read with its uncertainty
    p_error = float(np.sqrt(max(p_two_sided * (1 - p_two_sided), 1.0 / n_boot) / n_boot))
    return {"a": obs_a, "b": obs_b, "difference": obs_a - obs_b, "low": float(low), "high": float(high),
            "confidence": confidence, "n_boot": n_boot, "p_two_sided": p_two_sided, "p_floor": 2 / (n + 1),
            "p_monte_carlo_error": p_error, "direction": direction, **extra}


def seed_bootstrap(a_by_seed: dict, b_by_seed: dict, n_boot: int, confidence: float, rng: np.random.Generator,
                   direction: str) -> dict:
    """Difference of the mean over seeds of fold-mean values, A minus B, seeds resampled per model. For criteria that
    do not decompose over rows (silhouette, DBCV): the rows are fixed, only training randomness is resampled."""
    a = np.asarray(list(a_by_seed.values()), dtype=float)
    b = np.asarray(list(b_by_seed.values()), dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("need at least two seeds per model")
    draws_a = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    draws_b = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    return _summary(float(a.mean()), float(b.mean()), draws_a - draws_b, confidence, n_boot, direction,
                    {"n_seeds_a": int(a.size), "n_seeds_b": int(b.size), "resampled": "seeds only"})


class NoVariation(ValueError):
    """Every bootstrap draw gives the same difference: the configurations' values do not vary over the folds and
    seeds drawn (deterministic methods on one fold, for example), so there is nothing to estimate a spread from."""


def sign_flip_p(differences: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided p of a sign-flip (randomization) test that the mean of independent paired differences is 0: the
    share of sign assignments whose mean is at least as far from 0 as the observed one. Exact up to 16 differences
    (every assignment), else from 20,000 random assignments. It cannot go below 2 / 2^n, which is the point: three
    folds that agree can never give p below 0.25."""
    d = np.asarray(differences, dtype=float)
    observed = abs(float(d.mean()))
    if d.size <= 16:
        signs = 1 - 2 * ((np.arange(2 ** d.size)[:, None] >> np.arange(d.size)[None, :]) & 1)
    else:
        signs = rng.choice((-1, 1), size=(20000, d.size))
    means = np.abs((signs * d[None, :]).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12 * max(1.0, observed)))


def folds_and_seeds_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, confidence: float,
                              rng: np.random.Generator, direction: str) -> dict:
    """Difference of means, A minus B, of values measured per (fold, seed): a [folds, seeds of A], b [folds, seeds of
    B], the same folds in the same order. For criteria that do not decompose over rows (silhouette, DBCV) and for
    agreement with reference labels, whose variation comes from the data (folds, such as subsamples) and from the
    method's randomness (seeds); a deterministic method varies only over folds.

    The interval: each draw resamples the folds, paired (a drawn fold brings both configurations' values), then each
    configuration's seeds, the same seed draw in every fold. The p-value: the larger of the bootstrap's and a
    sign-flip test's over the folds' differences (sign_flip_p), because a bootstrap over a handful of folds that agree
    gives p = 0 however small the difference. With one fold the p covers the seeds' randomness only, and a difference
    that does not vary at all raises NoVariation (identical values give a difference of 0, reported as such)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b: [folds, seeds] with the same folds")
    folds = a.shape[0]
    if min(a.shape[1], b.shape[1]) < 1 or folds * min(a.shape[1], b.shape[1]) < 2:
        raise ValueError("need at least two folds or two seeds")
    picks = rng.integers(0, folds, size=(n_boot, folds))
    weights_a = rng.multinomial(a.shape[1], np.full(a.shape[1], 1 / a.shape[1]), size=n_boot) / a.shape[1]
    weights_b = rng.multinomial(b.shape[1], np.full(b.shape[1], 1 / b.shape[1]), size=n_boot) / b.shape[1]
    per_fold_a = a @ weights_a.T                       # [folds, draws]
    per_fold_b = b @ weights_b.T
    draw = np.arange(n_boot)[:, None]
    diffs = (per_fold_a[picks, draw] - per_fold_b[picks, draw]).mean(axis=1)
    if folds == 1 and np.ptp(diffs) == 0 and diffs[0] != 0:
        raise NoVariation(f"the values do not vary over the seeds of the one fold, so a difference of "
                          f"{float(a.mean() - b.mean()):+.4f} has no interval (deterministic methods on one fold)")
    out = _summary(float(a.mean()), float(b.mean()), diffs, confidence, n_boot, direction,
                   {"n_folds": folds, "n_seeds_a": int(a.shape[1]), "n_seeds_b": int(b.shape[1]),
                    "resampled": "folds (paired) and seeds"})
    if folds >= 2:
        out["p_bootstrap"] = out["p_two_sided"]
        out["p_sign_flip_folds"] = sign_flip_p(a.mean(axis=1) - b.mean(axis=1), rng)
        out["p_two_sided"] = max(out["p_bootstrap"], out["p_sign_flip_folds"])
    else:
        out["note"] = "one fold: the interval and p cover the seeds' randomness only, not the data's"
    return out


def paired_units_bootstrap(a_by_unit: dict, b_by_unit: dict, n_boot: int, confidence: float,
                           rng: np.random.Generator, direction: str, unit: str) -> dict:
    """Difference of means, A minus B, over the units both have (for example folds), resampled jointly: a drawn
    unit brings both models' values. For a value measured once per unit, such as stability within a fold. The p is
    the larger of the bootstrap's and a sign-flip test's over the units (sign_flip_p), so a few units that agree
    cannot give p = 0."""
    shared = sorted(set(a_by_unit) & set(b_by_unit))
    if len(shared) < 2:
        raise ValueError(f"need at least two shared {unit}s")
    a = np.asarray([a_by_unit[k] for k in shared], dtype=float)
    b = np.asarray([b_by_unit[k] for k in shared], dtype=float)
    picks = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    diffs = (a[picks] - b[picks]).mean(axis=1)
    out = _summary(float(a.mean()), float(b.mean()), diffs, confidence, n_boot, direction,
                   {f"n_{unit}s": len(shared), "resampled": f"{unit}s, paired"})
    out["p_bootstrap"] = out["p_two_sided"]
    out[f"p_sign_flip_{unit}s"] = sign_flip_p(a - b, rng)
    out["p_two_sided"] = max(out["p_bootstrap"], out[f"p_sign_flip_{unit}s"])
    return out


def probability_of_outperforming(a, b, direction: str) -> float:
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    better = (a > b) if direction == "max" else (a < b)
    return float((better + 0.5 * (a == b)).mean())


def probability_of_outperforming_interval(a, b, n_boot: int, confidence: float, rng: np.random.Generator,
                                          meaningful_threshold: float, direction: str) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = probability_of_outperforming(rng.choice(a, a.size), rng.choice(b, b.size), direction)
    alpha = 1 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"p_a_beats_b": probability_of_outperforming(a, b, direction), "low": float(low), "high": float(high),
            "confidence": confidence, "n_boot": n_boot, "significant": bool(low > 0.5),
            "meaningful": bool(high > meaningful_threshold), "meaningful_threshold": meaningful_threshold}


def probability_of_outperforming_by_fold(a, b, n_boot: int, confidence: float, rng: np.random.Generator,
                                          meaningful_threshold: float, direction: str) -> dict:
    """P(A beats B) within a fold, averaged over the folds: a [folds, seeds of A], b [folds, seeds of B], the same
    folds in the same order. In each fold every seed of A meets every seed of B, ties counted half; with one seed
    each, this is the share of folds where A is better. The interval resamples the folds."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] < 1 or min(a.shape[1], b.shape[1]) < 1:
        raise ValueError("a and b: [folds, seeds] with the same folds and at least one seed each")
    per_fold = np.array([probability_of_outperforming(a[f], b[f], direction) for f in range(a.shape[0])])
    picks = rng.integers(0, per_fold.size, size=(n_boot, per_fold.size))
    draws = per_fold[picks].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"p_a_beats_b": float(per_fold.mean()), "low": float(low), "high": float(high), "confidence": confidence,
            "n_boot": n_boot, "significant": bool(low > 0.5), "meaningful": bool(high > meaningful_threshold),
            "meaningful_threshold": meaningful_threshold, "units": "within each fold, averaged over the folds"}


def holm(pvalues) -> list:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0 or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a non-empty list of numbers in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def verdict(difference: float, adjusted_p: float, alpha: float, direction: str) -> str:
    """better, worse or "no detectable difference", for A against B, in the metric's direction."""
    if adjusted_p >= alpha:
        return "no detectable difference"
    a_better = difference > 0 if direction == "max" else difference < 0
    return "better" if a_better else "worse"


def round_half_up(value: float, digits: int) -> str:
    """Round once, half up, from the exact value: 0.6005 shows as 0.601 at 3 digits."""
    from decimal import ROUND_HALF_UP, Decimal

    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))
