"""Batch correction, for data measured in several batches (plates, runs, sites, sequencing lanes). The batch column
names each row's batch; every batch of the rows transformed must have fit rows.

  center    each batch's mean moved to the overall mean (fit rows)
  combat    ComBat with parametric empirical Bayes (Johnson, Li and Rabinovic 2007): each feature's batch mean and
            variance are estimated, shrunk towards the batch's average over features, and removed. This is the
            template's own implementation (MIT), iterated to `tolerance`
  harmony   Harmony (Korsunsky et al. 2019) on an embedding such as PCA (put a reduce step first); needs the GPL package
            harmonypy (the extras profile)
Batch correction learns from the fit rows; when the fit rows are every row (unsupervised `all`), it is applied with
the batches of every row, as is usual.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.prep.base import PrepStep
from sweepkit.registry import register_prep


@dataclasses.dataclass(frozen=True)
class BatchSettings:
    method: Literal["center", "combat", "harmony"]
    batch: str                      # the column holding each row's batch
    tolerance: float | None         # combat: stop when the empirical Bayes estimates change less than this
    max_iter: int | None            # combat, harmony: most iterations
    theta: float | None             # harmony: diversity penalty


def _combat_fit(x: np.ndarray, batches: np.ndarray, tolerance: float, max_iter: int) -> dict:
    names = np.unique(batches)
    n, n_features = x.shape
    sizes = np.array([(batches == b).sum() for b in names])
    if (sizes < 2).any():
        raise ValueError(f"prep batch_correct combat: batches {names[sizes < 2].tolist()} have fewer than 2 fit rows")
    batch_means = np.stack([x[batches == b].mean(axis=0) for b in names])
    grand = (sizes[:, None] * batch_means).sum(axis=0) / n
    residual = np.empty_like(x)
    for k, b in enumerate(names):
        residual[batches == b] = x[batches == b] - batch_means[k]
    pooled_sd = np.sqrt((residual ** 2).mean(axis=0))
    pooled_sd = np.where(pooled_sd > 0, pooled_sd, 1.0)
    z = (x - grand) / pooled_sd
    gamma_hat = np.stack([z[batches == b].mean(axis=0) for b in names])
    delta_hat = np.stack([z[batches == b].var(axis=0, ddof=1) for b in names])
    gamma_star, delta_star = np.empty_like(gamma_hat), np.empty_like(delta_hat)
    iterations = []
    for k, b in enumerate(names):
        g_bar = gamma_hat[k].mean()
        t2 = gamma_hat[k].var(ddof=1)
        m, s2 = delta_hat[k].mean(), delta_hat[k].var(ddof=1)
        if s2 <= 0 or t2 <= 0:
            gamma_star[k], delta_star[k] = gamma_hat[k], np.where(delta_hat[k] > 0, delta_hat[k], 1.0)
            iterations.append(0)
            continue
        lam = (m * m + 2 * s2) / s2
        theta = (m * m * m + m * s2) / s2
        zb = z[batches == b]
        nb = zb.shape[0]
        g_old, d_old = gamma_hat[k].copy(), delta_hat[k].copy()
        for it in range(1, max_iter + 1):
            g_new = (nb * t2 * gamma_hat[k] + d_old * g_bar) / (nb * t2 + d_old)
            sum2 = ((zb - g_new) ** 2).sum(axis=0)
            d_new = (theta + 0.5 * sum2) / (nb / 2.0 + lam - 1.0)
            change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
                         np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-12)))
            g_old, d_old = g_new, d_new
            if change < tolerance:
                break
        iterations.append(it)
        gamma_star[k], delta_star[k] = g_old, np.where(d_old > 0, d_old, 1.0)
    return {"names": names, "grand": grand, "sd": pooled_sd, "gamma": gamma_star, "delta": delta_star,
            "iterations": iterations}


@register_prep("batch_correct", settings=BatchSettings, kinds=("tabular",), profile="core", packages=(),
               description="remove batch effects: centring, ComBat, or Harmony (extras)", example="prep_batch")
class BatchCorrect(PrepStep):
    @staticmethod
    def check_job(job) -> None:
        need = {"center": (), "combat": ("tolerance", "max_iter"), "harmony": ("max_iter", "theta")}
        for name, s in job.prep:
            if name != "batch_correct":
                continue
            for field in ("tolerance", "max_iter", "theta"):
                if (getattr(s, field) is not None) != (field in need[s.method]):
                    raise ValueError(f"prep batch_correct {s.method}: set {list(need[s.method]) or 'no options'} and "
                                     "leave the others None")

    def _batches(self, positions):
        return self.ctx.table.column(self.s.batch).astype(str)[positions]

    def fit(self, x, positions):
        x = np.asarray(x, dtype=np.float64)
        self.num = self.numeric_columns(x.shape[1])
        batches = self._batches(positions)
        part = x[:, self.num]
        if self.s.method == "center":
            self.grand = part.mean(axis=0)
            self.means = {b: part[batches == b].mean(axis=0) for b in np.unique(batches)}
        elif self.s.method == "combat":
            self.fitted = _combat_fit(part, batches, self.s.tolerance, self.s.max_iter)
        else:
            self.fit_positions = np.asarray(positions)
            self.fit_x = part

    def _check_known(self, batches, known):
        unseen = sorted(set(batches.tolist()) - set(known))
        if unseen:
            raise ValueError(f"prep batch_correct: batches {unseen[:5]} have no fit rows")

    def transform(self, x, positions):
        x = np.array(x, dtype=np.float64, copy=True)
        batches = self._batches(positions)
        part = x[:, self.num]
        if self.s.method == "center":
            self._check_known(batches, self.means)
            for b in np.unique(batches):
                part[batches == b] = part[batches == b] - self.means[b] + self.grand
        elif self.s.method == "combat":
            f = self.fitted
            self._check_known(batches, f["names"].tolist())
            index = {b: k for k, b in enumerate(f["names"].tolist())}
            z = (part - f["grand"]) / f["sd"]
            for b in np.unique(batches):
                k = index[b]
                rows = batches == b
                z[rows] = (z[rows] - f["gamma"][k]) / np.sqrt(f["delta"][k])
            part = z * f["sd"] + f["grand"]
        else:
            if not np.array_equal(np.asarray(positions), self.fit_positions):
                raise ValueError("prep batch_correct harmony corrects the fit rows themselves; use it with a split "
                                 "whose evaluated rows are the fit rows (all, subsamples)")
            import harmonypy
            import pandas as pd

            from sweepkit.streams import int_seed

            # harmonypy 2.1.0 (its C++ backend): run_harmony(data_mat, meta_data, vars_use, theta, ...,
            # max_iter_harmony, verbose, random_state); Z_corr holds one row per row of data_mat. Releases before 2.0
            # returned it the other way round, which the shape check below also accepts.
            meta = pd.DataFrame({"batch": batches})
            result = harmonypy.run_harmony(part, meta, "batch", theta=self.s.theta, max_iter_harmony=self.s.max_iter,
                                           verbose=False, random_state=int_seed(self.ctx.seed, "harmony"))
            corrected = np.asarray(result.Z_corr, dtype=np.float64)
            part = corrected if corrected.shape == part.shape else corrected.T
        x[:, self.num] = part
        return x

    def record(self):
        out = {"method": self.s.method, "batch": self.s.batch}
        if self.s.method == "combat":
            out["batches"] = self.fitted["names"].tolist()
            out["iterations"] = self.fitted["iterations"]
        return out
