"""Deep clustering on tabular rows, in PyTorch (the torch profile).

autoencoder_kmeans  an autoencoder (sweepkit.models.embed.deep) trained on the fit rows, then k-means (scikit-learn,
                    k-means++ starts, n_init of them from their own stream, at most 300 iterations and tolerance 1e-4,
                    scikit-learn's defaults, recorded in model_params.json) on their codes; new rows are encoded and
                    given the nearest centre (fit.assign predict works)
dec                 Deep Embedded Clustering (Xie, Girshick and Farhadi 2016): the autoencoder is trained first; k-means
                    on the codes (as above) gives the first centres; then the encoder and the centres are refined
                    together with Adam. Each
                    row's soft assignment q is a Student t kernel on its distance to each centre,
                        q_ij = (1 + |z_i - mu_j|^2 / alpha)^(-(alpha + 1) / 2), normalized over j,
                    the target p sharpens it, p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j') with f_j = sum_i q_ij,
                    and the loss is the KL divergence KL(P || Q). P is recomputed from every fit row each
                    update_interval epochs, and refinement stops when fewer than tol of the rows changed cluster since
                    the last update (or after refine_epochs). The decoder is not used in refinement, as in the paper.
                    Labels are argmax q; the probability column is max q.

Random streams (sweepkit.streams): the autoencoder's own ("weights", "autoencoder batches"), "kmeans" for the
k-means starts, and "dec batches" for the refinement's row order, so a seed gives the same clusters every time on the
CPU (on a GPU the last digits can differ). The autoencoder needs inputs on a common scale (a scale step first). The
reported numbers (info): the autoencoder's last loss, the k-means inertia in the latent space, and for DEC the epochs
of refinement run and the share of rows that changed at the last update.
"""
import dataclasses

import numpy as np

from sweepkit.models.base import Clusterer
from sweepkit.models.embed.deep import (AutoencoderSettings, check_autoencoder, in_batches, settings_record,
                                        train_autoencoder)
from sweepkit.registry import register_model
from sweepkit.streams import generator, int_seed

CLUSTER = ("clustering",)
TAB = ("tabular",)


@dataclasses.dataclass(frozen=True)
class AutoencoderKMeansSettings(AutoencoderSettings):
    n_clusters: int
    n_init: int                     # k-means starts in the latent space; the one with the lowest inertia is kept


@dataclasses.dataclass(frozen=True)
class DECSettings(AutoencoderSettings):
    n_clusters: int
    n_init: int                     # k-means starts that give the first centres
    alpha: float                    # degrees of freedom of the Student t kernel (1 in the paper)
    refine_epochs: int              # at most this many epochs of refinement
    refine_lr: float                # Adam's learning rate in refinement
    update_interval: int            # epochs between two computations of the target distribution
    tol: float                      # stop when fewer than this share of rows changed cluster between two updates


def _kmeans(z: np.ndarray, s, seed: int):
    from sklearn.cluster import KMeans

    if s.n_clusters > z.shape[0]:
        raise ValueError(f"n_clusters {s.n_clusters} is more than the {z.shape[0]} fit rows")
    return KMeans(n_clusters=s.n_clusters, init="k-means++", n_init=s.n_init, max_iter=300, tol=1e-4,
                  random_state=int_seed(seed, "kmeans")).fit(z)


def _check_common(s, where: str) -> None:
    check_autoencoder(s, where)
    if s.n_clusters < 2 or s.n_init < 1:
        raise ValueError(f"{where}: n_clusters at least 2 and n_init at least 1")


@register_model("autoencoder_kmeans", settings=AutoencoderKMeansSettings, family="clusterer", tasks=CLUSTER,
                kinds=TAB, gpu="torch", profile="torch", packages=("torch", "sklearn"),
                description="k-means on an autoencoder's codes (PyTorch and scikit-learn)", example="cluster_deep")
class AutoencoderKMeans(Clusterer):
    @staticmethod
    def check_job(job) -> None:
        _check_common(job.model, "model autoencoder_kmeans")

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.net, self.record = train_autoencoder(x, self.s, self.seed, self.device, False)
        self.dev = next(self.net.parameters()).device
        z = in_batches(self.net.encode, x, self.s.batch_size, self.dev)
        self.kmeans = _kmeans(z, self.s, self.seed)

    def labels(self):
        return np.asarray(self.kmeans.labels_, dtype=np.int64)

    def can_assign(self):
        return True

    def assign(self, x_new):
        z = in_batches(self.net.encode, np.asarray(x_new, dtype=np.float32), self.s.batch_size, self.dev)
        return np.asarray(self.kmeans.predict(z), dtype=np.int64)

    def info(self):
        return {"autoencoder_final_loss": self.record["curve"][-1]["loss"],
                "latent_inertia": float(self.kmeans.inertia_)}

    def params(self):
        from sweepkit.models.base import library_params

        return {**settings_record(self.s), "device": self.record["device"], "parameters": self.record["parameters"],
                "optimizer": self.record["optimizer"],
                "loss_by_epoch": [e["loss"] for e in self.record["curve"]], "kmeans": library_params(self.kmeans)}


def soft_assignment(z, centres, alpha: float):
    """The Student t kernel of DEC: q [rows, clusters], each row summing to 1."""
    distance = ((z[:, None, :] - centres[None, :, :]) ** 2).sum(dim=2)
    q = (1.0 + distance / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q):
    """DEC's sharpened target p from q (rows by clusters)."""
    weight = q ** 2 / q.sum(dim=0, keepdim=True)
    return weight / weight.sum(dim=1, keepdim=True)


@register_model("dec", settings=DECSettings, family="clusterer", tasks=CLUSTER, kinds=TAB, gpu="torch",
                profile="torch", packages=("torch", "sklearn"),
                description="Deep Embedded Clustering (Xie et al. 2016; PyTorch)", example="cluster_deep")
class DEC(Clusterer):
    @staticmethod
    def check_job(job) -> None:
        s = job.model
        _check_common(s, "model dec")
        if not s.alpha > 0 or s.refine_epochs < 1 or not s.refine_lr > 0 or s.update_interval < 1:
            raise ValueError("model dec: alpha and refine_lr above 0, refine_epochs and update_interval at least 1")
        if not 0 <= s.tol < 1:
            raise ValueError("model dec: tol in [0, 1)")

    def _q(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            z = torch.from_numpy(in_batches(self.net.encode, x, self.s.batch_size, self.dev)).float().to(self.dev)
            return soft_assignment(z, self.centres, self.s.alpha).cpu().numpy().astype(np.float64)

    def fit(self, x):
        import torch

        from sweepkit.train.loop import optimizer_record

        s = self.s
        x = np.asarray(x, dtype=np.float32)
        self.net, self.record = train_autoencoder(x, s, self.seed, self.device, False)
        self.dev = next(self.net.parameters()).device
        start = _kmeans(in_batches(self.net.encode, x, s.batch_size, self.dev), s, self.seed)
        self.latent_inertia = float(start.inertia_)
        self.centres = torch.nn.Parameter(torch.from_numpy(np.asarray(start.cluster_centers_, dtype=np.float32))
                                          .to(self.dev))
        encoder = self.net.encoder
        optimizer = torch.optim.Adam(list(encoder.parameters()) + [self.centres], lr=s.refine_lr)
        data = torch.from_numpy(np.ascontiguousarray(x)).to(self.dev)
        order_rng = generator(self.seed, "dec batches")
        n = int(x.shape[0])
        steps = max(1, n // s.batch_size)
        previous = np.asarray(start.labels_, dtype=np.int64)
        changed, epochs_run, curve = None, 0, []
        target = None
        for epoch in range(s.refine_epochs):
            if epoch % s.update_interval == 0:
                encoder.eval()
                q_all = self._q(x)
                labels = q_all.argmax(axis=1)
                if epoch > 0:
                    changed = float(np.mean(labels != previous))
                    if changed < s.tol:
                        break
                previous = labels
                target = torch.from_numpy(target_distribution(torch.from_numpy(q_all)).numpy().astype(np.float32))
                target = target.to(self.dev)
            # the encoder trains in eval mode (gradients still flow): no dropout in the refinement, as in Xie et al.
            # 2016, so the q of the loss is the q the target distribution was computed from
            encoder.eval()
            order = order_rng.permutation(n)
            total = torch.zeros((), device=self.dev)
            for b in range(steps):
                picked = torch.from_numpy(order[b * s.batch_size:(b + 1) * s.batch_size]).to(self.dev)
                q = soft_assignment(encoder(data[picked]), self.centres, s.alpha)
                p = target[picked]
                loss = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(dim=1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total = total + loss.detach()
            epochs_run += 1
            curve.append(float(total.item()) / steps)
        self.net.eval()
        self.q = self._q(x)
        self.refine = {"epochs_run": epochs_run, "last_change": changed, "kl_by_epoch": curve,
                       "optimizer": optimizer_record(optimizer)}

    def labels(self):
        return self.q.argmax(axis=1).astype(np.int64)

    def soft(self):
        return {"probability": self.q.max(axis=1)}

    def can_assign(self):
        return True

    def assign(self, x_new):
        return self._q(np.asarray(x_new, dtype=np.float32)).argmax(axis=1).astype(np.int64)

    def info(self):
        out = {"autoencoder_final_loss": self.record["curve"][-1]["loss"], "latent_inertia": self.latent_inertia,
               "refine_epochs_run": self.refine["epochs_run"]}
        if self.refine["last_change"] is not None:
            out["last_change"] = self.refine["last_change"]
        return out

    def params(self):
        return {**settings_record(self.s), "device": self.record["device"], "parameters": self.record["parameters"],
                "pretraining_optimizer": self.record["optimizer"], "refine_optimizer": self.refine["optimizer"],
                "loss_by_epoch": [e["loss"] for e in self.record["curve"]], "kl_by_epoch": self.refine["kl_by_epoch"]}
