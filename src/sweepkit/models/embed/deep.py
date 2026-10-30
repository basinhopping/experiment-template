"""Deep embedders on tabular rows, in PyTorch (the torch profile): an autoencoder and a variational autoencoder.

autoencoder   an encoder of linear layers (the hidden widths, then the latent dimension) and a decoder that mirrors
              it, trained to rebuild the fit rows (mean squared error per value); the embedding is the encoder's
              output
vae           the same with a Gaussian latent (Kingma and Welling 2014): the encoder gives a mean and a log-variance,
              training minimizes the squared reconstruction error summed over each row's values plus beta times the
              KL divergence to a standard normal (beta 1 is the plain VAE; other values as in Higgins et al. 2017);
              the embedding is the mean

Both train with AdamW for a fixed number of epochs on mini-batches, from the run's own random streams
(sweepkit.streams): "weights" seeds torch before the network is built (its weights and its dropout), "autoencoder
batches" orders the rows, and "vae noise" draws the latent noise, so a seed gives the same embedding every time on the
CPU (on a GPU the last digits can differ). With fewer rows than batch_size, each epoch is one batch of all of them.
Both embed new rows (transform: fit.assign predict works) and rebuild rows from an embedding (inverse_transform), so
the runner reports the reconstruction error, the criterion "reconstruction_error"; info() adds the last epoch's
loss (and KL term). model_params.json records every setting, the network's size, the optimizer's settings with its
own defaults, and the loss of every epoch.

The inputs should share a scale (a scale step first): the decoder's last layer is linear and the loss weighs every
column alike. The device is the fit section's (cpu, or cuda when torch sees a GPU). Used as a prep step (reduce), an
autoencoder's embedding feeds any clusterer; sweepkit.models.cluster.deep adds k-means in the latent space and DEC.
"""
import dataclasses
import math
from typing import Literal

import numpy as np

from sweepkit.models.base import Embedder
from sweepkit.registry import register_model
from sweepkit.streams import generator, int_seed

EMBED = ("embedding",)
TAB = ("tabular",)


@dataclasses.dataclass(frozen=True)
class AutoencoderSettings:
    hidden: tuple[int, ...]         # widths of the encoder's hidden layers (the decoder mirrors them); may be empty
    latent_dim: int                 # the embedding's dimension
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # between layers
    dropout: float                  # after each hidden layer's activation, in training only
    epochs: int                     # passes over the fit rows
    batch_size: int                 # rows per step (the last incomplete batch of an epoch is dropped)
    lr: float                       # AdamW's learning rate
    weight_decay: float             # AdamW's decoupled weight decay


@dataclasses.dataclass(frozen=True)
class VAESettings(AutoencoderSettings):
    beta: float                     # weight of the KL term (1: the plain variational autoencoder)


def check_autoencoder(s, where: str) -> None:
    """The settings' own checks, without torch."""
    if any(h < 1 for h in s.hidden) or s.latent_dim < 1:
        raise ValueError(f"{where}: hidden widths and latent_dim must be positive")
    if not 0.0 <= s.dropout < 1.0:
        raise ValueError(f"{where}: dropout must be in [0, 1)")
    if s.epochs < 1 or s.batch_size < 1:
        raise ValueError(f"{where}: epochs and batch_size must be at least 1")
    if not s.lr > 0 or s.weight_decay < 0:
        raise ValueError(f"{where}: lr must be above 0 and weight_decay not negative")
    if not getattr(s, "beta", 1.0) > 0:
        raise ValueError(f"{where}: beta must be above 0")


def torch_device(device: str):
    """The torch device for the run's resolved device, refusing one torch cannot use here."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device cuda was resolved, but torch sees no CUDA GPU (a CPU-only torch build?)")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise ValueError("device mps was resolved, but torch's MPS backend is not available")
    return torch.device(device)


def _layers(widths: list, activation: str, dropout: float, last_activation: bool):
    """Linear layers between consecutive widths, each followed by the activation and dropout, except the last unless
    last_activation."""
    from torch import nn

    from sweepkit.models.torch.heads import activation_layer

    layers = []
    for k in range(len(widths) - 1):
        layers.append(nn.Linear(widths[k], widths[k + 1]))
        if k < len(widths) - 2 or last_activation:
            layers += [activation_layer(activation), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


def _module_class():
    from torch import nn

    class AutoencoderModule(nn.Module):
        """The encoder and the decoder. variational: the encoder's body ends in two linear heads, the mean and the
        log-variance of the latent."""

        def __init__(self, n_in: int, hidden: tuple, latent: int, activation: str, dropout: float, variational: bool):
            super().__init__()
            self.variational = variational
            widths = [n_in] + list(hidden)
            if variational:
                self.body = _layers(widths, activation, dropout, last_activation=True)
                self.mu = nn.Linear(widths[-1], latent)
                self.logvar = nn.Linear(widths[-1], latent)
            else:
                self.encoder = _layers(widths + [latent], activation, dropout, last_activation=False)
            self.decoder = _layers([latent] + list(reversed(hidden)) + [n_in], activation, dropout,
                                   last_activation=False)

        def encode(self, x):
            """The embedding: the code, or the latent mean."""
            if self.variational:
                return self.mu(self.body(x))
            return self.encoder(x)

        def posterior(self, x):
            h = self.body(x)
            return self.mu(h), self.logvar(h)

        def decode(self, z):
            return self.decoder(z)

    return AutoencoderModule


def parameter_count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def train_autoencoder(x: np.ndarray, s, seed: int, device: str, variational: bool) -> tuple:
    """Train on the rows of x. Returns (the network in eval mode on its device, a record: the loss of every epoch,
    steps per epoch, the optimizer's settings, the network's size)."""
    import torch

    from sweepkit.models.torch.base import deterministic_cudnn
    from sweepkit.train.loop import optimizer_record

    dev = torch_device(device)
    deterministic_cudnn(dev.type)
    torch.manual_seed(int_seed(seed, "weights"))
    net = _module_class()(int(x.shape[1]), tuple(s.hidden), int(s.latent_dim), s.activation, float(s.dropout),
                          variational).to(dev)
    data = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    order_rng = generator(seed, "autoencoder batches")
    noise = torch.Generator().manual_seed(int_seed(seed, "vae noise"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=s.lr, weight_decay=s.weight_decay)
    n = int(data.shape[0])
    if n < 2:
        raise ValueError("an autoencoder needs at least two fit rows")
    steps = max(1, n // s.batch_size)
    beta = float(s.beta) if variational else None
    curve = []
    for epoch in range(1, s.epochs + 1):
        net.train()
        order = order_rng.permutation(n)
        total = torch.zeros((), device=dev)
        total_kl = torch.zeros((), device=dev)
        for b in range(steps):
            picked = torch.from_numpy(order[b * s.batch_size:(b + 1) * s.batch_size]).to(dev)
            batch = data[picked]
            if variational:
                mu, logvar = net.posterior(batch)
                eps = torch.randn(tuple(mu.shape), generator=noise).to(dev)
                rebuilt = net.decode(mu + eps * torch.exp(0.5 * logvar))
                recon = ((rebuilt - batch) ** 2).sum(dim=1).mean()
                kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
                loss = recon + beta * kl
                total_kl = total_kl + kl.detach()
            else:
                loss = ((net.decode(net.encode(batch)) - batch) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total = total + loss.detach()
        entry = {"epoch": epoch, "loss": float(total.item()) / steps}
        if variational:
            entry["kl"] = float(total_kl.item()) / steps
        if not math.isfinite(entry["loss"]):
            raise FloatingPointError(f"the autoencoder's loss is {entry['loss']} in epoch {epoch}")
        curve.append(entry)
    net.eval()
    return net, {"curve": curve, "steps_per_epoch": steps, "optimizer": optimizer_record(optimizer),
                 "parameters": parameter_count(net), "device": str(dev)}


def in_batches(fn, x: np.ndarray, batch_size: int, device) -> np.ndarray:
    """fn over the rows of x, batch_size rows at a time, without gradients; float64 [rows, outputs]."""
    import torch

    parts = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            part = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size], dtype=np.float32)).to(device)
            parts.append(fn(part).float().cpu().numpy().astype(np.float64))
    if not parts:
        raise ValueError("no rows")
    return np.concatenate(parts)


def settings_record(s) -> dict:
    out = {}
    for key, value in dataclasses.asdict(s).items():
        out[key] = list(value) if isinstance(value, tuple) else value
    return out


class DeepEmbedder(Embedder):
    variational = False

    def fit_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.net, self.record = train_autoencoder(x, self.s, self.seed, self.device, self.variational)
        self.dev = next(self.net.parameters()).device
        return self.transform(x)

    def can_transform(self):
        return True

    def transform(self, x_new):
        return in_batches(self.net.encode, np.asarray(x_new, dtype=np.float32), self.s.batch_size,
                          self.dev).astype(np.float32)

    def can_inverse(self):
        return True

    def inverse_transform(self, embedding):
        return in_batches(self.net.decode, np.asarray(embedding, dtype=np.float32), self.s.batch_size, self.dev)

    def info(self):
        last = self.record["curve"][-1]
        out = {"final_loss": last["loss"]}
        if "kl" in last:
            out["final_kl"] = last["kl"]
        return out

    def params(self):
        return {**settings_record(self.s), "variational": self.variational, "device": self.record["device"],
                "parameters": self.record["parameters"], "steps_per_epoch": self.record["steps_per_epoch"],
                "optimizer": self.record["optimizer"], "loss_by_epoch": [e["loss"] for e in self.record["curve"]]}


@register_model("autoencoder", settings=AutoencoderSettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="torch",
                profile="torch", packages=("torch",),
                description="an autoencoder's code (PyTorch, linear layers, mean squared error)", example="embed_deep")
class Autoencoder(DeepEmbedder):
    @staticmethod
    def check_job(job) -> None:
        check_autoencoder(job.model, "model autoencoder")


@register_model("vae", settings=VAESettings, family="embedder", tasks=EMBED, kinds=TAB, gpu="torch", profile="torch",
                packages=("torch",),
                description="a variational autoencoder's latent mean (PyTorch, beta-weighted KL)", example="embed_deep")
class VAE(DeepEmbedder):
    variational = True

    @staticmethod
    def check_job(job) -> None:
        check_autoencoder(job.model, "model vae")
