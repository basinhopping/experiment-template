"""Losses of the torch family, and the concordance correlation the reports use.

    soft_cross_entropy            classification: cross entropy against a target distribution per row (one-hot, or
                                  smoothed: (1 - e) one-hot + e / classes), with optional class weights
    multilabel_loss               multi-label: binary cross entropy with logits per label, with optional positive
                                  weights (negatives / positives per label from the training rows)
    regression_loss               regression, on targets standardized with the training rows' mean and SD: mse, l1,
                                  or ccc (1 minus the mean concordance correlation coefficient over the targets)
    supervised_contrastive_loss   Khosla et al. (2020), equation 2 (L_out), one view of each row
    concordance                   Lin's (1989) concordance correlation coefficient per column
"""
import numpy as np


def soft_cross_entropy(logits, targets, weights):
    import torch

    per_class = -targets * torch.log_softmax(logits.float(), dim=-1)
    if weights is not None:
        per_class = per_class * weights[None, :]
    return per_class.sum(dim=-1).mean()


def multilabel_loss(logits, targets, pos_weight):
    import torch

    return torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float(),
                                                                pos_weight=pos_weight, reduction="mean")


def concordance(pred, target):
    """Lin's (1989) concordance correlation coefficient of each column: 2 cov / (var_pred + var_target + (mean_pred -
    mean_target)^2), with population (1 / n) moments. pred and target: [n, k] float tensors; returns [k]. A column
    whose denominator is 0 (both constant and equal) gives 0. Unchanged when the same affine map is applied to both."""
    pm, tm = pred.mean(dim=0), target.mean(dim=0)
    pc, tc = pred - pm, target - tm
    covariance = (pc * tc).mean(dim=0)
    denominator = (pc * pc).mean(dim=0) + (tc * tc).mean(dim=0) + (pm - tm) ** 2
    return 2.0 * covariance / denominator.clamp_min(1e-12)


def concordance_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """concordance in float64 numpy, for the reports: pred and target [n, k]; returns [k]."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    pm, tm = pred.mean(axis=0), target.mean(axis=0)
    pc, tc = pred - pm, target - tm
    denominator = (pc * pc).mean(axis=0) + (tc * tc).mean(axis=0) + (pm - tm) ** 2
    return 2.0 * (pc * tc).mean(axis=0) / np.maximum(denominator, 1e-12)


def regression_loss(pred, target, kind: str):
    """pred and target: [n, k], the target standardized; in float32 outside autocast."""
    import torch

    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        if kind == "mse":
            return ((p - t) ** 2).mean()
        if kind == "l1":
            return (p - t).abs().mean()
        if kind == "ccc":
            return 1.0 - concordance(p, t).mean()
    raise ValueError(f"unknown regression loss {kind!r}")


def supervised_contrastive_loss(projections, labels, temperature: float):
    """The supervised contrastive loss of one batch with one view of each row (Khosla et al. 2020, equation 2,
    L_out). projections: [n, d], n >= 2, any scale: each row is L2-normalized here, in float32 (float64 stays
    float64). labels: [n] class indices. For anchor i, with A(i) every row but i and P(i) the rows of A(i) with i's
    label:
        loss_i = -(1 / |P(i)|) sum over p in P(i) of log( exp(z_i.z_p / t) / sum over a in A(i) of exp(z_i.z_a / t) )
    Anchors with an empty P(i) are left out and the loss is the mean of loss_i over the others; it is 0, with
    gradients of 0, when every anchor's P(i) is empty. Equation 2 as written: the reference code of the paper also
    multiplies the loss by t / 0.07, which this does not. Computed without autocast, so the similarities are float32
    even when the caller is inside an autocast region."""
    import torch

    with torch.autocast(device_type=projections.device.type, enabled=False):
        z = projections if projections.dtype == torch.float64 else projections.float()
        z = torch.nn.functional.normalize(z, dim=1)
        labels = labels.to(z.device)
        n = z.shape[0]
        own = torch.eye(n, dtype=torch.bool, device=z.device)
        logits = (z @ z.T / temperature).masked_fill(own, float("-inf"))   # i itself is not in A(i)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive = (labels[:, None] == labels[None, :]) & ~own
        n_positive = positive.sum(dim=1)
        per_anchor = log_prob.masked_fill(~positive, 0.0).sum(dim=1) / n_positive.clamp_min(1)
        n_anchors = (n_positive > 0).sum()
        return -per_anchor.sum() / n_anchors.clamp_min(1)
