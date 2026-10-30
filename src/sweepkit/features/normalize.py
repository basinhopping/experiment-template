"""Normalization of a torch job's input, per channel, after its feature (sweepkit.features.sequence, image and audio
name the mode in their settings):

    none    the values as they are
    item    each row's channels scaled to mean 0 and SD 1 over that row's own steps (a sequence) or positions (an image)
    global  the same with one mean and SD per channel, from every training row (labels unused)
    group   the same with one mean and SD per channel for each value of a group column (a subject, a site, a device),
            from every row of that group among the job's training, validation and test rows. This uses the inputs
            of validation and test rows (never their labels), so it is transductive: the run records it in
            prep_record.json and any result that uses it should say so

After normalizing, a sequence's padded steps are set to 0. Statistics are computed in float64 and stored as float32.
"""
import numpy as np

NORMALIZATIONS = ("none", "item", "global", "group")


def check_mode(normalization: str, group_column, where: str) -> None:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"{where}: normalization must be one of {NORMALIZATIONS}")
    if (normalization == "group") != (group_column is not None):
        raise ValueError(f"{where}: normalization group names its group_column; the other modes take None")


def _mask(x, lengths):
    """[batch, 1, steps] float mask of a sequence batch, or None for an image."""
    import torch

    if x.dim() != 3:
        return None
    steps = x.shape[-1]
    return (torch.arange(steps, device=x.device)[None, :] < lengths[:, None]).to(x.dtype)[:, None, :]


def _shape(stat, x):
    """A [batch, channels] statistic shaped to broadcast over x."""
    return stat.reshape(tuple(stat.shape) + (1,) * (x.dim() - 2))


class Normalizer:
    """Called with a batch after the feature (x [batch, channels, steps] with lengths, or [batch, channels, height,
    width] with lengths None) and the batch rows' dataset positions (group mode only)."""

    def __init__(self, mode: str, mean=None, std=None, group_of=None):
        if mode in ("global", "group") and (mean is None or std is None):
            raise ValueError(f"{mode} normalization needs its statistics")
        if mode == "group" and group_of is None:
            raise ValueError("group normalization needs the position-to-group map")
        self.mode, self.mean, self.std, self.group_of = mode, mean, std, group_of

    def __call__(self, x, lengths, positions=None):
        import torch

        if self.mode == "none":                             # padding still holds zeros, as the models expect
            return x if lengths is None or x.dim() != 3 else x * _mask(x, lengths)
        mask = _mask(x, lengths)
        if self.mode == "item":
            axes = tuple(range(2, x.dim()))
            if mask is None:
                mean = x.mean(dim=axes, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=axes, keepdim=True)
            else:
                count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                mean = (x * mask).sum(dim=-1, keepdim=True) / count
                var = (((x - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / count
            x = (x - mean) / torch.sqrt(var + 1e-5)
        elif self.mode == "global":
            x = (x - _shape(self.mean[None, :], x)) / _shape(self.std[None, :], x)
        elif self.mode == "group":
            if positions is None or len(positions) != x.shape[0]:
                raise ValueError("group normalization needs the dataset positions of the batch's rows")
            idx = torch.as_tensor(self.group_of[np.asarray(positions)], device=x.device)
            x = (x - _shape(self.mean[idx], x)) / _shape(self.std[idx], x)
        else:
            raise ValueError(self.mode)
        return x if mask is None else x * mask

    def record(self) -> dict:
        """What the normalization learned, for prep_record.json."""
        out = {"mode": self.mode}
        if self.mode in ("global", "group"):
            out["mean"] = np.asarray(self.mean.detach().cpu(), dtype=np.float64).round(6).tolist()
            out["sd"] = np.asarray(self.std.detach().cpu(), dtype=np.float64).round(6).tolist()
        if self.mode == "group":
            out["transductive"] = True
        return out


def statistics(source, feature, batch_size: int, device, group_of=None, n_groups=None) -> tuple:
    """Per channel (and per group when group_of maps dataset positions to group indices): the mean and SD over every
    real step (sequences) or position (images) of the source's rows after the feature. Returns float32 tensors [channels]
    or [groups, channels]; a group with no row among the source's rows gets mean 0 and SD 1."""
    import torch

    from sweepkit.train.batches import eval_batches

    total = sq = count = None
    for ks, x, lengths in eval_batches(source, batch_size):
        # the feature runs on the device; the sums are kept in float64 on the CPU (Apple's MPS has no float64)
        xt = feature(torch.from_numpy(x).to(device)).to("cpu", torch.float64)
        lt = feature.lengths(torch.from_numpy(lengths)).clamp_max(xt.shape[-1]) if lengths is not None else None
        mask = _mask(xt, lt)
        if mask is None:
            weight = torch.ones((xt.shape[0], 1) + tuple(xt.shape[2:]), dtype=xt.dtype, device=xt.device)
        else:
            weight = mask
        axes = tuple(range(2, xt.dim()))
        s = (xt * weight).sum(dim=axes)                       # [batch, channels]
        s2 = (xt * xt * weight).sum(dim=axes)
        n = weight.sum(dim=axes).expand_as(s)                 # [batch, channels]
        if group_of is None:
            s, s2, n = s.sum(dim=0), s2.sum(dim=0), n.sum(dim=0)
            total, sq, count = (s, s2, n) if total is None else (total + s, sq + s2, count + n)
            continue
        if total is None:
            channels = xt.shape[1]
            total = torch.zeros(n_groups, channels, dtype=torch.float64, device=xt.device)
            sq, count = torch.zeros_like(total), torch.zeros_like(total)
        idx = torch.as_tensor(group_of[source.positions[ks]], device=xt.device)
        total.index_add_(0, idx, s)
        sq.index_add_(0, idx, s2)
        count.index_add_(0, idx, n)
    if total is None:
        raise ValueError("no rows to compute normalization statistics from")
    empty = count == 0
    if group_of is None and bool(empty.any()):
        raise ValueError("the training rows have no values for normalization statistics")
    mean = total / count.clamp_min(1.0)
    std = torch.sqrt((sq / count.clamp_min(1.0) - mean * mean).clamp_min(1e-10))
    if group_of is not None:
        # a group with no row among the source's rows is never normalized by the job: mean 0 and SD 1 keep it finite
        mean = torch.where(empty, torch.zeros_like(mean), mean)
        std = torch.where(empty, torch.ones_like(std), std)
    return mean.float().to(device), std.float().to(device)


def group_index(table, column: str) -> tuple:
    """(group names in sorted order, each dataset position's group index)."""
    values = table.column(column).astype(str)
    names = tuple(sorted(set(values.tolist())))
    lookup = {name: i for i, name in enumerate(names)}
    return names, np.array([lookup[v] for v in values.tolist()], dtype=np.int64)
