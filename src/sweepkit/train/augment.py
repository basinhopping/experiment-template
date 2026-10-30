"""Augmentations of training batches (the torch family's fit setting `augment`, a list of these sections). Validation,
test and external rows are never augmented.

Order inside a training step, each from its own random stream (sweepkit.streams), so adding one never changes the
draws of the others:
    speed   audio waveforms, before the feature: every time a row is used, one of `factors` is drawn with equal
            probability and the waveform is resampled by it (1.1 is 10% faster and higher, as sox "speed"); stream
            "augment speed"
    mask    after the feature and the normalization: row_masks bands of up to row_mask_width rows and col_masks spans of
            up to col_mask_width columns set to 0 (the mean, after normalization). A sequence's rows are its channels
            and its columns its steps, and column masks fall inside each row's own steps (SpecAugment, Park et al.
            2019, for spectrograms); an image's rows and columns are its height and width; stream "augment masks"
    noise   Gaussian noise of standard deviation sd added to the input (real steps only); stream "augment noise"
    mixup   each row is mixed with a partner from the same batch (a random permutation, so a row is sometimes its own
            partner): input = w * own + (1 - w) * partner, and every target the loss uses (main and auxiliary) is mixed
            with the same weight, each row's w drawn from Beta(alpha, alpha) (Zhang et al. 2018); a mixed sequence
            counts the steps of the longer of the two, since padding is 0; stream "augment mixup"
"""
import dataclasses
from fractions import Fraction

import numpy as np

from sweepkit.registry import register_augment

AFTER_FEATURES = ("tabular", "sequence", "image")


@dataclasses.dataclass(frozen=True)
class SpeedSettings:
    factors: tuple[float, ...]    # speeds drawn with equal probability, each a ratio of whole numbers up to 100


@dataclasses.dataclass(frozen=True)
class MaskSettings:
    row_masks: int                # masks across rows per training row (0 for none)
    row_mask_width: int           # widest row mask
    col_masks: int                # masks across columns per training row (0 for none)
    col_mask_width: int           # widest column mask


@dataclasses.dataclass(frozen=True)
class NoiseSettings:
    sd: float                     # standard deviation of the noise, in the units of the (normalized) input


@dataclasses.dataclass(frozen=True)
class MixupSettings:
    alpha: float                  # Beta(alpha, alpha) mixing weights; above 0


def speed_ratio(speed: float) -> tuple:
    """(up, down) with up / down = 1 / speed exactly, both at most 100. Raises ValueError when the speed is not such a
    ratio (0.9 is 10 / 9 and 1.1 is 10 / 11; 1.0 is 1 / 1)."""
    if not speed > 0:
        raise ValueError(f"speed {speed} must be positive")
    exact = Fraction(speed).limit_denominator(100)
    if abs(float(exact) - speed) > 1e-9:
        raise ValueError(f"speed {speed} is not a ratio of whole numbers up to 100 (such as 0.9 or 1.05)")
    return exact.denominator, exact.numerator


@register_augment("speed", settings=SpeedSettings, kinds=("audio",), profile="torch", packages=("scipy",),
                  description="speed perturbation of audio waveforms (polyphase resampling)")
class Speed:
    order = 0

    @staticmethod
    def check(s: SpeedSettings) -> None:
        if not s.factors:
            raise ValueError("augment speed: list at least one factor")
        if len(set(s.factors)) != len(s.factors):
            raise ValueError(f"augment speed: factors repeat: {list(s.factors)}")
        for factor in s.factors:
            if not 0.5 <= factor <= 2.0:
                raise ValueError(f"augment speed: {factor} is outside 0.5 to 2.0")
            speed_ratio(factor)


@register_augment("mask", settings=MaskSettings, kinds=("sequence", "image"), profile="torch", packages=("torch",),
                  description="masks across rows and columns (SpecAugment for spectrograms)")
class Mask:
    order = 1

    @staticmethod
    def check(s: MaskSettings) -> None:
        for name in ("row_masks", "row_mask_width", "col_masks", "col_mask_width"):
            if getattr(s, name) < 0:
                raise ValueError(f"augment mask: {name} must not be negative")
        if s.row_masks + s.col_masks == 0:
            raise ValueError("augment mask: give at least one mask (or leave the augmentation out)")


@register_augment("noise", settings=NoiseSettings, kinds=AFTER_FEATURES, profile="torch", packages=("torch",),
                  description="Gaussian noise added to the input")
class Noise:
    order = 2

    @staticmethod
    def check(s: NoiseSettings) -> None:
        if not s.sd > 0:
            raise ValueError("augment noise: sd must be above 0")


@register_augment("mixup", settings=MixupSettings, kinds=AFTER_FEATURES, profile="torch", packages=("torch",),
                  description="mixup of rows and targets within a batch")
class Mixup:
    order = 3

    @staticmethod
    def check(s: MixupSettings) -> None:
        if not s.alpha > 0:
            raise ValueError("augment mixup: alpha must be above 0")


# Speed perturbation ----------------------------------------------------------------------------------------------

_CACHE: dict = {}       # id(items) -> (items, {(position, speed): resampled waveform}), the newest last


def change_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """A float32 waveform played at the given speed (polyphase resampling, Kaiser window with beta 5, as sox does it
    closely enough), so a row of n samples becomes ceil(n / speed) samples long."""
    if samples.ndim != 1:
        raise ValueError("change_speed takes one waveform")
    if speed == 1.0:
        return samples
    from scipy.signal import resample_poly

    up, down = speed_ratio(speed)
    return resample_poly(samples.astype(np.float64), up, down, window=("kaiser", 5.0)).astype(np.float32)


def at_speed(items: list, position: int, speed: float) -> np.ndarray:
    """items[position] at the given speed, resampled once per process and kept (it depends only on the waveform and
    the speed, so keeping it changes no result)."""
    if speed == 1.0:
        return items[position]
    entry = _CACHE.get(id(items))
    if entry is None or entry[0] is not items:
        entry = (items, {})
        _CACHE.pop(id(items), None)
        _CACHE[id(items)] = entry
        while len(_CACHE) > 4:          # the item lists of the last four datasets, as the torch runner keeps them
            del _CACHE[next(iter(_CACHE))]
    key = (int(position), float(speed))
    if key not in entry[1]:
        entry[1][key] = change_speed(items[position], speed)
    return entry[1][key]


def draw_speeds(factors: tuple, n: int, rng: np.random.Generator) -> np.ndarray:
    """One speed per row, each factor with equal probability. With a single factor nothing is drawn."""
    values = np.asarray(factors, dtype=np.float64)
    if values.shape[0] == 1:
        return np.full(n, values[0])
    return values[rng.integers(0, values.shape[0], size=n)]


def cached_bytes() -> int:
    return int(sum(wave.nbytes for _, waves in _CACHE.values() for wave in waves.values()))


# Masks, noise and mixup (torch, on the job's device) --------------------------------------------------------------

def _band_mask(n_masks: int, max_width: int, limit, size: int, generator):
    """[batch, size] bool: n_masks spans of width 0..max_width starting inside [0, limit - width]."""
    import torch

    batch = limit.shape[0]
    widths = torch.randint(0, max_width + 1, (batch, n_masks), generator=generator)
    room = (limit[:, None] - widths).clamp_min(0)
    starts = (torch.rand(batch, n_masks, generator=generator) * (room + 1)).floor().long()
    position = torch.arange(size)[None, None, :]
    inside = (position >= starts[..., None]) & (position < (starts + widths)[..., None])
    return inside.any(dim=1)


def axis_masks(x, lengths, s: MaskSettings, generator):
    """x: [batch, rows, columns] (a sequence: channels by steps, lengths [batch]) or [batch, channels, rows,
    columns] (an image, lengths None). generator: a CPU torch.Generator, so the masks follow the job's seed."""
    import torch

    image = x.dim() == 4
    batch = x.shape[0]
    rows, cols = x.shape[-2], x.shape[-1]
    masked = x
    if s.row_masks > 0 and s.row_mask_width > 0:
        limit = torch.full((batch,), rows, dtype=torch.long)
        band = _band_mask(s.row_masks, s.row_mask_width, limit, rows, generator).to(x.device)
        masked = masked.masked_fill(band[:, None, :, None] if image else band[:, :, None], 0.0)
    if s.col_masks > 0 and s.col_mask_width > 0:
        if image or lengths is None:
            limit = torch.full((batch,), cols, dtype=torch.long)
        else:
            limit = lengths.detach().to("cpu").long().clamp_max(cols)
        span = _band_mask(s.col_masks, s.col_mask_width, limit, cols, generator).to(x.device)
        masked = masked.masked_fill(span[:, None, None, :] if image else span[:, None, :], 0.0)
    return masked


def add_noise(x, lengths, s: NoiseSettings, generator):
    """Gaussian noise on every value of the input (real steps only for a sequence)."""
    import torch

    noise = torch.randn(tuple(x.shape), generator=generator).to(device=x.device, dtype=x.dtype) * s.sd
    if lengths is not None and x.dim() == 3:
        steps = x.shape[-1]
        mask = (torch.arange(steps, device=x.device)[None, :] < lengths[:, None]).to(x.dtype)
        noise = noise * mask[:, None, :]
    return x + noise


def mixup_draw(alpha: float, n: int, rng: np.random.Generator) -> tuple:
    """(weights [n] float32 from Beta(alpha, alpha), partner of each row [n])."""
    if alpha <= 0:
        raise ValueError("mixup needs alpha above 0")
    weights = rng.beta(alpha, alpha, size=n).astype(np.float32)
    partner = rng.permutation(n)
    return weights, partner


def mixup_mix(x, lengths, targets: list, weights: np.ndarray, partner: np.ndarray) -> tuple:
    """x [batch, ...] and lengths [batch] (or None) on the device; targets: [batch, k] tensors (or None, kept None).
    Returns (mixed x, the longer of the two lengths, mixed targets in the same order)."""
    import torch

    w = torch.as_tensor(weights, device=x.device)
    index = torch.as_tensor(partner, device=x.device, dtype=torch.long)
    wx = w.to(x.dtype).reshape((-1,) + (1,) * (x.dim() - 1))
    mixed = wx * x + (1.0 - wx) * x[index]
    mixed_lengths = torch.maximum(lengths, lengths[index]) if lengths is not None else None
    out = []
    for t in targets:
        if t is None:
            out.append(None)
            continue
        wt = w.to(t.dtype).reshape((-1,) + (1,) * (t.dim() - 1))
        out.append(wt * t + (1.0 - wt) * t[index])
    return mixed, mixed_lengths, out
