"""features "logmel": a log-mel spectrogram of audio waveforms, computed in torch on the job's device, one batch at a
time. An optional example of a domain feature: audio rows become sequences of frames, one channel per mel band.

The mel filterbank is built with numpy exactly as librosa.filters.mel builds its default (Slaney mel scale, Slaney
area normalization); a test compares both where librosa is installed. Framing: a Hann window of win_ms, a hop of
hop_ms, centred frames with zero padding, so a waveform of n samples has 1 + n // hop frames. The normalization
(sweepkit.features.normalize) acts on the log-mel bands.

sample_rate is a setting, checked against the dataset's where the dataset declares one.
"""
import dataclasses
from typing import Literal

import numpy as np

from sweepkit.features.normalize import check_mode
from sweepkit.registry import register_feature


@dataclasses.dataclass(frozen=True)
class LogMelSettings:
    sample_rate: int              # the waveforms' sample rate (checked against the dataset's where it declares one)
    win_ms: float                 # analysis window length, milliseconds
    hop_ms: float                 # step between frames, milliseconds
    n_fft: int                    # FFT size, at least the window length in samples
    n_mels: int                   # number of mel bands
    f_min: float                  # lowest band edge, Hz
    f_max: float                  # highest band edge, Hz; at most half the sample rate
    log_offset: float             # added to the mel power before the log, so silence stays finite
    normalization: Literal["none", "item", "global", "group"]
    # item: each band scaled to mean 0, SD 1 over that row's frames; global: with one mean and SD per band from every
    # training frame; group: per value of group_column (for example a recording site), from every row of that group
    # (transductive); none: raw log-mel (loudness kept)
    group_column: str | None


def _slaney_constants() -> tuple:
    """Slaney's mel scale: linear below 1 kHz (200/3 Hz per mel), logarithmic above."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    return f_sp, min_log_hz, min_log_hz / f_sp, np.log(6.4) / 27.0


def hz_to_mel(freq):
    f_sp, min_log_hz, min_log_mel, logstep = _slaney_constants()
    freq = np.asarray(freq, dtype=np.float64)
    log_region = freq >= min_log_hz
    return np.where(log_region, min_log_mel + np.log(np.maximum(freq, 1e-12) / min_log_hz) / logstep, freq / f_sp)


def mel_to_hz(mels):
    f_sp, min_log_hz, min_log_mel, logstep = _slaney_constants()
    mels = np.asarray(mels, dtype=np.float64)
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), f_sp * mels)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """[n_mels, 1 + n_fft // 2] weights, as librosa.filters.mel(htk=False, norm="slaney")."""
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, fft_freqs.shape[0]))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


def window_samples(settings: LogMelSettings) -> tuple:
    """(window length, hop) in samples."""
    rate = settings.sample_rate
    return int(round(settings.win_ms * rate / 1000)), int(round(settings.hop_ms * rate / 1000))


def check(settings: LogMelSettings) -> None:
    """Raise ValueError for settings that cannot work, before any job runs."""
    check_mode(settings.normalization, settings.group_column, "features logmel")
    if settings.sample_rate < 1:
        raise ValueError("features logmel: sample_rate must be positive")
    win, hop = window_samples(settings)
    if hop < 1 or win < 1:
        raise ValueError(f"features logmel: window {win} and hop {hop} samples must be positive")
    if win > settings.n_fft:
        raise ValueError(f"features logmel: window of {win} samples is longer than n_fft {settings.n_fft}")
    if settings.n_fft % 2:
        # with an odd n_fft, torch.stft's centred frames number 1 + (samples - 1) // hop, not 1 + samples // hop
        raise ValueError(f"features logmel: n_fft must be even, got {settings.n_fft}")
    if not 0 <= settings.f_min < settings.f_max <= settings.sample_rate / 2:
        raise ValueError(f"features logmel: need 0 <= f_min < f_max <= {settings.sample_rate / 2}, got "
                         f"{settings.f_min}, {settings.f_max}")
    if settings.n_mels < 1 or settings.log_offset <= 0:
        raise ValueError("features logmel: n_mels must be positive and log_offset above zero")
    fb = mel_filterbank(settings.sample_rate, settings.n_fft, settings.n_mels, settings.f_min, settings.f_max)
    empty = int((fb.max(axis=1) == 0).sum())
    if empty:
        raise ValueError(f"features logmel: {empty} of {settings.n_mels} mel bands contain no FFT bin (n_fft "
                         f"{settings.n_fft}, f_max {settings.f_max}); use fewer bands or a larger n_fft")


def dataset_sample_rate(job) -> int | None:
    """The sample rate the job's dataset declares: the items dataset's wav setting, or the synthetic audio
    generator's; None when the dataset does not say."""
    if job.data_name == "items" and job.data.loader == "wav":
        return job.data.sample_rate
    if job.data_name == "synthetic" and job.data.generator == "audio":
        from sweepkit.data.synthetic import audio_sample_rate

        return audio_sample_rate()
    return None


@register_feature("logmel", settings=LogMelSettings, in_kinds=("audio",), out_kind="sequence",
                  description="log-mel spectrogram of audio waveforms (frames become steps, mel bands channels)",
                  profile="torch", packages=("torch",), example="audio_logmel")
class LogMel:
    """Callable on a batch of waveforms [batch, samples] -> log-mel [batch, n_mels, frames]."""

    @staticmethod
    def check_job(job) -> None:
        check(job.features)
        declared = dataset_sample_rate(job)
        if declared is not None and declared != job.features.sample_rate:
            raise ValueError(f"features logmel: sample_rate {job.features.sample_rate}, but the dataset's waveforms "
                             f"are at {declared} Hz")

    @staticmethod
    def declared_shape(settings, input_shape):
        return (settings.n_mels,)

    @staticmethod
    def is_transductive(section: dict) -> bool:
        """Group normalization takes its statistics from every row of the job (sweepkit.analysis.transductive)."""
        return section.get("normalization") == "group"

    def __init__(self, settings: LogMelSettings, device):
        import torch

        check(settings)
        self.settings = settings
        self.normalization = settings.normalization
        self.group_column = settings.group_column
        self.win, self.hop = window_samples(settings)
        self.n_fft = settings.n_fft
        self.n_mels = settings.n_mels
        self.window = torch.hann_window(self.win, periodic=True, dtype=torch.float32, device=device)
        fb = mel_filterbank(settings.sample_rate, settings.n_fft, settings.n_mels, settings.f_min, settings.f_max)
        self.filterbank = torch.tensor(fb, dtype=torch.float32, device=device)

    def out_shape(self, input_shape: tuple) -> tuple:
        return (self.n_mels,)

    def lengths(self, n_samples):
        """Frames produced for n samples (int or integer tensor)."""
        return 1 + n_samples // self.hop

    def __call__(self, wave):
        import torch

        spec = torch.stft(wave.float(), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window, center=True, pad_mode="constant", return_complex=True)
        power = spec.real.square() + spec.imag.square()            # [batch, freq, frames]
        mel = torch.matmul(self.filterbank, power)                  # [batch, n_mels, frames]
        return torch.log(mel + self.settings.log_offset)
