"""Bounded audio loading and pitch-aligned harmonic STFT features."""

import math
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .config import Config


def load_audio(path, sample_rate):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "decoded.wav"
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode:
                raise ValueError(f"FFmpeg could not decode audio: {r.stderr}")
            x, sr = sf.read(target, dtype="float32", always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Audio is empty or nonfinite")
    x = x.mean(axis=1)
    if sr != sample_rate:
        gcd = math.gcd(sr, sample_rate)
        x = resample_poly(x, sample_rate // gcd, sr // gcd).astype(np.float32)
    return x


@lru_cache(maxsize=16)
def _harmonic_bins(sample_rate, n_fft, harmonics):
    frequencies = 440 * 2 ** ((torch.arange(128, dtype=torch.float32) - 69) / 12)
    bins = torch.arange(1, harmonics + 1)[:, None] * frequencies[None, :] * n_fft / sample_rate
    valid = bins < n_fft / 2
    low = bins.floor().long().clamp(0, n_fft // 2)
    high = (low + 1).clamp(0, n_fft // 2)
    return low, high, (bins - bins.floor()), valid


def features(audio, config: Config):
    """Returns [2*harmonics, time, 128 pitches], centered at hop-spaced times."""
    x = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    if x.ndim != 1 or not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Expected finite nonempty mono audio")
    spec = torch.stft(
        x,
        n_fft=config.n_fft,
        hop_length=config.hop,
        window=torch.hann_window(config.n_fft),
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).abs()
    low, high, fraction, valid = _harmonic_bins(config.sample_rate, config.n_fft, config.harmonics)
    magnitude = (
        spec[low] * (1 - fraction[:, :, None]) + spec[high] * fraction[:, :, None]
    ) * valid[:, :, None]
    # Fixed normalization avoids per-file loudness destroying velocity information.
    harmonics = torch.log1p(magnitude * 4).permute(0, 2, 1)
    difference = torch.zeros_like(harmonics)
    difference[:, 1:] = torch.relu(harmonics[:, 1:] - harmonics[:, :-1])
    return torch.cat([harmonics, difference], dim=0)
