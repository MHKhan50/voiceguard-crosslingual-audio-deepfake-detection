"""Simulated telephone channel: narrowband filter, 8 kHz, G.711-style mu-law, optional noise."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000
_SOS = butter(6, [300, 3400], btype="bandpass", fs=8000, output="sos")


def mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Encode to 8-bit mu-law and decode back (quantisation loss of a G.711 line)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_sig = float(np.mean(x ** 2)) + 1e-12
    noise = rng.standard_normal(len(x))
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
    return x + noise


def telephone(x: np.ndarray, snr_db: Optional[float] = None, seed: int = 0) -> np.ndarray:
    """16 kHz waveform in, 16 kHz waveform out that has passed through a phone-like line."""
    x = np.asarray(x, dtype=np.float64)
    if snr_db is not None:
        x = add_noise(x, snr_db, np.random.default_rng(seed))
    nb = resample_poly(x, 1, 2)
    if len(nb) > 100:
        nb = sosfiltfilt(_SOS, nb)
    peak = np.max(np.abs(nb)) + 1e-9
    nb = mu_law(nb / peak) * peak if peak > 1 else mu_law(nb)
    y = resample_poly(nb, 2, 1)[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


CONDITIONS = {"clean": None, "phone": "phone", "phone_noise15": 15.0}


def apply_condition(x: np.ndarray, condition: str, seed: int = 0) -> np.ndarray:
    if condition == "clean":
        return np.asarray(x, dtype=np.float32)
    if condition == "phone":
        return telephone(x)
    if condition == "phone_noise15":
        return telephone(x, snr_db=15.0, seed=seed)
    raise ValueError("unknown condition: " + condition)
