"""Scoring logic of the VoiceGuard demo, kept apart from the user interface so it can be tested without Gradio.

The feature recipe repeats src/voiceguard/extract.py for one clip: first 4 s at 16 kHz, waveform normalised per clip,
WavLM Base+ hidden state 6 mean-pooled over the frames that cover real audio.
"""
from __future__ import annotations

from math import gcd
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

SR = 16000
MIN_SECONDS = 1.0
# Posterior thresholds for the three verdicts, at an even prior (a design choice of the demo, not a tuned value).
FAKE_AT = 0.95
REAL_AT = 0.05


def to_mono_16k(sample_rate: int, data: np.ndarray) -> np.ndarray:
    """Gradio numpy audio (any integer or float type, mono or multi-channel) to float32 mono at 16 kHz."""
    x = np.asarray(data)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    if sample_rate != SR:
        g = gcd(int(sample_rate), SR)
        x = resample_poly(x, SR // g, int(sample_rate) // g).astype(np.float32)
    return x


def check_clip(x: np.ndarray, max_seconds: float) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return the part of the clip the model uses, or an error message."""
    if len(x) < MIN_SECONDS * SR:
        return None, "The clip is shorter than 1 second. Please use 1 to 10 seconds of speech."
    if float(np.sqrt(np.mean(x ** 2))) < 1e-4:
        return None, "The clip is silent or almost silent."
    return x[: int(max_seconds * SR)].astype(np.float32), None


def wavlm_features(model, x: np.ndarray, layer: int) -> np.ndarray:
    """Mean-pooled hidden state `layer` of WavLM for one 16 kHz clip (matches the batch code used in training)."""
    import torch

    with torch.no_grad():
        wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        mean = wav.mean(1, keepdim=True)
        var = ((wav - mean) ** 2).mean(1, keepdim=True)
        norm = (wav - mean) / torch.sqrt(var + 1e-7)
        mask = torch.ones_like(norm, dtype=torch.long)
        out = model(norm, attention_mask=mask, output_hidden_states=True)
        frames = out.hidden_states[0].shape[1]
        n = int(min(max(len(x) // 320, 1), frames))
        pooled = out.hidden_states[layer][0, :n].float().mean(0)
    return np.nan_to_num(pooled.numpy()).astype(np.float32)


def score(bundle: Dict, features: np.ndarray) -> Dict[str, float]:
    """Calibrated log-likelihood ratio and the probability of a fake at an even prior."""
    raw = bundle["model"].predict_proba(features.reshape(1, -1).astype(np.float32))[:, 1]
    llr = float(bundle["calibrator"].llr(raw)[0])
    return {"llr": llr, "p_fake": float(1.0 / (1.0 + np.exp(-llr)))}


def verdict(p_fake: float) -> str:
    if p_fake >= FAKE_AT:
        return "Likely AI-generated"
    if p_fake <= REAL_AT:
        return "Likely real"
    return "Uncertain"


def overall(clean: Dict[str, float], phone: Optional[Dict[str, float]]) -> str:
    """Combine the two checks; disagreement between them is reported as uncertain."""
    v = verdict(clean["p_fake"])
    if phone is not None and verdict(phone["p_fake"]) != v:
        return "Uncertain (the result changes over a simulated phone line)"
    return v
