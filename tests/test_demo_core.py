"""Demo scoring logic without Gradio, torch or the real model."""
import importlib.util
import os

import numpy as np

from voiceguard import pipeline

_spec = importlib.util.spec_from_file_location(
    "demo_core", os.path.join(os.path.dirname(__file__), "..", "demo", "core.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


def test_to_mono_16k_handles_int16_stereo_44k():
    t = np.arange(44100) / 44100
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    x = core.to_mono_16k(44100, np.stack([tone, tone], axis=1))
    assert x.dtype == np.float32 and abs(len(x) - 16000) <= 1 and 0.45 < np.abs(x).max() < 0.55


def test_check_clip_rejects_short_and_silent_and_trims():
    assert core.check_clip(np.ones(8000, np.float32) * 0.1, 4.0)[0] is None
    assert core.check_clip(np.zeros(32000, np.float32), 4.0)[0] is None
    x, err = core.check_clip(np.random.default_rng(0).standard_normal(16000 * 6).astype(np.float32) * 0.1, 4.0)
    assert err is None and len(x) == 64000


def test_score_verdict_and_overall():
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 200)
    x = (rng.standard_normal((400, 8)) + 2 * y[:, None]).astype(np.float32)
    model = pipeline.make_model("logreg").fit(x, y)
    cal = pipeline.LLRCalibrator().fit(model.predict_proba(x)[:, 1], y)
    bundle = {"model": model, "calibrator": cal}
    fake, real = core.score(bundle, np.full(8, 4.0)), core.score(bundle, np.full(8, -2.0))
    assert fake["llr"] > 0 > real["llr"]
    assert core.verdict(fake["p_fake"]) == "Likely AI-generated" and core.verdict(real["p_fake"]) == "Likely real"
    assert core.verdict(0.5) == "Uncertain"
    assert core.overall(fake, real).startswith("Uncertain") and core.overall(fake, None) == "Likely AI-generated"
