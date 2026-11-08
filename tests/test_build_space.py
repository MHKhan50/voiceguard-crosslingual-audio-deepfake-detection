"""The NumPy head used for the browser model must reproduce the scikit-learn detector exactly."""
import importlib.util
import os

import numpy as np

from voiceguard import pipeline

_spec = importlib.util.spec_from_file_location(
    "build_space", os.path.join(os.path.dirname(__file__), "..", "demo", "build_space.py"))
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def _bundle():
    rng = np.random.default_rng(0)
    y = np.array([0] * 150 + [1] * 450)
    x = (rng.standard_normal((600, 12)) + 1.5 * y[:, None]).astype(np.float32)
    model = pipeline.make_model("mlp", 42).fit(x, y)
    cal = pipeline.LLRCalibrator().fit(model.predict_proba(x)[:, 1], y)
    return {"model": model, "calibrator": cal}


def test_head_matches_sklearn_including_saturated_scores():
    b = _bundle()
    head = bs.sklearn_head(b)
    probe = np.concatenate([np.random.default_rng(1).standard_normal((50, 12)) * 3, np.full((2, 12), 40.0),
                            np.full((2, 12), -40.0)]).astype(np.float32)
    ref = b["calibrator"].llr(b["model"].predict_proba(probe)[:, 1])
    assert np.max(np.abs(bs.head_llr(head, probe) - ref)) < 1e-4  # float32 rounding only


def test_verdicts_follow_thresholds():
    llr = np.array([-10.0, 0.0, 10.0, np.log(0.95 / 0.05) + 1e-6])
    assert list(bs.verdicts(llr)) == [0, 1, 2, 2]
