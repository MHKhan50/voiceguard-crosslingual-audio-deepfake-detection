import numpy as np

from voiceguard import metrics as M


def test_eer_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert M.eer(y, y.astype(float)) == 0.0
    rng = np.random.default_rng(0)
    assert 0.35 < M.eer(y, rng.random(100)) < 0.65


def test_eer_known_value():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.9, 0.95])
    assert abs(M.eer(y, s) - 0.25) < 1e-9


def test_ece_and_brier():
    y = np.array([1, 1, 0, 0])
    assert M.ece(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert abs(M.ece(y, np.array([0.9, 0.9, 0.9, 0.9])) - 0.4) < 1e-9
    assert M.brier(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0


def test_risk_coverage_drops_with_abstention():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.99, 0.95, 0.02, 0.6, 0.45])
    rc = {r["coverage"]: r["error"] for r in M.risk_coverage(y, p, (1.0, 0.6))}
    assert rc[0.6] == 0.0 and rc[1.0] == 0.4
