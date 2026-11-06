"""Detection and calibration metrics. Positive class = spoof (1)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eer(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Equal error rate: where false accepts of fakes equal false rejects of real speech."""
    y = np.asarray(labels)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, np.asarray(scores))
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels)
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, scores))


def brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def ece(labels: Sequence[int], probs: Sequence[float], n_bins: int = 15) -> float:
    """Expected calibration error on the confidence of the predicted class."""
    y = np.asarray(labels)
    p = np.asarray(probs, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def risk_coverage(labels: Sequence[int], probs: Sequence[float],
                  coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5)) -> List[Dict[str, float]]:
    """Error rate when the detector answers only its most confident share of clips."""
    y = np.asarray(labels)
    p = np.asarray(probs, dtype=float)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf, kind="stable")
    wrong = ((p >= 0.5).astype(int) != y)[order]
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        out.append({"coverage": float(c), "error": float(wrong[:k].mean())})
    return out


def bootstrap_eer_ci(labels: Sequence[int], scores: Sequence[float], n: int = 500,
                     seed: int = 42) -> List[float]:
    y = np.asarray(labels)
    s = np.asarray(scores)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = eer(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
