"""Train light classifiers on fixed embeddings, calibrate on dev, evaluate under channel shift."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M


def make_model(kind: str, seed: int = 42):
    if kind == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True,
                            max_iter=200, random_state=seed)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """One-feature logistic regression on the raw score, fitted on dev data only."""

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(_logit(raw_probs).reshape(-1, 1), labels)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(raw_probs).reshape(-1, 1))[:, 1]


def evaluate(labels: np.ndarray, probs: np.ndarray, attacks: List[str],
             known_attacks: List[str]) -> Dict:
    labels = np.asarray(labels)
    res = {
        "n": int(len(labels)),
        "eer": M.eer(labels, probs),
        "eer_ci95": M.bootstrap_eer_ci(labels, probs, n=300),
        "auc": M.auc(labels, probs),
        "ece": M.ece(labels, probs),
        "brier": M.brier(labels, probs),
        "risk_coverage": M.risk_coverage(labels, probs),
        "per_attack_eer": {},
    }
    att = np.asarray(attacks)
    bona = att == "bonafide"
    for a in sorted(set(attacks) - {"bonafide"}):
        m = bona | (att == a)
        res["per_attack_eer"][a] = M.eer(labels[m], probs[m])
    unseen = bona | ~np.isin(att, list(known_attacks) + ["bonafide"])
    res["eer_unseen_attacks"] = M.eer(labels[unseen], probs[unseen])
    return res


def run_experiment(train_x, train_y, dev_x, dev_y, eval_sets: Dict[str, Dict],
                   kind: str, known_attacks: List[str], seed: int = 42) -> Dict:
    """eval_sets: condition -> {"x", "y", "attacks"}."""
    model = make_model(kind, seed).fit(train_x, train_y)
    cal = PlattCalibrator().fit(model.predict_proba(dev_x)[:, 1], np.asarray(dev_y))
    out = {}
    for cond, d in eval_sets.items():
        raw = model.predict_proba(d["x"])[:, 1]
        r = evaluate(d["y"], cal.transform(raw), d["attacks"], known_attacks)
        r["ece_uncalibrated"] = M.ece(d["y"], raw)
        out[cond] = r
    return out


def majority_baseline(train_y, eval_y) -> Dict[str, float]:
    p = float(np.mean(train_y))
    probs = np.full(len(eval_y), p)
    return {"eer": 0.5, "brier": M.brier(eval_y, probs), "accuracy": float(np.mean(np.asarray(eval_y) == int(p >= 0.5)))}
