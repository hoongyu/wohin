"""Metric helpers shared by training and evaluation (pure numpy / scikit-learn)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising the negative log-likelihood of softmax(logits / T)."""

    def nll(log_t: float) -> float:
        t = np.exp(log_t)
        return -log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean()

    res = minimize_scalar(nll, bounds=(np.log(0.01), np.log(100.0)), method="bounded")
    return float(np.exp(res.x))


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": int(m.sum()),
                "accuracy": float(correct[m].mean()) if m.any() else None,
                "confidence": float(conf[m].mean()) if m.any() else None,
            }
        )
    return out


def choose_threshold(probs: np.ndarray, y: np.ndarray, target_accuracy: float = 0.95) -> float:
    """Lowest confidence threshold whose accepted predictions reach the target accuracy.

    Predictions with max probability below the threshold are not trusted: the app asks the user
    to pick the material from the top three instead. Falls back to the threshold with the best
    selective accuracy if the target is never reached.
    """
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    candidates = np.unique(np.round(conf, 3))
    best_t, best_acc = float(candidates.max()), -1.0
    for t in sorted(candidates):
        m = conf >= t
        if m.sum() < max(10, 0.05 * len(y)):
            break
        acc = correct[m].mean()
        if acc >= target_accuracy:
            return float(t)
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def selective(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    conf = probs.max(1)
    m = conf >= threshold
    correct = probs.argmax(1) == y
    top3 = np.argsort(-probs, axis=1)[:, :3]
    in_top3 = (top3 == y[:, None]).any(1)
    return {
        "threshold": float(threshold),
        "coverage": float(m.mean()),
        "accuracy_on_covered": float(correct[m].mean()) if m.any() else None,
        "abstained": int((~m).sum()),
        "top3_contains_true_on_abstained": float(in_top3[~m].mean()) if (~m).any() else None,
    }


def classification_report(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": float((top3 == y[:, None]).any(1).mean()),
        "ece": ece(probs, y),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
    }
