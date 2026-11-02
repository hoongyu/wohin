"""Prediction smoke test on tiny fake embeddings: runs without downloading any weights."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wohin import CLASSES
from wohin.metrics import choose_threshold, ece, fit_temperature, probs_from_logits, selective
from wohin.predict import is_confident, predict_from_embedding, probabilities


def _fake_bundle(dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(CLASSES), dim)) * 3
    y = np.repeat(np.arange(len(CLASSES)), 20)
    X = centers[y] + rng.normal(size=(len(y), dim))
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    bundle = {"classes": CLASSES, "head": head, "temperature": 1.5, "threshold": 0.5, "backbone": "fake"}
    return bundle, centers


def test_predict_from_embedding_topk():
    bundle, centers = _fake_bundle()
    top = predict_from_embedding(centers[2], bundle, k=3)
    assert len(top) == 3
    assert top[0][0] == CLASSES[2]
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert 0 < sum(probs) <= 1 + 1e-6
    assert is_confident(top, bundle)


def test_probabilities_sum_to_one():
    bundle, centers = _fake_bundle()
    p = probabilities(centers, bundle)
    assert p.shape == (len(CLASSES), len(CLASSES))
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)


def test_temperature_and_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 500)
    logits = rng.normal(size=(500, 4))
    logits[np.arange(500), y] += 2.0
    over = logits * 5  # overconfident
    t = fit_temperature(over, y)
    assert t > 1
    assert ece(probs_from_logits(over, t), y) < ece(probs_from_logits(over), y)
    p = probs_from_logits(logits)
    thr = choose_threshold(p, y, 0.9)
    s = selective(p, y, thr)
    assert 0 < s["coverage"] <= 1
