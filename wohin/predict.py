"""Image -> top-k materials with calibrated probabilities.

Needs only the committed head (artifacts/model/head.joblib) plus the backbone weights,
which are downloaded on first use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from wohin import ROOT
from wohin.metrics import probs_from_logits

HEAD_PATH = ROOT / "artifacts" / "model" / "head.joblib"


@lru_cache(maxsize=2)
def load_head(path: Path = HEAD_PATH) -> dict:
    return joblib.load(path)


def probabilities(embeddings: np.ndarray, bundle: dict) -> np.ndarray:
    """(n, dim) embeddings -> (n, n_classes) temperature-scaled probabilities."""
    X = np.atleast_2d(embeddings)
    logits = bundle["head"].decision_function(X)
    return probs_from_logits(logits, bundle["temperature"])


def predict_from_embedding(embedding: np.ndarray, bundle: dict, k: int = 3) -> list[tuple[str, float]]:
    p = probabilities(embedding, bundle)[0]
    order = np.argsort(-p)[:k]
    return [(bundle["classes"][i], float(p[i])) for i in order]


def is_confident(top: list[tuple[str, float]], bundle: dict) -> bool:
    """True if the top prediction clears the abstain threshold chosen on validation."""
    return bool(top) and top[0][1] >= bundle["threshold"]


def predict(image: Image.Image, k: int = 3, bundle: dict | None = None) -> list[tuple[str, float]]:
    from wohin.embed import embed_pil

    bundle = bundle or load_head()
    emb = embed_pil([image], bundle["backbone"])
    return predict_from_embedding(emb, bundle, k)


if __name__ == "__main__":
    import sys

    b = load_head()
    for p in sys.argv[1:]:
        top = predict(Image.open(p), bundle=b)
        flag = "confident" if is_confident(top, b) else "ask user"
        print(p, [(c, round(v, 3)) for c, v in top], flag)
