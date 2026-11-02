"""Fit the linear head on cached embeddings, calibrate it, choose the abstain threshold.

Usage: uv run python -m wohin.train [--backbone clip]   (then evaluates on the test split)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wohin import CLASSES, ROOT
from wohin.embed import BACKBONES, load_cached
from wohin.metrics import choose_threshold, classification_report, fit_temperature, probs_from_logits

MODEL_DIR = ROOT / "artifacts" / "model"
METRICS_DIR = ROOT / "artifacts" / "metrics"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
TARGET_ACCURACY = 0.95
SEED = 42


def load_split_data(backbone: str) -> dict[str, dict]:
    """Return {split: {"X", "y", "source", "path"}} aligned with data/manifest.csv."""
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    X, paths = load_cached(backbone)
    index = {p: i for i, p in enumerate(paths)}
    missing = [p for p in manifest["path"] if p not in index]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest rows have no cached embedding; run wohin.embed")
    rows = manifest["path"].map(index).to_numpy()
    y_all = manifest["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    out = {}
    for split in ["train", "val", "test"]:
        m = (manifest["split"] == split).to_numpy()
        out[split] = {
            "X": X[rows[m]],
            "y": y_all[m],
            "source": manifest["source"].to_numpy()[m],
            "path": manifest["path"].to_numpy()[m],
        }
    return out


def make_head(C: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED),
    )


def train(backbone: str = "clip", save: bool = True) -> dict:
    data = load_split_data(backbone)
    tr, va = data["train"], data["val"]
    t0 = time.time()
    search = []
    for C in C_GRID:
        head = make_head(C).fit(tr["X"], tr["y"])
        f1 = f1_score(va["y"], head.predict(va["X"]), average="macro")
        search.append({"C": C, "val_macro_f1": float(f1)})
        print(f"[train] {backbone} C={C:<6} val macro-F1={f1:.4f}", flush=True)
    best_C = max(search, key=lambda r: r["val_macro_f1"])["C"]
    head = make_head(best_C).fit(tr["X"], tr["y"])
    fit_seconds = time.time() - t0

    val_logits = head.decision_function(va["X"])
    temperature = fit_temperature(val_logits, va["y"])
    val_probs_raw = probs_from_logits(val_logits)
    val_probs = probs_from_logits(val_logits, temperature)
    threshold = choose_threshold(val_probs, va["y"], TARGET_ACCURACY)

    bundle = {
        "backbone": backbone,
        "backbone_config": {k: v for k, v in BACKBONES[backbone].items() if k != "file"},
        "classes": CLASSES,
        "head": head,
        "C": best_C,
        "temperature": temperature,
        "threshold": threshold,
        "target_accuracy": TARGET_ACCURACY,
    }
    report = {
        "backbone": backbone,
        "n_train": int(len(tr["y"])),
        "n_val": int(len(va["y"])),
        "C_search": search,
        "best_C": best_C,
        "temperature": temperature,
        "threshold": threshold,
        "threshold_rule": f"lowest max-probability threshold with validation accuracy >= {TARGET_ACCURACY}",
        "val_uncalibrated": classification_report(val_probs_raw, va["y"], CLASSES),
        "val_calibrated": classification_report(val_probs, va["y"], CLASSES),
        "fit_seconds": round(fit_seconds, 1),
    }
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        name = "head.joblib" if backbone == "clip" else f"head_{backbone}.joblib"
        joblib.dump(bundle, MODEL_DIR / name, compress=3)
        suffix = "" if backbone == "clip" else f"_{backbone}"
        (METRICS_DIR / f"train{suffix}.json").write_text(json.dumps(report, indent=2))
        print(f"[train] saved artifacts/model/{name} (C={best_C}, T={temperature:.3f}, threshold={threshold:.3f})")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="all", choices=["all", *sorted(BACKBONES)])
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    from wohin.embed import cache_path

    backbones = [b for b in BACKBONES if cache_path(b).exists()] if args.backbone == "all" else [args.backbone]
    if "clip" not in backbones:
        print("[train] no cached CLIP embeddings; run `uv run python -m wohin.embed` first")
        return 1
    for b in backbones:
        train(b)
    if not args.no_eval:
        from wohin import evaluate

        evaluate.main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
