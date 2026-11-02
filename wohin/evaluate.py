"""Test-set evaluation: CLIP zero-shot baseline vs. trained linear head(s), plus plots.

Usage: uv run python -m wohin.evaluate
Writes artifacts/metrics/test.json, docs/img/confusion_matrix.png, docs/img/reliability.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from wohin import CLASSES, ROOT
from wohin.embed import BACKBONES, cache_path
from wohin.metrics import (
    choose_threshold,
    classification_report,
    probs_from_logits,
    reliability_bins,
    selective,
)
from wohin.train import METRICS_DIR, MODEL_DIR, TARGET_ACCURACY, load_split_data

IMG_DIR = ROOT / "docs" / "img"
ZS_TEXT = MODEL_DIR / "zeroshot_text_clip.npy"

# Prompt ensemble for the zero-shot baseline: several names per class x several templates.
ZS_NAMES = {
    "paper": ["paper", "newspaper", "paper waste"],
    "cardboard": ["cardboard", "a cardboard box", "corrugated cardboard"],
    "glass": ["glass", "a glass bottle", "a glass jar"],
    "metal": ["metal", "a metal can", "aluminium"],
    "plastic": ["plastic", "a plastic bottle", "plastic packaging"],
    "organic": ["food waste", "organic waste", "vegetable scraps", "garden waste"],
    "textile": ["textile", "clothes", "fabric"],
    "residual": ["general trash", "mixed rubbish", "miscellaneous garbage"],
}
ZS_TEMPLATES = [
    "a photo of {}.",
    "a photo of {} waste.",
    "a photo of {} in the trash.",
    "a close-up photo of {}.",
]


def zeroshot_text_embeddings() -> np.ndarray:
    """(n_classes, dim) averaged, normalised text embeddings; cached in artifacts/model/."""
    if ZS_TEXT.exists():
        return np.load(ZS_TEXT)
    import torch

    from wohin.embed import load_backbone

    model, _, tokenizer = load_backbone("clip")
    rows = []
    with torch.no_grad():
        for c in CLASSES:
            prompts = [t.format(n) for n in ZS_NAMES[c] for t in ZS_TEMPLATES]
            e = model.encode_text(tokenizer(prompts)).float()
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            rows.append(torch.nn.functional.normalize(e, dim=0).numpy())
    W = np.stack(rows).astype(np.float32)
    ZS_TEXT.parent.mkdir(parents=True, exist_ok=True)
    np.save(ZS_TEXT, W)
    return W


def _block(probs: np.ndarray, d: dict, threshold: float) -> dict:
    out = classification_report(probs, d["y"], CLASSES)
    out["selective"] = selective(probs, d["y"], threshold)
    out["per_source"] = {}
    for s in sorted(set(d["source"])):
        m = d["source"] == s
        r = classification_report(probs[m], d["y"][m], CLASSES)
        present = {c: v for c, v in r["per_class"].items() if v["support"] > 0}
        # macro-F1 over the classes that exist in this source
        idx = [CLASSES.index(c) for c in present]
        from sklearn.metrics import f1_score

        r["macro_f1"] = float(f1_score(d["y"][m], probs[m].argmax(1), labels=idx, average="macro", zero_division=0))
        r["per_class"] = present
        r["selective"] = selective(probs[m], d["y"][m], threshold)
        out["per_source"][s] = r
    return out


def _confusion(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, probs.argmax(1)):
        cm[t, p] += 1
    return cm


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=130)
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(curves: dict[str, tuple[list[dict], float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.4), dpi=130)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    markers = ["o", "s", "^", "D"]
    for (name, (bins, e)), mk in zip(curves.items(), markers):
        xs = [b["confidence"] for b in bins if b["count"] > 0]
        ys = [b["accuracy"] for b in bins if b["count"] > 0]
        ax.plot(xs, ys, marker=mk, label=f"{name} (ECE {e:.3f})")
    ax.set_xlabel("confidence (max probability)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability on the test split")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate() -> dict:
    data = load_split_data("clip")
    va, te = data["val"], data["test"]
    results: dict = {
        "split_sizes": {s: int(len(d["y"])) for s, d in data.items()},
        "classes": CLASSES,
        "threshold_rule": f"lowest max-probability threshold reaching {TARGET_ACCURACY} accuracy on validation",
        "models": {},
    }

    # Zero-shot baseline (same backbone, no training).
    W = zeroshot_text_embeddings()
    scale = 100.0  # CLIP logit scale
    zs_val = probs_from_logits(scale * va["X"] @ W.T)
    zs_test = probs_from_logits(scale * te["X"] @ W.T)
    zs_t = choose_threshold(zs_val, va["y"], TARGET_ACCURACY)
    results["models"]["zeroshot_clip"] = {"description": "OpenCLIP ViT-B/32 zero-shot, prompt ensemble", **_block(zs_test, te, zs_t)}

    curves = {"zero-shot CLIP": (reliability_bins(zs_test, te["y"]), results["models"]["zeroshot_clip"]["ece"])}
    latency = {}
    for b in BACKBONES:
        name = "head.joblib" if b == "clip" else f"head_{b}.joblib"
        if not (MODEL_DIR / name).exists() or not cache_path(b).exists():
            continue
        bundle = joblib.load(MODEL_DIR / name)
        d = load_split_data(b)["test"]
        logits = bundle["head"].decision_function(d["X"])
        raw = probs_from_logits(logits)
        cal = probs_from_logits(logits, bundle["temperature"])
        block = _block(cal, d, bundle["threshold"])
        block["ece_before_temperature"] = classification_report(raw, d["y"], CLASSES)["ece"]
        block["temperature"] = bundle["temperature"]
        block["C"] = bundle["C"]
        block["description"] = f"{b} frozen embeddings + logistic regression, temperature-scaled"
        results["models"][f"head_{b}"] = block
        meta = METRICS_DIR / f"embed_{b}.json"
        if meta.exists():
            latency[b] = json.loads(meta.read_text())["latency_ms_single_image"]
        if b == "clip":
            curves["linear head, before temperature"] = (reliability_bins(raw, d["y"]), block["ece_before_temperature"])
            curves["linear head, temperature-scaled"] = (reliability_bins(cal, d["y"]), block["ece"])
            plot_confusion(_confusion(cal, d["y"]), "Linear head on CLIP ViT-B/32, test split", IMG_DIR / "confusion_matrix.png")
            plot_confusion(_confusion(zs_test, te["y"]), "Zero-shot CLIP ViT-B/32, test split", IMG_DIR / "confusion_matrix_zeroshot.png")
    results["latency_ms_single_image_cpu"] = latency
    plot_reliability(curves, IMG_DIR / "reliability.png")
    return results


def summary_table(results: dict) -> str:
    lines = ["| model | accuracy | macro-F1 | top-3 | ECE | coverage @ threshold | accuracy on covered |", "|---|---|---|---|---|---|---|"]
    for name, r in results["models"].items():
        s = r["selective"]
        lines.append(
            f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['top3_accuracy']:.3f} | {r['ece']:.3f} | "
            f"{s['coverage']:.3f} (t={s['threshold']:.2f}) | {s['accuracy_on_covered']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate()
    (METRICS_DIR / "test.json").write_text(json.dumps(results, indent=2))
    print(summary_table(results))
    for name, r in results["models"].items():
        per = ", ".join(f"{s}: acc {v['accuracy']:.3f} / F1 {v['macro_f1']:.3f}" for s, v in r["per_source"].items())
        print(f"[evaluate] {name} per source -> {per}")
    print("[evaluate] wrote artifacts/metrics/test.json and docs/img/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
