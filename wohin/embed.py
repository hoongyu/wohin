"""Load a frozen backbone (with download fallbacks) and cache image embeddings.

Usage: uv run python -m wohin.embed [--backbone clip|effnet]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wohin import ROOT

EMB_DIR = ROOT / "artifacts" / "embeddings"
WEIGHTS_DIR = ROOT / "artifacts" / "weights"

# The design asks for OpenCLIP ViT-B/32 laion2b_s34b_b79k (hosted on Hugging Face). Hugging Face was
# blocked in the build environment, so we use the LAION-2B checkpoint that OpenCLIP publishes as a
# GitHub release (`laion2b_e16`). Training and the app must use the same weights, so this is pinned.
BACKBONES = {
    "clip": {
        "kind": "open_clip",
        "arch": "ViT-B-32",
        "pretrained": "laion2b_e16",
        "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth",
        "file": "vit_b_32-laion2b_e16-af8dbd0c.pth",
    },
    "effnet": {
        "kind": "timm",
        "arch": "efficientnet_b0",
        "url": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth",
        "file": "efficientnet_b0_ra-3dd342df.pth",
    },
}


def _log(msg: str) -> None:
    print(f"[embed] {msg}", flush=True)


def _fetch_weights(name: str) -> Path:
    cfg = BACKBONES[name]
    path = WEIGHTS_DIR / cfg["file"]
    if not path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _log(f"downloading {cfg['url']}")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(cfg["url"], timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


@lru_cache(maxsize=2)
def load_backbone(name: str = "clip"):
    """Return (model, preprocess, tokenizer_or_None) in eval mode on CPU."""
    cfg = BACKBONES[name]
    torch.set_grad_enabled(False)
    if cfg["kind"] == "open_clip":
        import open_clip

        local = WEIGHTS_DIR / cfg["file"]
        if local.exists():
            pretrained = str(local)
        else:
            try:  # OpenCLIP's own download (GitHub release or HF mirror for this tag)
                model, _, preprocess = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"])
                return model.eval(), preprocess, open_clip.get_tokenizer(cfg["arch"])
            except Exception as e:  # noqa: BLE001
                _log(f"open_clip download failed ({e}); fetching {cfg['url']}")
                pretrained = str(_fetch_weights(name))
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["arch"], pretrained=pretrained)
        return model.eval(), preprocess, open_clip.get_tokenizer(cfg["arch"])

    import timm

    try:
        model = timm.create_model(cfg["arch"], pretrained=True, num_classes=0)
    except Exception as e:  # noqa: BLE001
        _log(f"timm download failed ({e}); fetching {cfg['url']}")
        model = timm.create_model(cfg["arch"], pretrained=False, num_classes=0)
        state = torch.load(_fetch_weights(name), map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        model.load_state_dict(state)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    preprocess = timm.data.create_transform(**data_cfg)
    return model.eval(), preprocess, None


def encode_images(model, images: torch.Tensor, name: str) -> np.ndarray:
    """Images (already preprocessed) -> L2-normalised embeddings."""
    with torch.no_grad():
        if BACKBONES[name]["kind"] == "open_clip":
            feats = model.encode_image(images)
        else:
            feats = model(images)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def embed_pil(images: list[Image.Image], name: str = "clip") -> np.ndarray:
    model, preprocess, _ = load_backbone(name)
    batch = torch.stack([preprocess(im.convert("RGB")) for im in images])
    return encode_images(model, batch, name)


class _Files(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(ROOT / self.paths[i]) as im:
            return self.preprocess(im.convert("RGB"))


def embed(paths: list[str], name: str = "clip", batch_size: int = 64, workers: int = 3) -> np.ndarray:
    """Embed image files (paths relative to the repo root)."""
    model, preprocess, _ = load_backbone(name)
    loader = torch.utils.data.DataLoader(_Files(paths, preprocess), batch_size=batch_size, num_workers=workers)
    out = []
    done = 0
    t0 = time.time()
    for batch in loader:
        out.append(encode_images(model, batch, name))
        done += len(batch)
        if done % (batch_size * 10) == 0 or done == len(paths):
            _log(f"{done}/{len(paths)} images, {done / (time.time() - t0):.1f} img/s")
    return np.concatenate(out)


def cache_path(name: str) -> Path:
    return EMB_DIR / f"{name}.npz"


def load_cached(name: str = "clip") -> tuple[np.ndarray, list[str]]:
    d = np.load(cache_path(name), allow_pickle=False)
    return d["X"], [str(p) for p in d["paths"]]


def measure_latency(name: str, n: int = 30) -> float:
    """Median single-image latency in ms (preprocess + forward, batch of 1, CPU)."""
    model, preprocess, _ = load_backbone(name)
    im = Image.fromarray((np.random.default_rng(0).random((384, 512, 3)) * 255).astype(np.uint8))
    times = []
    for _ in range(n):
        t = time.perf_counter()
        encode_images(model, preprocess(im).unsqueeze(0), name)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times[5:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=sorted(BACKBONES))
    args = ap.parse_args()
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    paths = manifest["path"].tolist()
    torch.set_num_threads(4)
    t0 = time.time()
    load_backbone(args.backbone)
    t_load = time.time() - t0
    t0 = time.time()
    X = embed(paths, args.backbone)
    t_embed = time.time() - t0
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path(args.backbone), X=X, paths=np.array(paths))
    latency = measure_latency(args.backbone)
    meta = {
        "backbone": args.backbone,
        "config": {k: v for k, v in BACKBONES[args.backbone].items() if k != "file"},
        "n_images": len(paths),
        "dim": int(X.shape[1]),
        "load_seconds": round(t_load, 1),
        "embed_seconds": round(t_embed, 1),
        "images_per_second": round(len(paths) / t_embed, 1),
        "latency_ms_single_image": round(latency, 1),
        "torch_threads": torch.get_num_threads(),
    }
    (EMB_DIR / f"{args.backbone}.json").write_text(json.dumps(meta, indent=2))
    # keep a committed copy of the timing for the build log
    timing_dir = ROOT / "artifacts" / "metrics"
    timing_dir.mkdir(parents=True, exist_ok=True)
    (timing_dir / f"embed_{args.backbone}.json").write_text(json.dumps(meta, indent=2))
    _log(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
