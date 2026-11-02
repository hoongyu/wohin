"""Download TrashNet and RealWaste, map labels to 8 unified classes, write a split manifest.

Usage: uv run python -m wohin.data
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from wohin import CLASSES, ROOT

RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
SEED = 42
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

TRASHNET_ZIP_URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
TRASHNET_GIT = "https://github.com/garythung/trashnet"
REALWASTE_ZIP_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
REALWASTE_GIT = "https://github.com/sam-single/realwaste"

# Original label -> unified label.
TRASHNET_MAP = {
    "glass": "glass",
    "paper": "paper",
    "cardboard": "cardboard",
    "plastic": "plastic",
    "metal": "metal",
    "trash": "residual",
}
REALWASTE_MAP = {
    "Cardboard": "cardboard",
    "Food Organics": "organic",
    "Glass": "glass",
    "Metal": "metal",
    "Miscellaneous Trash": "residual",
    "Paper": "paper",
    "Plastic": "plastic",
    "Textile Trash": "textile",
    "Vegetation": "organic",
}


def _log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def _download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns False on any network error (caller falls back)."""
    try:
        _log(f"downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception as e:  # noqa: BLE001 - any failure triggers the fallback
        _log(f"download failed ({e}); using fallback")
        return False


def _git_clone(url: str, dest: Path) -> None:
    _log(f"git clone --depth 1 {url}")
    subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dest)], check=True)


def fetch_trashnet() -> Path:
    """Return the directory holding the six TrashNet class folders."""
    out = RAW / "trashnet"
    if (out / "dataset-resized").is_dir():
        return out / "dataset-resized"
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "trashnet.zip"
    if not zpath.exists() and not _download(TRASHNET_ZIP_URL, zpath):
        repo = RAW / "trashnet-git"
        if not repo.exists():
            _git_clone(TRASHNET_GIT, repo)
        zpath = repo / "data" / "dataset-resized.zip"
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(out, members=members)
    return out / "dataset-resized"


def fetch_realwaste() -> Path:
    """Return the directory holding the nine RealWaste class folders."""
    candidates = [RAW / "realwaste" / "RealWaste", RAW / "realwaste-uci" / "realwaste-main" / "RealWaste"]
    for c in candidates:
        if c.is_dir():
            return c
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "realwaste.zip"
    if zpath.exists() or _download(REALWASTE_ZIP_URL, zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW / "realwaste-uci")
        for d in (RAW / "realwaste-uci").rglob("RealWaste"):
            if d.is_dir():
                return d
    _git_clone(REALWASTE_GIT, RAW / "realwaste")
    return RAW / "realwaste" / "RealWaste"


def _scan(root: Path, source: str, mapping: dict[str, str]) -> list[dict]:
    rows = []
    for original, unified in mapping.items():
        folder = root / original
        if not folder.is_dir():
            raise FileNotFoundError(f"missing class folder {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMAGE_EXT:
                rows.append(
                    {
                        "path": p.relative_to(ROOT).as_posix(),
                        "source": source,
                        "original_label": original,
                        "label": unified,
                    }
                )
    return rows


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified 70/15/15 split by unified label, separately within each source dataset."""
    df = df.copy()
    df["split"] = ""
    for _, g in df.groupby("source"):
        train_idx, rest_idx = train_test_split(g.index, test_size=0.30, stratify=g["label"], random_state=seed)
        rest = g.loc[rest_idx]
        val_idx, test_idx = train_test_split(rest.index, test_size=0.50, stratify=rest["label"], random_state=seed)
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"
    return df


def build_manifest() -> pd.DataFrame:
    rows = _scan(fetch_trashnet(), "trashnet", TRASHNET_MAP) + _scan(fetch_realwaste(), "realwaste", REALWASTE_MAP)
    df = assign_splits(pd.DataFrame(rows))
    assert set(df["label"]) <= set(CLASSES)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    return df


def main() -> int:
    df = build_manifest()
    _log(f"wrote {MANIFEST.relative_to(ROOT)} with {len(df)} rows")
    counts = df.pivot_table(index="label", columns=["source", "split"], aggfunc="size", fill_value=0)
    counts = counts.reindex(CLASSES).fillna(0).astype(int)
    print(counts.to_string())
    print(df.groupby(["source", "split"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
