import pandas as pd
import pytest

from wohin import CLASSES, ROOT
from wohin.data import assign_splits

MANIFEST = ROOT / "data" / "manifest.csv"


def _toy():
    rows = []
    for src in ["a", "b"]:
        for c in CLASSES[:4]:
            for i in range(20):
                rows.append({"path": f"{src}/{c}/{i}.jpg", "source": src, "original_label": c, "label": c})
    return pd.DataFrame(rows)


def test_assign_splits_disjoint_and_stratified():
    df = assign_splits(_toy())
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["path"].is_unique
    fr = df.groupby("split").size() / len(df)
    assert abs(fr["train"] - 0.70) < 0.03
    # every class appears in every split for every source
    assert (df.groupby(["source", "label", "split"]).size() > 0).all()
    assert len(df.groupby(["source", "label", "split"])) == 2 * 4 * 3


def test_assign_splits_deterministic():
    a = assign_splits(_toy())
    b = assign_splits(_toy())
    assert (a["split"] == b["split"]).all()


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not built")
def test_manifest_no_path_in_two_splits():
    df = pd.read_csv(MANIFEST)
    assert df["path"].is_unique
    assert df.groupby("path")["split"].nunique().max() == 1
    assert set(df["label"]) <= set(CLASSES)
    assert set(df["split"]) == {"train", "val", "test"}
