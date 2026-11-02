"""Copy what the Hugging Face Docker Space needs into a cloned Space repository.

Usage (from the repository root): python deploy/huggingface/make_space.py <path-to-cloned-space>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FILES = ["app.py", "requirements.txt"]
DIRS = ["wohin", "rules", "artifacts/model"]


def main(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(ROOT / name, dest / name)
    for name in DIRS:
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / name, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(HERE / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(HERE / "README.md", dest / "README.md")  # carries the Space settings block
    print(f"Space files written to {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python deploy/huggingface/make_space.py <path-to-cloned-space>")
    main(Path(sys.argv[1]))
