"""Streamlit Community Cloud entry point: runs the app in the repository root.

Community Cloud installs the first dependency file it finds, looking next to the entry point before the
repository root. The root holds uv.lock, which pins the CUDA build of torch on Linux (several GB), so this
folder carries its own requirements.txt with CPU-only torch.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
