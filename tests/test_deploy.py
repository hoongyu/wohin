"""Deployment files: the Streamlit Community Cloud entry point and its requirements."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from wohin import ROOT

CLOUD = ROOT / "deploy" / "streamlit"


def _requirements(path: Path) -> list[str]:
    lines = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def test_cloud_requirements_match_root():
    assert _requirements(CLOUD / "requirements.txt") == _requirements(ROOT / "requirements.txt")


def test_cloud_entry_point_runs_the_app():
    at = AppTest.from_file(str(CLOUD / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.text_input(key="query").input("pizza box").run()
    assert not at.exception
    assert any("Cardboard" in m.value for m in at.markdown)
