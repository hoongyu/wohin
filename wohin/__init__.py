"""Wohin: photo-based waste sorting assistant for Berlin."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["paper", "cardboard", "glass", "metal", "plastic", "organic", "textile", "residual"]
