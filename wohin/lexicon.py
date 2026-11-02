"""Text search over common item names in German and English (rules/items.yaml)."""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from wohin import ROOT
from wohin.rules import _key

ITEMS_PATH = ROOT / "rules" / "items.yaml"


@dataclass(frozen=True)
class Item:
    id: str
    names: dict[str, list[str]]
    material: str
    answers: dict[str, str] = field(default_factory=dict)
    verified: bool = True
    hint: dict[str, str] | None = None

    def name(self, lang: str) -> str:
        return self.names[lang][0]


def _norm(text: str) -> str:
    """Lowercase, fold umlauts (ü -> ue) and strip accents, collapse whitespace and dashes."""
    t = text.lower().strip()
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}.items():
        t = t.replace(a, b)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("-", " ")
    return " ".join(t.split())


@lru_cache(maxsize=1)
def load_items() -> tuple[Item, ...]:
    with open(ITEMS_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["items"]
    items = []
    for r in raw:
        answers = {str(k): _key(v) for k, v in (r.get("answers") or {}).items()}
        items.append(
            Item(
                id=r["id"],
                names={"de": list(r["de"]), "en": list(r["en"])},
                material=r["material"],
                answers=answers,
                verified=r.get("verified", True) is not False,
                hint=r.get("hint"),
            )
        )
    return tuple(items)


def search(text: str, limit: int = 5) -> list[tuple[Item, str, dict[str, str]]]:
    """Return up to `limit` matches as (item, material, preset_answers), best first.

    Scoring: exact name match > name starts with query > query contained in name (or vice versa)
    > fuzzy similarity (difflib ratio >= 0.75).
    """
    q = _norm(text)
    if not q:
        return []
    scored: list[tuple[float, int, Item]] = []
    for order, item in enumerate(load_items()):
        best = 0.0
        for name in item.names["de"] + item.names["en"]:
            n = _norm(name)
            if n == q:
                s = 4.0
            elif n.startswith(q) or q.startswith(n):
                s = 3.0 - abs(len(n) - len(q)) / 100
            elif q in n or n in q:
                s = 2.0 - abs(len(n) - len(q)) / 100
            else:
                r = difflib.SequenceMatcher(None, q, n).ratio()
                s = r if r >= 0.75 else 0.0
            best = max(best, s)
        if best > 0:
            scored.append((best, order, item))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(it, it.material, dict(it.answers)) for _, _, it in scored[:limit]]
