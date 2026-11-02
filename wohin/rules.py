"""Berlin rule engine: material + follow-up answers -> bin outcome.

The rules live in rules/berlin.yaml. Each material has a small decision tree:
a node is either {"ask": question_id, "then": {answer: node}} or {"outcome": outcome_id}.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from wohin import ROOT

RULES_PATH = ROOT / "rules" / "berlin.yaml"
UNSURE = "unsure"
UNSURE_LABEL = {"de": "Ich bin nicht sicher", "en": "I'm not sure"}


@dataclass(frozen=True)
class Question:
    id: str
    text: dict[str, str]
    options: dict[str, dict[str, str]]  # answer value -> {"de":..., "en":...}; includes "unsure"


@dataclass(frozen=True)
class Outcome:
    id: str
    bin_id: str
    bin_label: dict[str, str]
    colour: str
    colour_verified: bool
    text: dict[str, str]
    source_url: str
    verified: str  # "true" | "partial" | "false"
    bsr_az: str


@lru_cache(maxsize=4)
def load_rules(path: Path = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    # YAML 1.1 turns the bare keys yes/no into booleans; normalise them back to strings.
    def norm(obj):
        if isinstance(obj, dict):
            return {_key(k): norm(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [norm(v) for v in obj]
        return obj

    return norm(rules)


def _key(k) -> str:
    if k is True:
        return "yes"
    if k is False:
        return "no"
    return str(k)


def _verified(v) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def materials() -> list[str]:
    return list(load_rules()["flows"].keys())


def get_question(qid: str) -> Question:
    q = load_rules()["questions"][qid]
    options = dict(q["options"])
    options[UNSURE] = UNSURE_LABEL
    return Question(id=qid, text=q["text"], options=options)


def get_outcome(oid: str) -> Outcome:
    rules = load_rules()
    o = rules["outcomes"][oid]
    b = rules["bins"][o["bin"]]
    return Outcome(
        id=oid,
        bin_id=o["bin"],
        bin_label=b["label"],
        colour=b["colour"],
        colour_verified=bool(b.get("colour_verified", False)),
        text=o["text"],
        source_url=rules["sources"][o["source"]],
        verified=_verified(o["verified"]),
        bsr_az=rules["meta"]["bsr_az"],
    )


def _walk(material: str, answers: dict[str, str]) -> dict:
    """Follow the tree as far as the given answers allow; return the node where we stop."""
    flows = load_rules()["flows"]
    if material not in flows:
        raise KeyError(f"unknown material {material!r}")
    node = flows[material]
    while "ask" in node:
        qid = node["ask"]
        if qid not in answers:
            return node
        answer = answers[qid]
        branches = node["then"]
        if answer not in branches:
            # Unknown answers are treated like "I'm not sure".
            answer = UNSURE
        node = branches[answer]
    return node


def next_question(material: str, answers: dict[str, str] | None = None) -> Question | None:
    """The next question to ask, or None when an outcome is reached."""
    node = _walk(material, answers or {})
    return get_question(node["ask"]) if "ask" in node else None


def outcome(material: str, answers: dict[str, str] | None = None) -> Outcome | None:
    """The outcome for these answers, or None if more questions are needed."""
    node = _walk(material, answers or {})
    return get_outcome(node["outcome"]) if "outcome" in node else None


def all_paths(material: str) -> list[tuple[dict[str, str], str]]:
    """Enumerate every (answers, outcome_id) path through a material's tree."""
    out: list[tuple[dict[str, str], str]] = []

    def rec(node: dict, answers: dict[str, str]) -> None:
        if "outcome" in node:
            out.append((dict(answers), node["outcome"]))
            return
        for ans, child in node["then"].items():
            rec(child, {**answers, node["ask"]: ans})

    rec(load_rules()["flows"][material], {})
    return out
