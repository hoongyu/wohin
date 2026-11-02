import pytest

from wohin import CLASSES
from wohin.lexicon import load_items, search
from wohin.rules import get_question, load_rules, next_question, outcome


@pytest.mark.parametrize(
    "query,material",
    [
        ("Pizzakarton", "cardboard"),
        ("receipt", "paper"),
        ("Kassenbon", "paper"),
        ("tea bag", "organic"),
        ("Teebeutel", "organic"),
        ("Joghurtbecher", "plastic"),
        ("Kronkorken", "metal"),
        ("Alufolie", "metal"),
        ("tetra pak", "paper"),
        ("Batterie", "residual"),
        ("Glühbirne", "glass"),
        ("gluehbirne", "glass"),
        ("banana peel", "organic"),
        ("Windel", "residual"),
        ("jar", "glass"),
    ],
)
def test_lookup_material(query, material):
    hits = search(query)
    assert hits, query
    assert hits[0][1] == material


def test_fuzzy_typo():
    assert search("Joghurtbcher")[0][0].id == "yogurt_pot"


def test_no_match():
    assert search("xqzzy") == []
    assert search("") == []


def test_about_fifty_items_with_valid_materials():
    items = load_items()
    assert len(items) >= 50
    assert len({i.id for i in items}) == len(items)
    for it in items:
        assert it.material in CLASSES
        assert it.names["de"] and it.names["en"]


def test_preset_answers_are_valid_and_verified_items_resolve():
    questions = load_rules()["questions"]
    for it in load_items():
        for qid, ans in it.answers.items():
            assert qid in questions, (it.id, qid)
            assert ans in get_question(qid).options, (it.id, ans)
        if it.verified:
            assert it.hint is None
            # verified items should resolve fully (or leave at most a question the user can answer)
            o = outcome(it.material, it.answers)
            assert o is not None or next_question(it.material, it.answers) is not None
        else:
            assert it.hint and it.hint["de"] and it.hint["en"], it.id
