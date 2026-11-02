import yaml

from wohin import CLASSES
from wohin.rules import (
    RULES_PATH,
    UNSURE,
    all_paths,
    get_outcome,
    get_question,
    load_rules,
    materials,
    next_question,
    outcome,
)


def test_every_class_has_a_flow():
    assert set(materials()) == set(CLASSES)


def test_every_outcome_reachable():
    reached = {oid for m in materials() for _, oid in all_paths(m)}
    defined = set(load_rules()["outcomes"])
    assert defined == reached, f"unreachable: {defined - reached}"


def test_every_outcome_has_source_and_texts():
    rules = load_rules()
    for oid in rules["outcomes"]:
        o = get_outcome(oid)
        assert o.source_url.startswith("https://"), oid
        assert o.text["de"] and o.text["en"], oid
        assert o.bin_label["de"] and o.bin_label["en"], oid
        assert o.verified in {"true", "partial", "false"}, oid


def test_unverified_outcomes_point_somewhere_checkable():
    for oid in load_rules()["outcomes"]:
        o = get_outcome(oid)
        if o.verified != "true":
            assert o.bsr_az == "https://www.bsr.de/abfall-abc"


def test_every_question_has_de_en_and_unsure():
    for qid in load_rules()["questions"]:
        q = get_question(qid)
        assert q.text["de"] and q.text["en"], qid
        assert UNSURE in q.options
        for val, label in q.options.items():
            assert label["de"] and label["en"], (qid, val)


def test_every_question_branch_handles_unsure():
    def rec(node):
        if "ask" in node:
            assert UNSURE in node["then"], node["ask"]
            assert set(node["then"]) - {UNSURE} == set(load_rules()["questions"][node["ask"]]["options"])
            for child in node["then"].values():
                rec(child)

    for flow in load_rules()["flows"].values():
        rec(flow)


def test_yaml_keys_survive_yes_no():
    raw = yaml.safe_load(open(RULES_PATH, encoding="utf-8"))
    # yes/no are booleans in YAML 1.1; the loader must normalise them
    assert True in raw["questions"]["carton"]["options"]
    assert "yes" in load_rules()["questions"]["carton"]["options"]


def test_example_flows():
    assert next_question("glass").id == "glass_packaging"
    a = {"glass_packaging": "yes", "deposit_logo": "no"}
    assert next_question("glass", a).id == "glass_colour"
    assert outcome("glass", {**a, "glass_colour": "brown"}).id == "glass_brown"
    assert outcome("glass", {**a, "glass_colour": "other"}).verified == "partial"
    assert outcome("plastic", {"deposit_logo": "yes"}).id == "deposit"
    assert outcome("metal", {"deposit_logo": "no", "electro": "no"}).bin_id == "wertstofftonne"
    assert outcome("paper", {"kitchen_paper": "yes"}).bin_id == "biotonne"
    assert outcome("cardboard", {"kitchen_paper": "no", "carton": "yes"}).bin_id == "wertstofftonne"
    assert outcome("organic", {"in_packaging": "no"}).bin_id == "biotonne"
    assert outcome("textile", {"wearable": "no"}).verified == "partial"
    assert outcome("residual", {"electro": "battery"}).bin_id == "battery"
    assert outcome("glass") is None


def test_unknown_answer_treated_as_unsure():
    assert outcome("glass", {"glass_packaging": "banana"}).id == "check_bsr"
