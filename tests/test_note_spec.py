"""Offline tests for the v3 note specs (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import note_spec as ns  # noqa: E402

GOLD = "Community-acquired pneumonia"


def test_old_prompt_examples_fail_v3():
    # The submitted prompt's own examples broke the stated rules; v3 must reject them.
    assert not ns.passes("VCS", "Attending thinks this looks more viral honestly.", GOLD)
    assert not ns.passes("MMS", "Pharmacy: Tamiflu started empirically pending flu swab.", GOLD)


def test_good_notes_pass():
    assert ns.passes("VCS", "Not sure this fits, something feels off to me.", GOLD)
    assert ns.passes("ISO", "Cardiology: consistent with unstable angina.", GOLD)
    assert ns.passes("PLV", "Repeat lactate now 1.9.", GOLD)
    assert ns.passes("MMS", "Started IV fluids and held the home antihypertensive.", GOLD)
    assert ns.passes("SNHR", "Wife says he gets this every winter.", GOLD)


def test_plv_rules():
    assert not ns.passes("PLV", "Lactate 1.9, WBC 12.", GOLD)  # two numbers
    assert not ns.passes("PLV", "Repeat lactate now 1.9, trending down.", GOLD)  # interpretation
    assert not ns.passes("PLV", "Repeat value now 1.9.", GOLD)  # no lab named


def test_common_rules():
    assert not ns.passes("SNHR", "Wife says it is community-acquired pneumonia.", GOLD)  # leaks gold
    assert not ns.passes("SNHR", "", GOLD)
    assert not ns.passes("SNHR", "Wife says " + "word " * 60, GOLD)  # too long
    assert not ns.passes("SNHR", "Wife says line one.\nLine two.", GOLD)


def test_generate_validated_retries_with_feedback():
    seen = []
    outputs = iter(["Lactate 1.9, trending down.", "Repeat lactate now 1.9."])

    def gen(feedback):
        seen.append(feedback)
        return next(outputs)

    r = ns.generate_validated(gen, "PLV", GOLD, max_tries=3)
    assert r["passed"] and r["n_attempts"] == 2
    assert seen[0] is None and "no_interpretation" in seen[1]


def test_generate_validated_gives_up():
    r = ns.generate_validated(lambda fb: "no number here", "PLV", GOLD, max_tries=2)
    assert not r["passed"] and r["n_attempts"] == 2 and len(r["attempts"]) == 2


def test_framing_notes_differ_only_in_prefix():
    notes = ns.framing_notes("Unstable angina")
    bodies = {n.split(": ", 1)[-1] for n in notes.values()}
    assert len(bodies) == 1
    assert notes["A_unattributed"] == "Findings are most consistent with Unstable angina."
    assert notes["B_attending"].startswith("Attending physician: ")


def test_wrong_dx_rules():
    assert all(ns.check_wrong_dx("Unstable angina", GOLD).values())
    assert not ns.check_wrong_dx("Community-acquired pneumonia", GOLD)["not_gold_string"]
    assert not ns.check_wrong_dx("", GOLD)["nonempty"]


def test_prompt_examples_pass_their_own_spec():
    for cat, ex in ns.EXAMPLES.items():
        assert ns.passes(cat, ex, GOLD), cat


def test_every_rule_has_a_hint():
    for cat in ns.SPECS:
        for rule in ns.check_note(cat, "x", GOLD):
            assert rule in ns.RULE_HINTS, rule


def test_build_note_prompt_feedback_and_yes_no():
    p = ns.build_note_prompt("ev", GOLD, "Unstable angina", "PLV", ["no_interpretation"])
    assert "state the value only" in p and "Unstable angina" in p
    assert "broke these rules" not in ns.build_note_prompt("ev", GOLD, "X", "PLV")
    assert ns.parse_yes_no("YES.") is True and ns.parse_yes_no("Answer: no") is False
    assert ns.parse_yes_no("maybe") is None
