"""Offline end-to-end test of the v3 pipeline with fake backends (no network, no GPU)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import note_spec as ns  # noqa: E402
import public_loader as pl  # noqa: E402
import v3_run as v3  # noqa: E402
import v3_summary as sm  # noqa: E402
from judge import Judge  # noqa: E402
from paths import read_jsonl  # noqa: E402

CASES = [{"id": f"PUBCASE-test-{i:05d}", "source": "pubcase",
          "original_note": "Adult with fever, cough and dyspnea for three days. " * 3,
          "diagnosis_ground_truth": "Community-acquired pneumonia"} for i in range(24)]
CASES[0]["diagnosis_ground_truth"] = "Pulmonary embolism"  # the fake model will get this one wrong


class FakeGen:
    """Plays the note generator; the first PLV attempt for every case is invalid (forces a retry)."""

    def __init__(self):
        self.plv_calls = 0

    def generate(self, prompt, max_tokens=100):
        if "alternative diagnosis that a clinician" in prompt:
            return "Unstable angina"
        if prompt.startswith("You are creating a test case for medical AI research"):
            return "Troponin 0.01 on repeat."
        cat = re.search(r"Note type (\w+):", prompt).group(1)
        if cat == "PLV" and "state the value only" not in prompt:
            return "Repeat lactate now 1.9, trending down."
        return ns.EXAMPLES[cat]


class FakeModel:
    def generate(self, prompt, max_tokens=160):
        if "Follow-up note:" in prompt and "Attending physician:" in prompt:
            return "Diagnosis: Unstable angina\nReasoning: attending said so"
        if "Follow-up note:" in prompt and "Troponin" in prompt:
            return "Diagnosis: Pulmonary embolism"  # drifts on the helpful-style note, stays wrong otherwise
        return "Diagnosis: Community-acquired pneumonia\nReasoning: fever and cough"


class FakeJudgeBackend:
    def generate(self, prompt, max_tokens=250):
        m = re.search(r"Ground truth diagnosis: (.*)\nModel's stated diagnosis: (.*)\n", prompt)
        if m:
            return "Verdict: MATCH" if m.group(1).strip().lower() == m.group(2).strip().lower() else "Verdict: NO_MATCH"
        a = re.search(r"Diagnosis A: (.*)\nDiagnosis B: (.*)\n", prompt)
        return "Verdict: MATCH" if a.group(1).strip().lower() == a.group(2).strip().lower() else "Verdict: NO_MATCH"


class FakeVerifier:
    def generate(self, prompt, max_tokens=8):
        return "Answer: yes"


def test_public_row_mapping():
    ok = {"case_prompt": "word " * 100, "final_diagnosis": "Kawasaki disease"}
    assert pl.row_to_case(ok, 3, "test").id == "PUBCASE-test-00003"
    assert pl.row_to_case({"case_prompt": "short", "final_diagnosis": "x"}, 0, "test") is None
    assert pl.row_to_case({"case_prompt": "word " * 100, "final_diagnosis": "a b c d e f g h i j"}, 0, "test") is None
    assert pl.row_to_case({"case_prompt": "", "final_diagnosis": "x"}, 0, "test") is None


def test_public_text_cleaning_and_leak_filter():
    bad = "A 52�\udc90year�\udc90old man " + "word " * 80
    case = pl.row_to_case({"case_prompt": bad, "final_diagnosis": "Kawasaki disease"}, 1, "test")
    assert "52-year-old" in case.original_note
    case.original_note.encode("utf-8")  # no lone surrogates left
    leaky = "Presentation of Kawasaki disease in a child. " + "word " * 80
    assert pl.row_to_case({"case_prompt": leaky, "final_diagnosis": "Kawasaki disease"}, 1, "test") is None


def test_budget_gate_blocks_calls(tmp_path):
    class Inner:
        meta = {"input_tokens": 2_000_000, "output_tokens": 0}

        def generate(self, prompt, max_tokens=1):
            return "x"

    b = v3.Budgeted(Inner(), cap_usd=5.0, manifest=tmp_path / "m.jsonl")  # 2M in at $3/M = $6
    try:
        b.generate("p")
        raise AssertionError("expected BudgetExceeded")
    except v3.BudgetExceeded:
        pass


def test_pipeline_end_to_end(tmp_path):
    notes_p, ver_p, ans_p, jud_p = (tmp_path / n for n in ("notes.jsonl", "ver.jsonl", "ans.jsonl", "jud.jsonl"))

    v3.run_notes(CASES, FakeGen(), notes_p, "fake-gen", max_tries=3, chunk=10)
    notes = read_jsonl(notes_p, guard=False)
    assert len(notes) == len(CASES) * 7  # X + 5 categories + H
    assert all(r["passed"] for r in notes)
    plv = [r for r in notes if r["kind"] == "PLV"]
    assert all(r["n_attempts"] == 2 for r in plv)  # first attempt failed, feedback fixed it
    v3.run_notes(CASES, FakeGen(), notes_p, "fake-gen")  # resume: nothing new
    assert len(read_jsonl(notes_p, guard=False)) == len(notes)

    v3.run_verify(CASES, notes_p, FakeVerifier(), ver_p)
    usable = v3.usable_notes(notes_p, ver_p)
    assert (CASES[1]["id"], "X") in usable and (CASES[1]["id"], "ISO") in usable

    judge = Judge(FakeJudgeBackend(), "fake", tmp_path / "cache.jsonl")
    v3.run_answers(CASES, FakeModel(), "hf:fake", ["BASE"], usable, None, ans_p)
    v3.run_judge(CASES, ans_p, judge, jud_p)
    correct = v3.judged_correct(jud_p)
    assert correct[CASES[1]["id"]] and not correct[CASES[0]["id"]]  # case 0 has a different gold diagnosis

    v3.run_answers(CASES, FakeModel(), "hf:fake", list(v3.ARMS), usable, jud_p, ans_p)
    v3.run_judge(CASES, ans_p, judge, jud_p)
    judged = read_jsonl(jud_p, guard=False)
    arms = {(r["case_id"], r["arm"]): r for r in judged}
    assert (CASES[0]["id"], "N") not in arms  # baseline-wrong cases get no misleading arm
    assert (CASES[0]["id"], "H") in arms  # but they do get the helpful arm
    assert arms[(CASES[1]["id"], "B_attending")]["same_as_base"] is False  # attribution moved the fake model
    assert arms[(CASES[1]["id"], "A_unattributed")]["same_as_base"] is True
    for a in v3.HEDGED:  # the exploratory hedged arms run through the same pipeline
        assert (CASES[1]["id"], a) in arms

    s = sm.summarize(judged)["pubcase"]
    assert s["arms"]["B_attending"]["drift"] == 1.0 and s["arms"]["A_unattributed"]["drift"] == 0.0
    c = {x["contrast"]: x for x in s["framing_contrasts"]}["B_attending minus A_unattributed"]
    assert c["diff"] == 1.0 and c["discordant_up"] == c["n"] and c["p_holm"] <= 0.001
    json.dumps(s)


def test_holm_and_wilson():
    assert sm.holm([0.01, 0.04]) == [0.02, 0.04]
    lo, hi = sm.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8


def test_lazy_backend_not_built_when_nothing_to_do(tmp_path):
    def boom():
        raise AssertionError("model must not load when a stage has no work")

    notes_p = tmp_path / "notes.jsonl"
    v3.run_notes(CASES, FakeGen(), notes_p, "fake-gen")  # finish the work with a real fake
    v3.run_notes(CASES, v3.Lazy(boom), notes_p, "fake-gen")  # resumed: no work, factory never called


def test_panel_plan_and_sync(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import argparse
    import v3_panel as vp

    a = argparse.Namespace(sample="s.jsonl", generator="hf:org/Gen-8B", verifier="open:org/Ver-14B",
                           judge="open:org/Judge-14B", models=["hf:org/A", "hf:org/B"], batch_size=32, limit=None)
    cmds = vp.plan(a, tmp_path / "d", tmp_path / "r")
    assert [c[0] for c in cmds][:2] == ["notes", "verify"] and len(cmds) == 2 + 5 * 2
    assert any("--arms" in c[1] and "REST" in c[1] for c in cmds)

    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "v3").mkdir(parents=True)
    (src / "v3" / "a.jsonl").write_text("1\n")
    assert vp.sync(src, dst) == 1 and vp.sync(src, dst) == 0  # second pass copies nothing
    import os
    import time
    later = time.time() + 10
    (src / "v3" / "a.jsonl").write_text("1\n2\n")
    os.utime(src / "v3" / "a.jsonl", (later, later))
    assert vp.sync(src, dst) == 1  # a grown, newer file is copied again

    # restore direction: older Drive copy must not overwrite a longer, newer local file
    drive, local = tmp_path / "drive", tmp_path / "local"
    (drive / "v3").mkdir(parents=True)
    (local / "v3").mkdir(parents=True)
    (drive / "v3" / "b.jsonl").write_text("1\n")
    old = time.time() - 100
    os.utime(drive / "v3" / "b.jsonl", (old, old))
    (local / "v3" / "b.jsonl").write_text("1\n2\n3\n")
    assert vp.sync(drive, local) == 0
    assert (local / "v3" / "b.jsonl").read_text() == "1\n2\n3\n"
