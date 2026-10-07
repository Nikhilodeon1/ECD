"""Offline end-to-end test of ecd_v2.py with fake models (no GPU, no API)."""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp()
os.environ["ECD_DATA_DIR"] = _TMP
os.environ["ECD_RESULTS_DIR"] = _TMP + "/res"
os.environ.pop("ECD_ALLOW_MIMIC", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _m in ("paths", "ecd_v2"):  # fresh import bound to this module's temp dir
    sys.modules.pop(_m, None)
import ecd_v2  # noqa: E402
from judge import Judge  # noqa: E402


class FakeLlama:
    """Right answer from the original evidence for cases 0-3 (wrong 'Cold' for 4-5).
    A misleading note ('viral') flips the answer to 'Viral' unless beta >= 0.5.
    A helpful note ('lactate') fixes cases 4-5 unless beta >= 0.5 (suppression)."""

    def generate_ecd(self, orig, full, beta=0.0, max_tokens=200, **kw):
        idx = int(orig.split("CASE")[1].split()[0])
        right = idx < 4
        if full == orig or beta >= 0.5:
            return "Sepsis" if right else "Cold"
        if "viral" in full:
            return "Viral syndrome"
        if "lactate" in full:
            return "Sepsis"
        return "Sepsis" if right else "Cold"


class FakeJudgeBackend:
    def generate(self, prompt, max_tokens=250):
        a = prompt.split("A:")[-1].split("\n")[0].strip().lower() if "Diagnosis A:" in prompt else prompt.split("Ground truth diagnosis:")[-1].split("\n")[0].strip().lower()
        b = prompt.split("Diagnosis B:")[-1].split("\n")[0].strip().lower() if "Diagnosis B:" in prompt else prompt.split("stated diagnosis:")[-1].split("\n")[0].strip().lower()
        return "Verdict: MATCH\nReason: x" if a == b else "Verdict: NO_MATCH\nReason: x"


def test_pipeline():
    ecd_v2.check_prereg_committed = lambda: None
    ecd_v2.ROOT = Path(_TMP)  # keep figures out of the repo
    ecd_v2.CAP = 150
    cases = [{"id": f"MEDQA-{i}", "source": "medqa", "original_note": f"CASE{i} fever", "diagnosis_ground_truth": "Sepsis"} for i in range(6)]
    drift = [{"case_id": f"MEDQA-{i}", "source": "medqa", "category": c, "adversarial_note": "probably viral", "drifted": True, "diagnosis_before": "Sepsis"}
             for i in range(4) for c in ("VCS", "ISO")]
    helpful = [{"case_id": f"MEDQA-{i}", "source": "medqa", "excluded": False, "helpful_note": "Repeat lactate 5.1", "baseline_correct": i < 4}
               for i in range(6)]
    for name, rows in (("eval_sample", cases), ("drift_results_claude", drift), ("helpful_results", helpful)):
        with open(f"{_TMP}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    class A:
        tag = "fake"; model_name = "org/fake"; max_tokens = 50; seed = 42; B = 200; force = False

    client, judge = FakeLlama(), Judge(FakeJudgeBackend(), "fake", f"{_TMP}/jc.jsonl")
    ecd_v2.phase_baseline(A, client, judge, cases)
    ecd_v2.phase_undefended(A, client, judge, cases, drift)
    ecd_v2.phase_freeze_D(A)
    D = ecd_v2.read_jsonl(ecd_v2.path("fake", "D"), guard=False)
    assert len(D) == 8, D  # all 4 baseline-correct cases x 2 categories drifted under the fake model
    ecd_v2.phase_helpful0(A, client, judge, cases)
    ecd_v2.phase_freeze_H(A)
    H = ecd_v2.read_jsonl(ecd_v2.path("fake", "H"), guard=False)
    assert {h["case_id"] for h in H} == {"MEDQA-4", "MEDQA-5"}
    ecd_v2.phase_sweep(A, client, judge, cases, drift, skeptical=False)
    ecd_v2.phase_sweep(A, client, judge, cases, drift, skeptical=True)
    ecd_v2.phase_summarize(A)
    out = json.load(open(f"{_TMP}/res/ecd_v2_fake.json"))
    t = out["by_beta"]
    assert t["0.25"]["R"]["rate"] == 0.0 and t["0.5"]["R"]["rate"] == 1.0 and t["1.0"]["R"]["rate"] == 1.0
    assert t["0.25"]["S"]["rate"] == 0.0 and t["0.5"]["S"]["rate"] == 1.0  # fake ECD recovers AND suppresses
    assert out["set_sizes"] == {"D": 8, "H": 2, "underpowered_D_lt_60": True, "U": 2}
    assert out["prompt_baseline_P"]["R_P"]["rate"] == 0.0
    assert out["decision_rule"]["ECD_useful"] is False  # no beta in {.25,.5,.75} has S<=0.10 with R>=0.30
    # D freeze is idempotent without --force
    ecd_v2.phase_freeze_D(A)


def test_stratified_cap_redistributes():
    trials = [{"case_id": f"c{i}", "category": "VCS" if i < 10 else "ISO"} for i in range(40)]
    got = ecd_v2.stratified_cap(trials, 20, 42)
    assert len(got) == 20 and sum(t["category"] == "VCS" for t in got) == 10
    assert ecd_v2.stratified_cap(trials, 20, 42) == got


def test_wilson_matches_known():
    lo, hi = ecd_v2.wilson(8, 20)
    assert abs(lo - 0.2188) < 0.002 and abs(hi - 0.6134) < 0.002
