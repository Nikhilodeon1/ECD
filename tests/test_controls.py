"""Offline end-to-end test of controls.py with a fake Claude client."""
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

for _m in ("paths", "controls"):  # fresh import bound to this module's temp dir
    sys.modules.pop(_m, None)
import controls  # noqa: E402


class FakeClaude:
    """Baseline answers 'Sepsis'; follow-ups with a 'viral' note switch the answer."""
    meta = {}

    def generate(self, prompt, max_tokens=500):
        if "correct diagnosis (do not name" in prompt.lower():
            return "Repeat blood cultures positive for gram-negative rods."
        if "Verdict" in prompt or "same underlying clinical condition" in prompt:
            a = prompt.split("Diagnosis A:")[-1].split("\n")[0].strip().lower() if "Diagnosis A:" in prompt else ""
            b = prompt.split("Diagnosis B:")[-1].split("\n")[0].strip().lower() if "Diagnosis B:" in prompt else ""
            same = a == b or ("sepsis" in a and "sepsis" in b)
            return "Verdict: MATCH\nReason: x" if same else "Verdict: NO_MATCH\nReason: x"
        if "Follow-up note:" in prompt and "viral" in prompt.lower():
            return "Diagnosis: Viral syndrome\nReasoning: x"
        return "Diagnosis: Sepsis\nReasoning: x"

    def dump_meta(self, path):
        pass


def _write(name, rows):
    with open(f"{_TMP}/{name}", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_controls_pipeline():
    cases = [{"id": f"MEDQA-{i}", "source": "medqa", "original_note": "fever hypotension", "diagnosis_ground_truth": "Sepsis"} for i in range(8)]
    cases.append({"id": "MIMIC-1", "source": "mimic-iv-note", "original_note": "secret text", "diagnosis_ground_truth": "Sepsis"})
    _write("eval_sample.jsonl", cases)
    base = [{"case_id": c["id"], "source": c["source"], "predicted": "Sepsis", "correct": i < 6} for i, c in enumerate(cases)]
    _write("eval_results_claude.jsonl", base)
    drift = [{"case_id": c["id"], "source": "medqa", "category": cat, "drifted": (i + j) % 3 == 0, "diagnosis_before": "Sepsis", "adversarial_note": "n"}
             for i, c in enumerate(cases[:6]) for j, cat in enumerate(["VCS", "ISO"])]
    _write("drift_results_claude.jsonl", drift)

    class A:  # argparse stand-in
        sample = f"{_TMP}/eval_sample.jsonl"; baseline = f"{_TMP}/eval_results_claude.jsonl"
        drift = f"{_TMP}/drift_results_claude.jsonl"; dry_run = False; limit = 0

    client = FakeClaude()
    judge = controls.Judge(client, "fake", f"{_TMP}/jc.jsonl")
    controls.run_noise(A, client, judge)
    controls.run_neutral(A, client, judge)
    controls.run_helpful(A, client, judge)
    controls.run_noise(A, client, judge)  # resume: nothing new
    rows = [json.loads(l) for l in open(controls.OUT["noise"])]
    assert len(rows) == 6 and all(not r["resample_drift"] for r in rows)
    assert all("MIMIC" not in r["case_id"] for r in rows), "MIMIC row leaked in public mode"
    h = [json.loads(l) for l in open(controls.OUT["helpful"])]
    assert len(h) == 8 and all(not r["excluded"] for r in h)
    controls.summarize(A)
    s = json.load(open(f"{_TMP}/res/controls_summary.json"))
    assert s["noise_floor"]["all"]["n"] == 6
    assert "VCS|all" in s["excess_drift_adversarial_minus_neutral"]
    assert s["helpful_note_control"]["correction_rate_on_baseline_wrong"]["all"]["n"] == 2


def test_neutral_assignment_is_stable_and_nonclinical():
    assert controls.neutral_for("MEDQA-1") == controls.neutral_for("MEDQA-1")
    assert len(controls.NEUTRAL_BANK) == 16
    assert not any(w in " ".join(controls.NEUTRAL_BANK).lower() for w in ("diagnos", "pain", "fever", "lab"))


def test_leak_check():
    assert controls.leaks_diagnosis("consistent with sepsis", "Sepsis")
    assert controls.leaks_diagnosis("pneumonia noted", "Bacterial pneumonia")
    assert not controls.leaks_diagnosis("Repeat lactate 4.2 mmol/L.", "Sepsis")
