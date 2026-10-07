import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import judge_validation as jv  # noqa: E402


def _rows():
    base = [{"case_id": f"m{i}", "source": "medqa", "acc": i % 10 < 8, "claude": i % 10 < 8} for i in range(100)]
    base += [{"case_id": f"x{i}", "source": "mimic-iv-note", "acc": i % 10 < 5, "claude": i % 10 < 5} for i in range(100)]
    drift = []
    for i in range(100):
        for j, c in enumerate(["VCS", "ISO", "PLV", "MMS", "SNHR"]):
            drift.append({"case_id": f"m{i}", "source": "medqa", "category": c, "d": (i + j) % 8 == 0})
            drift.append({"case_id": f"x{i}", "source": "mimic-iv-note", "category": c, "d": (i + j) % 3 == 0})
    return base, drift


def test_headline_detects_gaps():
    base, drift = _rows()
    h = jv.headline(base, drift, "acc", "d")
    assert h["baseline_accuracy"]["medqa"] == 0.8 and h["baseline_accuracy"]["mimic-iv-note"] == 0.5
    assert h["accuracy_gap_fisher"]["p_value"] < 1e-4
    assert h["drift_gap_mann_whitney"]["p_value"] < 1e-4
    assert h["category_cochrans_q"]["n_cases"] == 200


def test_agreement_and_kappa():
    rows = [{"case_id": f"c{i}", "source": "medqa", "a": i % 2 == 0, "b": i % 2 == 0 if i % 10 else not (i % 2 == 0)} for i in range(200)]
    ag = jv.agreement(rows, "a", "b")["all"]
    assert 0.85 < ag["raw_agreement"] < 0.95 and ag["kappa"] > 0.7 and ag["ci_95"][0] < ag["kappa"] < ag["ci_95"][1]
    assert jv.kappa_ci([1, 1, 1], [0, 1, 0], ["a", "b", "c"])["kappa"] is None  # constant rater
