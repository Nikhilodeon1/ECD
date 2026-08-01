"""Week 6: adversarial injection across all 5 taxonomy categories.

For every case Claude got right in the Week 5 baseline, generates an
adversarial note per category (adversarial_notes.py), re-asks for a
diagnosis with the note appended, and checks whether the diagnosis changed
from the model's own original answer -- that's Diagnosis Drift. Compared
against the model's own prior answer, not against ground truth, since drift
is about the model changing its mind, not about accuracy per se (though for
this sample the two are closely related since we only test cases that
started out correct).

Usage (from src/, after Week 5's eval_baseline.py has already produced
eval_results_claude.jsonl):
    python eval_drift.py
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from adversarial_notes import CATEGORY_DEFINITIONS, build_generate_note_prompt
from grade import build_drift_check_prompt, parse_verdict
from llm_clients import AnthropicClient
from prompts import build_followup_prompt, parse_diagnosis


def load_baseline_correct(baseline_results_path: str, sample_path: str) -> list[dict]:
    with open(baseline_results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    with open(sample_path, encoding="utf-8") as f:
        cases_by_id = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    correct = []
    for r in results:
        if not r["correct"]:
            continue
        case = cases_by_id.get(r["case_id"])
        if case is None:
            continue
        correct.append({**case, "baseline_diagnosis": r["predicted"]})
    return correct


def run_drift_eval(correct_cases: list[dict], client, out_path: str) -> list[dict]:
    results = []
    categories = list(CATEGORY_DEFINITIONS.keys())
    total = len(correct_cases) * len(categories)
    i = 0
    for case in correct_cases:
        for category in categories:
            i += 1
            try:
                note_prompt = build_generate_note_prompt(
                    case["original_note"], case["baseline_diagnosis"], category
                )
                adversarial_note = client.generate(note_prompt, max_tokens=200).strip()

                followup_prompt = build_followup_prompt(case["original_note"], adversarial_note)
                raw_after = client.generate(followup_prompt, max_tokens=300)
                diagnosis_after = parse_diagnosis(raw_after)

                drift_prompt = build_drift_check_prompt(case["baseline_diagnosis"], diagnosis_after)
                drift_raw = client.generate(drift_prompt, max_tokens=250)
                same = parse_verdict(drift_raw)
                drifted = not same
            except Exception as e:
                print(f"[{i}/{total}] {case['id']} / {category} failed: {e}")
                continue

            results.append(
                {
                    "case_id": case["id"],
                    "source": case["source"],
                    "category": category,
                    "diagnosis_before": case["baseline_diagnosis"],
                    "adversarial_note": adversarial_note,
                    "diagnosis_after": diagnosis_after,
                    "drifted": drifted,
                }
            )
            print(f"[{i}/{total}] {case['id']} / {category} drifted={drifted}")
            time.sleep(0.2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def drift_rate_table(results: list[dict]) -> dict:
    by_cat = defaultdict(lambda: {"drifted": 0, "total": 0})
    by_source = defaultdict(lambda: {"drifted": 0, "total": 0})
    overall = {"drifted": 0, "total": 0}

    for r in results:
        by_cat[r["category"]]["total"] += 1
        by_cat[r["category"]]["drifted"] += int(r["drifted"])
        by_source[r["source"]]["total"] += 1
        by_source[r["source"]]["drifted"] += int(r["drifted"])
        overall["total"] += 1
        overall["drifted"] += int(r["drifted"])

    def rate(d):
        return round(d["drifted"] / d["total"], 3) if d["total"] else 0

    return {
        "overall_ddr": rate(overall),
        "overall_n": overall["total"],
        "by_category": {k: {"ddr": rate(v), "n": v["total"]} for k, v in by_cat.items()},
        "by_source": {k: {"ddr": rate(v), "n": v["total"]} for k, v in by_source.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default="../data/processed/eval_results_claude.jsonl")
    parser.add_argument("--sample", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--out", default="../data/processed/drift_results_claude.jsonl")
    args = parser.parse_args()

    correct_cases = load_baseline_correct(args.baseline_results, args.sample)
    print(
        f"{len(correct_cases)} baseline-correct cases x 5 categories "
        f"= {len(correct_cases) * 5} drift trials"
    )

    client = AnthropicClient()
    results = run_drift_eval(correct_cases, client, args.out)

    table = drift_rate_table(results)
    print(json.dumps(table, indent=2))

    table_path = Path(args.out).with_name("drift_rate_table.json")
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"drift rate table -> {table_path}")
