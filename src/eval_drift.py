"""Adversarial injection across all 5 taxonomy categories.

For each baseline-correct case, generates an adversarial note per category,
re-elicits a diagnosis, and checks (via LLM judge) whether it changed from
the model's own prior answer. Writes each trial immediately and resumes on
rerun (keyed by case_id, category).
"""
import argparse
import json
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


def load_existing_trials(out_path: str) -> tuple[set[tuple[str, str]], list[dict]]:
    p = Path(out_path)
    if not p.exists():
        return set(), []
    results = []
    done = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results.append(r)
                done.add((r["case_id"], r["category"]))
    return done, results


def run_drift_eval(correct_cases: list[dict], client, out_path: str) -> list[dict]:
    categories = list(CATEGORY_DEFINITIONS.keys())
    already_done, results = load_existing_trials(out_path)
    if already_done:
        print(f"resuming: {len(already_done)} trials already done, skipping those")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = len(correct_cases) * len(categories)
    i = 0
    with out.open("a", encoding="utf-8") as f:
        for case in correct_cases:
            for category in categories:
                i += 1
                if (case["id"], category) in already_done:
                    continue
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
                    drifted = not parse_verdict(drift_raw)
                except Exception as e:
                    print(f"[{i}/{total}] {case['id']} / {category} failed: {e}")
                    continue

                result = {
                    "case_id": case["id"],
                    "source": case["source"],
                    "category": category,
                    "diagnosis_before": case["baseline_diagnosis"],
                    "adversarial_note": adversarial_note,
                    "diagnosis_after": diagnosis_after,
                    "drifted": drifted,
                }
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(f"[{i}/{total}] {case['id']} / {category} drifted={drifted}")

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
