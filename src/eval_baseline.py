"""Runs baseline (no adversarial note) diagnosis eval and produces an accuracy table.

Usage (from src/, with .env containing ANTHROPIC_API_KEY):
    python subsample.py --n 300
    python eval_baseline.py
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from grade import build_grade_prompt, parse_verdict
from llm_clients import AnthropicClient
from prompts import build_baseline_prompt, parse_diagnosis


def load_sample(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(cases: list[dict], generator_client, grader_client, out_path: str) -> list[dict]:
    results = []
    for i, c in enumerate(cases):
        prompt = build_baseline_prompt(c["original_note"])
        try:
            raw = generator_client.generate(prompt)
        except Exception as e:
            print(f"[{i}] generation failed for {c['id']}: {e}")
            continue
        predicted = parse_diagnosis(raw)

        grade_prompt = build_grade_prompt(c["diagnosis_ground_truth"], predicted)
        try:
            grade_raw = grader_client.generate(grade_prompt, max_tokens=250)
        except Exception as e:
            print(f"[{i}] grading failed for {c['id']}: {e}")
            continue
        correct = parse_verdict(grade_raw)

        results.append(
            {
                "case_id": c["id"],
                "source": c["source"],
                "model": generator_client.name,
                "ground_truth": c["diagnosis_ground_truth"],
                "predicted": predicted,
                "correct": correct,
                "raw_generation": raw,
                "raw_grade": grade_raw,
            }
        )
        print(f"[{i + 1}/{len(cases)}] {c['id']} correct={correct}")
        time.sleep(0.2)  # light rate-limit courtesy

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def accuracy_table(results: list[dict]) -> dict:
    by_model_source = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        key = (r["model"], r["source"])
        by_model_source[key]["total"] += 1
        by_model_source[key]["correct"] += int(r["correct"])

    table = {}
    for (model, source), counts in by_model_source.items():
        acc = counts["correct"] / counts["total"] if counts["total"] else 0
        table.setdefault(model, {})[source] = {
            "accuracy": round(acc, 3),
            "n": counts["total"],
        }
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--out", default="../data/processed/eval_results_claude.jsonl")
    args = parser.parse_args()

    cases = load_sample(args.sample)
    client = AnthropicClient()
    # NOTE: same client grading its own output -- self-grading bias risk,
    # see grade.py docstring. Fine for a first pass, revisit once GPT-5/
    # Gemini budgets exist so a different model can cross-grade instead.
    results = run_eval(cases, generator_client=client, grader_client=client, out_path=args.out)

    table = accuracy_table(results)
    print(json.dumps(table, indent=2))

    table_path = Path(args.out).with_name("accuracy_table.json")
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"accuracy table -> {table_path}")
