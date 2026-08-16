"""Runs baseline (no adversarial note) diagnosis eval and produces an accuracy table.

Usage (from src/, with .env containing ANTHROPIC_API_KEY):
    python subsample.py --n 300
    python eval_baseline.py

Structured as two batched stages (generate all diagnoses, then grade all of
them) rather than interleaving one case at a time -- in CLAUDE_MODE=manual
this means one paste-many-get-many round trip per stage instead of 600
individual round trips. In normal API mode this is behaviorally identical
to the old interleaved version, generate_batch() just loops generate()
under the hood (see llm_clients.py).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from grade import build_grade_prompt, parse_verdict
from llm_clients import get_claude_client
from prompts import build_baseline_prompt, parse_diagnosis


def load_sample(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(
    cases: list[dict],
    generator_client,
    grader_client,
    out_path: str,
    long_chunk_size: int = 40,
    short_chunk_size: int = 400,
) -> list[dict]:
    # stage 1: generate a diagnosis for every case
    # long prompts (embed a full clinical note, ~1200 tokens worst-case) --
    # 40/chunk keeps well under a 200k context window with room for output
    gen_prompts = [build_baseline_prompt(c["original_note"]) for c in cases]
    raw_generations = generator_client.generate_batch(
        gen_prompts, max_tokens=500, chunk_size=long_chunk_size
    )
    predicted = [parse_diagnosis(r) if r else "" for r in raw_generations]

    # stage 2: grade every prediction against ground truth
    # short prompts (just two diagnosis strings, ~250 tokens) -- can go big
    grade_prompts = [
        build_grade_prompt(c["diagnosis_ground_truth"], p) for c, p in zip(cases, predicted)
    ]
    raw_grades = grader_client.generate_batch(
        grade_prompts, max_tokens=250, chunk_size=short_chunk_size
    )

    results = []
    for c, raw, pred, grade_raw in zip(cases, raw_generations, predicted, raw_grades):
        if not raw or not grade_raw:
            print(f"skipping {c['id']}: missing generation or grade (see batch warnings above)")
            continue
        correct = parse_verdict(grade_raw)
        results.append(
            {
                "case_id": c["id"],
                "source": c["source"],
                "model": generator_client.name,
                "ground_truth": c["diagnosis_ground_truth"],
                "predicted": pred,
                "correct": correct,
                "raw_generation": raw,
                "raw_grade": grade_raw,
            }
        )
        print(f"{c['id']} correct={correct}")

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
    parser.add_argument(
        "--long-chunk-size", type=int, default=40, help="manual mode only -- prompts with full notes"
    )
    parser.add_argument(
        "--short-chunk-size", type=int, default=400, help="manual mode only -- grading prompts"
    )
    args = parser.parse_args()

    cases = load_sample(args.sample)
    client = get_claude_client()
    # NOTE: same client grading its own output -- self-grading bias risk,
    # see grade.py docstring. Fine for a first pass, revisit once GPT-5/
    # Gemini budgets exist so a different model can cross-grade instead.
    results = run_eval(
        cases,
        generator_client=client,
        grader_client=client,
        out_path=args.out,
        long_chunk_size=args.long_chunk_size,
        short_chunk_size=args.short_chunk_size,
    )

    table = accuracy_table(results)
    print(json.dumps(table, indent=2))

    table_path = Path(args.out).with_name("accuracy_table.json")
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"accuracy table -> {table_path}")
