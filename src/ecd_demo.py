"""Runs ECD against real cases from the Week 6 drift results, reusing the
adversarial notes already generated there instead of paying to regenerate
them. Shows plain generation (susceptible to drift) vs. ECD at a couple
alpha values, so you can see the defense actually doing something.

Needs a GPU pod with Llama-Med set up (see LlamaMedClient in llm_clients.py
-- swap the default model_name if aaditya/Llama3-OpenBioLLM-8B isn't the
one you want).

Usage (from src/, on the pod, after Week 6's eval_drift.py has produced
drift_results_claude.jsonl):
    python ecd_demo.py --n 5
"""
import argparse
import json

from llm_clients import LlamaMedClient
from prompts import build_baseline_prompt, build_followup_prompt, parse_diagnosis


def load_drifted_cases(drift_results_path: str, sample_path: str, n: int) -> list[dict]:
    """Pulls cases that actually drifted under Claude -- the interesting
    ones to test ECD against -- and joins back to eval_sample.jsonl to get
    the real case evidence text (drift_results.jsonl only has the short
    diagnosis labels and the adversarial note, not the original evidence)."""
    with open(drift_results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    with open(sample_path, encoding="utf-8") as f:
        cases_by_id = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    drifted = []
    for r in results:
        if not r["drifted"]:
            continue
        case = cases_by_id.get(r["case_id"])
        if case is None:
            continue
        drifted.append({**r, "original_note": case["original_note"]})
        if len(drifted) >= n:
            break
    return drifted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-results", default="../data/processed/drift_results_claude.jsonl")
    parser.add_argument("--sample", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--alphas", default="1,2", help="comma-separated nonzero alpha values to test")
    parser.add_argument("--model-name", default="aaditya/Llama3-OpenBioLLM-8B")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    cases = load_drifted_cases(args.drift_results, args.sample, args.n)
    print(f"testing ECD on {len(cases)} cases that drifted under Claude")

    client = LlamaMedClient(model_name=args.model_name)

    for c in cases:
        print(f"\n=== {c['case_id']} ({c['category']}) ===")
        print(f"Claude's diagnosis before: {c['diagnosis_before']}")
        print(f"Claude's diagnosis after (drifted): {c['diagnosis_after']}")
        print(f"adversarial note: {c['adversarial_note'][:200]}")

        original_prompt = build_baseline_prompt(c["original_note"])
        full_prompt = build_followup_prompt(c["original_note"], c["adversarial_note"])

        # 100 was cutting real responses off mid-sentence in practice --
        # this base model is also less reliable than Claude at sticking to
        # the "Diagnosis: X" format, so parse_diagnosis often falls back to
        # the raw text; more headroom at least avoids truncating that too
        plain = parse_diagnosis(client.generate(full_prompt, max_tokens=200))
        print(f"Llama-Med plain (alpha=0, no defense): {plain}")

        for alpha in alphas:
            ecd_out = parse_diagnosis(
                client.generate_ecd(original_prompt, full_prompt, alpha=alpha, max_tokens=200)
            )
            print(f"Llama-Med ECD alpha={alpha}: {ecd_out}")
