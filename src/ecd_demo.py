"""Qualitative ECD demo: plain vs ECD-defended output on cases that drifted
under Claude, GPU required. Reuses Week 6's adversarial notes."""
import argparse
import json

from llm_clients import LlamaMedClient
from prompts import build_llama_baseline_prompt, build_llama_followup_prompt, parse_llama_diagnosis


def load_drifted_cases(
    drift_results_path: str, sample_path: str, n: int, source: str | None = None
) -> list[dict]:
    """Drifted cases joined to eval_sample.jsonl for the evidence text.
    source='mimic-iv-note' must only run where that data is cleared to be."""
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
        if source is not None and case["source"] != source:
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
    parser.add_argument("--source", default=None, choices=["medqa", "mimic-iv-note"])
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    cases = load_drifted_cases(args.drift_results, args.sample, args.n, args.source)
    print(f"testing ECD on {len(cases)} cases that drifted under Claude")

    client = LlamaMedClient(model_name=args.model_name)

    for c in cases:
        print(f"\n=== {c['case_id']} ({c['category']}) ===")
        print(f"Claude's diagnosis before: {c['diagnosis_before']}")
        print(f"Claude's diagnosis after (drifted): {c['diagnosis_after']}")
        print(f"adversarial note: {c['adversarial_note'][:200]}")

        original_prompt = build_llama_baseline_prompt(c["original_note"])
        full_prompt = build_llama_followup_prompt(c["original_note"], c["adversarial_note"])

        plain = parse_llama_diagnosis(client.generate(full_prompt, max_tokens=200))
        print(f"Llama-Med plain (alpha=0, no defense): {plain}")

        for alpha in alphas:
            ecd_out = parse_llama_diagnosis(
                client.generate_ecd(original_prompt, full_prompt, alpha=alpha, max_tokens=200)
            )
            print(f"Llama-Med ECD alpha={alpha}: {ecd_out}")
