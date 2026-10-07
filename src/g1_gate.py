"""Gate G1 (T1d): does the corrected ECD move toward the original-only answer?

20 trials where Claude drifted, OpenBioLLM-8B, beta in {0, 0.5, 1, 1.5}. The
parsed output at each beta is compared to the model's own original-only greedy
diagnosis by normalized string match (no API calls). Expect the match rate to
rise with beta and to be ~100% at beta=1. If not, stop and debug.

GPU required (pod). Per-trial text stays in data/ (MIMIC-derived); only the
aggregate goes to results/g1_gate.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

from judge import normalize
from paths import DATA, RESULTS, read_jsonl
from llm_clients import LlamaMedClient
from prompts import build_llama_baseline_prompt, build_llama_followup_prompt, parse_llama_diagnosis


def load_drifted(drift_path: str, sample_path: str, n: int, seed: int) -> list[dict]:
    cases = {c["id"]: c for c in read_jsonl(sample_path)}
    trials = [r for r in read_jsonl(drift_path) if r["drifted"] and r["case_id"] in cases]
    random.Random(seed).shuffle(trials)
    return [{**t, "original_note": cases[t["case_id"]]["original_note"]} for t in trials[:n]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-results", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--sample", default=str(DATA / "eval_sample.jsonl"))
    ap.add_argument("--model-name", default="aaditya/Llama3-OpenBioLLM-8B")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--betas", default="0,0.5,1,1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-trials", default=str(DATA / "g1_gate_trials.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "g1_gate.json"))
    args = ap.parse_args()

    betas = [float(b) for b in args.betas.split(",")]
    trials = load_drifted(args.drift_results, args.sample, args.n, args.seed)
    client = LlamaMedClient(model_name=args.model_name)

    rows, hits = [], {b: 0 for b in betas}
    parsed_ok = {b: 0 for b in betas}
    for t in trials:
        orig_p = build_llama_baseline_prompt(t["original_note"])
        full_p = build_llama_followup_prompt(t["original_note"], t["adversarial_note"])
        ref = parse_llama_diagnosis(client.generate(orig_p, max_tokens=200))  # HF generate, original only
        row = {"case_id": t["case_id"], "category": t["category"], "reference_original_only": ref, "by_beta": {}}
        for b in betas:
            out = parse_llama_diagnosis(client.generate_ecd(orig_p, full_p, beta=b, max_tokens=200))
            match = normalize(out) == normalize(ref) and normalize(ref) != ""
            row["by_beta"][str(b)] = {"output": out, "match": match}
            hits[b] += int(match)
            parsed_ok[b] += int(bool(normalize(out)))
        rows.append(row)
        print(f"{t['case_id']}/{t['category']}: " + " ".join(f"b={b}:{int(row['by_beta'][str(b)]['match'])}" for b in betas))

    Path(args.out_trials).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_trials, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    agg = {
        "script": "src/g1_gate.py",
        "model": args.model_name,
        "dtype": client.dtype,
        "n_trials": n,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "match_rate_vs_original_only_greedy": {str(b): round(hits[b] / n, 3) if n else None for b in betas},
        "parse_rate": {str(b): round(parsed_ok[b] / n, 3) if n else None for b in betas},
    }
    up_to_one = [agg["match_rate_vs_original_only_greedy"][str(b)] for b in betas if b <= 1.0]
    agg["monotone_nondecreasing_up_to_beta1"] = all(x <= y for x, y in zip(up_to_one, up_to_one[1:])) if n else None
    agg["pass_beta1_ge_0.95"] = (hits.get(1.0, 0) / n >= 0.95) if n and 1.0 in hits else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
