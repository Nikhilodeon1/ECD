"""Builds a fixed, reproducible eval subsample from the full cases.jsonl.

The same sample (same seed, same file) gets reused across Weeks 5-8 so
baseline accuracy, adversarial drift, and defense results are all measured
on the same cases -- otherwise comparisons across weeks aren't valid.

Caps how many times a single diagnosis can appear (--max-per-diagnosis) so
the sample isn't dominated by the handful of extremely common ICU
diagnoses (sepsis, pneumonia, UTI, etc. each appear 2,000-5,000+ times in
the raw MIMIC data).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_sample(
    cases: list[dict], n: int, max_per_diagnosis: int, seed: int
) -> list[dict]:
    random.seed(seed)
    by_source = defaultdict(list)
    for c in cases:
        by_source[c["source"]].append(c)

    per_dx_count = defaultdict(int)
    capped = []
    for source_cases in by_source.values():
        shuffled = source_cases[:]
        random.shuffle(shuffled)
        for c in shuffled:
            dx = c["diagnosis_ground_truth"]
            if per_dx_count[dx] >= max_per_diagnosis:
                continue
            per_dx_count[dx] += 1
            capped.append(c)

    random.shuffle(capped)
    return capped[:n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="../data/processed/cases.jsonl")
    parser.add_argument("--out", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--max-per-diagnosis", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    sample = stratified_sample(cases, args.n, args.max_per_diagnosis, args.seed)
    by_source = defaultdict(int)
    for c in sample:
        by_source[c["source"]] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in sample:
            f.write(json.dumps(c) + "\n")

    print(f"sampled {len(sample)} cases -> {out_path}")
    print(f"by source: {dict(by_source)}")
