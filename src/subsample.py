"""Fixed reproducible eval subsample (reused across all experiments).

Capped per diagnosis so common ICU diagnoses don't dominate; even split
per source by default (the MIMIC pool dwarfs MedQA's). Override with
--source-n.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _capped_pool(source_cases: list[dict], max_per_diagnosis: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = source_cases[:]
    rng.shuffle(shuffled)
    per_dx_count = defaultdict(int)
    capped = []
    for c in shuffled:
        dx = c["diagnosis_ground_truth"]
        if per_dx_count[dx] >= max_per_diagnosis:
            continue
        per_dx_count[dx] += 1
        capped.append(c)
    return capped


def stratified_sample(
    cases: list[dict],
    n: int,
    max_per_diagnosis: int,
    seed: int,
    target_per_source: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    by_source = defaultdict(list)
    for c in cases:
        by_source[c["source"]].append(c)
    sources = sorted(by_source.keys())

    if target_per_source is None:
        base = n // len(sources)
        target_per_source = {s: base for s in sources}
        remainder = n - base * len(sources)
        for s in sources[:remainder]:
            target_per_source[s] += 1

    sample = []
    actual_counts = {}
    for s in sources:
        capped = _capped_pool(by_source[s], max_per_diagnosis, seed)
        take = min(target_per_source.get(s, 0), len(capped))
        if take < target_per_source.get(s, 0):
            print(
                f"warning: wanted {target_per_source[s]} from '{s}' but only "
                f"{len(capped)} available after per-diagnosis capping"
            )
        sample.extend(capped[:take])
        actual_counts[s] = take

    random.Random(seed).shuffle(sample)
    return sample, actual_counts


def parse_source_n(raw: str) -> dict[str, int]:
    """'medqa=150,mimic-iv-note=150' -> dict."""
    out = {}
    for pair in raw.split(","):
        source, count = pair.split("=")
        out[source.strip()] = int(count)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="../data/processed/cases.jsonl")
    parser.add_argument("--out", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--max-per-diagnosis", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-n",
        default=None,
        help="override the even split, e.g. 'medqa=150,mimic-iv-note=150'",
    )
    args = parser.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    target = parse_source_n(args.source_n) if args.source_n else None
    sample, actual_counts = stratified_sample(
        cases, args.n, args.max_per_diagnosis, args.seed, target
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in sample:
            f.write(json.dumps(c) + "\n")

    print(f"sampled {len(sample)} cases -> {out_path}")
    print(f"by source: {actual_counts}")
