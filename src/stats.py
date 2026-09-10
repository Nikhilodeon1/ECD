"""Descriptive stats over a cases.jsonl file (aggregate counts only, no note text)."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_stats(cases: list[dict]) -> dict:
    by_source = Counter(c["source"] for c in cases)
    lengths = [len(c["original_note"].split()) for c in cases]
    top_diagnoses = Counter(c["diagnosis_ground_truth"] for c in cases).most_common(15)

    return {
        "total_cases": len(cases),
        "by_source": dict(by_source),
        "note_length_words": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 1) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
        },
        "top_15_diagnoses": top_diagnoses,
        "unique_diagnoses": len({c["diagnosis_ground_truth"] for c in cases}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="../data/processed/cases.jsonl")
    parser.add_argument("--out", default="../data/processed/stats.json")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    stats = compute_stats(cases)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
