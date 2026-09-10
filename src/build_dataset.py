"""Combine MedQA + MIMIC-IV-Note into one cases file. MIMIC args optional."""
import argparse
import json
from pathlib import Path

from medqa_loader import load_medqa
from mimic_loader import load_mimic_notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medqa-split", default="test")
    parser.add_argument("--discharge", help="path to MIMIC discharge.csv.gz (omit to skip MIMIC)")
    parser.add_argument("--diagnoses-icd")
    parser.add_argument("--d-icd")
    parser.add_argument("--out", default="../data/processed/cases.jsonl")
    args = parser.parse_args()

    cases = load_medqa(args.medqa_split)
    print(f"medqa: {len(cases)} diagnosis-style cases")

    if args.discharge:
        mimic_cases = load_mimic_notes(args.discharge, args.diagnoses_icd, args.d_icd)
        print(f"mimic: {len(mimic_cases)} cases")
        cases += mimic_cases
    else:
        print("no --discharge given, skipping MIMIC")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict()) + "\n")

    print(f"wrote {len(cases)} total cases to {out_path}")


if __name__ == "__main__":
    main()
