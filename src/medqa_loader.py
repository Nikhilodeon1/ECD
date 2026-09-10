"""Load MedQA (USMLE), keep only diagnosis-style questions (~18% of the test split)."""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from schema import BaseCase

DIAGNOSIS_PATTERN = re.compile(
    r"most likely diagnosis|likely cause of|most likely cause|diagnosis is"
    r"|underlying diagnosis|best explains this patient",
    re.IGNORECASE,
)


def is_diagnosis_question(question: str) -> bool:
    return bool(DIAGNOSIS_PATTERN.search(question))


def load_medqa(split: str = "test") -> list[BaseCase]:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    cases = []
    for i, ex in enumerate(ds):
        if not is_diagnosis_question(ex["question"]):
            continue
        answer_text = ex["options"][ex["answer_idx"]]
        cases.append(
            BaseCase(
                id=f"MEDQA-{split}-{i:05d}",
                source="medqa",
                case_summary=ex["question"],
                original_note=ex["question"],
                diagnosis_ground_truth=answer_text,
                metadata={
                    "options": ex["options"],
                    "answer_idx": ex["answer_idx"],
                    "meta_info": ex["meta_info"],
                },
            )
        )
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="train | test")
    parser.add_argument("--out", default="../data/processed/medqa_cases.jsonl")
    args = parser.parse_args()

    cases = load_medqa(args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict()) + "\n")

    print(f"wrote {len(cases)} diagnosis-style cases to {out_path}")
