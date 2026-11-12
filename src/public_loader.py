"""Public real-case tier: open-access case reports with a gold diagnosis (MedCaseReasoning).

Public text may be run on any model and released in full. Fields used: case_prompt (the
presentation, before any differential) and final_diagnosis; splits are train, val and test. The
dataset card lists CC-BY 4.0 (confirmed by the author).
"""
import argparse
import json
import re
from pathlib import Path

from judge import normalize
from schema import BaseCase
from subsample import stratified_sample

HF_NAME = "zou-lab/MedCaseReasoning"
MIN_WORDS, MAX_WORDS, MAX_DX_WORDS = 60, 700, 8
# a replacement character followed by a lone surrogate, or a lone surrogate alone
_MANGLED = re.compile("�?[\ud800-\udfff]+")


def clean_text(text: str | None) -> str:
    """The dataset has mangled non-breaking hyphens ("52<bad>year<bad>old"). Lone surrogates make
    tokenizers and API clients raise on encode, so they become hyphens; stray replacement
    characters are dropped."""
    return _MANGLED.sub("-", text or "").replace("�", "").strip()


def row_to_case(row: dict, i: int, split: str) -> BaseCase | None:
    """None when the row is unusable: missing fields, too short or long, a verbose label, or a
    presentation that already contains the diagnosis string."""
    text = clean_text(row.get("case_prompt"))
    dx = clean_text(row.get("final_diagnosis"))
    if not text or not dx:
        return None
    if not (MIN_WORDS <= len(text.split()) <= MAX_WORDS) or len(dx.split()) > MAX_DX_WORDS:
        return None
    if normalize(dx) and normalize(dx) in normalize(text):
        return None
    return BaseCase(id=f"PUBCASE-{split}-{i:05d}", source="pubcase", case_summary=text[:300],
                    original_note=text, diagnosis_ground_truth=dx, metadata={"hf": HF_NAME})


def load_public(split: str = "test") -> list[BaseCase]:
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, split=split)
    cases = [row_to_case(ex, i, split) for i, ex in enumerate(ds)]
    return [c for c in cases if c is not None]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-per-diagnosis", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../data/processed/pubcase_sample.jsonl")
    args = ap.parse_args()

    cases = [c.to_dict() for c in load_public(args.split)]
    sample, counts = stratified_sample(cases, args.n, args.max_per_diagnosis, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in sample:
            f.write(json.dumps(c) + "\n")
    print(f"usable {len(cases)}; sampled {len(sample)} -> {out}; by source {counts}")
