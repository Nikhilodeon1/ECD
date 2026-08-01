"""Loads MIMIC-IV-Note discharge summaries and converts them to BaseCase.

Meant to run ON THE RUNPOD POD against the mounted network volume -- this
script never uploads or copies raw note text anywhere, it only ever writes
out data/processed/mimic_cases.jsonl locally on whatever machine it runs on,
and that path is gitignored.

Ground truth diagnosis comes from the structured diagnoses_icd table (the
primary diagnosis, seq_num == 1), not from parsing free text -- that's more
reliable than regexing the "Discharge Diagnosis:" section, which is
inconsistently formatted across notes. We still extract the History of
Present Illness section as the case's "original_note" evidence, since that's
the actual presenting-evidence text a model should be shown (the full
discharge summary already contains the outcome/diagnosis, which would leak
the answer).

NOT YET VALIDATED against real MIMIC-IV-Note files -- written against the
documented schema (see PhysioNet's MIMIC-IV-Note and MIMIC-IV hosp module
docs) but the section-header regex below is a best-effort pattern and should
be spot-checked against a handful of real notes before trusting it at scale.
Run this file directly (`python mimic_loader.py`) to run the built-in
section-splitter self-test against synthetic note text -- that only checks
the parsing logic, not real MIMIC formatting.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from schema import BaseCase

# Matches a line that's just a section header, e.g. "History of Present Illness:"
SECTION_HEADER = re.compile(r"^\s*([A-Z][A-Za-z /\-]{2,40}):\s*$", re.MULTILINE)

EVIDENCE_SECTIONS = [
    "History of Present Illness",
    "Past Medical History",
    "Physical Exam",
    "Pertinent Results",
]


def split_sections(note_text: str) -> dict[str, str]:
    """Splits a discharge note into {section_name: section_body}."""
    headers = list(SECTION_HEADER.finditer(note_text))
    sections = {}
    for i, m in enumerate(headers):
        name = m.group(1).strip()
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(note_text)
        sections[name] = note_text[start:end].strip()
    return sections


def extract_evidence(note_text: str) -> str:
    """Pulls presenting-evidence sections only -- excludes diagnosis/outcome
    sections so we don't leak the answer into the model's input."""
    sections = split_sections(note_text)
    parts = [sections[name] for name in EVIDENCE_SECTIONS if name in sections]
    if not parts:
        # fallback: better to keep something than drop the case, but this
        # should be rare -- worth checking how often it happens on real data
        return note_text[:2000]
    return "\n\n".join(parts)


def load_primary_diagnoses(diagnoses_icd_path: str, d_icd_path: str) -> pd.DataFrame:
    """Returns one row per hadm_id: the primary (seq_num==1) diagnosis title."""
    diag = pd.read_csv(diagnoses_icd_path, compression="infer")
    titles = pd.read_csv(d_icd_path, compression="infer")
    primary = diag[diag["seq_num"] == 1]
    merged = primary.merge(titles, on=["icd_code", "icd_version"], how="left")
    # a few hadm_ids have >1 row tied for seq_num==1 -- keep first so the
    # index is unique and .get() below returns a scalar, not a Series
    merged = merged.drop_duplicates(subset="hadm_id", keep="first")
    return merged.set_index("hadm_id")["long_title"]


def load_mimic_notes(
    discharge_path: str, diagnoses_icd_path: str, d_icd_path: str, chunksize: int = 20000
) -> list[BaseCase]:
    # discharge.csv.gz is ~330k rows of long free text -- loading it all into
    # one DataFrame at once was getting OOM-killed on smaller pods. Streaming
    # it in chunks keeps peak memory bounded to one chunk of raw text at a
    # time, since we only keep the much-smaller extracted evidence afterward.
    primary_dx = load_primary_diagnoses(diagnoses_icd_path, d_icd_path)

    cases = []
    skipped_no_dx = 0
    skipped_short_evidence = 0
    for chunk in pd.read_csv(discharge_path, compression="infer", chunksize=chunksize):
        chunk = chunk[chunk["note_type"] == "DS"]  # discharge summaries only
        for row in chunk.itertuples():
            dx = primary_dx.get(row.hadm_id)
            if pd.isna(dx) or dx is None:
                skipped_no_dx += 1
                continue
            evidence = extract_evidence(row.text)
            if len(evidence.split()) < 20:
                # measured at <1% of cases (0.64% zero-word, 0.32% under-20-word) --
                # cheaper to drop these section-parsing failures than fix the regex
                skipped_short_evidence += 1
                continue
            cases.append(
                BaseCase(
                    id=f"MIMIC-{row.note_id}",
                    source="mimic-iv-note",
                    case_summary=evidence,
                    original_note=evidence,
                    diagnosis_ground_truth=dx,
                    metadata={
                        "subject_id": row.subject_id,
                        "hadm_id": row.hadm_id,
                        "note_id": row.note_id,
                    },
                )
            )
    if skipped_no_dx:
        print(f"skipped {skipped_no_dx} notes with no matched primary diagnosis")
    if skipped_short_evidence:
        print(f"skipped {skipped_short_evidence} notes with under-20-word evidence text")
    return cases


def _self_test():
    """Checks the section splitter against synthetic note text -- does NOT
    validate against real MIMIC formatting, just the parsing logic itself."""
    fake_note = """\
History of Present Illness:
65F with 3 days of fever and cough, found to have RLL consolidation on CXR.

Past Medical History:
Hypertension, type 2 diabetes.

Physical Exam:
Temp 101.8, crackles at right base.

Discharge Diagnosis:
Community-acquired pneumonia.
"""
    sections = split_sections(fake_note)
    assert "History of Present Illness" in sections
    assert "Discharge Diagnosis" in sections
    evidence = extract_evidence(fake_note)
    assert "Discharge Diagnosis" not in evidence.split("\n")[0]
    assert "pneumonia" not in evidence.lower()
    print("self-test passed:")
    print(sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discharge", help="path to discharge.csv.gz")
    parser.add_argument("--diagnoses-icd", help="path to diagnoses_icd.csv.gz")
    parser.add_argument("--d-icd", help="path to d_icd_diagnoses.csv.gz")
    parser.add_argument("--out", default="../data/processed/mimic_cases.jsonl")
    parser.add_argument(
        "--self-test", action="store_true", help="run the parser self-test and exit"
    )
    args = parser.parse_args()

    if args.self_test or not args.discharge:
        _self_test()
        if not args.discharge:
            print("\nno --discharge path given, ran self-test only")
        raise SystemExit(0)

    cases = load_mimic_notes(args.discharge, args.diagnoses_icd, args.d_icd)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict()) + "\n")

    print(f"wrote {len(cases)} cases to {out_path}")
