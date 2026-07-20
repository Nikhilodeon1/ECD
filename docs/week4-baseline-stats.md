# Week 4: Dataset Pipeline + Baseline Stats

## What the pipeline does
- `src/schema.py` - shared `BaseCase` format both loaders output (id, source, case text, ground-truth diagnosis, metadata). This is the "clean" pre-adversarial case - adversarial notes get attached to these in Week 6 per the taxonomy in `docs/week3-taxonomy-and-vignettes.md`.
- `src/medqa_loader.py` - pulls MedQA (USMLE) from Hugging Face and filters it down to diagnosis-style questions only.
- `src/mimic_loader.py` - parses MIMIC-IV-Note discharge summaries, pulls the presenting-evidence sections (History of Present Illness, Past Medical History, Physical Exam, Pertinent Results), and gets ground truth from the primary ICD diagnosis rather than parsing free text. Meant to run on the RunPod pod against the network volume - not tested against real MIMIC files yet, only against synthetic note text (see the self-test: `python mimic_loader.py --self-test`).
- `src/build_dataset.py` - runs both loaders and combines them into one `cases.jsonl`.
- `src/stats.py` - computes aggregate stats only (counts, distributions, lengths - no note text), which is why this file is safe to commit even though the actual data isn't.

## An important finding from building this
MedQA is not a diagnosis-question dataset - it's a general USMLE question bank that also covers ethics, pharmacology, next-step-in-management, etc. Only **236 of 1,273 test questions (about 18.5%)** matched a diagnosis-style filter. The loader filters for this automatically now; worth knowing in case the filter needs tightening later (a manual spot check of the matches would be a good Week 5 sanity check).

## Baseline stats (MedQA portion, test split only, run locally)
```
total cases:        236
note length (words): min 36, max 268, mean 125.2, median 120
unique diagnoses:    229 (out of 236 cases - most diagnoses only appear once)
```
MedQA train split (~10k questions) hasn't been pulled yet - filtering that in too would give a much bigger pool if 236 test cases turns out to be too few once adversarial notes are added on top in Week 6.

## MIMIC portion - not run yet
This needs to run on the pod where the network volume is mounted:
```
python build_dataset.py --medqa-split test \
  --discharge /path/to/discharge.csv.gz \
  --diagnoses-icd /path/to/diagnoses_icd.csv.gz \
  --d-icd /path/to/d_icd_diagnoses.csv.gz
python stats.py
```
Once that runs, drop the real MIMIC numbers into this doc alongside the MedQA ones above. The output `cases.jsonl` itself stays on the pod / in `data/` locally - never gets committed.

## Known rough edges to fix before Week 5
- One MedQA answer text had a stray `\n"` artifact in it (`"Benzodiazepine intoxication\n\""`) - raw HF dataset text isn't fully clean, worth a quick pass to strip stray whitespace/quote characters.
- The MIMIC section-header regex has only been checked against synthetic text, not real notes - spot-check it against a handful of real discharge summaries on the pod before trusting it at scale.
- Severity/category fields aren't in this dataset yet - that's Week 6.
