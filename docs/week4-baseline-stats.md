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

## MIMIC portion - run complete, real numbers
Ran on the pod against real MIMIC-IV-Note + MIMIC-IV hosp tables:
```
total_cases:       331,840  (236 medqa + 331,604 mimic-iv-note)
skipped:           189 notes with no matched primary diagnosis (dropped)
unique_diagnoses:  11,387
note length (words): min 0, max 4,265, mean 406.1, median 352.0
```
Top diagnoses are dominated by common inpatient conditions - sepsis, pneumonia, UTI, coronary disease, acute kidney failure, chemo encounters - which tracks with what a general hospital population should look like.

This means dataset size is no longer a constraint at all - 331k+ raw cases is far more than needed even for a much bigger benchmark than originally planned. The real work from here is picking a well-chosen, high-quality subset, not finding more data.

**Known issue found in this run:** `min: 0` words means some notes produced empty evidence text - the section-header regex is missing headers on at least some real notes (exactly the risk flagged before this ran). Need to quantify how common this is before trusting the data at scale - checking now.

## Known rough edges to fix before Week 5
- One MedQA answer text had a stray `\n"` artifact in it (`"Benzodiazepine intoxication\n\""`) - raw HF dataset text isn't fully clean, worth a quick pass to strip stray whitespace/quote characters.
- Section-header regex misses some real notes (see above) - need to quantify and likely fix before this data is trustworthy at scale.
- 189 MIMIC notes dropped for no matched primary diagnosis - small relative to 331k, but worth spot-checking a few to make sure it's a real absence and not a join bug.
- Severity/category fields aren't in this dataset yet - that's Week 6.
- Need to decide how to subsample down from 331k+331k to whatever final benchmark size makes sense - random sample, stratified by diagnosis frequency, or something else.
