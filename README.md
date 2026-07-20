# ECD (Evidential Consistency Decoding)

Studying "temporal sycophancy" in clinical LLMs: whether a model abandons a correct diagnosis after a misleading follow-up note, even when the original evidence hasn't changed.

## Status
- Week 2: technical background written up, see `docs/week2-technologies-report.md`
- Week 3: adversarial taxonomy (5 note types) + 20 draft vignettes, see `docs/week3-taxonomy-and-vignettes.md`
- Week 4 (current): dataset prep pipeline for MIMIC-IV-Note + MedQA

## Data
MIMIC-IV-Note is PhysioNet-credentialed data and lives on a RunPod network volume, not in this repo. This repo only ever contains code, docs, and aggregate statistics — never raw notes or anything derived that could re-identify a patient. `.gitignore` excludes `data/` and common raw-data file types as a safety net, but treat that as a backstop, not the actual control.
