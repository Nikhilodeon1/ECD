# ECD (Evidential Consistency Decoding)

Studying "temporal sycophancy" in clinical LLMs: whether a model abandons a correct diagnosis after a misleading follow-up note, even when the original evidence hasn't changed.

## Status
- Week 2: technical background written up, see `docs/week2-technologies-report.md`
- Week 3: adversarial taxonomy (5 note types) + 20 draft vignettes, see `docs/week3-taxonomy-and-vignettes.md`
- Week 4: dataset prep pipeline (`src/`) + baseline stats, see `docs/week4-baseline-stats.md`. Both MedQA and MIMIC sides ran successfully (331,840 total cases).
- Week 5: baseline diagnosis accuracy eval, see `docs/week5-baseline-eval.md`. Claude only so far (78.0% MedQA / 54.7% MIMIC, full 300-case clean run) - GPT-5/Gemini/Llama-Med still pending.
- Week 6: adversarial injection across all 5 taxonomy categories, see `docs/week6-drift-results.md` and `notebooks/week6-drift-analysis.ipynb`. Overall Diagnosis Drift Rate: **28.2%** (n=994, final run) - real MIMIC cases drift ~2.5x more (43.3%) than MedQA cases (17.6%).
- Week 7: ECD implementation (`src/ecd_decode.py`, `src/ecd_demo.py`), see `docs/week7-ecd-results.md`. Verified correct (α=0 matches plain decoding exactly) and shown working on a real GPU run - recovered a correct pre-drift diagnosis at α=1 on one case. Two real limitations found: α=2 degenerates into incoherent output, and the open model tested doesn't reliably follow the structured output format Claude does.
- Week 8 (current): gating network + accuracy-drift tradeoff curve, code built (`src/gating_network.py`, `src/ecd_tradeoff.py`) - not yet run for real.

## Pipeline usage
```
cd src
python build_dataset.py --medqa-split test          # MedQA only, runs anywhere
python build_dataset.py --medqa-split test \         # add MIMIC on the pod
  --discharge /path/discharge.csv.gz \
  --diagnoses-icd /path/diagnoses_icd.csv.gz --d-icd /path/d_icd_diagnoses.csv.gz
python stats.py
```

## Data
MIMIC-IV-Note is PhysioNet-credentialed data and lives on a RunPod network volume, not in this repo. This repo only ever contains code, docs, and aggregate statistics — never raw notes or anything derived that could re-identify a patient. `.gitignore` excludes `data/` and common raw-data file types as a safety net, but treat that as a backstop, not the actual control.
