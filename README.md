# Diagnosis Drift Benchmark + Evidential Consistency Decoding

Code for measuring *temporal sycophancy* in clinical LLMs (a model abandoning
a correct diagnosis after a misleading follow-up note) and evaluating ECD, a
contrastive-decoding mitigation.

## Data

- **MedQA** (`GBaker/MedQA-USMLE-4-options`) is fetched from Hugging Face.
- **MIMIC-IV-Note** requires PhysioNet credentialed access. Its files are not
  in this repo; pass their paths to `build_dataset.py`. No raw note text is
  ever committed (`.gitignore` blocks `data/` and raw-data formats).

## Setup

```
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

## Pipeline (run from `src/`)

```
python build_dataset.py --medqa-split test \
  --discharge <discharge.csv.gz> \
  --diagnoses-icd <diagnoses_icd.csv.gz> --d-icd <d_icd_diagnoses.csv.gz>
python subsample.py                 # fixed stratified eval sample
python stats.py                     # aggregate dataset stats (optional)
python eval_baseline.py             # baseline accuracy (Claude API)
python eval_drift.py                # adversarial injection + drift (Claude API)
python gating_network.py            # drift-risk classifier (CPU)
python ecd_v2.py <phase>            # corrected ECD, see scripts/RUNBOOK.md (the old alpha sweep had a sign error)
python week9_significance.py        # all significance tests -> significance_report.json
```

`eval_baseline.py`, `eval_drift.py`, `controls.py` and `ecd_v2.py` write results
incrementally and resume on rerun.

`notebooks/week6-drift-analysis.ipynb` reads the same result files for a
quick look at drift rate by category and source; not part of the pipeline.

## Layout

| Path | |
|---|---|
| `src/schema.py`, `*_loader.py`, `build_dataset.py`, `subsample.py` | data pipeline |
| `src/stats.py` | aggregate dataset stats |
| `src/prompts.py`, `grade.py`, `llm_clients.py` | prompting + LLM-as-judge |
| `src/adversarial_notes.py` | taxonomy definitions + note generation |
| `src/eval_baseline.py`, `eval_drift.py` | main benchmark |
| `src/ecd_decode.py` | the ECD algorithm (run directly for the self-test) |
| `src/ecd_demo.py`, `gating_network.py` | qualitative ECD demo, drift-risk classifier |
| `src/significance.py`, `week9_significance.py` | statistics |
| `src/ecd_v2.py`, `ecd_decode.py` | corrected ECD (beta) and its pre-registered R/S evaluation; `*_legacy.py` is the sign-error version, do not use |
| `src/controls.py`, `judge_validation.py`, `judge.py` | noise / neutral / helpful-note controls, second-judge validation, verdict cache |
| `src/note_audit.py`, `classifier_validity.py`, `stats_v2.py`, `power_check.py` | note-rule audit, classifier validity, GEE + cluster bootstrap, power-formula check |
| `scripts/` | pod setup (`setup_pod.sh`, `pod_env.sh`), preflight, public-subset export, `RUNBOOK.md` |
| `tests/` | offline unit tests (`python -m pytest tests -q`) |
| `notebooks/` | ad-hoc result analysis, not part of the pipeline |
