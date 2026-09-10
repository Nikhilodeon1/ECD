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
python eval_baseline.py             # baseline accuracy (Claude API)
python eval_drift.py                # adversarial injection + drift (Claude API)
python gating_network.py            # drift-risk classifier (CPU)
python ecd_tradeoff.py              # alpha sweep, ECD vs plain (GPU + Claude judge)
python week9_significance.py        # all significance tests -> significance_report.json
```

`eval_baseline.py`, `eval_drift.py`, and `ecd_tradeoff.py` write results
incrementally and resume on rerun.

## Layout

| Path | |
|---|---|
| `src/schema.py`, `*_loader.py`, `build_dataset.py`, `subsample.py` | data pipeline |
| `src/prompts.py`, `grade.py`, `llm_clients.py` | prompting + LLM-as-judge |
| `src/adversarial_notes.py` | taxonomy definitions + note generation |
| `src/eval_baseline.py`, `eval_drift.py` | main benchmark |
| `src/ecd_decode.py` | the ECD algorithm (run directly for the self-test) |
| `src/ecd_tradeoff.py`, `ecd_demo.py`, `gating_network.py` | ECD evaluation |
| `src/significance.py`, `week9_significance.py` | statistics |
