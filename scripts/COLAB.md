# Running the v3 pipeline on Colab (public text only)

Runtime: **A100**. It runs bf16 and holds a 14B judge (about 28 GB of weights) plus its KV cache in
40 GB, and with `--batch-size 48` an 8B model fills the card instead of idling. T4 has no bf16 and
16 GB (an 8B model barely fits and cannot batch); L4 and A10 (24 GB) cannot hold a 14B judge in fp16;
TPUs would need a different inference stack. Use batch 48 for 7-9B models and 16 for 14B.

Do NOT put MIMIC text on Colab. `paths.py` drops MIMIC rows unless `ECD_ALLOW_MIMIC=1`; never set
it here. MIMIC runs go on the cleared RunPod pod or the laptop.

## How resuming works
- Work happens on local disk (`/content/...`). `scripts/v3_panel.py` copies outputs to Drive after
  every stage (also after a failed stage) and restores them from Drive at the start.
- Every stage appends one row at a time and skips rows already present, so a disconnect loses at
  most one chunk (64 items). A stage with nothing left to do loads no model.
- Model weights are re-downloaded each session to local disk (fast, and avoids Drive quota and slow
  FUSE reads). Outputs and results are the only things kept on Drive.
- After a disconnect: reconnect, run Cell 1 and Cell 3 again. Nothing else.

## Cell 1: setup (every session)
```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["ECD_DATA_DIR"] = "/content/ecd_data"
os.environ["ECD_RESULTS_DIR"] = "/content/ecd_results"
os.environ["HF_HOME"] = "/content/hf"
!git clone <your repo url> /content/ecd 2>/dev/null || (cd /content/ecd && git pull)
!pip -q install -r /content/ecd/requirements.txt accelerate datasets scipy
```
Gated models (Llama): run `from huggingface_hub import login; login()` first.

## Cell 2: public sample (first session only; it is then restored from Drive)
```
!mkdir -p $ECD_DATA_DIR && cd /content/ecd/src && python public_loader.py --split test --n 400 --out $ECD_DATA_DIR/pubcase_sample.jsonl
!mkdir -p /content/drive/MyDrive/ecd/data && cp $ECD_DATA_DIR/pubcase_sample.jsonl /content/drive/MyDrive/ecd/data/
```
Check the usable-case count it prints.

## Cell 3: the whole panel (rerun this after any disconnect)
Smoke test first, with `--limit 20` and one model, and read the per-category pass rates
(`passed`, `n_attempts`) in the notes file before the full run.
```
!cd /content/ecd && python scripts/v3_panel.py \
    --sample $ECD_DATA_DIR/pubcase_sample.jsonl \
    --generator hf:<generator-repo> --verifier open:<verifier-repo> --judge open:<judge-repo> \
    --models hf:<tested-repo-1> hf:<tested-repo-2> \
    --drive-dir /content/drive/MyDrive/ecd --batch-size 48 --limit 20
```
Drop `--limit` for the full 400. Add `--dry-run` to print the commands without running them.
Generator and verifier must be different model families. `--batch-size` applies only to `hf:`
models (generator and tested models); `open:` judges and verifiers run one prompt at a time. If a
14B model is the generator or a tested model, run it in its own invocation with `--batch-size 16`.

Summaries land in `$ECD_RESULTS_DIR/v3_summary_judged__<model>__<judge>.json` and are synced to Drive.

## Claude (budget gate)
Use `claude:claude-sonnet-5` as a tested model; the runner stops at an estimated $9 (token counts
at $3/$15 per M), logging to `results/api_manifest_v3.jsonl`. Judge Claude's answers with the open
judge (no extra spend). Check the counts with `python src/v3_run.py answers --dry-run ...` first.
