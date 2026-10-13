# Run book (rebuttal round)

## Where each kind of work runs

| Place | What may be there | What runs there |
|---|---|---|
| **Laptop** (holds MIMIC under the author's credentialed access; keep `data/` out of any cloud sync) | MedQA + MIMIC, `ECD_ALLOW_MIMIC=1` | Everything that only needs the Claude API or CPU: drift re-run, controls, judge cache, note audit, classifier, stats, tiered judge |
| **RunPod GPU pod, uploaded fresh each session** | MIMIC-derived files, uploaded by hand, pod deleted afterwards (no network volume) | MIMIC GPU steps: ECD v2 on MIMIC trials, open-weight judge, stronger-model pilot on MIMIC |
| **UCSC shared pod** | MedQA only (public). `ECD_ALLOW_MIMIC` stays unset: loaders drop MIMIC rows | MedQA-only GPU steps if an A100 is available |
| **Git / GitHub** | code, tests, `results/` aggregates, `figures/` | never any note text, never `data/` |

MIMIC-derived text (the evidence in `eval_sample.jsonl`, generated notes, model outputs,
judge caches) is never committed and never uploaded to the UCSC cluster. The loaders
enforce this: without `ECD_ALLOW_MIMIC=1` MIMIC rows are dropped with a `[guard]` warning.

## Laptop (PowerShell, repo root)

    cd src
    $env:ECD_ALLOW_MIMIC = "1"
    python ../scripts/pod_preflight.py                  # sample 300, baseline 117+82 correct; drift file missing until re-run
    python -m pytest ../tests -q
    # paid steps: always --dry-run first, and only after the budget is confirmed
    python eval_drift.py --dry-run                      # (see below) then the real run, resumable

Backed-up inputs live in `data/processed/` (copied from the Aug 31 backup): `eval_sample.jsonl`
(300 cases, 150/150) and `eval_results_claude.jsonl` (199 baseline-correct, 117 MedQA + 82
MIMIC, identical to the paper). The 994-trial drift file was lost with the pod; only a
175-trial partial from Aug 31 survives (`drift_results_claude_PARTIAL_aug31_175trials.jsonl`,
first 35 cases, run config unknown). The drift run must be redone, and its numbers will differ
from the submitted ones (the sampling noise T3a measures).

## RunPod GPU pod (MIMIC GPU steps): 1x A100 80 GB, 16 cores, 64-128 GB RAM

    # laptop: bundle the private inputs (MIMIC-derived, never git, never UCSC)
    bash scripts/make_private_bundle.sh           # -> ecd_private_bundle.tgz
    # upload that file and the repo to the pod (web terminal / scp), then on the pod:
    bash scripts/setup_pod.sh && source scripts/pod_env.sh
    mkdir -p ~/ECD/data && tar xzf ~/ecd_private_bundle.tgz -C ~/ECD/data/
    export ECD_ALLOW_MIMIC=1
    python ../scripts/pod_preflight.py
    python -m pytest ../tests -q
    python g1_gate.py                             # STOP here if match rate does not rise with beta / ~100% at beta=1
    # corrected ECD on the 8B. Judge = free open-weight model on the same GPU (no Claude calls):
    python ecd_v2.py baseline && python ecd_v2.py undefended
    python ecd_v2.py freeze-D && python ecd_v2.py helpful0 && python ecd_v2.py freeze-H
    # --- beta > 0 starts: refuses until docs/prereg-v2.md is committed ---
    python ecd_v2.py sweep && python ecd_v2.py prompt && python ecd_v2.py summarize
    # validate the open judge against Claude's recorded verdicts (free)
    python judge_validation.py --judge open --model-name Qwen/Qwen2.5-7B-Instruct
    # copy results/*.json and figures/*.png back (aggregates only), then DELETE the pod

## UCSC shared pod (MedQA only)

    cd ~ && git clone <repo> ECD && cd ECD
    bash scripts/setup_pod.sh          # check it prints a CUDA build: torch ...+cu... cuda True
    source scripts/pod_env.sh          # unsets ECD_ALLOW_MIMIC
    # copy a MedQA-only bundle in: python scripts/export_public_subset.py (run on the laptop with
    # ECD_ALLOW_MIMIC=1; it writes ecd_public_medqa.tgz with MedQA rows only), then
    mkdir -p ~/ECD/data && tar xzf ~/ecd_public_medqa.tgz -C ~/ECD/data/

A 16 GB V100 cannot hold the 8B model in fp16; use an A100/A6000/L40 (>= 40 GB).

## Steps (same commands anywhere; only the data they can see differs)

Free (no Claude calls):

    python power_check.py
    python note_audit.py ; python classifier_validity.py ; python stats_v2.py
    python g1_gate.py                                   # GPU: gate for the corrected ECD

Paid (Claude API; `--dry-run` first):

    python controls.py noise --dry-run                  # T3a
    python controls.py noise ; python controls.py neutral ; python controls.py helpful
    python controls.py summary
    python judge_validation.py --judge tiered           # T4b
    python judge_validation.py --judge open --model-name Qwen/Qwen2.5-7B-Instruct   # T4a, GPU, free of Claude cost

ECD v2 (GPU; judge calls are Claude; needs the helpful notes from `controls.py helpful`):

    python ecd_v2.py baseline ; python ecd_v2.py undefended
    python ecd_v2.py freeze-D ; python ecd_v2.py helpful0 ; python ecd_v2.py freeze-H
    # beta > 0 starts here: refuses until docs/prereg-v2.md is committed
    python ecd_v2.py sweep ; python ecd_v2.py prompt ; python ecd_v2.py summarize

Outputs under `data/` hold per-trial text and are never committed; `results/` and `figures/`
hold aggregates only.
