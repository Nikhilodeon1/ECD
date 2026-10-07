# Pod runbook (rebuttal round)

Repo in persistent home (`~/ECD`), venv and model caches in `/tmp` (rebuilt on every
new pod). MIMIC text is only ever loaded when `ECD_ALLOW_MIMIC=1`, which you set
**only** on the cleared RunPod volume. On the shared pod, leave it unset and every
loader silently drops MIMIC rows (it prints `[guard] dropped ...`).

## 0. Once, on RunPod (has the MIMIC volume)

    cd <repo> && ECD_ALLOW_MIMIC=1 python scripts/export_public_subset.py
    # -> ecd_public_medqa.tgz : MedQA-only eval_sample / baseline / drift results (public data)

Download that file (a few hundred KB) and put it on the shared pod.

## 1. Every new shared pod

    cd ~ && git clone <your repo url> ECD && cd ECD
    bash scripts/setup_pod.sh            # venv in /tmp/venv, CUDA torch, requirements (few min)
    source scripts/pod_env.sh            # every new shell: venv + caches in /tmp, cd to src/
    mkdir -p ~/ECD/data && tar xzf ~/ecd_public_medqa.tgz -C ~/ECD/data/
    printf 'ClaudeKey=<your key>\n' > ~/ECD/.env && chmod 600 ~/ECD/.env   # only needed for paid steps
    nvidia-smi --query-gpu=name,memory.total --format=csv   # 16 GB V100 cannot hold the 8B in fp16; 32 GB can

## 2. Free steps (no Claude calls). Run these first.

    python ../scripts/pod_preflight.py             # public mode: expects MedQA 150 / 117 correct / ~585 trials
    python -m pytest ../tests -q                   # 20 offline tests
    python power_check.py                          # G-5: McNemar formula vs exact simulation
    python note_audit.py                           # T5 (MedQA notes here; MIMIC notes on RunPod)
    python classifier_validity.py                  # T9
    python stats_v2.py                             # T8 (source term needs MIMIC: rerun on RunPod)
    python g1_gate.py                              # T1d gate: 20 trials, 8B, no API (downloads ~16 GB to /tmp/hf)

Then send back `results/*.json`. The G1 gate must show match rate rising with beta and
~100% at beta=1; if not, stop.

## 3. Paid steps (Claude API). Do NOT start until the budget is confirmed.

Always `--dry-run` first; it prints the paid calls still to make and makes none.

    python controls.py noise   --dry-run    # T3a ~ $0.5 on the MedQA half
    python controls.py neutral --dry-run    # T3b
    python controls.py helpful --dry-run    # T3c
    python controls.py noise  --limit 20    # pilot, then drop --limit
    python controls.py noise && python controls.py neutral && python controls.py helpful
    python controls.py summary              # T3e -> results/controls_summary.json

Judge validation (T4a open judge is free of Claude cost; needs a GPU, ~15 GB for a 7B in fp16):

    python judge_validation.py --judge open --model-name Qwen/Qwen2.5-7B-Instruct --dry-run
    python judge_validation.py --judge open --model-name Qwen/Qwen2.5-7B-Instruct
    python judge_validation.py --judge tiered      # T4b, Claude, ~$1.2 on the full 1,294

ECD v2 on the 8B (needs T3c first for the helpful notes; judge calls are Claude):

    python ecd_v2.py baseline && python ecd_v2.py undefended
    python ecd_v2.py freeze-D && python ecd_v2.py helpful0 && python ecd_v2.py freeze-H
    # --- beta > 0 starts here: refuses to run until docs/prereg-v2.md is committed ---
    python ecd_v2.py sweep && python ecd_v2.py prompt && python ecd_v2.py summarize

## 4. Same commands on RunPod for the MIMIC half

    source scripts/pod_env.sh && export ECD_ALLOW_MIMIC=1
    # then repeat sections 2-3. Same scripts, same outputs, all MIMIC-derived rows stay under data/.

Outputs under `data/` (per-trial text) are never committed. Aggregates in `results/`
and `figures/` are safe to commit.
