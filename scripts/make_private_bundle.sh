#!/usr/bin/env bash
# Run on the laptop (repo root). Bundles the MIMIC-derived inputs the GPU steps need.
# The bundle contains MIMIC note text: upload it only to a cleared pod by hand, never git,
# never the shared UCSC cluster. Delete the pod when finished.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
FILES=(eval_sample.jsonl eval_results_claude.jsonl drift_results_claude.jsonl helpful_results.jsonl)
for f in "${FILES[@]}"; do [ -f "data/processed/$f" ] || { echo "missing data/processed/$f"; exit 1; }; done
tar czf ecd_private_bundle.tgz -C data processed/eval_sample.jsonl processed/eval_results_claude.jsonl \
    processed/drift_results_claude.jsonl processed/helpful_results.jsonl
ls -la ecd_private_bundle.tgz
echo "bundle holds MIMIC-derived text. Upload to the RunPod pod only. It is gitignored? check: git check-ignore -v ecd_private_bundle.tgz"
