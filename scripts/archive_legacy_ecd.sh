#!/usr/bin/env bash
# Run on the pod from the repo root. Moves legacy (sign-error) ECD outputs out of
# the way. Per-trial files hold MIMIC-derived text -> data/ (gitignored).
set -euo pipefail
mkdir -p data/legacy_signerror results/legacy_signerror
for f in tradeoff_results.jsonl tradeoff_clean_accuracy.jsonl; do
  [ -f "data/processed/$f" ] && mv -v "data/processed/$f" "data/legacy_signerror/$f"
done
# aggregate only (rates, n, CI): safe for results/
[ -f data/processed/tradeoff_curve.json ] && mv -v data/processed/tradeoff_curve.json results/legacy_signerror/tradeoff_curve.json
echo "archived. Do not report anything in these locations."
