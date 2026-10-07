# DO NOT REPORT: legacy ECD results (sign error)

Everything produced by `src/ecd_decode_legacy.py` / `src/ecd_tradeoff_legacy.py`
(the submitted ECD, `log p_final = (1+alpha) log p_full - alpha log p_orig`).
That blend amplifies the note-conditioned distribution. Old alpha corresponds to
beta = -alpha in the corrected form. See `docs/erratum.md`.

Aggregate numbers as printed in the submitted paper, kept only so the erratum can
cite them (n = 54 drifted trials per alpha, 8B open-weight model):

| old alpha | = beta | recovery of original diagnosis |
|---|---|---|
| 0 | 0 | 5.6% |
| 0.5 | -0.5 | 5.6% |
| 1.0 | -1.0 | 5.6% |
| 1.5 | -1.5 | 1.9% |
| 2.0 | -2.0 | 0.0% |

Clean-case accuracy 38.2% (n = 55) is unaffected by the sign error (no note).

Per-trial files (`tradeoff_results.jsonl`, `tradeoff_clean_accuracy.jsonl`) contain
model outputs on MIMIC-derived text. They stay on the pod in
`data/legacy_signerror/` and are never committed. `scripts/archive_legacy_ecd.sh`
does the move and copies only the aggregate `tradeoff_curve.json` here.
