# Evidence ledger

Every number in paper/main.tex comes from a row here. Source for public rows: `results/v3_summary_judged__<model>__open-microsoft-phi-4.json`, copied from Drive on 2026-11-19; the Colab runs wrote them on 2026-11-13. Source for MIMIC rows: `results/v3_summary_judged__claude-claude-sonnet-5__claude-claude-sonnet-5.json`. Derived values are computed in paper/make_ledger.py.

Prespecified in docs/prereg-v3.md: R, N, the five categories, framing arms A, B, C, contrasts B-A and C-A, the helpful note. Exploratory: hedged arms (Ah, Bh, Ch) and the adoption string match.

## Gemma 4 12B, public cases (n_cases 400, baseline-correct 114, accuracy 28.5%)

| id | arm | n | drift (%) | excess over neutral, points [95% CI] | status |
|---|---|---|---|---|---|
| gemma4-0 | R | 114 | 2.6 | -20.2 [-28.1, -12.3] | prespecified |
| gemma4-1 | N | 114 | 22.8 |  | prespecified |
| gemma4-2 | VCS | 87 | 36.8 | +13.8 [+3.4, +24.1] | prespecified |
| gemma4-3 | ISO | 108 | 37.0 | +14.8 [+6.5, +24.1] | prespecified |
| gemma4-4 | PLV | 93 | 22.6 | +1.1 [-5.4, +8.6] | prespecified |
| gemma4-5 | MMS | 89 | 20.2 | -1.1 [-10.1, +7.9] | prespecified |
| gemma4-6 | SNHR | 109 | 30.3 | +9.2 [+0.9, +17.4] | prespecified |
| gemma4-7 | A | 108 | 69.4 | +49.1 [+38.9, +59.3] | prespecified |
| gemma4-8 | B | 108 | 61.1 | +40.7 [+30.6, +50.9] | prespecified |
| gemma4-9 | C | 108 | 49.1 | +28.7 [+18.5, +38.9] | prespecified |
| gemma4-10 | Ah | 108 | 45.4 | +25.0 [+14.8, +35.2] | exploratory |
| gemma4-11 | Bh | 108 | 38.0 | +17.6 [+8.3, +26.9] | exploratory |
| gemma4-12 | Ch | 108 | 35.2 | +14.8 [+6.5, +23.1] | exploratory |

Derived: N minus R = 20.2 points; CI from the paired R-minus-N bootstrap with the sign flipped: [12.3, 28.1] (R excess -20.2 [-28.1, -12.3]).

| contrast | diff (points) | 95% CI | discordant up/down | exact p | Holm p |
|---|---|---|---|---|---|
| B-A | -8.3 | [-13.9, -3.7] | 0/9 | 0.0039 | 0.0039 |
| C-A | -20.4 | [-28.7, -13.0] | 1/23 | 0.0 | 0.0 |
| Bh-Ah | -7.4 | [-13.9, -0.9] | 3/11 | 0.0574 | 0.0574 |
| Ch-Ah | -10.2 | [-18.5, -1.9] | 5/16 | 0.0266 | 0.0532 |

Helpful note: corrects a wrong baseline in 19.7% of 213 cases; stays correct in 79.8% of 94 baseline-correct cases (so 20.2% do not).
Adoption (exploratory string match; the answer contains the inserted wrong diagnosis): unattributed 66 of 108, attending 61 of 108, nursing 47 of 108.

## Llama 3.1 8B, public cases (n_cases 400, baseline-correct 83, accuracy 20.7%)

| id | arm | n | drift (%) | excess over neutral, points [95% CI] | status |
|---|---|---|---|---|---|
| llama31-0 | R | 83 | 4.8 | -21.7 [-32.5, -12.0] | prespecified |
| llama31-1 | N | 83 | 26.5 |  | prespecified |
| llama31-2 | VCS | 65 | 27.7 | +7.7 [-1.5, +16.9] | prespecified |
| llama31-3 | ISO | 78 | 59.0 | +34.6 [+20.5, +47.4] | prespecified |
| llama31-4 | PLV | 66 | 27.3 | +4.5 [-4.5, +13.6] | prespecified |
| llama31-5 | MMS | 60 | 31.7 | +3.3 [-6.7, +13.3] | prespecified |
| llama31-6 | SNHR | 79 | 30.4 | +5.1 [-3.8, +13.9] | prespecified |
| llama31-7 | A | 77 | 83.1 | +57.1 [+44.2, +68.8] | prespecified |
| llama31-8 | B | 77 | 81.8 | +55.8 [+42.9, +67.5] | prespecified |
| llama31-9 | C | 77 | 80.5 | +54.5 [+41.6, +66.2] | prespecified |
| llama31-10 | Ah | 77 | 64.9 | +39.0 [+26.0, +51.9] | exploratory |
| llama31-11 | Bh | 77 | 66.2 | +40.3 [+27.3, +51.9] | exploratory |
| llama31-12 | Ch | 77 | 61.0 | +35.1 [+22.1, +48.1] | exploratory |

Derived: N minus R = 21.7 points; CI from the paired R-minus-N bootstrap with the sign flipped: [12.0, 32.5] (R excess -21.7 [-32.5, -12.0]).

| contrast | diff (points) | 95% CI | discordant up/down | exact p | Holm p |
|---|---|---|---|---|---|
| B-A | -1.3 | [-5.2, +2.6] | 1/2 | 1.0 | 1.0 |
| C-A | -2.6 | [-7.8, +2.6] | 1/3 | 0.625 | 1.0 |
| Bh-Ah | +1.3 | [-2.6, +6.5] | 2/1 | 1.0 | 1.0 |
| Ch-Ah | -3.9 | [-9.1, +0.0] | 0/3 | 0.25 | 0.5 |

Helpful note: corrects a wrong baseline in 19.4% of 242 cases; stays correct in 78.5% of 65 baseline-correct cases (so 21.5% do not).
Adoption (exploratory string match; the answer contains the inserted wrong diagnosis): unattributed 66 of 77, attending 62 of 77, nursing 61 of 77.

## Claude Sonnet 5 on MIMIC-IV-Note (150 cases)

Baseline-correct 77 of 150 (51.3%).

| arm | n | drift (%) | excess over neutral, points [95% CI] |
|---|---|---|---|
| R | 77 | 3.9 | -5.2 [-13.0, +1.3] |
| N | 77 | 9.1 |  |
| ISO | 26 | 42.3 | +34.6 [+11.5, +53.8] |
| A_unattributed | 61 | 78.7 | +70.5 [+59.0, +82.0] |
| B_attending | 61 | 68.9 | +60.7 [+47.5, +72.1] |
| C_nursing | 61 | 57.4 | +49.2 [+36.1, +62.3] |

Derived: N minus R = 5.2 points, CI [-1.3, 13.0] (sign-flipped R excess).

| contrast | diff (points) | 95% CI | discordant up/down | exact p | Holm p |
|---|---|---|---|---|---|
| B_attending minus A_unattributed | -9.8 | [-19.7, +0.0] | 2/8 | 0.1094 | 0.1094 |
| C_nursing minus A_unattributed | -21.3 | [-31.1, -11.5] | 0/13 | 0.0002 | 0.0004 |

## Note generation (public cases, Qwen2.5-7B-Instruct, up to 3 attempts)

| kind | notes | passed rules (%) | mean attempts |
|---|---|---|---|
| X | 400 | 96.3 | 1.08 |
| ISO | 385 | 94.3 | 1.16 |
| SNHR | 385 | 98.7 | 1.09 |
| PLV | 385 | 89.6 | 1.25 |
| VCS | 385 | 84.9 | 1.54 |
| MMS | 385 | 84.2 | 1.66 |
| H | 400 | 76.7 | 1.49 |

Verifier (phi-4) answered yes to 97.6% of 2124 notes. In a calibration check it answered yes to 96 of 96 notes judged against their own category and to 11 of 96 judged against a wrong category.

## Setup facts

- Public cases: MedCaseReasoning test split (897 rows on the dataset viewer), 400 sampled, at most 3 per diagnosis, seed 42. Presentation 60 to 700 words, diagnosis at most 8 words, diagnosis string not in the presentation.
- MIMIC cases: 150 MIMIC-IV-Note cases from the evaluation sample (median 333 words, 12 over 800 words and truncated).
- Neutral arm: 16 fixed non-clinical lines, one chosen by a hash of the case id. Re-ask arm: the baseline prompt sampled again.
- Open models: greedy decoding, 160 new tokens. Claude: thinking disabled, 500 tokens for baseline and re-ask, 300 for note arms.
- Bootstrap: 10000 case resamples. Exact McNemar test on discordant pairs. Holm correction over two contrasts per family.
- MIMIC run: 1430 Claude API calls (745 Haiku 5.5, 685 Sonnet 5), estimated cost 6.01 dollars before the final judging pass and 6.51 dollars after it, priced at Sonnet rates.
- Models: Gemma 4 12B instruction-tuned, Llama 3.1 8B Instruct, Qwen2.5 7B Instruct (generator), phi-4 14B (verifier and public judge), Claude Sonnet 5 (tested model and MIMIC judge), Claude Haiku 5.5 (MIMIC notes and verifier).
- Token limits and sizes: 160 new tokens for open models, 500 and 300 for Claude, evidence cut to 800 words, public presentations of 60 to 700 words, diagnoses of at most 8 words, notes of at most 40 words, 16 neutral lines, 3 attempts.
- Holm p printed as 0.0 in the summary file is reported in the paper as below 0.001.
- Judge reading check: of the first 25 baseline answers that phi-4 judged incorrect for Gemma 4, the author's non-clinician reading found about 3 to 4 that look like a parent or subtype of the reference diagnosis.
- Count of verifier notes: 2124.
- Judge validation for the v3 judges: none. Clinician audit: none.
