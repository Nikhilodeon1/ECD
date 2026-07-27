# Week 5: Baseline Evaluation

## Method
- Sample: 300 cases targeted (150 MedQA + 150 MIMIC-IV-Note, even split per `src/subsample.py`), seed 42, max 3 cases per diagnosis to avoid a handful of common ICU diagnoses dominating the set.
- Model tested: Claude (only model with a confirmed budget + verified enterprise Zero Data Retention agreement, required before any MIMIC note text could be sent to a cloud API).
- Prompt: shown only the presenting-evidence section (`src/prompts.py`), asked for a single most-likely diagnosis in a fixed format.
- Grading: LLM-as-judge (`src/grade.py`) rather than exact string match, since MIMIC ground truth is a verbose ICD long-title a model won't say verbatim. **Caveat: Claude is grading its own output right now** - self-grading bias is a real risk, worth cross-grading with a different model once GPT-5/Gemini budgets exist.

## Results
```
                accuracy    n
medqa           0.801      146
mimic-iv-note   0.521      140
```
(286 of 300 targeted cases completed - ~14 dropped to generation/grading call failures, worth a quick look at which ones and why before Week 6, but not alarming at this rate.)

## What this shows
Claude is meaningfully worse on real clinical notes (52.1%) than on curated USMLE-style vignettes (80.1%). This is a legitimate and useful finding, not just noise: MedQA vignettes are written to contain exactly the clues needed to point at one diagnosis, while real discharge notes are messier, cover comorbidities, and the presenting-evidence section alone doesn't always cleanly determine the specific primary ICD diagnosis. Worth stating explicitly in the eventual paper - it's a meaningful baseline-difficulty gap between the two data sources, and it means MIMIC cases are the harder, more realistic test of the actual research question.

## Before trusting this fully
- Only one model tested so far (Claude). GPT-5/Gemini need separate budgets + their own DUA/retention check before touching MIMIC data - same process as was done for Anthropic.
- Self-grading bias unaddressed - see caveat above.
- ~14 dropped cases unexplained - check what failed before Week 6 adversarial injection runs on the same sample.
