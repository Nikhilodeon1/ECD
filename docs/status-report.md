# ECD Project Status Report

**Project**: Evidential Consistency Decoding (ECD) — measuring "temporal sycophancy" in clinical LLMs (models abandoning a correct diagnosis after a misleading follow-up note).
**Repo**: private GitHub repo, `Nikhilodeon1/ECD`.

## Code / Infrastructure
- `src/schema.py` — shared `BaseCase` data format for all cases.
- `src/medqa_loader.py` — pulls MedQA (USMLE) from Hugging Face, filters to diagnosis-style questions only (236 of 1,273 test questions qualify — MedQA is mostly non-diagnosis question types).
- `src/mimic_loader.py` — parses MIMIC-IV-Note discharge summaries, extracts presenting-evidence sections (HPI, PMH, Physical Exam, Pertinent Results), gets ground truth from the primary ICD diagnosis. Runs on the RunPod pod against the network volume.
- `src/build_dataset.py`, `src/stats.py` — pipeline orchestration and aggregate dataset statistics.
- `src/subsample.py` — reproducible, stratified eval sample (even split across data sources, capped per-diagnosis so common ICU diagnoses don't dominate). Same sample reused across weeks for comparability.
- `src/llm_clients.py`, `src/prompts.py`, `src/grade.py`, `src/eval_baseline.py` — baseline diagnosis-accuracy eval harness against Claude's API, with LLM-as-judge grading (needed because MIMIC ground truth is verbose ICD text a model won't say verbatim).
- All data (MIMIC notes, derived cases, eval results) stays out of git via `.gitignore` — repo only ever contains code, docs, and aggregate stats/numbers, never raw note text.

## Data
- Full combined dataset: **331,840 cases** (331,604 from MIMIC-IV-Note + 236 from MedQA), built and verified on the pod.
- Data quality: <1% of MIMIC cases dropped for unparseable evidence sections (0.64% zero-word, 0.32% under-20-word) — filtered out rather than fixing the parser, given the low rate.
- Compliance: verified before any MIMIC text was sent to a cloud API. Confirmed the Anthropic account has an enterprise Zero Data Retention agreement (not just a self-serve toggle), satisfying PhysioNet's cloud-API guidance. GPT-5/Gemini are NOT yet cleared for MIMIC data — same verification needs to happen for those providers before they touch real note text.

## Deliverables completed
- **Week 3**: adversarial taxonomy — 5 categories (Vague Contradictory Statement, Incorrect Specialist Opinion, Perturbed Lab Values, Misleading Medication Suggestion, Subtle Nurse Handoff Redirection), each mapped to a distinct hypothesized failure mechanism (tone bias, authority bias, anchoring, inference-from-action, informal-narrative bias) rather than arbitrary templates. Severity rubric (1-5) + 20 draft example vignettes. (`docs/week3-taxonomy-and-vignettes.md`)
- **Week 4**: dataset pipeline + baseline stats, real numbers above. (`docs/week4-baseline-stats.md`)
- **Week 5**: baseline diagnosis accuracy, no adversarial notes. Claude only (only model cleared for MIMIC so far): **78.7% accuracy on MedQA, 51.3% on MIMIC** (n=150 each, full clean run, zero drops). The gap is a real and useful finding — curated USMLE vignettes are easier than messy real clinical notes. (`docs/week5-baseline-eval.md`)

## Known limitations / open items
- **Only Claude tested so far.** GPT-5, Gemini, and Llama-Med are not yet running — GPT-5/Gemini need separate API budgets plus their own data-retention verification before touching MIMIC data; Llama-Med needs pod GPU + weights setup (not started).
- **Self-grading bias**: Claude is currently grading its own diagnosis outputs (LLM-as-judge). Worth cross-grading with a different model once another provider's budget/compliance is sorted.
- **Diagnosis Drift Rate (the project's core metric) has not been measured yet.** Week 5 only measured baseline accuracy with no adversarial note present. That's the next step (Week 6).
- An earlier open question from project scoping was never fully resolved: target publication venue (NeurIPS main track is unrealistic for this scope; NeurIPS Evaluations & Datasets track was identified as a realistic target), clinician access for validating ground truth/severity labels, and full API budget across all 4 model providers. These affect how large the final benchmark needs to be and haven't been revisited since Week 4.

## Immediate next step
Week 6: adversarial injection across all 5 taxonomy categories, applied to the cases Claude got right in the Week 5 baseline, to produce the first real Diagnosis Drift Rate numbers.
