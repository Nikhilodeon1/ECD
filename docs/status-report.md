# ECD Project Status Report

**Project**: Evidential Consistency Decoding (ECD) — measuring "temporal sycophancy" in clinical LLMs (models abandoning a correct diagnosis after a misleading follow-up note).
**Repo**: private GitHub repo, `Nikhilodeon1/ECD`.

## Code / Infrastructure
- `src/schema.py` — shared `BaseCase` data format for all cases.
- `src/medqa_loader.py` — pulls MedQA (USMLE) from Hugging Face, filters to diagnosis-style questions only (236 of 1,273 test questions qualify — MedQA is mostly non-diagnosis question types).
- `src/mimic_loader.py` — parses MIMIC-IV-Note discharge summaries, extracts presenting-evidence sections (HPI, PMH, Physical Exam, Pertinent Results), gets ground truth from the primary ICD diagnosis. Runs on the RunPod pod against the network volume. Streams the file in chunks (fixed after an out-of-memory kill on a smaller pod) rather than loading all ~330k rows at once.
- `src/build_dataset.py`, `src/stats.py` — pipeline orchestration and aggregate dataset statistics.
- `src/subsample.py` — reproducible, stratified eval sample (even split across data sources, capped per-diagnosis so common ICU diagnoses don't dominate). Same sample reused across weeks for comparability.
- `src/llm_clients.py`, `src/prompts.py`, `src/grade.py`, `src/eval_baseline.py` — baseline diagnosis-accuracy eval harness against Claude's API, with LLM-as-judge grading (needed because MIMIC ground truth is verbose ICD text a model won't say verbatim).
- `src/adversarial_notes.py`, `src/eval_drift.py` — new this update. Generates a category-specific adversarial follow-up note per case (Claude, guided by the Week 3 taxonomy definitions), re-queries the diagnosis, and checks via LLM-as-judge whether it changed from the model's own original answer.
- `notebooks/week6-drift-analysis.ipynb` — loads drift results, produces the drift-rate tables and a by-category bar chart.
- All data (MIMIC notes, derived cases, eval results) stays out of git via `.gitignore` — repo only ever contains code, docs, and aggregate stats/numbers, never raw note text.

## Data
- Full combined dataset: **331,840 cases** (331,604 from MIMIC-IV-Note + 236 from MedQA), built and verified on the pod.
- Data quality: <1% of MIMIC cases dropped for unparseable evidence sections (0.64% zero-word, 0.32% under-20-word) — filtered out rather than fixing the parser, given the low rate.
- Compliance: verified before any MIMIC text was sent to a cloud API. Confirmed the Anthropic account has an enterprise Zero Data Retention agreement (not just a self-serve toggle), satisfying PhysioNet's cloud-API guidance. GPT-5/Gemini are NOT yet cleared for MIMIC data — same verification needs to happen for those providers before they touch real note text.
- **Operational risk found this update**: generated data (`data/processed/`) lives only on the pod's local disk, not the persistent network volume. A pod reset wiped it once already, forcing a full rerun of the dataset build + baseline eval. Not yet fixed — still a risk for future pod resets. Worth pointing the pipeline's output path at the actual persistent volume mount.

## Deliverables completed
- **Week 3**: adversarial taxonomy — 5 categories (Vague Contradictory Statement, Incorrect Specialist Opinion, Perturbed Lab Values, Misleading Medication Suggestion, Subtle Nurse Handoff Redirection), each mapped to a distinct hypothesized failure mechanism (tone bias, authority bias, anchoring, inference-from-action, informal-narrative bias) rather than arbitrary templates. Severity rubric (1-5) + 20 draft example vignettes. (`docs/week3-taxonomy-and-vignettes.md`)
- **Week 4**: dataset pipeline + baseline stats, real numbers above. (`docs/week4-baseline-stats.md`)
- **Week 5**: baseline diagnosis accuracy, no adversarial notes. Claude only (only model cleared for MIMIC so far): **78.0% accuracy on MedQA, 54.7% on MIMIC** (n=150 each, full clean run, zero drops). The gap is a real and useful finding — curated USMLE vignettes are easier than messy real clinical notes. (`docs/week5-baseline-eval.md`)
- **Week 6**: adversarial injection across all 5 taxonomy categories, applied to the cases Claude got right in Week 5. **Overall Diagnosis Drift Rate: 28.2%** (n=994, final run against the current pipeline) — Claude abandons an already-correct diagnosis about 3 in 10 times after a single misleading note. Sharper finding: **MIMIC cases drift 43.3% vs. MedQA's 17.6%**, a ~2.5x gap — consistent with Week 5's finding that real clinical notes are already harder to diagnose correctly. Category-level differences (26-34%) remain flat at this sample size, not a strong signal yet. (`docs/week6-drift-results.md`, `notebooks/week6-drift-analysis.ipynb`)
- **Week 7**: ECD algorithm implemented (`src/ecd_decode.py`) — Context-Aware Decoding formula, KV-cached token-by-token loop. Verified two ways: CPU self-test proves α=0 is byte-identical to plain greedy decoding (exact invariant), and a real H100 run against Llama3-OpenBioLLM-8B recovered a correct pre-drift diagnosis at α=1 from a wrong baseline. Two documented limitations: α=2 degrades into incoherent output, and this open model doesn't reliably follow the structured output format Claude does. (`docs/week7-ecd-results.md`)
- **Week 8 (in progress)**: `src/gating_network.py` (drift-risk classifier trained on Week 6's labeled trials) and `src/ecd_tradeoff.py` (accuracy-drift tradeoff sweep) are built and smoke-tested but not yet run for real — blocked on Claude API budget (see below).

## Infra note this update
A manual Claude.ai copy-paste batching mode was built to work around an API quota limit, then **fully removed** after it was found to introduce a real confound — batching many prompts into one shared context measurably changed results (~11-point swing on MedQA) vs. isolated API calls. Reverted to pure API-only. In its place: `eval_baseline.py`, `eval_drift.py`, and `ecd_tradeoff.py` now write each result to disk immediately and auto-resume from wherever a run stopped (verified with a simulated interruption test) — this matters because pod resets have wiped `data/processed/` (gitignored, correctly) at least 3 times now, each time requiring careful restoration from a local backup rather than a full costly rerun.

## Known limitations / open items
- **Only Claude tested so far.** GPT-5, Gemini are not yet running — need separate API budgets plus their own data-retention verification before touching MIMIC data. Llama-Med is running (Week 7/8), but only as the open-weight side of ECD, not as a fourth model in the baseline/drift comparison.
- **Self-grading bias**: Claude grades its own diagnosis outputs, its own generated adversarial notes, and its own drift verdicts throughout. No independent validation beyond a 5-example manual spot-check, which mostly held up but surfaced a genuinely ambiguous case (judge called "no drift" when the model added an unconfirmed etiology detail rather than cleanly changing its answer) — a real, not just hypothetical, source of noise in the DDR numbers.
- Mentor feedback received: (1) asked how often temporal sycophancy actually occurs — answered, 28.2% overall, 43.3% on real notes. (2) suggested narrowing scope to one model + one defense rather than the full 4-model, multi-defense plan — not yet formally resolved; worth deciding before Week 9 writing locks in scope.
- An earlier open question from project scoping was never fully resolved: target publication venue (NeurIPS main track is unrealistic for this scope; NeurIPS Evaluations & Datasets track was identified as a realistic target), clinician access for validating ground truth/severity labels, and full API budget across all 4 model providers.
- **Claude API budget is the current binding constraint** on finishing Week 8 (~$6 available, full remaining run estimated ~$7-13). Not a data-loss risk given the resume logic, just a pacing one.

## Immediate next step
Finish Week 8 (`gating_network.py` + `ecd_tradeoff.py`) once budget allows → write up results → Week 9 (stats/significance testing, ablations) per the syllabus. See `docs/strategy-agent-briefing.md` for the full deep-dive version of this report, including the complete debugging/infra history.
