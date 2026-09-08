# ECD Project — Full Briefing for Strategy Agent

This is a complete handoff. Read it fully before advising — later sections depend on earlier ones (compliance rules in particular constrain what's allowed in every subsequent step).

## 1. What this project is

**Project name**: ECD (Evidential Consistency Decoding). AIEA summer internship project, mentored by Leilani.

**Research question**: does an LLM abandon an *already-correct* diagnosis when it receives a misleading follow-up note, even though nothing in the original evidence changed? Called this "temporal sycophancy" — the model caving to whatever arrived most recently instead of weighing evidence.

**Why it matters**: LLMs are starting to appear in clinical decision-support. Real EHRs accumulate notes over time from multiple sources (nurses, specialists, labs) — some of that later information can be wrong, vague, or misleading, and a model that flips its diagnosis under that pressure is a real safety risk.

## 2. Novelty assessment (established early, still holds)

- **The real contribution**: a taxonomy of adversarial follow-up note types + a benchmark measuring diagnosis drift under them, grounded in real clinical notes (MIMIC-IV) not just synthetic vignettes. This combination — general LLM sycophancy research + longitudinal clinical reasoning research — hadn't been done together.
- **What is NOT novel**: the ECD decoding mechanism (dual-context contrastive decoding) is architecturally identical to **Context-Aware Decoding** (Shi et al.). Frame ECD as "CAD applied to temporal diagnostic sycophancy," not a new algorithm — this is already reflected correctly in the codebase's docstrings and Week 7 writeup.
- **Realistic venue**: NeurIPS main track is a stretch for this scope. NeurIPS Evaluations & Datasets track (formerly Datasets & Benchmarks) is a realistic target — it's specifically built for benchmark contributions like this one. Never formally decided/locked in; worth revisiting once more results exist.
- **Unresolved from early scoping** (never fully closed out): clinician access for validating severity/ground-truth labels, full API budget across all 4 originally-planned model providers. These affect how large the final benchmark needs to be.

## 3. Compliance — the load-bearing constraint on everything

MIMIC-IV is PhysioNet-credentialed data. Two compliance questions got resolved carefully and must stay respected in all future work:

1. **MIMIC data never leaves a controlled environment.** Raw notes/derived note text live only on the RunPod network volume (or wherever else is explicitly cleared). `.gitignore` blocks `data/` and raw-data file extensions as a backstop, but the real rule is architectural: the git repo only ever contains code, docs, and aggregate statistics (counts, distributions) — never note text.
2. **Cloud APIs need verified data handling before touching MIMIC text.** PhysioNet's own guidance says: don't use a cloud API on restricted data unless zero data retention is verified — "if a service's data handling practices are unclear or cannot be fully verified, do not use the service." This was checked for Anthropic specifically: the user's Anthropic account is an **enterprise account (via UC Davis) with a genuine Zero Data Retention agreement** (not just a self-serve training-opt-out toggle) — confirmed, not assumed. **GPT-5 and Gemini have NOT been cleared this way** — do not send MIMIC text through them without doing the same verification first.
3. A related but distinct question came up later: could GPU inference run on Nautilus (a shared multi-tenant national research cluster, free GPUs) instead of paying for RunPod GPU time? Resolution: **yes, but only for MedQA-sourced cases** (fully public). MIMIC-sourced cases must stay on the cleared RunPod environment — moving the derived `eval_sample.jsonl` (which embeds real extracted MIMIC evidence text) to Nautilus would put that text on infrastructure that was never vetted the way RunPod was. `ecd_tradeoff.py` and `ecd_demo.py` both have a `--source medqa`/`--source mimic-iv-note` filter for exactly this split, plus a warning in the code that pre-filtering the sample file (not just filtering at read time) matters — the raw file containing MIMIC text must not physically land on Nautilus even if the script only reads the medqa subset from it.

## 4. Infrastructure

- **Repo**: private GitHub, `Nikhilodeon1/ECD`. Local dev machine + RunPod GPU pods (ephemeral — see §8 for a real gotcha here).
- **Compute**: RunPod pods, GPU tier matters — settled on A100/H100-class (80GB), explicitly avoided RTX 4000 Ada (20GB, too tight for an 8B model's dual KV caches under ECD).
- **Models in use**: Claude Sonnet 5 via Anthropic API, **extended thinking explicitly disabled** (`thinking={"type": "disabled"}` — this was fixing a real bug, see §8) — locked as the fixed model config for the whole project, never to be changed mid-project since that would break comparability across weeks. Llama3-OpenBioLLM-8B (`aaditya/Llama3-OpenBioLLM-8B`) is the open-weight model for local GPU inference — "Llama-4-Med" from the original proposal isn't a real released checkpoint, this was substituted in.

## 5. Codebase map (`src/`)

| File | Purpose |
|---|---|
| `schema.py` | Shared `BaseCase` dataclass — id, source, case_summary, original_note, diagnosis_ground_truth, metadata |
| `medqa_loader.py` | Pulls MedQA from HF, filters to diagnosis-style questions only (236/1273 test questions qualify — MedQA is mostly non-diagnosis question types) |
| `mimic_loader.py` | Parses MIMIC-IV-Note discharge summaries, extracts presenting-evidence sections (HPI/PMH/Physical Exam/Pertinent Results — explicitly excludes the Discharge Diagnosis section so the answer doesn't leak into the prompt), gets ground truth from primary ICD diagnosis. Streams the CSV in chunks (fixed an OOM crash on a lower-RAM pod) |
| `build_dataset.py`, `stats.py` | Orchestration + aggregate dataset stats |
| `subsample.py` | Reproducible stratified eval sample — even split across sources, capped per-diagnosis so common ICU diagnoses (sepsis, pneumonia) don't dominate. Same sample (seed 42) reused across weeks for comparability. Has `--source-n` override |
| `llm_clients.py` | `AnthropicClient` (API) and `LlamaMedClient` (local GPU, has both `generate()` and `generate_ecd()`). A `ManualClaudeClient` batch-paste mode was built and then **fully removed** — see §8 |
| `prompts.py`, `grade.py` | Prompt templates + LLM-as-judge grading (needed because MIMIC ground truth is verbose ICD text a model won't say verbatim) |
| `eval_baseline.py` | Week 5: baseline diagnosis accuracy, no adversarial note. Writes each result immediately + auto-resumes from a partial run |
| `adversarial_notes.py`, `eval_drift.py` | Week 6: generates a category-specific adversarial note per case (Claude, guided by the taxonomy), re-diagnoses, checks drift via LLM-judge. Also incremental-write + auto-resume |
| `ecd_decode.py` | Week 7: the actual ECD algorithm — Context-Aware Decoding formula `log p_final = (1+α)·log p_full − α·log p_original`, efficient KV-cached token-by-token loop |
| `ecd_demo.py` | Qualitative demo: plain vs. ECD-defended output on real Week 6 drift cases |
| `gating_network.py` | Week 8: TF-IDF + logistic regression trained on Week 6's 960 labeled trials, predicts drift risk from a note's text+category |
| `ecd_tradeoff.py` | Week 8: sweeps α across drifted cases, measures recovery rate via Claude-judge, plus a one-time clean-case accuracy measurement (see §7 for the math reasoning behind why that's only measured once) |

## 6. Taxonomy (Week 3) — the 5 adversarial note categories

Each maps to a distinct hypothesized failure mechanism, not arbitrary templates:
1. **VCS (Vague Contradictory Statement)** — pure tone/confidence, zero new evidence. Tests bare sycophancy.
2. **ISO (Incorrect Specialist Opinion)** — wrong claim, but from an authoritative-sounding source. Tests authority bias.
3. **PLV (Perturbed Lab Values)** — one data point changed/emphasized, rest of evidence unchanged. Tests anchoring on recency.
4. **MMS (Misleading Medication Suggestion)** — a treatment order implying a different diagnosis. Tests inference-from-action.
5. **SNHR (Subtle Nurse Handoff Redirection)** — informal, low-authority, family-relayed reframing. Tests susceptibility to narrative from non-clinical sources.

Severity rubric: 1 (no consequence) to 5 (could be fatal). 20 hand-written draft vignettes exist (`docs/week3-taxonomy-and-vignettes.md`) as illustrations, separate from the real eval sample.

## 7. Results so far (real numbers, with caveats)

**Week 4 — dataset**: 331,840 total cases built (331,604 MIMIC-IV-Note + 236 MedQA). <1% of MIMIC cases dropped for unparseable evidence sections.

**Week 5 — baseline accuracy** (most recent clean API run, n=150/150): **Claude 78.0% MedQA, 54.7% MIMIC**. Consistent with an earlier run (78.7%/51.3%) — small run-to-run variance is expected/normal for a stochastic model. The gap itself (MIMIC harder than MedQA) is a real, useful finding: curated USMLE vignettes are easier than messy real clinical notes.

**Week 6 — drift** (earlier full run, computed against the 78.7%/51.3% baseline draw): **Overall Diagnosis Drift Rate 27.0%** (n=960, zero failed trials). By category: SNHR 30.2%, PLV 29.7%, MMS 26.0%, VCS 25.0%, ISO 24.0% (fairly flat, not a strong signal yet at this n). By source: **MIMIC 44.0% vs. MedQA 16.1%** — the headline finding: real clinical notes drift ~3x more than clean vignettes, consistent with Week 5's accuracy gap. **Caveat: this run predates the manual-mode removal and predates the current baseline rerun — a fresh `eval_drift.py` run is in progress (currently paused, see §9) to get numbers computed against the current, consistent pipeline. Don't treat 27.0% as final.**

Quality spot-check (5 trials, manual review): notes read as plausible and in-character for their category. One real grading ambiguity found: a case where the model *added* an unconfirmed detail rather than cleanly changing its diagnosis — judge called it "no drift," a defensible but genuinely fuzzy call. Documented as a real source of noise in the DDR numbers.

**Week 7 — ECD implementation**: algorithm verified two ways — (a) CPU self-test proves α=0 produces byte-identical output to plain greedy decoding, the one exact mathematical invariant available; (b) real H100 run against Llama3-OpenBioLLM-8B recovered a correct pre-drift diagnosis ("chronic granulomatous disease") at α=1 from a plain baseline that was completely wrong. Two real, documented limitations: α=2 degrades into incoherent output (expected CAD failure mode, motivates Week 8's tuned interpolation), and this particular open model doesn't reliably follow the structured `Diagnosis: X` output format the way Claude does (parsing sometimes falls back to raw text).

**Week 8 — in progress, not yet run for real.** `gating_network.py` and `ecd_tradeoff.py` are built and smoke-tested (synthetic data / fake clients) but have not produced real results yet.

## 8. Things tried, found broken, and fixed — worth knowing so they aren't repeated

- **OOM on `mimic_loader.py`**: loading the full ~330k-row discharge notes file into one DataFrame got OOM-killed on a lower-RAM pod. Fixed by streaming via `pd.read_csv(..., chunksize=...)`.
- **Claude API silent-failure bug**: ~4.7% of calls returned a response with *only* a `ThinkingBlock` and no text at all — the model spent its whole `max_tokens` budget on an unrequested thinking step and never got to the answer. Fixed by explicitly passing `thinking={"type": "disabled"}` plus raising the default `max_tokens` ceiling (free — Anthropic bills actual tokens used, not the cap).
- **Manual Claude.ai copy-paste mode — built, then fully removed.** Context: user hit Anthropic API quota mid-project. Built a `ManualClaudeClient` that batched many prompts into one paste (numbered `===PROMPT N===`/`===ANSWER N===` format) to reduce round trips. **It worked mechanically** (verified batching/parsing/ordering), but when actually used, **results differed meaningfully from the isolated-API-call baseline** (68.0%/55.3% vs. the established 78.7%/51.3% — an ~11-point swing on MedQA, bigger than normal run-to-run noise). Conclusion: batching many cases into one shared context is not methodologically equivalent to independent calls — real cross-prompt interference, not just stochastic variance. **Removed entirely** (deleted `ManualClaudeClient`, `get_claude_client()` toggle, `generate_batch()`). Reverted to pure `AnthropicClient()` everywhere. This is worth remembering if budget pressure comes up again — the fix is more budget or fewer calls, not manual-paste batching.
- **Incremental-write + auto-resume added everywhere real money gets spent** (`eval_baseline.py`, `eval_drift.py`, `ecd_tradeoff.py`): each result is written to disk and flushed immediately, and every rerun of the same command skips whatever's already done (keyed by case_id, or (case_id, category) for drift trials, or (case_id, alpha) for the tradeoff sweep). Verified with a simulated "funds run out mid-run" test. This exists specifically because funds/pod interruptions have happened repeatedly and used to mean re-paying for completed work.
- **Ephemeral pod storage has bitten this project multiple times.** `data/processed/` and `.env` are both gitignored (correctly — they're either derived-from-restricted-data or secrets) and live only on whichever pod they were generated on. Every fresh pod / fresh clone loses them, requiring a full pipeline rebuild from `build_dataset.py` onward. This has happened at least 3 times. Not yet solved with a persistent-storage fix — worth raising if it keeps happening.
- **Batch-size/context tuning**: when manual mode still existed, chunk sizes were tuned way up (long prompts embedding full notes: 20→40/chunk; short grading prompts: 150→400/chunk) before it was determined batching itself was the problem, not the size. Irrelevant now that manual mode is gone, but explains some git history churn if reviewed.

## 9. Current status, right now

- Week 5 baseline: done, clean, current (78.0%/54.7%).
- Week 6 drift eval: **mid-rerun**, currently **paused** (user stepped away, safely Ctrl+C'd — RunPod pod may still be running and billing, was told to actually stop the pod not just the script). Resume with `python3.12 eval_drift.py` once funds are available; it picks up automatically.
- Week 7: complete, documented, not yet committed to GitHub (user handles commits themselves).
- Week 8: code complete, not yet run for real.

## 10. Blockers

1. **Claude API budget** — the binding constraint. User has ~$6; a full clean run (remaining `eval_drift.py` + `gating_network.py` (free) + `ecd_tradeoff.py`) was estimated at ~$7-13 total for everything from scratch. User is asking parents for more funds. Not a data-loss risk (resume logic covers that), just a pacing one.
2. **GPT-5/Gemini cross-model comparison** — blocked on (a) separate API budgets, (b) same DUA/retention verification Anthropic went through, not yet done for either provider. Currently the whole project only has evidence for one model (Claude).
3. **Mentor feedback, partially addressed**: (a) "how often does drift happen" — answered, 27% (pending refresh). (b) "narrow scope to one model + one defense" — raised, never formally decided. Worth resolving before Week 9 writing locks in scope.
4. **Self-grading bias** — Claude grades its own diagnosis outputs, its own generated adversarial notes, and its own drift verdicts throughout. No independent validation beyond a single 5-example manual spot-check in Week 6. A real, acknowledged limitation, not yet mitigated.

## 11. Immediate next steps (in order)

1. Resume/finish `eval_drift.py` once funds allow → get final, current Week 6 numbers.
2. Run `gating_network.py` (free) and `ecd_tradeoff.py` (GPU + Claude judge calls) for Week 8's real accuracy-drift tradeoff curve.
3. Write up `docs/week8-tradeoff-results.md` from real output.
4. Week 9 (per syllabus): stats/significance testing (McNemar's test is the right tool for the paired before/after drift comparisons — flagged back in Week 2, not yet implemented) + ablations, feeding into a Methods+Results paper draft.
5. Resolve the "narrow scope" question from the mentor before Week 9 locks in what the paper actually claims.

## 12. Where full detail lives

`docs/status-report.md` (living project status doc), `docs/week{2,3,4,5,6,7}-*.md` (per-week deliverables with full methodology/results), `README.md` (quick pipeline usage). This briefing is a synthesis of all of them plus context that isn't written down anywhere else (the manual-mode saga, the compliance reasoning, the exact blockers) — the per-week docs are the source of truth for exact numbers.
