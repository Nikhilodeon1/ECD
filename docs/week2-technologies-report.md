# Week 2: Technologies & CS Behind ECD

## The core idea, in plain terms
- An LLM picks its next word based on a probability distribution over words, given whatever context it's seen.
- "Diagnosis drift" = that distribution shifts enough after the adversarial note that the model's answer changes.
- Need a consistent way to pull "the diagnosis" out of free text (e.g. force output format like `Diagnosis: X`) so you can actually compare before/after.

## Why models do this (sycophancy)
- Comes from RLHF/preference training — reward models like agreeable, confident-sounding responses.
- Model ends up over-weighting the *newest* input instead of the *best* evidence.

## ECD's actual mechanism (heads up: this already exists)
- ECD = run two forward passes per token (original note only vs. full context), then blend the two distributions.
- This is basically **Context-Aware Decoding (CAD)**, a known technique (Shi et al.), plus close cousins DoLa and DeCoRe.
- Not a knock — just means: cite it honestly, position ECD as "CAD applied to temporal sycophancy," not a brand-new algorithm.
- Formula, if useful: `p_final(y) ∝ p_full(y) * [p_full(y)/p_original(y)]^α` — α controls how hard you push away from the "ignore new context" distribution.

## The catch that affects your whole plan
- This trick needs raw token-level probabilities at every decoding step, which means you need direct access to model weights.
- **GPT-5, Claude, Gemini APIs don't give you this** — no full logprobs, no custom decoding loop.
- So ECD can only really be built end-to-end on **Llama-4-Med** (open weights, via `transformers`/`vllm`).
- For the closed models, you'll need a workaround — generate full completions under each context, then rank/vote instead of blending token-by-token. Decide this now, not in week 7.

## The two prompting baselines (no infra needed, work on any model)
- **Self-consistency**: sample the answer multiple times, take the majority vote.
- **Evidence-anchored prompting**: explicitly remind the model of the original evidence and make it justify any change.

## Stats you'll actually need
- **DDR** = % of cases where diagnosis changes after the adversarial note.
- **Severity-weighted DDR** = same, but you build a harm-weighting rubric first (a wrong-but-similar diagnosis matters less than a dangerous miss).
- Since each case gets tested twice (before/after, or with/without defense) → use **McNemar's test**, not a plain two-sample test.
- Report effect sizes (odds ratio or Cohen's h) with bootstrapped confidence intervals.
- Accuracy-vs-drift tradeoff = just sweep α and plot accuracy vs. DDR.

## Tools
- Local model + logits: `transformers`, `vllm`
- Closed APIs: official `openai`, `anthropic`, `google-genai` SDKs
- Stats: `scipy.stats`, `statsmodels`
- Data: `pandas`
- Skip LangChain — plain Python scripts are enough for this pipeline.

## Datasets, quick notes
- **MIMIC-IV-Note**: free-text discharge/radiology notes, linkable to structured MIMIC-IV tables (labs, diagnoses) if you want real lab values for the "perturbed labs" note type.
- **MedQA**: clean vignettes with a labeled correct answer — useful since MIMIC notes have no ground-truth diagnosis label, you'd have to build that yourself.
