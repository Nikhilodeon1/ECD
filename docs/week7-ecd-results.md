# Week 7: ECD Implementation

## What was built
- `src/ecd_decode.py` — the actual ECD algorithm: Context-Aware Decoding (Shi et al.) applied to temporal sycophancy. At each generated token, blends log-probabilities from two contexts (original note only vs. full context with adversarial note): `log p_final = (1+α)·log p_full − α·log p_original`. Uses KV-caching on both passes so it's an efficient token-by-token loop, not a full re-run each step.
- `src/llm_clients.py` — `LlamaMedClient` now does real local inference via `transformers` (model: `aaditya/Llama3-OpenBioLLM-8B`, an open 8B medical-tuned model — "Llama-4-Med" from the original proposal isn't an actual released checkpoint). `generate()` matches the same interface as `AnthropicClient` so this model can run through the existing Week 5/6 eval harnesses unchanged; `generate_ecd()` is the defense itself.
- `src/ecd_demo.py` — runs plain vs. ECD-defended generation on real cases that drifted under Claude in Week 6, reusing those already-generated adversarial notes rather than paying to regenerate them.

## Verification
- **CPU self-test (`ecd_decode.py`, run locally)**: confirms α=0 produces byte-identical output to plain greedy decoding — the one exact mathematical invariant that can be asserted, not just eyeballed. Passed. Also confirmed the two-context KV-cache plumbing runs cleanly with divergent contexts at α=1.
- **Real GPU run (H100, `aaditya/Llama3-OpenBioLLM-8B`)**: ran on 5 real cases that drifted under Claude in Week 6, at α=1 and α=2.

## Results from the real run

**The mechanism works.** Case `MEDQA-test-00922` (PLV category — a perturbed lab value note): plain generation (α=0) gave a completely wrong, off-topic answer ("Atopic dermatitis"). At α=1, ECD correctly recovered **"chronic granulomatous disease"** — the actual pre-drift correct diagnosis, explicitly citing the NBT test result as supporting evidence. This is the algorithm doing exactly what it's designed to do: pulling the output back toward what the original evidence supports.

**Two real limitations surfaced, not code bugs:**
1. **α=2.0 degenerates into incoherent output** in multiple cases (garbled fragments, repeated/truncated text). This is a documented failure mode of contrastive decoding pushed too far — amplifying the divergence too aggressively pushes token selection into low-probability, incoherent territory. This is exactly the problem Week 8's tuned interpolation strength (learned or swept α) is meant to solve — Week 7 wasn't supposed to find the "right" α, just prove the mechanism works.
2. **This 8B open model is much less reliable than Claude at following the structured `Diagnosis: X` output format.** Several responses were raw reasoning text with no parseable diagnosis line, so `parse_diagnosis()` fell back to returning the whole response. Bumped `max_tokens` from 100→200 after the run since some outputs were visibly truncated mid-sentence, but the format-following gap is a real model-capability difference, not something more tokens fixes on its own.

## Before trusting this fully
- Only 5 cases tested manually so far — this is a qualitative proof-of-concept run, not a quantitative eval. A real accuracy-drift tradeoff measurement (Week 8's deliverable) needs this run at scale across the full sample, with more robust parsing/grading than simple string matching — likely needs the same LLM-as-judge approach used in `grade.py`, extended to grade Llama-Med's outputs too, since exact-format parsing is clearly not reliable enough for this model.
- Only one open model tested (`aaditya/Llama3-OpenBioLLM-8B`) — the format-following issue might be specific to this checkpoint; worth keeping in mind if this becomes a bottleneck later.
- The demo intentionally used α=1 and α=2 as round-number spot-checks, not a tuned sweep — Week 8 is where the actual interpolation-strength tuning happens.
