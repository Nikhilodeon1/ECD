"""Evidential Consistency Decoding (ECD) -- the core algorithm from the proposal.

This is Context-Aware Decoding (Shi et al.) applied to temporal sycophancy --
see docs/week2-technologies-report.md for why this is CAD and not a new
algorithm, and why it can only run against a local open-weight model (needs
raw per-token logit access, which no closed API exposes).

At each generated token, runs two forward passes:
  - "full" context: original note + adversarial follow-up note
  - "original" context: original note only
and blends their log-probabilities:

    log p_final = (1 + alpha) * log p_full - alpha * log p_original

alpha=0 reduces to plain greedy decoding on the full context (verified by
the self-test below). Higher alpha pushes the output away from whatever the
adversarial note alone would cause and back toward what the original
evidence alone supports.

Uses KV-caching on both passes (each step only feeds the single new token +
past_key_values) rather than re-running the full sequence every step, since
that would be O(n^2) and unusably slow even on small models.

REQUIRES A GPU for any real medical model -- this file's own self-test runs
on CPU against tiny gpt2 purely to verify the blending math and cache
plumbing are correct, not to validate anything about medical reasoning.
"""
import torch


@torch.no_grad()
def generate_with_ecd(
    model,
    tokenizer,
    original_prompt: str,
    full_prompt: str,
    alpha: float = 1.0,
    max_new_tokens: int = 200,
    device: str = "cpu",
) -> str:
    model.eval()
    full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    original_ids = tokenizer(original_prompt, return_tensors="pt").input_ids.to(device)

    full_past = None
    original_past = None
    input_full = full_ids
    input_original = original_ids
    generated = []

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        out_full = model(input_full, past_key_values=full_past, use_cache=True)
        out_original = model(input_original, past_key_values=original_past, use_cache=True)
        full_past = out_full.past_key_values
        original_past = out_original.past_key_values

        full_logprobs = torch.log_softmax(out_full.logits[:, -1, :], dim=-1)
        original_logprobs = torch.log_softmax(out_original.logits[:, -1, :], dim=-1)

        ecd_logprobs = (1 + alpha) * full_logprobs - alpha * original_logprobs
        next_token = torch.argmax(ecd_logprobs, dim=-1, keepdim=True)

        if eos_id is not None and next_token.item() == eos_id:
            break
        generated.append(next_token.item())

        # next step only needs the new token -- KV cache carries the rest
        input_full = next_token
        input_original = next_token

    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_plain_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 200, device: str = "cpu") -> str:
    """Plain greedy decoding on a single context -- used only to verify
    alpha=0 reduces to this exactly."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def _self_test():
    """CPU-runnable check that the ECD math/caching is correct. Uses tiny
    gpt2, not a medical model -- this validates the mechanism, not any
    medical reasoning. Run with: python ecd_decode.py"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading gpt2 for self-test (small, CPU, no gating)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    prompt = "The capital of France is"

    # alpha=0 must reduce to plain greedy decoding on the full context --
    # the one thing we can assert exactly, not just "looks reasonable"
    ecd_alpha0 = generate_with_ecd(
        model, tokenizer, original_prompt=prompt, full_prompt=prompt, alpha=0.0, max_new_tokens=10
    )
    plain = generate_plain_greedy(model, tokenizer, prompt, max_new_tokens=10)
    assert ecd_alpha0 == plain, f"alpha=0 mismatch:\n  ecd:   {ecd_alpha0!r}\n  plain: {plain!r}"
    print(f"alpha=0 matches plain greedy decoding exactly: {ecd_alpha0!r}")

    # sanity check that a nonzero alpha with different original/full
    # contexts actually runs without shape errors and produces *something*
    original = "The weather today is"
    full = "The weather today is sunny and the sky is completely clear with"
    ecd_alpha1 = generate_with_ecd(
        model, tokenizer, original_prompt=original, full_prompt=full, alpha=1.0, max_new_tokens=10
    )
    print(f"alpha=1.0 with divergent contexts ran cleanly: {ecd_alpha1!r}")

    print("self-test passed")


if __name__ == "__main__":
    _self_test()
