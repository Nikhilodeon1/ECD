"""Evidential Consistency Decoding (ECD), sign-corrected.

Per token, two KV-cached forward passes: the full context (original evidence
plus the follow-up note) and the original context (original evidence only).
Their log-probabilities are blended as

    log p_final = (1 - beta) * log p_full + beta * log p_orig
                = log p_full + beta * (log p_orig - log p_full),   beta >= 0

beta = 0: plain greedy decoding on the full context (undefended).
beta = 1: greedy decoding on the original context (the note is ignored; the
          ceiling on recovery by construction, not a result).
beta > 1: extrapolation away from the note-conditioned distribution.

This is context-aware decoding (Shi et al. 2023) with the amplified context
set to the original evidence, in the regime beta = 1 + alpha_CAD. The earlier
implementation (ecd_decode_legacy.py) used the opposite sign: its alpha is
beta = -alpha here. See docs/erratum.md.

Run with no args for the CPU self-test (gpt2; checks the math and endpoints,
not medical reasoning). Offline unit tests: tests/test_ecd.py.
"""
import math

import torch


def blend_logprobs(
    full_logprobs: torch.Tensor,
    original_logprobs: torch.Tensor,
    beta: float,
    plausibility_mask: bool = False,
    mask_ratio: float = 0.1,
) -> torch.Tensor:
    """Blended per-token scores. The (1-beta)/beta form makes beta=0 and beta=1
    exactly the full and original log-probs.

    plausibility_mask (ablation, default off): for beta > 0, tokens with
    p_orig < mask_ratio * max p_orig get -inf, so extrapolation cannot select
    tokens the original evidence considers implausible.
    """
    if beta == 0:
        return full_logprobs
    if beta == 1 and not plausibility_mask:
        return original_logprobs
    scores = (1.0 - beta) * full_logprobs + beta * original_logprobs
    if plausibility_mask:
        floor = original_logprobs.max(dim=-1, keepdim=True).values + math.log(mask_ratio)
        scores = torch.where(original_logprobs >= floor, scores, torch.full_like(scores, float("-inf")))
    return scores


def _eos_ids(model, tokenizer) -> set[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    gen_cfg = getattr(model, "generation_config", None)
    extra = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(extra, int):
        ids.add(extra)
    elif extra:
        ids.update(int(i) for i in extra)
    return ids


@torch.no_grad()
def ecd_generate_ids(
    model,
    full_ids: torch.Tensor,
    original_ids: torch.Tensor,
    beta: float,
    max_new_tokens: int,
    eos_ids: set[int],
    plausibility_mask: bool = False,
    stop_after: callable = None,
) -> list[int]:
    """Greedy decode on blended scores. full_ids / original_ids: [1, L] on the
    model's device. stop_after(generated_ids) -> True ends generation early
    (used to stop once the first answer line is complete)."""
    model.eval()
    full_past = original_past = None
    input_full, input_original = full_ids, original_ids
    generated: list[int] = []
    # beta == 0 never needs the original context; skip its forward passes
    use_original = beta != 0

    for _ in range(max_new_tokens):
        out_full = model(input_full, past_key_values=full_past, use_cache=True)
        full_past = out_full.past_key_values
        full_logprobs = torch.log_softmax(out_full.logits[:, -1, :].float(), dim=-1)

        if use_original:
            out_original = model(input_original, past_key_values=original_past, use_cache=True)
            original_past = out_original.past_key_values
            original_logprobs = torch.log_softmax(out_original.logits[:, -1, :].float(), dim=-1)
            scores = blend_logprobs(full_logprobs, original_logprobs, beta, plausibility_mask)
        else:
            scores = full_logprobs

        next_token = torch.argmax(scores, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        if stop_after is not None and stop_after(generated):
            break

        input_full = next_token  # KV cache carries the rest
        input_original = next_token

    return generated


def _first_line_done(tokenizer):
    """True once the decoded text has a non-empty line followed by a newline."""

    def check(ids: list[int]) -> bool:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        stripped = text.lstrip()
        return "\n" in stripped

    return check


@torch.no_grad()
def generate_with_ecd(
    model,
    tokenizer,
    original_prompt: str,
    full_prompt: str,
    beta: float = 0.0,
    max_new_tokens: int = 200,
    device: str = "cpu",
    plausibility_mask: bool = False,
    stop_at_first_line: bool = False,
) -> str:
    """stop_at_first_line: end once the first answer line is complete. The
    experiment parsers read only the first non-empty line, so the parse is
    unchanged; leave False for byte-identity checks against model.generate."""
    full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    original_ids = tokenizer(original_prompt, return_tensors="pt").input_ids.to(device)
    generated = ecd_generate_ids(
        model,
        full_ids,
        original_ids,
        beta,
        max_new_tokens,
        _eos_ids(model, tokenizer),
        plausibility_mask=plausibility_mask,
        stop_after=_first_line_done(tokenizer) if stop_at_first_line else None,
    )
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_plain_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 200, device: str = "cpu") -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def _self_test():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    original = "The weather today is"
    full = "The weather today is sunny and the sky is completely clear with"
    n = 12

    plain_full = generate_plain_greedy(model, tokenizer, full, max_new_tokens=n)
    plain_orig = generate_plain_greedy(model, tokenizer, original, max_new_tokens=n)

    b0 = generate_with_ecd(model, tokenizer, original, full, beta=0.0, max_new_tokens=n)
    assert b0 == plain_full, f"beta=0 != greedy(full): {b0!r} vs {plain_full!r}"
    print(f"beta=0 == greedy(full):     {b0!r}")

    b1 = generate_with_ecd(model, tokenizer, original, full, beta=1.0, max_new_tokens=n)
    assert b1 == plain_orig, f"beta=1 != greedy(original): {b1!r} vs {plain_orig!r}"
    print(f"beta=1 == greedy(original): {b1!r}")

    for beta in (0.5, 1.5):
        out = generate_with_ecd(model, tokenizer, original, full, beta=beta, max_new_tokens=n)
        print(f"beta={beta}: {out!r}")
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
