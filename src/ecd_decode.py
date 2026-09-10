"""Evidential Consistency Decoding: Context-Aware Decoding (Shi et al. 2023)
applied to temporal sycophancy.

Per token, two KV-cached forward passes (full context = original + adversarial
note; original context = original only), blended:

    log p_final = (1 + alpha) * log p_full - alpha * log p_original

alpha=0 == plain greedy decoding on the full context. Run with no args for
the CPU self-test (gpt2, checks the math, not medical reasoning).
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

        input_full = next_token  # KV cache carries the rest
        input_original = next_token

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

    prompt = "The capital of France is"
    ecd_alpha0 = generate_with_ecd(
        model, tokenizer, original_prompt=prompt, full_prompt=prompt, alpha=0.0, max_new_tokens=10
    )
    plain = generate_plain_greedy(model, tokenizer, prompt, max_new_tokens=10)
    assert ecd_alpha0 == plain, f"alpha=0 mismatch: {ecd_alpha0!r} vs {plain!r}"
    print(f"alpha=0 == plain greedy: {ecd_alpha0!r}")

    ecd_alpha1 = generate_with_ecd(
        model,
        tokenizer,
        original_prompt="The weather today is",
        full_prompt="The weather today is sunny and the sky is completely clear with",
        alpha=1.0,
        max_new_tokens=10,
    )
    print(f"alpha=1.0 divergent contexts: {ecd_alpha1!r}")
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
