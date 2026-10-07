"""LLM clients.

AnthropicClient: closed API, used for the drift benchmark and all grading
(the account has a verified zero-data-retention agreement, required before
sending MIMIC text to a cloud API).

LlamaMedClient: local GPU inference, used for ECD (needs raw logit access).
"""
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        ...


class AnthropicClient(LLMClient):
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ClaudeKey")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set -- check .env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        # run-config bookkeeping only; request parameters are unchanged
        self.meta = {"requested_model": model, "response_models": {}, "n_calls": 0,
                     "input_tokens": 0, "output_tokens": 0}

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        # thinking disabled: otherwise ~5% of calls spend the whole token
        # budget on an unrequested thinking block and return no text
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        self._record(resp)
        for block in resp.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(f"no text block in response: {resp.content}")

    def _record(self, resp) -> None:
        m = self.meta
        m["n_calls"] += 1
        m["response_models"][resp.model] = m["response_models"].get(resp.model, 0) + 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            m["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            m["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

    def dump_meta(self, path: str) -> None:
        """Append this run's call/model/token counts (no text) to a JSONL manifest."""
        import json
        import time

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**self.meta, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


class LlamaMedClient(LLMClient):
    """Local GPU inference. generate() matches AnthropicClient's interface;
    generate_ecd() is the ECD defense (see ecd_decode.py)."""

    name = "llama-med"

    def __init__(self, model_name: str = "aaditya/Llama3-OpenBioLLM-8B", device: str = "cuda", dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.model_name = model_name
        if dtype == "auto":
            # V100 (sm70) has no native bf16; emulation is slow and numerically different
            native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            dtype = "bfloat16" if native_bf16 else "float16"
        self.dtype = dtype
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, device_map=device)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device)
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        from ecd_decode import generate_plain_greedy

        return generate_plain_greedy(
            self.model, self.tokenizer, prompt, max_new_tokens=max_tokens, device=self.device
        )

    def generate_ecd(
        self,
        original_prompt: str,
        full_prompt: str,
        beta: float = 0.0,
        max_tokens: int = 300,
        plausibility_mask: bool = False,
        stop_at_first_line: bool = True,
    ) -> str:
        """Sign-corrected ECD: beta=0 undefended, beta=1 ignores the note."""
        from ecd_decode import generate_with_ecd

        return generate_with_ecd(
            self.model,
            self.tokenizer,
            original_prompt=original_prompt,
            full_prompt=full_prompt,
            beta=beta,
            max_new_tokens=max_tokens,
            device=self.device,
            plausibility_mask=plausibility_mask,
            stop_at_first_line=stop_at_first_line,
        )

    def generate_ecd_legacy(
        self, original_prompt: str, full_prompt: str, alpha: float = 1.0, max_tokens: int = 300
    ) -> str:
        """Sign-error implementation, kept only for ecd_tradeoff_legacy.py. DO NOT REPORT."""
        from ecd_decode_legacy import generate_with_ecd

        return generate_with_ecd(
            self.model,
            self.tokenizer,
            original_prompt=original_prompt,
            full_prompt=full_prompt,
            alpha=alpha,
            max_new_tokens=max_tokens,
            device=self.device,
        )


class OpenJudgeClient(LLMClient):
    """Open-weight instruct model used as a second judge (T4a). Chat template,
    greedy, answer primed with 'Verdict:' so the locked output format is followed."""

    name = "open-judge"

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "auto", prime: str = "Verdict:"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name, self.device, self.prime = model_name, device, prime
        if dtype == "auto":
            native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            dtype = "bfloat16" if native_bf16 else "float16"
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, device_map=device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device)
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        ) + self.prime
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=min(max_tokens, 100), do_sample=False, pad_token_id=pad)
        return self.prime + self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
