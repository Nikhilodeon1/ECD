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

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        # thinking disabled: otherwise ~5% of calls spend the whole token
        # budget on an unrequested thinking block and return no text
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(f"no text block in response: {resp.content}")


class LlamaMedClient(LLMClient):
    """Local GPU inference. generate() matches AnthropicClient's interface;
    generate_ecd() is the ECD defense (see ecd_decode.py)."""

    name = "llama-med"

    def __init__(self, model_name: str = "aaditya/Llama3-OpenBioLLM-8B", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        from ecd_decode import generate_plain_greedy

        return generate_plain_greedy(
            self.model, self.tokenizer, prompt, max_new_tokens=max_tokens, device=self.device
        )

    def generate_ecd(
        self, original_prompt: str, full_prompt: str, alpha: float = 1.0, max_tokens: int = 300
    ) -> str:
        from ecd_decode import generate_with_ecd

        return generate_with_ecd(
            self.model,
            self.tokenizer,
            original_prompt=original_prompt,
            full_prompt=full_prompt,
            alpha=alpha,
            max_new_tokens=max_tokens,
            device=self.device,
        )
