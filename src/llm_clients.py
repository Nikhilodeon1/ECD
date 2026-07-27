"""Pluggable LLM client interface.

Only Anthropic is wired up right now -- that's the only model with a
confirmed API budget and a verified Zero Data Retention agreement (enterprise
Anthropic account), which matters because this is the only client that's
allowed to see real MIMIC note text per PhysioNet's cloud-API guidance.

GPT-5 / Gemini stay as stubs until separate budgets + their own retention
policies are confirmed -- don't wire real MIMIC data through them without
doing the same DUA check done for Anthropic.

Llama-Med runs locally on the pod (transformers/vllm), not through this
client interface at all -- no cloud call, no retention question, but needs
pod GPU/weights details before it can be implemented for real.
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

        # accepts ClaudeKey too for now -- rename your .env var to
        # ANTHROPIC_API_KEY when convenient, this fallback is just so it
        # works today without you having to edit .env mid-eval
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ClaudeKey")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY (or ClaudeKey) not set -- check .env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # content[0] isn't always the text block -- a ThinkingBlock can come
        # first, so scan for the actual text block instead of assuming position
        for block in resp.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(f"no text block in response: {resp.content}")


class LlamaMedClient(LLMClient):
    """Not implemented yet -- needs pod GPU + exact model weights to build
    for real (transformers/vllm local inference, no API call)."""

    name = "llama-med"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LlamaMedClient needs pod-side model weights/GPU setup before this can run"
        )

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        raise NotImplementedError
