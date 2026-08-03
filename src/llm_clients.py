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

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            # the actual bug: without this, the model sometimes spends the
            # whole max_tokens budget on an unrequested thinking block and
            # never gets to emit any text at all (~4.7% of calls in a real
            # run) -- not just a block-ordering issue, there was no text
            # block present at all. Disabling thinking removes the failure
            # mode outright; the higher default max_tokens is a cheap
            # safety net on top (billed on actual tokens used, not the cap).
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(f"no text block in response: {resp.content}")


class LlamaMedClient(LLMClient):
    """Local GPU inference -- runs on the pod, never makes a network call, so
    there's no PhysioNet cloud-API question for this one (unlike Anthropic/
    GPT-5/Gemini, which needed the DUA/retention check before touching MIMIC
    text).

    Default model is a placeholder -- "Llama-4-Med" from the proposal isn't
    an actual released checkpoint, this needs a real HF model id. Swap via
    the `model_name` arg once you've picked/confirmed one (aaditya/Llama3-
    OpenBioLLM-8B is a reasonable real option: ~8B params, openly available,
    fits a single GPU). NOT YET RUN against a real model -- only
    ecd_decode.py's own algorithm has been verified (against tiny gpt2 on
    CPU, see its self-test). Loading and generation here is written against
    the documented transformers API but unverified end-to-end.

    generate() gives plain single-context output, matching the same
    interface as AnthropicClient so this model can run through the existing
    eval_baseline.py / eval_drift.py harnesses unchanged. generate_ecd()
    is the actual Week 7 defense -- see ecd_decode.py for the algorithm.
    """

    name = "llama-med"

    def __init__(
        self,
        model_name: str = "aaditya/Llama3-OpenBioLLM-8B",
        device: str = "cuda",
    ):
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
