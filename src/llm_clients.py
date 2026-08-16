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
from pathlib import Path

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


def _read_multiline_input() -> str:
    """Reads pasted lines until a line that's exactly END, or EOF."""
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


class ManualClaudeClient(LLMClient):
    """No API calls at all -- for when you've hit Anthropic's account quota
    but still have normal Claude.ai access. Writes each prompt to a file
    (safer than printing to terminal for long MIMIC-note prompts, which can
    wrap/truncate in some terminals), you paste it into Claude.ai by hand,
    then paste the response back into this terminal.

    Same generate() interface as AnthropicClient, so any script that uses
    it doesn't need to change -- see get_claude_client() below for the
    toggle. No temperature/other API params to preserve: AnthropicClient
    never set any beyond model/max_tokens, so this is behaviorally
    equivalent on that front already.

    max_tokens is accepted for interface compatibility but not enforced --
    there's no API knob for it here. Mention it to Claude yourself in the
    web UI if a response is running long.

    Only realistic for smaller call volumes -- eval_drift.py alone is
    ~2,880 calls, not something to paste by hand. Fine for ecd_tradeoff.py
    (~120 calls by default, still a lot -- consider smaller --n-drifted/
    --n-clean in manual mode) or small spot-check reruns.
    """

    name = "claude-manual"

    def __init__(self, prompt_file: str = "../data/processed/_manual_prompt.txt"):
        self.prompt_file = prompt_file

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        Path(self.prompt_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.prompt_file).write_text(prompt, encoding="utf-8")

        print("\n" + "=" * 60)
        print(f"MANUAL MODE -- prompt written to {self.prompt_file}")
        print("Paste it into Claude.ai, copy the full response, paste it below.")
        print("Type END on its own line when done pasting the response.")
        print("=" * 60)

        return _read_multiline_input()


def get_claude_client(model: str = "claude-sonnet-5") -> LLMClient:
    """Toggle between real API calls and manual copy-paste mode via the
    CLAUDE_MODE env var (in .env or exported in the shell) -- defaults to
    'api'. Set CLAUDE_MODE=manual when you've hit an API quota/budget
    limit. Use this everywhere instead of instantiating AnthropicClient
    directly, so scripts don't need code changes to switch modes."""
    mode = os.environ.get("CLAUDE_MODE", "api").lower()
    if mode == "manual":
        return ManualClaudeClient()
    return AnthropicClient(model=model)


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
