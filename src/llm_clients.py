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

    def generate_batch(
        self, prompts: list[str], max_tokens: int = 500, chunk_size: int = 20
    ) -> list[str]:
        """Default: just loops generate() -- real for API/local clients,
        there's no batching benefit there. ManualClaudeClient overrides
        this with actual paste-many-at-once batching. Scripts should call
        generate_batch() uniformly so they work the same either way."""
        return [self.generate(p, max_tokens=max_tokens) for p in prompts]


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
    """No API calls at all -- batches many prompts into one paste instead of
    one prompt per round trip, for when you've hit Anthropic's account
    quota but still have normal Claude.ai web access (which has its own,
    separate usage limits, not the API budget).

    generate() is a single-prompt convenience wrapper around
    generate_batch([prompt])[0] -- same interface as AnthropicClient, so it
    still works as a drop-in anywhere generate() is called directly.

    generate_batch() is the real interface for this mode: pass ALL the
    prompts for a pipeline stage at once, it chunks them (chunk_size --
    pick this based on how long each individual prompt is, there's no one
    right number):
      - short prompts (e.g. grading two diagnosis strings against each
        other, ~100-200 tokens each): chunk_size 100-300 is fine
      - long prompts (this codebase's diagnosis-generation prompts embed a
        full clinical note, ~1000+ tokens each after truncation):
        chunk_size 15-20 -- batching 500 of these would be 500k+ tokens,
        not a reasonable single paste
    For each chunk: writes ALL its prompts to one file in a numbered
    ===PROMPT N=== / ===ANSWER N=== format, you paste the whole thing into
    Claude.ai, copy the FULL reply back, paste it here. Parses answers back
    out by number so order is preserved even if Claude doesn't answer in
    strict order.

    max_tokens is accepted for interface compatibility but not enforced --
    no API knob for it here, mention it to Claude yourself if needed.
    """

    name = "claude-manual"

    def __init__(self, prompt_file: str = "../data/processed/_manual_batch_prompt.txt"):
        self.prompt_file = prompt_file

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return self.generate_batch([prompt], max_tokens=max_tokens, chunk_size=1)[0]

    def generate_batch(
        self, prompts: list[str], max_tokens: int = 500, chunk_size: int = 20
    ) -> list[str]:
        import re

        all_outputs: list[str] = []
        chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        for chunk_i, chunk in enumerate(chunks):
            header = (
                "Respond to each numbered prompt below, independently. For "
                "each one, output exactly:\n\n===ANSWER <N>===\n<your response "
                "to prompt N, nothing else>\n\nDo this for every prompt, in "
                "order, using the same number. Nothing before ===ANSWER 1=== "
                "and nothing after the last answer.\n\n"
            )
            body = "\n\n".join(f"===PROMPT {i + 1}===\n{p}" for i, p in enumerate(chunk))
            batch_prompt = header + body

            Path(self.prompt_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.prompt_file).write_text(batch_prompt, encoding="utf-8")

            print("\n" + "=" * 60)
            print(f"MANUAL BATCH MODE -- chunk {chunk_i + 1}/{len(chunks)} ({len(chunk)} prompts)")
            print(f"Prompt written to {self.prompt_file}")
            print("Paste it into Claude.ai, copy the FULL response, paste it below.")
            print("Type END on its own line when done pasting the response.")
            print("=" * 60)

            response = _read_multiline_input()

            pattern = re.compile(
                r"===ANSWER\s+(\d+)===\s*\n(.*?)(?=(?:===ANSWER\s+\d+===)|\Z)", re.DOTALL
            )
            by_number = {int(n): text.strip() for n, text in pattern.findall(response)}

            missing = [i for i in range(1, len(chunk) + 1) if i not in by_number]
            if missing:
                print(
                    f"WARNING: missing/unparsed answers for prompt numbers {missing} "
                    f"in this chunk (got {len(by_number)}/{len(chunk)}) -- these will "
                    f"be blank, likely to fail downstream parsing/grading"
                )

            all_outputs.extend(by_number.get(i, "") for i in range(1, len(chunk) + 1))

        return all_outputs


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
