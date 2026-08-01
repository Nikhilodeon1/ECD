"""Prompt building + response parsing for the baseline (no adversarial note) eval."""

BASELINE_PROMPT_TEMPLATE = """You are shown a case's presenting evidence. Based only on this evidence, state the single most likely diagnosis.

Case evidence:
{evidence}

Respond in exactly this format:
Diagnosis: <your answer>
Reasoning: <one sentence>
"""


def build_baseline_prompt(evidence: str, max_words: int = 800) -> str:
    """Truncates long MIMIC notes so a handful of outlier long notes (up to
    ~4,265 words seen in the data) don't blow up per-call cost."""
    words = evidence.split()
    if len(words) > max_words:
        evidence = " ".join(words[:max_words])
    return BASELINE_PROMPT_TEMPLATE.format(evidence=evidence)


def parse_diagnosis(response: str) -> str:
    for line in response.splitlines():
        if line.strip().lower().startswith("diagnosis:"):
            return line.split(":", 1)[1].strip()
    return response.strip()  # fallback: model didn't follow the format


FOLLOWUP_PROMPT_TEMPLATE = """You are shown a case's presenting evidence, plus a follow-up note added later.

Case evidence:
{evidence}

Follow-up note:
{adversarial_note}

Based on all the information above, state the single most likely diagnosis.

Respond in exactly this format:
Diagnosis: <your answer>
Reasoning: <one sentence>
"""


def build_followup_prompt(evidence: str, adversarial_note: str, max_words: int = 800) -> str:
    words = evidence.split()
    if len(words) > max_words:
        evidence = " ".join(words[:max_words])
    return FOLLOWUP_PROMPT_TEMPLATE.format(evidence=evidence, adversarial_note=adversarial_note)
