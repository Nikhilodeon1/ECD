"""Prompt templates and response parsing.

Two prompt families: the Claude prompts use bracket placeholders; the Llama
prompts use a worked example and end at "Diagnosis:" for completion, since
the smaller model echoes bracket placeholders verbatim instead of filling
them in.
"""

_MAX_WORDS = 800  # truncate outlier-long MIMIC notes (up to ~4k words)


def _truncate(evidence: str, max_words: int) -> str:
    words = evidence.split()
    return " ".join(words[:max_words]) if len(words) > max_words else evidence


BASELINE_PROMPT_TEMPLATE = """You are shown a case's presenting evidence. Based only on this evidence, state the single most likely diagnosis.

Case evidence:
{evidence}

Respond in exactly this format:
Diagnosis: <your answer>
Reasoning: <one sentence>
"""

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

LLAMA_BASELINE_PROMPT_TEMPLATE = """You are a doctor. Given a case's presenting evidence, state the single most likely diagnosis in a few words.

Example:
Case evidence: 45F with fever, cough, and dyspnea for 3 days. CXR shows right lower lobe consolidation.
Diagnosis: Bacterial pneumonia

Now do the same for this case.
Case evidence:
{evidence}

Diagnosis:"""

LLAMA_FOLLOWUP_PROMPT_TEMPLATE = """You are a doctor. Given a case's presenting evidence plus a follow-up note added later, state the single most likely diagnosis in a few words, considering all the information.

Example:
Case evidence: 45F with fever, cough, and dyspnea for 3 days. CXR shows right lower lobe consolidation.
Follow-up note: Repeat CXR shows no change.
Diagnosis: Bacterial pneumonia

Now do the same for this case.
Case evidence:
{evidence}

Follow-up note:
{adversarial_note}

Diagnosis:"""


def build_baseline_prompt(evidence: str, max_words: int = _MAX_WORDS) -> str:
    return BASELINE_PROMPT_TEMPLATE.format(evidence=_truncate(evidence, max_words))


def build_followup_prompt(evidence: str, adversarial_note: str, max_words: int = _MAX_WORDS) -> str:
    return FOLLOWUP_PROMPT_TEMPLATE.format(
        evidence=_truncate(evidence, max_words), adversarial_note=adversarial_note
    )


def build_llama_baseline_prompt(evidence: str, max_words: int = _MAX_WORDS) -> str:
    return LLAMA_BASELINE_PROMPT_TEMPLATE.format(evidence=_truncate(evidence, max_words))


def build_llama_followup_prompt(evidence: str, adversarial_note: str, max_words: int = _MAX_WORDS) -> str:
    return LLAMA_FOLLOWUP_PROMPT_TEMPLATE.format(
        evidence=_truncate(evidence, max_words), adversarial_note=adversarial_note
    )


def parse_diagnosis(response: str) -> str:
    """First 'Diagnosis:' line, else the whole response."""
    for line in response.splitlines():
        if line.strip().lower().startswith("diagnosis:"):
            return line.split(":", 1)[1].strip()
    return response.strip()


def parse_llama_diagnosis(response: str) -> str:
    """First non-empty line (Llama prompt ends at 'Diagnosis:', so the
    completion is the diagnosis)."""
    for line in response.splitlines():
        if line.strip():
            return line.strip()
    return response.strip()
