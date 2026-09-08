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


# --- Llama-Med-specific prompts (Week 8 ecd_tradeoff.py / ecd_demo.py only) ---
#
# The BASELINE_PROMPT_TEMPLATE / FOLLOWUP_PROMPT_TEMPLATE above use bracket
# placeholders ("<your answer>") that Claude correctly treats as fill-in-the-
# blank instructions. Real bug found inspecting Week 8 tradeoff results at
# scale: Llama3-OpenBioLLM-8B doesn't reliably understand that convention and
# instead echoes the literal placeholder text back ("...is <your answer>."),
# or regurgitates prompt content verbatim -- not a sign the model can't
# diagnose, a sign the prompt format doesn't suit this weaker instruction-
# follower. Fixed here with a worked example instead of an abstract
# placeholder, and by ending the prompt at "Diagnosis:" so the model just
# continues naturally rather than having to reproduce a label+placeholder
# pattern itself. Claude's prompts above are untouched -- they're already
# validated (Week 5/6 results depend on them), don't touch that pattern.

LLAMA_BASELINE_PROMPT_TEMPLATE = """You are a doctor. Given a case's presenting evidence, state the single most likely diagnosis in a few words.

Example:
Case evidence: 45F with fever, cough, and dyspnea for 3 days. CXR shows right lower lobe consolidation.
Diagnosis: Bacterial pneumonia

Now do the same for this case.
Case evidence:
{evidence}

Diagnosis:"""


def build_llama_baseline_prompt(evidence: str, max_words: int = 800) -> str:
    words = evidence.split()
    if len(words) > max_words:
        evidence = " ".join(words[:max_words])
    return LLAMA_BASELINE_PROMPT_TEMPLATE.format(evidence=evidence)


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


def build_llama_followup_prompt(evidence: str, adversarial_note: str, max_words: int = 800) -> str:
    words = evidence.split()
    if len(words) > max_words:
        evidence = " ".join(words[:max_words])
    return LLAMA_FOLLOWUP_PROMPT_TEMPLATE.format(evidence=evidence, adversarial_note=adversarial_note)


def parse_llama_diagnosis(response: str) -> str:
    """The Llama prompts above end at 'Diagnosis:', so the model's completion
    IS the diagnosis directly -- no 'Diagnosis:' prefix to search for in the
    output itself. Just take the first non-empty line, trimmed."""
    for line in response.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return response.strip()
