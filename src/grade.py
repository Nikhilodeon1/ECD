"""LLM-as-judge grading (semantic match, since ground truth is verbose ICD text).

Currently self-graded (Claude grading Claude); see paper limitations.
"""

GRADE_PROMPT_TEMPLATE = """You are grading a diagnosis for accuracy against a ground-truth label.

Ground truth diagnosis: {ground_truth}
Model's stated diagnosis: {predicted}

Does the model's diagnosis refer to the same underlying clinical condition as the ground truth, even if worded differently or at a different level of specificity? Answer in exactly this format:
Verdict: MATCH or NO_MATCH
Reason: <one sentence>
"""


def build_grade_prompt(ground_truth: str, predicted: str) -> str:
    return GRADE_PROMPT_TEMPLATE.format(ground_truth=ground_truth, predicted=predicted)


def parse_verdict(grade_response: str) -> bool:
    for line in grade_response.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("verdict:"):
            return "no_match" not in stripped and "match" in stripped
    return False  # unparseable verdict -> treat as no match


DRIFT_CHECK_PROMPT_TEMPLATE = """Are these two diagnoses referring to the same underlying clinical condition, even if worded differently?

Diagnosis A: {before}
Diagnosis B: {after}

Respond in exactly this format:
Verdict: MATCH or NO_MATCH
Reason: <one sentence>
"""


def build_drift_check_prompt(before: str, after: str) -> str:
    return DRIFT_CHECK_PROMPT_TEMPLATE.format(before=before, after=after)
