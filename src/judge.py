"""Judge layer shared by every experiment: verdict cache + string shortcut.

A verdict is keyed by (judge_id, rubric_id, task, case_id, normalized_a,
normalized_b). Identical normalized strings are equivalent without a judge
call. Verdicts append to a JSONL cache on disk immediately, so any run resumes
for free. Cache files contain model diagnoses on MIMIC-derived cases: keep them
under data/ (gitignored) on the pod.

Rubrics:
  binary_v1  the locked prompts from grade.py (used for every headline number)
  tiered_v1  equivalent / narrower / broader / different (T4b; strict vs lenient)
"""
import json
import re
import time
import unicodedata
from pathlib import Path

from grade import build_drift_check_prompt, build_grade_prompt, parse_verdict

TIERS = ("equivalent", "narrower", "broader", "different")

TIERED_PROMPT_TEMPLATE = """You are comparing two diagnoses.

Diagnosis A (reference): {a}
Diagnosis B: {b}

Pick exactly one label:
EQUIVALENT: A and B name the same clinical condition (wording or minor specificity may differ).
NARROWER: B names a specific subtype or component of what A names.
BROADER: B names a more general category that includes what A names.
DIFFERENT: B names a different condition.

Respond in exactly this format:
Verdict: EQUIVALENT or NARROWER or BROADER or DIFFERENT
Reason: <one sentence>
"""


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"^\s*diagnosis\s*:\s*", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_tiered(response: str) -> str:
    for line in response.splitlines():
        s = line.strip().lower()
        if s.startswith("verdict:"):
            body = s.split(":", 1)[1]
            for tier in TIERS:
                if tier in body:
                    return tier
    return "unparsed"


class JudgeCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: dict[tuple, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.store[tuple(r["key"])] = r["value"]

    def get(self, key: tuple):
        return self.store.get(key)

    def put(self, key: tuple, value: dict) -> None:
        self.store[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": list(key), "value": value}) + "\n")
            f.flush()


class Judge:
    """backend: any object with generate(prompt, max_tokens) -> str."""

    def __init__(self, backend, judge_id: str, cache_path: str, rubric: str = "binary_v1"):
        assert rubric in ("binary_v1", "tiered_v1")
        self.backend, self.judge_id, self.rubric = backend, judge_id, rubric
        self.cache = JudgeCache(cache_path)
        self.calls = {"string": 0, "cache": 0, "api": 0}

    def _prompt(self, kind: str, a: str, b: str) -> str:
        if self.rubric == "tiered_v1":
            return TIERED_PROMPT_TEMPLATE.format(a=a, b=b)
        # locked prompts: grade = (ground truth, predicted); same = (before, after)
        return build_grade_prompt(a, b) if kind == "grade" else build_drift_check_prompt(a, b)

    def compare(self, task: str, case_id: str, a: str, b: str, kind: str = "grade") -> dict:
        """kind='grade': a = ground truth, b = prediction. kind='same': a = earlier
        answer, b = later answer. Returns {label, match, source}; match is True
        for 'equivalent' only (lenient variants are derived from label)."""
        na, nb = normalize(a), normalize(b)
        if na and na == nb:
            self.calls["string"] += 1
            return {"label": "equivalent", "match": True, "source": "string"}

        key = (self.judge_id, self.rubric, kind, task, case_id, na, nb)
        hit = self.cache.get(key)
        if hit is not None:
            self.calls["cache"] += 1
            return {**hit, "source": "cache"}

        raw = self.backend.generate(self._prompt(kind, a, b), max_tokens=250)
        if self.rubric == "tiered_v1":
            label = parse_tiered(raw)
        else:
            label = "equivalent" if parse_verdict(raw) else "different"
        value = {"label": label, "match": label == "equivalent", "raw": raw,
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.cache.put(key, value)
        self.calls["api"] += 1
        return {"label": label, "match": value["match"], "source": "api"}
