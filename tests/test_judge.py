"""Offline tests for the judge cache and string shortcut."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge import Judge, normalize, parse_tiered  # noqa: E402


class Fake:
    def __init__(self, reply):
        self.reply, self.n = reply, 0

    def generate(self, prompt, max_tokens=250):
        self.n += 1
        return self.reply


def test_normalize():
    assert normalize("Diagnosis:  Bacterial Pneumonia. ") == "bacterial pneumonia"
    assert normalize("COPD (chronic)") == "copd chronic"


def test_string_shortcut_makes_no_call():
    b = Fake("Verdict: NO_MATCH\nReason: x")
    with tempfile.TemporaryDirectory() as d:
        j = Judge(b, "fake", f"{d}/c.jsonl")
        r = j.compare("baseline", "c1", "Sepsis", "diagnosis: sepsis.")
        assert r == {"label": "equivalent", "match": True, "source": "string"}
        assert b.n == 0


def test_cache_hit_and_resume_from_disk():
    b = Fake("Verdict: MATCH\nReason: same")
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/c.jsonl"
        j = Judge(b, "fake", path)
        assert j.compare("baseline", "c1", "MI", "heart attack")["source"] == "api"
        assert j.compare("baseline", "c1", "MI", "heart attack")["source"] == "cache"
        assert b.n == 1
        j2 = Judge(b, "fake", path)  # new process, same file
        assert j2.compare("baseline", "c1", "MI", "heart attack")["source"] == "cache"
        assert b.n == 1
        # different case_id or judge_id must not collide
        assert j2.compare("baseline", "c2", "MI", "heart attack")["source"] == "api"
        assert Judge(b, "other", path).compare("baseline", "c1", "MI", "heart attack")["source"] == "api"


def test_tiered_labels():
    assert parse_tiered("Verdict: NARROWER\nReason: y") == "narrower"
    assert parse_tiered("Verdict: BROADER") == "broader"
    assert parse_tiered("gibberish") == "unparsed"
    with tempfile.TemporaryDirectory() as d:
        j = Judge(Fake("Verdict: BROADER\nReason: z"), "fake", f"{d}/t.jsonl", rubric="tiered_v1")
        r = j.compare("baseline", "c1", "Lobar pneumonia", "Pneumonia")
        assert r["label"] == "broader" and r["match"] is False
