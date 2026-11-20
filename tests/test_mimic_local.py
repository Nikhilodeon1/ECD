"""Offline tests for the laptop MIMIC runner (no network, no GPU, no real MIMIC text)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mimic_local as ml  # noqa: E402
import test_v3 as t3  # noqa: E402
import v3_run as v3  # noqa: E402
from paths import read_jsonl  # noqa: E402


def args(**kw):
    return argparse.Namespace(**{"limit": None, "batch_size": 4, "budget_usd": 5.0, "with_hedged": False, "api_only": False, "generator": ml.GENERATOR,
                                 "verifier": ml.VERIFIER, "judge": ml.JUDGE, "cats": ml.CATS, "notes_chunk": 8, **kw})


def test_plan_order_and_flags():
    cmds = ml.plan(args())
    assert [c[0] for c in cmds] == ml.STAGES
    text = {c[0]: " ".join(c[1]) for c in cmds}
    assert "--no-helpful" in text["notes"] and "--cats ISO,VCS,SNHR" in text["notes"] and "--chunk 8" in text["notes"]
    assert "claude:claude-sonnet-5" in text["base"] and "--budget-usd 5.0" in text["base"]
    assert "A_unattributed" in text["arms"] and "--judge claude:claude-sonnet-5" in text["arms"]
    assert "A_hedged" not in text["arms"]  # exploratory arms are opt-in to save budget
    assert "A_hedged" in " ".join(dict(ml.plan(args(with_hedged=True)))["arms"])
    # cross-family design: generator, verifier and the tested model are three different families
    assert "Qwen" in ml.GENERATOR and "Phi" in ml.VERIFIER and "claude" in ml.MODEL
    assert "--limit 5" in " ".join(ml.plan(args(limit=5))[1][1])


def test_build_sample_keeps_only_mimic(tmp_path):
    rows = [{"id": "MIMIC-1", "source": "mimic-iv-note"}, {"id": "MEDQA-1", "source": "medqa"}]
    src = tmp_path / "eval.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert ml.build_sample(src, tmp_path / "out" / "m.jsonl") == 1
    assert [r["id"] for r in read_jsonl(tmp_path / "out" / "m.jsonl", guard=False)] == ["MIMIC-1"]


def test_refuses_on_colab(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    try:
        ml.refuse_if_not_local()
    except SystemExit as e:
        assert "MIMIC stays on the laptop" in str(e)
    else:
        raise AssertionError("should refuse on Colab")


def test_run_notes_restricted_categories(tmp_path):
    p = tmp_path / "notes.jsonl"
    v3.run_notes(t3.CASES[:6], t3.FakeGen(), p, "fake", chunk=2, cats=("ISO", "VCS"), helpful=False)
    kinds = {r["kind"] for r in read_jsonl(p, guard=False)}
    assert kinds == {"X", "ISO", "VCS"}
    n = len(read_jsonl(p, guard=False))
    v3.run_notes(t3.CASES[:6], t3.FakeGen(), p, "fake", chunk=2, cats=("ISO", "VCS"), helpful=False)
    assert len(read_jsonl(p, guard=False)) == n  # resumed: nothing regenerated


def test_pod_overrides_use_open_models_and_open_judge():
    a = args(generator="hf:Qwen/Qwen2.5-7B-Instruct", verifier="open:microsoft/phi-4",
             judge="open:microsoft/phi-4", batch_size=32, notes_chunk=64)
    text = {c[0]: " ".join(c[1]) for c in ml.plan(a)}
    assert "Qwen2.5-7B-Instruct" in text["notes"] and "--chunk 64" in text["notes"] and "--batch-size 32" in text["notes"]
    assert "--judge open:microsoft/phi-4" in text["judge_base"] and "claude:claude-sonnet-5" in text["base"]
    assert "judged__claude-claude-sonnet-5__open-microsoft-phi-4.jsonl" in text["summary"]


def test_api_only_mode_has_no_local_models_and_fewer_arms():
    a = args(api_only=True, generator=ml.API_GENERATOR, verifier=ml.API_VERIFIER, judge=ml.API_JUDGE, cats=ml.API_CATS)
    text = {c[0]: " ".join(c[1]) for c in ml.plan(a)}
    assert "hf:" not in " ".join(text.values()) and "open:" not in " ".join(text.values())
    assert "--arms R,N,ISO,A_unattributed,B_attending,C_nursing " in text["arms"] + " "
    assert "--cats ISO" in text["notes"]
