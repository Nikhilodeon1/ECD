"""Offline unit tests for the sign-corrected ECD (no downloads, CPU).

Run:  python -m pytest tests/test_ecd.py -q     (from the repo root)
  or: python tests/test_ecd.py
"""
import math
import sys
from pathlib import Path

import torch

# Local-env guard: a broken torchvision install makes `transformers` fail to import
# GPT-2. Nothing here needs torchvision, so hide it if it cannot load.
try:
    import torchvision  # noqa: F401
except Exception:
    for _k in [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]:
        del sys.modules[_k]
    sys.modules["torchvision"] = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ecd_decode as new  # noqa: E402
import ecd_decode_legacy as legacy  # noqa: E402

# Toy vocabulary: token 0 = diagnosis the NOTE pushes (A), token 1 = diagnosis the
# ORIGINAL evidence supports (B), token 2 = other (C).
P_FULL = torch.tensor([[0.60, 0.30, 0.10]])  # note-conditioned: favors A
P_ORIG = torch.tensor([[0.10, 0.80, 0.10]])  # original-only: favors B
LF, LO = P_FULL.log(), P_ORIG.log()


def legacy_blend(alpha):
    return (1 + alpha) * LF - alpha * LO


def new_blend(beta):
    return new.blend_logprobs(LF, LO, beta)


def probs(scores):
    return torch.softmax(scores, dim=-1)[0]


def kl(p, q):
    return float((p * (p / q).log()).sum())


def test_old_formula_moves_toward_note():
    """Old formula: P(original diagnosis B) FALLS and P(note diagnosis A) RISES with alpha."""
    pb = [float(probs(legacy_blend(a))[1]) for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    pa = [float(probs(legacy_blend(a))[0]) for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert all(x > y for x, y in zip(pb, pb[1:])), pb
    assert all(x < y for x, y in zip(pa, pa[1:])), pa
    d = [kl(probs(legacy_blend(a)), P_ORIG[0]) for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert all(x < y for x, y in zip(d, d[1:])), f"KL to original should grow: {d}"


def test_new_formula_moves_toward_original():
    """New formula: P(B) RISES, P(A) FALLS, KL to original-only SHRINKS as beta grows."""
    betas = (0.0, 0.25, 0.5, 0.75, 1.0)
    pb = [float(probs(new_blend(b))[1]) for b in betas]
    pa = [float(probs(new_blend(b))[0]) for b in betas]
    assert all(x < y for x, y in zip(pb, pb[1:])), pb
    assert all(x > y for x, y in zip(pa, pa[1:])), pa
    d = [kl(probs(new_blend(b)), P_ORIG[0]) for b in betas]
    assert all(x > y for x, y in zip(d, d[1:])), f"KL to original should shrink: {d}"


def test_endpoints_are_exact():
    assert torch.equal(new_blend(0.0), LF)
    assert torch.equal(new_blend(1.0), LO)


def test_old_alpha_is_new_beta_minus_alpha():
    for a in (0.0, 0.5, 1.0, 1.5, 2.0):
        assert torch.allclose(probs(legacy_blend(a)), probs(new_blend(-a)), atol=1e-6), a
        # unnormalized scores agree too
        assert torch.allclose(legacy_blend(a), new_blend(-a), atol=1e-5), a


def test_beta_above_one_extrapolates_past_original():
    p1, p2 = probs(new_blend(1.0)), probs(new_blend(2.0))
    assert p2[1] > p1[1] and p2[0] < p1[0]


def test_plausibility_mask_blocks_implausible_tokens():
    # token 0 has p_orig=0.10 = 0.125 * max(0.80): kept at ratio 0.1, dropped at 0.2
    keep = new.blend_logprobs(LF, LO, 2.0, plausibility_mask=True, mask_ratio=0.1)
    drop = new.blend_logprobs(LF, LO, 2.0, plausibility_mask=True, mask_ratio=0.2)
    assert torch.isfinite(keep).all()
    assert torch.isinf(drop[0, 0]) and drop[0, 0] < 0 and torch.isfinite(drop[0, 1])
    assert torch.argmax(drop, dim=-1).item() == 1


# ---- end-to-end with a scripted fake LM -------------------------------------


class Out:
    def __init__(self, logits, past):
        self.logits, self.past_key_values = logits, past


class FakeLM:
    """Context kind is read from the first call: a NOTE token (id 9) in the prompt
    means the full context. Step-0 logits follow P_FULL / P_ORIG; later steps emit EOS."""

    NOTE, EOS = 9, 3

    def eval(self):
        return self

    def __call__(self, ids, past_key_values=None, use_cache=True):
        if past_key_values is None:
            kind, step = ("full" if (ids == self.NOTE).any() else "orig"), 0
        else:
            kind, step = past_key_values
        if step == 0:
            probs0 = P_FULL if kind == "full" else P_ORIG
            row = torch.cat([probs0.log()[0], torch.full((7,), -30.0)])
        else:
            row = torch.full((10,), -30.0)
            row[self.EOS] = 0.0
        return Out(row.view(1, 1, -1), (kind, step + 1))


def test_end_to_end_direction_fake_lm():
    lm = FakeLM()
    full = torch.tensor([[5, 6, 9]])
    orig = torch.tensor([[5, 6]])
    first = {}
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        ids = new.ecd_generate_ids(lm, full, orig, beta, 3, {FakeLM.EOS})
        first[beta] = ids[0]
    assert first[0.0] == 0, first  # undefended -> note's diagnosis A
    assert first[1.0] == 1, first  # ignores the note -> original diagnosis B
    assert first[1.5] == 1, first
    flips = [first[b] for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert flips == sorted(flips), f"should switch A->B once, monotone: {flips}"
    # legacy with positive alpha never recovers B
    for alpha in (0.5, 1.0, 2.0):
        ids = new.ecd_generate_ids(lm, full, orig, -alpha, 3, {FakeLM.EOS})
        assert ids[0] == 0, (alpha, ids)


# ---- byte-identity on a tiny random GPT-2 ------------------------------------


class StubTok:
    """'12 7 9' -> [12, 7, 9]; decode -> same style string."""

    eos_token_id = None

    class _B:
        def __init__(self, t):
            self.input_ids = t

    def __call__(self, text, return_tensors="pt"):
        return StubTok._B(torch.tensor([[int(x) for x in text.split()]]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=120, n_positions=64, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=None, eos_token_id=None, pad_token_id=0)
    return GPT2LMHeadModel(cfg).eval()


def _greedy_ids(model, text, n):
    ids = StubTok()(text).input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0, eos_token_id=None)
    return out[0, ids.shape[1]:].tolist()


def test_tiny_gpt2_endpoints_byte_identical():
    model, tok = _tiny_model(), StubTok()
    orig, full, n = "5 17 33 41 8", "5 17 33 41 8 77 90 12 60", 12
    b0 = new.generate_with_ecd(model, tok, orig, full, beta=0.0, max_new_tokens=n)
    b1 = new.generate_with_ecd(model, tok, orig, full, beta=1.0, max_new_tokens=n)
    assert b0 == " ".join(map(str, _greedy_ids(model, full, n)))
    assert b1 == " ".join(map(str, _greedy_ids(model, orig, n)))
    assert b0 != b1, "test contexts must actually disagree"


def test_tiny_gpt2_legacy_alpha_equals_new_negative_beta():
    model, tok = _tiny_model(), StubTok()
    orig, full, n = "5 17 33 41 8", "5 17 33 41 8 77 90 12 60", 12
    for alpha in (0.5, 1.0, 2.0):
        old = legacy.generate_with_ecd(model, tok, orig, full, alpha=alpha, max_new_tokens=n)
        neu = new.generate_with_ecd(model, tok, orig, full, beta=-alpha, max_new_tokens=n)
        assert old == neu, (alpha, old, neu)
    assert legacy.generate_with_ecd(model, tok, orig, full, alpha=0.0, max_new_tokens=n) == new.generate_with_ecd(
        model, tok, orig, full, beta=0.0, max_new_tokens=n
    )


def test_stop_at_first_line_matches_first_line_of_full_run():
    class Tok(StubTok):
        # token 5 renders as newline so a "first line" exists
        def decode(self, ids, skip_special_tokens=True):
            return "".join("\n" if int(i) % 11 == 5 else "a" for i in ids)

    model, tok = _tiny_model(), Tok()
    orig, full = "5 17 33 41 8", "5 17 33 41 8 77 90 12 60"
    long = new.generate_with_ecd(model, tok, orig, full, beta=0.5, max_new_tokens=40)
    short = new.generate_with_ecd(model, tok, orig, full, beta=0.5, max_new_tokens=40, stop_at_first_line=True)
    first = lambda t: next((l for l in t.splitlines() if l.strip()), "")
    assert first(long) == first(short)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tests passed")
