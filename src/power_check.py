"""G-5: verify the McNemar sample-size formula used in Appendix D.

Formula (Connor 1987, Biometrics 43:207-211; also Miettinen 1968):
    n = [ z_{a/2} sqrt(psi) + z_b sqrt(psi - delta^2) ]^2 / delta^2
with psi = p10 + p01 (discordant proportion) and delta = p10 - p01.
It is a normal approximation; here it is checked against the EXACT two-sided
McNemar (binomial) test by simulation in three scenarios. Pure math, no data.
Output -> results/power_check.json.
"""
import json
import math
import time

import numpy as np
from scipy.stats import binom, norm

from paths import RESULTS


def connor_n(p10: float, p01: float, alpha: float = 0.05, power: float = 0.8) -> float:
    psi, delta = p10 + p01, abs(p10 - p01)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta**2)) ** 2 / delta**2


def exact_power(n: int, p10: float, p01: float, alpha: float = 0.05, sims: int = 40000, seed: int = 1) -> float:
    """Share of simulated paired samples where the exact two-sided McNemar test rejects."""
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [p10, p01, 1 - p10 - p01], size=sims)
    b, c = draws[:, 0], draws[:, 1]
    m = b + c
    k = np.minimum(b, c)
    p = np.where(m > 0, np.minimum(1.0, 2 * binom.cdf(k, np.maximum(m, 1), 0.5)), 1.0)
    return float((p < alpha).mean())


def smallest_n_exact(p10, p01, target=0.8, alpha=0.05, sims=20000) -> int:
    lo, hi = 5, 5000
    while lo < hi:
        mid = (lo + hi) // 2
        if exact_power(mid, p10, p01, alpha, sims) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    scenarios = {
        "S1 one-directional, delta=0.077 (category best case)": (0.077, 0.0),
        "S2 one-directional, delta=0.056 (small-effect best case)": (0.056, 0.0),
        "S3 bidirectional, delta=0.077, psi=0.263": (0.170, 0.093),
    }
    out = {"script": "src/power_check.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "reference": "Connor RJ (1987) Sample size for testing differences in proportions for the paired-sample design. Biometrics 43:207-211",
           "alpha": 0.05, "target_power": 0.8, "scenarios": {}}
    for name, (p10, p01) in scenarios.items():
        n = connor_n(p10, p01)
        n_up = math.ceil(n)
        out["scenarios"][name] = {
            "p10": p10, "p01": p01, "connor_n": round(n, 1),
            "exact_power_at_connor_n": round(exact_power(n_up, p10, p01), 3),
            "smallest_n_for_80pct_exact_power": smallest_n_exact(p10, p01),
        }
        print(name, out["scenarios"][name])
    worst = min(v["exact_power_at_connor_n"] for v in out["scenarios"].values())
    out["verdict"] = ("Connor's formula reaches >= 0.75 exact power in every scenario"
                      if worst >= 0.75 else
                      f"formula is optimistic: exact power at Connor's n falls to {worst}; use the exact-search n")
    (RESULTS / "power_check.json").parent.mkdir(exist_ok=True)
    (RESULTS / "power_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out["verdict"])


if __name__ == "__main__":
    main()
