"""
significance_tests.py — McNemar's test + bootstrap CIs for the
curated-vs-baseline differential lift on the 200-task / 40-skill holdout.

WHY
    The v1.9 / v2.0 reports state Δ = pass(curated) − pass(baseline) as
    +0.075 and +0.045 respectively. The paper draft hedges with "within
    one standard error of each other" but does not run a real test. This
    script does:
      1. Per-model McNemar's test (paired binary outcome) → is Δ
         significantly different from 0 in each model?
      2. Per-model bootstrap 95% CI on Δ (resample task_uids with
         replacement) → reports point estimate and uncertainty.
      3. Cross-model comparison: is v2.0's Δ significantly smaller
         than v1.9's? Bootstrap test on the difference of Δs.

USAGE
    python training/significance_tests.py
    # uses default eval dirs; pass --episodes <path> for a custom run

OUTPUT
    Prints a compact summary suitable for paste into the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


# --- McNemar ---------------------------------------------------------------

def mcnemar_paired(eps: List[dict]) -> dict:
    """Pair each task's BL and CU verdicts; run McNemar's test on the
    discordant pairs. Returns dict of stats."""
    by_task: dict[str, dict[str, bool]] = defaultdict(dict)
    for e in eps:
        by_task[e["task_uid"]][e["condition"]] = bool(e["passed"])

    # 2x2 contingency
    a = b = c = d = 0  # a=both pass, b=BL pass CU fail, c=BL fail CU pass, d=both fail
    for uid, conds in by_task.items():
        if "baseline" not in conds or "curated" not in conds:
            continue
        bl = conds["baseline"]; cu = conds["curated"]
        if bl and cu: a += 1
        elif bl and not cu: b += 1
        elif not bl and cu: c += 1
        else: d += 1

    n_pairs = a + b + c + d
    discordant = b + c

    # Continuity-corrected McNemar's chi-squared:
    #   chi^2 = (|b - c| - 1)^2 / (b + c)
    # Two-sided p-value via chi-squared(df=1) survival function.
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    elif discordant < 25:
        # exact binomial: P(X >= max(b,c) | n=b+c, p=0.5), two-sided
        k = max(b, c)
        n = discordant
        # two-sided p = 2 * sum_{i=k..n} C(n,i) * 0.5^n
        log_half_n = -n * math.log(2)
        log_p_one_sided = -math.inf
        for i in range(k, n + 1):
            log_binom = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1))
            log_term = log_binom + log_half_n
            log_p_one_sided = log_sum_exp(log_p_one_sided, log_term)
        p_value = min(1.0, 2.0 * math.exp(log_p_one_sided))
        chi2 = float("nan")
    else:
        chi2 = ((abs(b - c) - 1) ** 2) / discordant
        # chi-squared(df=1) survival = erfc(sqrt(chi2 / 2))
        p_value = math.erfc(math.sqrt(chi2 / 2))

    return {
        "n_pairs": n_pairs,
        "a_both_pass": a,
        "b_bl_only": b,   # BL passed, CU failed (curated worse)
        "c_cu_only": c,   # BL failed, CU passed (curated better)
        "d_both_fail": d,
        "delta": (a + c) / n_pairs - (a + b) / n_pairs if n_pairs else 0.0,
        "chi2": chi2,
        "p_value": p_value,
        "discordant_used_exact": discordant < 25,
    }


def log_sum_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# --- Bootstrap CI ----------------------------------------------------------

def bootstrap_delta(eps: List[dict], n_resamples: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for Δ = pass_rate(CU) - pass_rate(BL), resampling
    task_uids (preserves the per-task pairing). Returns (point, lo, hi)."""
    by_task: dict[str, dict[str, int]] = defaultdict(dict)
    for e in eps:
        by_task[e["task_uid"]][e["condition"]] = 1 if e["passed"] else 0

    # only keep tasks with both conditions
    paired = [(t["baseline"], t["curated"]) for t in by_task.values()
              if "baseline" in t and "curated" in t]
    n = len(paired)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        cu_sum = sum(s[1] for s in sample)
        bl_sum = sum(s[0] for s in sample)
        deltas.append((cu_sum - bl_sum) / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[int(0.975 * n_resamples)]
    point = sum(s[1] - s[0] for s in paired) / n
    return point, lo, hi


def bootstrap_delta_diff(eps_a: List[dict], eps_b: List[dict],
                         n_resamples: int = 10000, seed: int = 42) -> dict:
    """Cross-model: is Δ_a > Δ_b? Bootstrap CI on (Δ_a - Δ_b), resampling
    task_uids INDEPENDENTLY for each model (correct under the assumption
    that the models were evaluated on the same tasks but the eval runs
    are independent)."""
    rng = random.Random(seed)

    by_task_a: dict[str, dict[str, int]] = defaultdict(dict)
    by_task_b: dict[str, dict[str, int]] = defaultdict(dict)
    for e in eps_a:
        by_task_a[e["task_uid"]][e["condition"]] = 1 if e["passed"] else 0
    for e in eps_b:
        by_task_b[e["task_uid"]][e["condition"]] = 1 if e["passed"] else 0

    paired_a = [(t["baseline"], t["curated"]) for t in by_task_a.values() if len(t) == 2]
    paired_b = [(t["baseline"], t["curated"]) for t in by_task_b.values() if len(t) == 2]
    n_a, n_b = len(paired_a), len(paired_b)

    diffs = []
    for _ in range(n_resamples):
        sa = [paired_a[rng.randrange(n_a)] for _ in range(n_a)]
        sb = [paired_b[rng.randrange(n_b)] for _ in range(n_b)]
        d_a = sum(s[1] - s[0] for s in sa) / n_a
        d_b = sum(s[1] - s[0] for s in sb) / n_b
        diffs.append(d_a - d_b)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    point_a = sum(s[1] - s[0] for s in paired_a) / n_a
    point_b = sum(s[1] - s[0] for s in paired_b) / n_b
    point = point_a - point_b
    # one-sided p: fraction of bootstrap diffs <= 0 (testing "Δ_a > Δ_b")
    p_one_sided = sum(1 for d in diffs if d <= 0) / n_resamples
    return {
        "delta_a": point_a, "delta_b": point_b, "diff": point,
        "ci_lo": lo, "ci_hi": hi, "p_one_sided": p_one_sided,
    }


# --- main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--v1_9", type=Path,
                        default="data/pipeline-runs/default/bench-eval-post-sft-v1_9/episodes.json",
                        help="v1.9 episodes.json (post-SFT 2B)")
    parser.add_argument("--v2_0", type=Path,
                        default="data/pipeline-runs/default/bench-eval-post-sft-v2_0/episodes.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    print(f"loading v1.9: {args.v1_9}")
    eps_v19 = json.loads(args.v1_9.read_text())
    print(f"loading v2.0: {args.v2_0}")
    eps_v20 = json.loads(args.v2_0.read_text())

    for label, eps in [("v1.9", eps_v19), ("v2.0", eps_v20)]:
        print(f"\n=== {label} (n_eps={len(eps)}) ===")
        m = mcnemar_paired(eps)
        print(f"  contingency  a={m['a_both_pass']}  b(BL only)={m['b_bl_only']}  "
              f"c(CU only)={m['c_cu_only']}  d(both fail)={m['d_both_fail']}")
        print(f"  Δ = {m['delta']:+.4f}")
        chi2_str = "exact-binomial" if m["discordant_used_exact"] else f"chi^2={m['chi2']:.3f}"
        print(f"  McNemar two-sided  {chi2_str}  p={m['p_value']:.4g}")
        point, lo, hi = bootstrap_delta(eps, n_resamples=args.bootstrap)
        print(f"  bootstrap 95% CI on Δ:  [{lo:+.4f}, {hi:+.4f}]  (point {point:+.4f})")

    print("\n=== v1.9 vs v2.0 (is v1.9's Δ larger than v2.0's?) ===")
    diff = bootstrap_delta_diff(eps_v19, eps_v20, n_resamples=args.bootstrap)
    print(f"  Δ_v1.9 - Δ_v2.0 = {diff['diff']:+.4f}")
    print(f"  bootstrap 95% CI on (Δ_v1.9 - Δ_v2.0):  [{diff['ci_lo']:+.4f}, {diff['ci_hi']:+.4f}]")
    print(f"  one-sided bootstrap p (H0: Δ_v1.9 ≤ Δ_v2.0):  p={diff['p_one_sided']:.4g}")
    if diff["ci_lo"] > 0:
        print("  → v1.9 Δ is significantly larger (95% CI excludes 0): bench-saturation supported.")
    elif diff["ci_hi"] < 0:
        print("  → v2.0 Δ is significantly larger (sign reversed from the saturation hypothesis).")
    else:
        print("  → v1.9 and v2.0 Δs are NOT significantly different at α=0.05.")
        print("    The saturation claim is directional only; corroborate with larger n or harder bench.")


if __name__ == "__main__":
    main()
