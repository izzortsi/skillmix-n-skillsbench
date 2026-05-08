"""
judge_agreement.py — Compare two judge runs over the same model responses.

Given two episodes.json files that differ only in which judge graded the
responses (e.g. Opus 4.7 vs GPT-5.4 over the same v2.0 model output),
compute:

  - per-episode verdict agreement rate
  - Cohen's kappa (corrects for chance agreement)
  - confusion matrix (Opus pass × GPT pass)
  - per-condition agreement (separately for baseline vs curated)
  - headline Δ shift between judges
  - disagreement breakdown: which side passes more often, by skill

USAGE
    python training/judge_agreement.py \\
        --judge-a data/pipeline-runs/default/bench-eval-post-sft-v2_0-llm-only/episodes.json \\
        --judge-b data/pipeline-runs/default/bench-eval-post-sft-v2_0-gpt/episodes.json \\
        --label-a opus --label-b gpt

OUTPUT
    Compact summary suitable for paste into the paper or an engineering report.

USAGE FOR FULL PAPER COVERAGE
    Loop over the (Opus LLM-only, GPT) pairs for v1, v1.9, v2.0, haiku,
    pre-SFT 2B, pre-SFT 4B. The headline cross-family bound is the maximum
    Δ shift across all six datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _key(ep: dict) -> Tuple[str, str]:
    """Episode identity = (task_uid, condition). Pairs episodes across judges."""
    return (ep["task_uid"], ep["condition"])


def cohens_kappa(a: List[bool], b: List[bool]) -> float:
    """Cohen's κ for two raters over the same N items, binary verdicts."""
    n = len(a)
    if n == 0:
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if p_e >= 1.0:
        return 1.0  # both raters always pass or always fail; vacuous
    return (p_o - p_e) / (1 - p_e)


def per_condition_pass(eps: List[dict]) -> Dict[str, Tuple[int, int]]:
    """Returns {condition: (passed, total)}."""
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for e in eps:
        out[e["condition"]][1] += 1
        if e["passed"]:
            out[e["condition"]][0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--judge-a", type=Path, required=True,
                   help="episodes.json from judge A (e.g. Opus run)")
    p.add_argument("--judge-b", type=Path, required=True,
                   help="episodes.json from judge B (e.g. GPT run)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--task-skill-map", type=Path,
                   default=ROOT / "data/pipeline-runs/default/synthesis/task_skill_map_eval.json",
                   help="for per-skill disagreement breakdown")
    p.add_argument("--show-disagreements", action="store_true",
                   help="Print per-episode disagreements (task_uid + verdicts).")
    args = p.parse_args()

    eps_a = {_key(e): e for e in json.loads(args.judge_a.read_text())}
    eps_b = {_key(e): e for e in json.loads(args.judge_b.read_text())}

    common_keys = sorted(set(eps_a) & set(eps_b))
    only_a = sorted(set(eps_a) - set(eps_b))
    only_b = sorted(set(eps_b) - set(eps_a))

    print(f"Judge A  ({args.label_a}): {args.judge_a}")
    print(f"Judge B  ({args.label_b}): {args.judge_b}")
    print(f"Common episodes: {len(common_keys)}")
    if only_a:
        print(f"  in A only: {len(only_a)}  (skipping)")
    if only_b:
        print(f"  in B only: {len(only_b)}  (skipping)")
    if not common_keys:
        print("No paired episodes; nothing to compare.")
        return

    # ---- per-episode agreement + κ ----
    a_passed = [bool(eps_a[k]["passed"]) for k in common_keys]
    b_passed = [bool(eps_b[k]["passed"]) for k in common_keys]
    n = len(common_keys)
    n_agree = sum(1 for x, y in zip(a_passed, b_passed) if x == y)
    agreement = n_agree / n
    kappa = cohens_kappa(a_passed, b_passed)

    # confusion matrix
    pp = sum(1 for x, y in zip(a_passed, b_passed) if x and y)
    pf = sum(1 for x, y in zip(a_passed, b_passed) if x and not y)   # A pass, B fail
    fp = sum(1 for x, y in zip(a_passed, b_passed) if not x and y)   # A fail, B pass
    ff = sum(1 for x, y in zip(a_passed, b_passed) if not x and not y)

    print(f"\n=== overall (n={n}) ===")
    print(f"  agreement rate: {n_agree}/{n} = {agreement:.4f}")
    print(f"  Cohen's κ:      {kappa:.4f}")
    print(f"  confusion (A row, B col):")
    print(f"            B=PASS  B=FAIL")
    print(f"    A=PASS    {pp:>4}    {pf:>4}")
    print(f"    A=FAIL    {fp:>4}    {ff:>4}")

    # ---- per-condition split ----
    print(f"\n=== per-condition ===")
    by_cond: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for k in common_keys:
        cond = eps_a[k]["condition"]
        by_cond[cond].append((a_passed[common_keys.index(k)], b_passed[common_keys.index(k)]))
    # Recompute by direct iteration (the index() above is O(n²); fix it)
    by_cond = defaultdict(list)
    for i, k in enumerate(common_keys):
        cond = eps_a[k]["condition"]
        by_cond[cond].append((a_passed[i], b_passed[i]))

    for cond, pairs in sorted(by_cond.items()):
        nc = len(pairs)
        agree_c = sum(1 for x, y in pairs if x == y)
        a_rate = sum(1 for x, _ in pairs if x) / nc
        b_rate = sum(1 for _, y in pairs if y) / nc
        kap_c = cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])
        print(f"  {cond:<10} n={nc}  A pass={a_rate:.3f}  B pass={b_rate:.3f}  "
              f"agreement={agree_c/nc:.4f}  κ={kap_c:.4f}")

    # ---- headline Δ shift ----
    pc_a = per_condition_pass([eps_a[k] for k in common_keys])
    pc_b = per_condition_pass([eps_b[k] for k in common_keys])
    if "baseline" in pc_a and "curated" in pc_a:
        a_bl = pc_a["baseline"][0] / pc_a["baseline"][1]
        a_cu = pc_a["curated"][0] / pc_a["curated"][1]
        b_bl = pc_b["baseline"][0] / pc_b["baseline"][1]
        b_cu = pc_b["curated"][0] / pc_b["curated"][1]
        d_a = a_cu - a_bl
        d_b = b_cu - b_bl
        print(f"\n=== headline Δ shift (the load-bearing number for §7.1) ===")
        print(f"  judge A ({args.label_a}):  BL={a_bl:.3f}  CU={a_cu:.3f}  Δ={d_a:+.4f}")
        print(f"  judge B ({args.label_b}):  BL={b_bl:.3f}  CU={b_cu:.3f}  Δ={d_b:+.4f}")
        print(f"  Δ shift (B - A): {d_b - d_a:+.4f}")
        print(f"  BL shift (B - A): {b_bl - a_bl:+.4f}")
        print(f"  CU shift (B - A): {b_cu - a_cu:+.4f}")

    # ---- per-skill disagreement breakdown ----
    if args.task_skill_map.exists():
        skill_map = json.loads(args.task_skill_map.read_text())
        ds_by_skill: Dict[str, List[str]] = defaultdict(list)  # skill -> ["A>B", "B>A", "agree"]
        for i, k in enumerate(common_keys):
            sk = skill_map.get(k[0], "unknown")
            if a_passed[i] == b_passed[i]:
                ds_by_skill[sk].append("agree")
            elif a_passed[i] and not b_passed[i]:
                ds_by_skill[sk].append(f"{args.label_a}>{args.label_b}")
            else:
                ds_by_skill[sk].append(f"{args.label_b}>{args.label_a}")

        print(f"\n=== per-skill disagreement (sorted by disagreement count) ===")
        rows = []
        for sk, tags in ds_by_skill.items():
            c = Counter(tags)
            n_sk = sum(c.values())
            n_dis = n_sk - c["agree"]
            rows.append((n_dis, sk, c["agree"], c.get(f"{args.label_a}>{args.label_b}", 0),
                         c.get(f"{args.label_b}>{args.label_a}", 0), n_sk))
        for n_dis, sk, ag, ab, ba, total in sorted(rows, key=lambda r: -r[0]):
            if n_dis == 0:
                continue
            print(f"  {sk:<55} agree={ag:>2}/{total}  "
                  f"{args.label_a}>{args.label_b}={ab:>2}  {args.label_b}>{args.label_a}={ba:>2}")

    # ---- per-episode disagreements (optional dump) ----
    if args.show_disagreements:
        print(f"\n=== disagreements (episode-level) ===")
        for i, k in enumerate(common_keys):
            if a_passed[i] != b_passed[i]:
                ea = eps_a[k]
                eb = eps_b[k]
                tag_a = "PASS" if a_passed[i] else "FAIL"
                tag_b = "PASS" if b_passed[i] else "FAIL"
                print(f"  task={k[0]} cond={k[1]}")
                print(f"    {args.label_a}: {tag_a} score={ea['score']:.2f} "
                      f"rationale={(ea.get('judge_rationale') or '')[:120]!r}")
                print(f"    {args.label_b}: {tag_b} score={eb['score']:.2f} "
                      f"rationale={(eb.get('judge_rationale') or '')[:120]!r}")


if __name__ == "__main__":
    main()
