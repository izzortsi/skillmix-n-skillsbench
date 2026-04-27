"""
compare_pre_post.py — Per-skill diff between pre-SFT and post-SFT eval.

Joins bench-eval-pre-sft/episodes.json (multi-student, includes the SFT
target as a base student) with bench-eval-post-sft/episodes.json (the
SFT'd model), filters to a chosen student, aggregates per (skill, condition,
train_state), and prints:
  - per-skill table: pre BL, pre CU, post BL, post CU, Δ pre, Δ post
  - aggregate row
  - cluster summary (lift / flat / regression skills under post-SFT)

Optional --csv writes the table to disk for downstream plotting.

USAGE
    cd skillmix-n-skillsbench
    python training/compare_pre_post.py
    # or with explicit paths:
    python training/compare_pre_post.py \
        --pre  data/pipeline-runs/default/bench-eval-pre-sft/episodes.json \
        --post data/pipeline-runs/default/bench-eval-post-sft/episodes.json \
        --tasks data/pipeline-runs/default/synthesis/tasks.json \
        --student-pattern qwen \
        --csv data/pipeline-runs/default/sft_dataset/per_skill_diff.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_skill_of_task(tasks_path: Path) -> Dict[str, str]:
    """Build {task_uid -> skill_name} from a tasks.json (full, train, or eval)."""
    with tasks_path.open(encoding="utf-8") as f:
        tasks = json.load(f)
    out: Dict[str, str] = {}
    for t in tasks:
        names = (t.get("acceptance_criteria") or {}).get("skill_names") or []
        if names:
            out[t["task_uid"]] = names[0]
    return out


def load_episodes(path: Path) -> List[Dict]:
    """Read episodes.json (a JSON array, not jsonl)."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_by_skill_condition(
    episodes: List[Dict],
    skill_of: Dict[str, str],
    student_pattern: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(skill, condition): {n, passed, score_sum}} filtered to
    episodes whose model contains `student_pattern` (substring match)."""
    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "passed": 0, "score_sum": 0.0}
    )
    for e in episodes:
        if student_pattern and student_pattern not in e.get("model", ""):
            continue
        skill = skill_of.get(e.get("task_uid", ""))
        if not skill:
            continue
        key = (skill, e.get("condition", ""))
        bucket = out[key]
        bucket["n"] += 1
        bucket["passed"] += int(bool(e.get("passed")))
        bucket["score_sum"] += float(e.get("score") or 0.0)
    return out


def rate(bucket: Dict[str, float], field: str = "passed") -> float:
    n = bucket["n"]
    if not n:
        return 0.0
    return bucket[field] / n if field == "passed" else bucket["score_sum"] / n


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


HEADERS = ["skill", "n", "pre_BL", "pre_CU", "post_BL", "post_CU", "Δ pre", "Δ post", "ΔΔ"]


def build_rows(
    pre_agg: Dict, post_agg: Dict, skills: List[str],
) -> List[Dict]:
    rows = []
    for sk in skills:
        pb = pre_agg.get((sk, "baseline"), {"n": 0, "passed": 0, "score_sum": 0})
        pc = pre_agg.get((sk, "curated"),  {"n": 0, "passed": 0, "score_sum": 0})
        qb = post_agg.get((sk, "baseline"), {"n": 0, "passed": 0, "score_sum": 0})
        qc = post_agg.get((sk, "curated"),  {"n": 0, "passed": 0, "score_sum": 0})
        pre_d = rate(pc) - rate(pb)
        post_d = rate(qc) - rate(qb)
        rows.append({
            "skill":   sk,
            "n":       pc["n"] or qc["n"],
            "pre_BL":  rate(pb),
            "pre_CU":  rate(pc),
            "post_BL": rate(qb),
            "post_CU": rate(qc),
            "Δ pre":   pre_d,
            "Δ post":  post_d,
            "ΔΔ":      post_d - pre_d,
        })
    return rows


def print_table(rows: List[Dict]) -> None:
    name_w = max(len(r["skill"]) for r in rows + [{"skill": "AGGREGATE"}])
    name_w = max(name_w, len("skill"))
    # header
    h = (
        f'{"skill":<{name_w}} {"n":>4}  '
        f'{"pre_BL":>7} {"pre_CU":>7}   '
        f'{"post_BL":>8} {"post_CU":>8}  '
        f'{"Δ pre":>7} {"Δ post":>7}   {"ΔΔ":>7}'
    )
    print(h)
    print("-" * len(h))
    for r in rows:
        print(
            f'{r["skill"]:<{name_w}} {r["n"]:>4}  '
            f'{r["pre_BL"]:>7.3f} {r["pre_CU"]:>7.3f}   '
            f'{r["post_BL"]:>8.3f} {r["post_CU"]:>8.3f}  '
            f'{r["Δ pre"]:>+7.3f} {r["Δ post"]:>+7.3f}   {r["ΔΔ"]:>+7.3f}'
        )


def cluster_summary(rows: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    """Group skills by post-SFT Δ sign: lift (Δ post > +0.05),
    flat (-0.05 <= Δ post <= +0.05), regression (Δ post < -0.05)."""
    lift, flat, regress = [], [], []
    for r in rows:
        d = r["Δ post"]
        if d > 0.05:
            lift.append(r["skill"])
        elif d < -0.05:
            regress.append(r["skill"])
        else:
            flat.append(r["skill"])
    return lift, flat, regress


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pre", type=Path,
        default=Path("data/pipeline-runs/default/bench-eval-pre-sft/episodes.json"),
        help="Pre-SFT episodes.json (corpus_harness multi-student output)",
    )
    parser.add_argument(
        "--post", type=Path,
        default=Path("data/pipeline-runs/default/bench-eval-post-sft/episodes.json"),
        help="Post-SFT episodes.json (eval_qwen35_lora.py output)",
    )
    parser.add_argument(
        "--tasks", type=Path,
        default=Path("data/pipeline-runs/default/synthesis/tasks.json"),
        help="tasks.json — used to map task_uid -> skill_name",
    )
    parser.add_argument(
        "--student-pattern", default="qwen",
        help="Substring to match the SFT-target student in episode.model "
             "fields. Default 'qwen' picks both pre-SFT base "
             "(ollama:qwen3.5:0.8b) and post-SFT LoRA "
             "(Qwen/Qwen3.5-0.8B+lora:...)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    skill_of = load_skill_of_task(args.tasks)
    pre_eps = load_episodes(args.pre)
    post_eps = load_episodes(args.post)

    pre_agg = aggregate_by_skill_condition(pre_eps, skill_of, args.student_pattern)
    post_agg = aggregate_by_skill_condition(post_eps, skill_of, args.student_pattern)

    skills = sorted(
        {sk for (sk, _) in pre_agg.keys()} | {sk for (sk, _) in post_agg.keys()}
    )
    rows = build_rows(pre_agg, post_agg, skills)

    # Sort by post-Δ descending so lifts top, regressions bottom
    rows.sort(key=lambda r: -r["Δ post"])

    print(f"Pre-SFT  : {args.pre}")
    print(f"Post-SFT : {args.post}")
    print(f"Tasks    : {args.tasks}")
    print(f"Student pattern: {args.student_pattern!r}")
    n_pre = sum(1 for e in pre_eps if args.student_pattern in e.get("model", ""))
    n_post = sum(1 for e in post_eps if args.student_pattern in e.get("model", ""))
    print(f"Episodes : pre={n_pre}, post={n_post}\n")

    print_table(rows)

    # Aggregate row
    def agg_rate(d: Dict, cond: str, field: str = "passed") -> float:
        n = sum(b["n"] for (s, c), b in d.items() if c == cond)
        v = sum(b[field if field == "passed" else "score_sum"]
                for (s, c), b in d.items() if c == cond)
        return v / n if n else 0.0

    pre_BL = agg_rate(pre_agg, "baseline")
    pre_CU = agg_rate(pre_agg, "curated")
    post_BL = agg_rate(post_agg, "baseline")
    post_CU = agg_rate(post_agg, "curated")
    pre_d = pre_CU - pre_BL
    post_d = post_CU - post_BL
    print("-" * 100)
    print(
        f'{"AGGREGATE":<{max(len(r["skill"]) for r in rows)}} {sum(r["n"] for r in rows):>4}  '
        f'{pre_BL:>7.3f} {pre_CU:>7.3f}   '
        f'{post_BL:>8.3f} {post_CU:>8.3f}  '
        f'{pre_d:>+7.3f} {post_d:>+7.3f}   {post_d - pre_d:>+7.3f}'
    )

    # Cluster summary
    lift, flat, regress = cluster_summary(rows)
    print("\n" + "=" * 60)
    print(f"LIFT cluster (Δ post > +0.05) — {len(lift)} skills")
    for sk in lift:
        print(f"  {sk}")
    print(f"\nFLAT cluster (-0.05 ≤ Δ post ≤ +0.05) — {len(flat)} skills")
    for sk in flat:
        print(f"  {sk}")
    print(f"\nREGRESSION cluster (Δ post < -0.05) — {len(regress)} skills")
    for sk in regress:
        print(f"  {sk}")
    print("=" * 60)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
