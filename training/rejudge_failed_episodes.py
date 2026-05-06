"""
rejudge_failed_episodes.py — Re-run the LLM judge on episodes whose
original judge call failed (e.g. OAuth/auth/network errors mid-eval).

WHY
    eval_qwen35_lora.py catches judge exceptions and returns a fallback
    JudgeResult(passed=False, score=0.0, rationale="judge evaluation
    failed: {e}"). On a long eval, an intermittent auth/network blip
    can corrupt 10-30% of episodes. The model responses are saved
    correctly; only the judge verdicts are wrong. This script reruns
    the judge on those episodes and updates episodes.json + summary.json
    in place — no need to regenerate model responses.

USAGE
    # 1. Fix the auth issue (export ANTHROPIC_API_KEY=... or refresh OAuth)
    # 2. Sanity-check the judge:
    python -c "from core.providers import create_provider; p = create_provider('anthropic:claude-opus-4-7'); print(p.chat([{'role':'user','content':'say ok'}]).text)"
    # 3. Run this:
    python training/rejudge_failed_episodes.py \\
        --episodes data/pipeline-runs/default/bench-eval-post-sft-v2_0/episodes.json \\
        --tasks data/pipeline-runs/default/synthesis/tasks_eval.json \\
        --judge anthropic:claude-opus-4-7

    # 4. summary.json is recomputed; re-read pass rates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# repo-relative imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b2_benchmarks.skillsbench.llm_judge import LLMJudgeEvaluator
from core.providers import create_provider
from core.schemas import load_extracted_tasks


FAILURE_PREFIX = "judge evaluation failed:"


def _is_failed(ep: dict) -> bool:
    """Detect an episode whose stored verdict came from the fallback path."""
    return (ep.get("judge_rationale") or "").startswith(FAILURE_PREFIX)


def _recompute_summary(episodes: list[dict], summary_path: Path) -> dict:
    """Rebuild summary.json from episodes (matches eval_qwen35_lora.py shape)."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_cond[ep["condition"]].append(ep)

    per_condition = {}
    for cond, eps in by_cond.items():
        n = len(eps)
        passed = sum(1 for e in eps if e.get("passed"))
        mean_score = round(sum(float(e.get("score", 0.0)) for e in eps) / n, 4) if n else 0.0
        per_condition[cond] = {
            "n": n,
            "passed": passed,
            "pass_rate": round(passed / n, 4) if n else 0.0,
            "mean_score": mean_score,
        }

    # Preserve any pre-existing top-level keys (model, base, adapter, tasks)
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass

    summary["total_episodes"] = len(episodes)
    summary["per_condition"] = per_condition
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=Path, required=True,
                        help="Path to episodes.json from a previous eval run")
    parser.add_argument("--tasks", type=Path, required=True,
                        help="Path to tasks_eval.json (needed to recover "
                             "passage / challenge / acceptance_criteria)")
    parser.add_argument("--judge", default="anthropic:claude-opus-4-7",
                        help="Judge provider:model id (default: opus-4-7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts of failed episodes; do not re-judge.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Re-judge only the first N failed episodes (0 = all)")
    args = parser.parse_args()

    # ---- Load episodes ----
    episodes = json.loads(args.episodes.read_text())
    failed_idx = [i for i, ep in enumerate(episodes) if _is_failed(ep)]
    print(f"Total episodes: {len(episodes)}")
    print(f"Failed (judge errored): {len(failed_idx)}")
    if not failed_idx:
        print("Nothing to do.")
        return

    if args.dry_run:
        # Sample the first few failure rationales to confirm the pattern
        for i in failed_idx[:5]:
            ep = episodes[i]
            print(f"  [{i}] task={ep['task_uid']} cond={ep['condition']} "
                  f"rationale={ep['judge_rationale'][:120]!r}")
        return

    # ---- Load tasks for context lookup ----
    tasks_by_uid = {t.task_uid: t for t in load_extracted_tasks(args.tasks)}
    print(f"Tasks loaded: {len(tasks_by_uid)}")

    # ---- Build judge ----
    # provider spec is "<name>:<model>" e.g. "anthropic:claude-opus-4-7";
    # split into the two args create_provider expects.
    print(f"Building judge: {args.judge}")
    if ":" in args.judge:
        p_name, p_model = args.judge.split(":", 1)
    else:
        p_name, p_model = args.judge, ""
    provider = create_provider(p_name, p_model)
    judge = LLMJudgeEvaluator(provider)

    # ---- Re-judge ----
    todo = failed_idx if not args.limit else failed_idx[:args.limit]
    n_recovered = 0
    n_still_failed = 0
    for k, idx in enumerate(todo, 1):
        ep = episodes[idx]
        task = tasks_by_uid.get(ep["task_uid"])
        if task is None:
            print(f"  [{k}/{len(todo)}] task {ep['task_uid']} not found in tasks file; skipping")
            continue

        t0 = time.time()
        try:
            verdict = judge.evaluate(
                response=ep["response"],
                passage=task.passage,
                challenge=task.challenge,
                acceptance_criteria=task.acceptance_criteria,
                query_type=getattr(task, "query_type", "FREE_FORM"),
            )
        except Exception as e:
            print(f"  [{k}/{len(todo)}] re-judge raised: {e}; leaving as-is")
            n_still_failed += 1
            continue
        elapsed = time.time() - t0

        if (verdict.rationale or "").startswith(FAILURE_PREFIX):
            n_still_failed += 1
            tag = "STILL FAILED"
        else:
            n_recovered += 1
            tag = "PASS" if verdict.passed else "FAIL"

        ep["passed"] = bool(verdict.passed)
        ep["score"] = float(verdict.score)
        ep["judge_rationale"] = verdict.rationale
        # Don't overwrite elapsed_s — that's the original gen+judge time

        print(f"  [{k}/{len(todo)}] {tag} score={verdict.score:.2f} "
              f"({elapsed:.1f}s)  {ep['task_uid']} / {ep['condition']}")

    # ---- Persist ----
    args.episodes.write_text(json.dumps(episodes, indent=2))
    print(f"\nWrote {len(episodes)} episodes -> {args.episodes}")

    summary_path = args.episodes.parent / "summary.json"
    summary = _recompute_summary(episodes, summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {summary_path}")
    print(f"\nRecovered: {n_recovered}  Still failed: {n_still_failed}")
    print(f"\nNew per-condition pass rates:")
    for cond, stats in summary["per_condition"].items():
        print(f"  {cond}: pass_rate={stats['pass_rate']} ({stats['passed']}/{stats['n']})")


if __name__ == "__main__":
    main()
