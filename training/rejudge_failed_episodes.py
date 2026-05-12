"""
rejudge_failed_episodes.py — Re-run the judge over an existing
episodes.json. Three modes:

  (1) DEFAULT: target only episodes whose original judge call errored
      (rationale starts with "judge evaluation failed:"). Recovers from
      transient auth/network blips during the original eval.

  (2) --all: target every episode in the file. Combined with
      --force-llm-judge, this enables apples-to-apples re-scoring of
      a full run via the LLM judge — useful when comparing against a
      pre-SFT base eval that itself was forced through the LLM judge
      (because the deterministic judge requires literal ANSWER: lines
      that base models don't reliably emit).

  (3) --judge deterministic: re-score every non-FREE_FORM episode using
      the deterministic extractor (_score_deterministic in llm_judge.py).
      No LLM calls. FREE_FORM episodes are LEFT UNCHANGED — assumes the
      source file already has valid LLM-judged FREE_FORM scores (true
      whenever the source was generated with --force-llm-judge, since
      FREE_FORM tasks under that flag are routed to the same LLM/FREE_FORM
      verifier path Table 1's deterministic-mixed dispatch uses).
      Use case: take an --force-llm-judge episodes file and produce
      matched-path Table 1 scoring (deterministic for YES_NO / SINGLE_WORD
      / RANKING, LLM for FREE_FORM) without spending any API budget.

  --force-llm-judge routes every targeted episode through the LLM
      judge by overriding query_type to FREE_FORM, regardless of the
      task's original query_type. Use this when the model under test
      doesn't follow the deterministic-judge format reliably.
      Mutually exclusive with --judge deterministic.

  --out-dir writes the rejudged episodes/summary to a NEW directory
      instead of overwriting the source. Recommended whenever you're
      re-judging ALL episodes, so the original headline numbers stay
      in place for reference.

USAGE
    # mode (1): recover transient judge failures in-place (default)
    python training/rejudge_failed_episodes.py \\
        --episodes data/pipeline-runs/default/bench-eval-post-sft-v2_0/episodes.json \\
        --tasks data/pipeline-runs/default/synthesis/tasks_eval.json

    # mode (2): full LLM-judge re-score of v2.0 for apples-to-apples
    # comparison with pre-SFT 4B (which was forced through LLM judge).
    # Writes to a parallel output dir.
    python training/rejudge_failed_episodes.py \\
        --episodes data/pipeline-runs/default/bench-eval-post-sft-v2_0/episodes.json \\
        --tasks data/pipeline-runs/default/synthesis/tasks_eval.json \\
        --all --force-llm-judge \\
        --out-dir data/pipeline-runs/default/bench-eval-post-sft-v2_0-llm-only

    # mode (3): produce matched-path Table 1 scoring for a pre-SFT 0.8B
    # HF run originally captured under --force-llm-judge. Zero API cost.
    python training/rejudge_failed_episodes.py \\
        --episodes data/pipeline-runs/default/bench-eval-pre-sft-0.8b-hf-llm/episodes.json \\
        --tasks data/pipeline-runs/default/synthesis/tasks_eval.json \\
        --all --judge deterministic \\
        --out-dir data/pipeline-runs/default/bench-eval-pre-sft-0.8b-hf-det
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

from b2_benchmarks.skillsbench.llm_judge import (
    LLMJudgeEvaluator,
    _score_deterministic,
)
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
    parser.add_argument("--all", action="store_true", dest="rejudge_all",
                        help="Re-judge ALL episodes, not just the ones whose "
                             "original judge call errored. Combined with "
                             "--force-llm-judge this gives a full LLM-judge "
                             "re-score. Recommended with --out-dir so the "
                             "original numbers are preserved.")
    parser.add_argument("--force-llm-judge", action="store_true",
                        help="Override every targeted episode's query_type "
                             "to FREE_FORM, routing it through the LLM "
                             "judge regardless of original task type. "
                             "Apples-to-apples comparison with pre-SFT "
                             "evals that ran with --force-llm-judge in "
                             "eval_qwen35_lora.py.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Write rejudged episodes.json + summary.json "
                             "to this directory instead of overwriting the "
                             "source. Strongly recommended with --all so "
                             "the original eval is preserved.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts of episodes that would be re-judged; "
                             "do not actually re-judge.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Re-judge only the first N targeted episodes "
                             "(0 = all). Useful for smoke-testing.")
    args = parser.parse_args()

    # ---- Load episodes ----
    episodes = json.loads(args.episodes.read_text())
    failed_idx = [i for i, ep in enumerate(episodes) if _is_failed(ep)]
    print(f"Total episodes: {len(episodes)}")
    print(f"Failed (judge errored): {len(failed_idx)}")

    # Decide what to re-judge
    if args.rejudge_all:
        target_idx = list(range(len(episodes)))
        print(f"Mode: --all  (target all {len(target_idx)} episodes)")
    else:
        target_idx = failed_idx
        print(f"Mode: failed-only  (target {len(target_idx)} episodes)")

    if not target_idx:
        print("Nothing to do.")
        return

    deterministic_mode = args.judge.strip().lower() == "deterministic"
    if deterministic_mode and args.force_llm_judge:
        print("ERROR: --judge deterministic and --force-llm-judge are mutually exclusive.")
        sys.exit(2)

    if args.force_llm_judge:
        print("Force LLM judge: ON  (every episode routed through LLM judge)")
    if deterministic_mode:
        print("Mode: --judge deterministic  (no LLM calls; FREE_FORM episodes left untouched)")

    if args.dry_run:
        for i in target_idx[:5]:
            ep = episodes[i]
            print(f"  [{i}] task={ep['task_uid']} cond={ep['condition']} "
                  f"qtype={getattr(ep, 'query_type', 'n/a')}  "
                  f"current passed={ep.get('passed')}  "
                  f"rationale={(ep.get('judge_rationale') or '')[:80]!r}")
        return

    # ---- Load tasks for context lookup ----
    tasks_by_uid = {t.task_uid: t for t in load_extracted_tasks(args.tasks)}
    print(f"Tasks loaded: {len(tasks_by_uid)}")

    # ---- Build judge (skipped in deterministic mode — no LLM calls) ----
    judge = None
    if not deterministic_mode:
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
    todo = target_idx if not args.limit else target_idx[:args.limit]
    n_recovered = 0
    n_still_failed = 0
    n_kept_freeform = 0
    for k, idx in enumerate(todo, 1):
        ep = episodes[idx]
        task = tasks_by_uid.get(ep["task_uid"])
        if task is None:
            print(f"  [{k}/{len(todo)}] task {ep['task_uid']} not found in tasks file; skipping")
            continue

        if deterministic_mode:
            # Skip FREE_FORM — leave existing score (assumed already
            # LLM-judged with the FREE_FORM verifier in the source file).
            qt = getattr(task, "query_type", "FREE_FORM")
            if qt == "FREE_FORM":
                n_kept_freeform += 1
                continue

            t0 = time.time()
            verdict = _score_deterministic(
                response=ep["response"],
                acceptance_criteria=task.acceptance_criteria,
                query_type=qt,
            )
            elapsed = time.time() - t0
            n_recovered += 1
            tag = "PASS" if verdict.passed else "FAIL"
            ep["passed"] = bool(verdict.passed)
            ep["score"] = float(verdict.score)
            ep["judge_rationale"] = verdict.rationale
            print(f"  [{k}/{len(todo)}] {tag} score={verdict.score:.2f} "
                  f"({elapsed*1000:.1f}ms) [det] {ep['task_uid']} / {ep['condition']}")
            continue

        # If --force-llm-judge, override query_type to FREE_FORM so the
        # judge skips its deterministic-extractor short-circuit.
        if args.force_llm_judge:
            effective_qt = "FREE_FORM"
        else:
            effective_qt = getattr(task, "query_type", "FREE_FORM")

        t0 = time.time()
        try:
            verdict = judge.evaluate(
                response=ep["response"],
                passage=task.passage,
                challenge=task.challenge,
                acceptance_criteria=task.acceptance_criteria,
                query_type=effective_qt,
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
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        episodes_out = args.out_dir / "episodes.json"
        summary_out = args.out_dir / "summary.json"
        # Preserve any pre-existing summary metadata if user pre-populated
        # the out-dir; otherwise we'll seed from scratch.
    else:
        episodes_out = args.episodes
        summary_out = args.episodes.parent / "summary.json"

    episodes_out.write_text(json.dumps(episodes, indent=2))
    print(f"\nWrote {len(episodes)} episodes -> {episodes_out}")

    # Seed summary from the source's summary.json so model-label / base /
    # adapter / tasks fields persist into the new dir.
    src_summary_path = args.episodes.parent / "summary.json"
    seed_summary = {}
    if src_summary_path.exists() and (args.out_dir or not summary_out.exists()):
        try:
            seed_summary = json.loads(src_summary_path.read_text())
        except json.JSONDecodeError:
            pass
    if args.out_dir and not summary_out.exists():
        summary_out.write_text(json.dumps(seed_summary, indent=2))

    summary = _recompute_summary(episodes, summary_out)
    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {summary_out}")
    if deterministic_mode:
        print(f"\nRe-scored deterministically: {n_recovered}  "
              f"Kept FREE_FORM (untouched): {n_kept_freeform}")
    else:
        print(f"\nRecovered: {n_recovered}  Still failed: {n_still_failed}")
    print(f"\nNew per-condition pass rates:")
    for cond, stats in summary["per_condition"].items():
        print(f"  {cond}: pass_rate={stats['pass_rate']} ({stats['passed']}/{stats['n']})")


if __name__ == "__main__":
    main()
