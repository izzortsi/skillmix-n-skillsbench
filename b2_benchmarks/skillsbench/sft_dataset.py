"""
b2_benchmarks.skillsbench.sft_dataset

Build SFT training data from bench_traced episodes.

Reads:  bench_traced/episodes.jsonl  (one episode per line, see s4.m4
        --mode inject for shape)
Writes: sft_dataset/dataset.jsonl   (one SFT row per line, chat format)
        sft_dataset/summary.json    (counts per skill / model / condition)

Filter (default): passed=True AND condition=curated
  -> demonstrations where skill injection produced a correct answer with
     visible CoT. This is the "apply given skill" SFT contract — the model
     learns to follow the procedural skill block in its system prompt.

Format: HuggingFace SFTTrainer / axolotl / unsloth-compatible chat schema.
Each row:
    {
      "messages": [
        {"role": "system",    "content": "<base prompt + injected skill block>"},
        {"role": "user",      "content": "<task input + question>"},
        {"role": "assistant", "content": "<CoT response ending with ANSWER: ...>"}
      ],
      "metadata": {...}   # task_uid, skill, judge verdict, model_source, etc.
    }

Trainers ignore the metadata field; we keep it for inspection, slicing,
and downstream filtering (e.g., 'train only on examples where score >= 0.8').

Usage:
    python -m b2_benchmarks.skillsbench.sft_dataset \\
        --episodes data/pipeline-runs/default/bench_traced/episodes.jsonl \\
        --out      data/pipeline-runs/default/sft_dataset/dataset.jsonl \\
        -v
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Episode -> SFT row
# ---------------------------------------------------------------------------


def _episode_metadata(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve provenance + judge verdict on every row.

    Trainers ignore unknown fields — this is for downstream filtering and
    debugging (e.g. 'show me all SFT rows where the spatial-reasoning skill
    was injected and judge_score < 0.7' to spot weak demos).
    """
    return {
        "task_uid":                 ep.get("task_uid", ""),
        "title":                    ep.get("title", ""),
        "skill_uid":                ep.get("injected_skill_uid", ""),
        "skill_name":               ep.get("injected_skill_name", ""),
        "condition":                ep.get("condition", ""),
        "judge_passed":             bool(ep.get("passed", False)),
        "judge_score":              float(ep.get("score", 0.0) or 0.0),
        "judge_conclusion_reached": bool(ep.get("conclusion_reached", False)),
        "judge_rationale":          ep.get("judge_rationale", ""),
        "model_source":             ep.get("model", ""),
        "extraction_method":        ep.get("extraction_method", ""),
    }


def _episode_to_sft_row(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one bench_traced episode into one chat-format SFT row.

    Reuses `trace.system_prompt` and `trace.user_prompt` verbatim — these
    are the exact prompts the model saw during generation, so the SFT
    distribution matches the inference distribution.
    """
    trace = ep.get("trace", {}) or {}
    system_prompt = (trace.get("system_prompt", "") or "").strip()
    user_prompt = (trace.get("user_prompt", "") or "").strip()
    # Prefer the full response (CoT + ANSWER line) over the parsed `answer`
    # field — the CoT IS the SFT signal we want to teach.
    response_text = (trace.get("response", "") or ep.get("answer", "") or "").strip()

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": response_text},
        ],
        "metadata": _episode_metadata(ep),
    }


# ---------------------------------------------------------------------------
# Filtering + dedup
# ---------------------------------------------------------------------------


def filter_episodes(
    episodes: Iterable[Dict[str, Any]],
    *,
    require_passed: bool = True,
    conditions: Optional[List[str]] = None,
    min_response_chars: int = 50,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Apply quality + condition filters.

    Defaults give the canonical "apply-given-skill" SFT v1 corpus:
      - condition = curated only
      - judge passed = True
      - response is non-trivially long (filters out "ANSWER: yes" one-liners
        even if they passed; we want the CoT, not just the conclusion)
    """
    cond_set = set(conditions) if conditions else None
    out: List[Dict[str, Any]] = []
    for ep in episodes:
        if cond_set is not None and ep.get("condition") not in cond_set:
            continue
        if require_passed and not ep.get("passed"):
            continue
        if min_score > 0.0 and float(ep.get("score", 0.0) or 0.0) < min_score:
            continue
        if min_response_chars > 0:
            response = (ep.get("trace", {}) or {}).get("response", "") or ep.get("answer", "")
            if len(response.strip()) < min_response_chars:
                continue
        out.append(ep)
    return out


def dedupe_by_keys(
    episodes: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Keep the highest-scoring episode per dedup key.

    Default key: (task_uid, condition, model). Same triple → same intended
    demonstration; if bench_traced was rerun we keep the best one.
    """
    keys = keys or ["task_uid", "condition", "model"]
    seen: Dict[tuple, Dict[str, Any]] = {}
    for ep in episodes:
        key = tuple(ep.get(k, "") for k in keys)
        cur = seen.get(key)
        if cur is None or float(ep.get("score", 0) or 0) > float(cur.get("score", 0) or 0):
            seen[key] = ep
    return list(seen.values())


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def build_dataset(
    episodes_path: Path,
    out_path: Path,
    *,
    require_passed: bool = True,
    conditions: Optional[List[str]] = None,
    min_score: float = 0.0,
    min_response_chars: int = 50,
    dedup: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Filter → dedup → format → write JSONL. Returns summary stats.

    Output files:
      <out_path>                          one SFT row per line (JSONL)
      <out_path>.parent / summary.json    counts + filter stats
    """
    if conditions is None:
        conditions = ["curated"]

    with episodes_path.open(encoding="utf-8") as fh:
        episodes = [json.loads(l) for l in fh if l.strip()]
    n_total = len(episodes)

    filtered = filter_episodes(
        episodes,
        require_passed=require_passed,
        conditions=conditions,
        min_score=min_score,
        min_response_chars=min_response_chars,
    )
    n_after_filter = len(filtered)

    if dedup:
        filtered = dedupe_by_keys(filtered)
    n_after_dedup = len(filtered)

    rows = [_episode_to_sft_row(ep) for ep in filtered]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- summary stats -----------------------------------------------------
    by_skill = Counter(r["metadata"]["skill_name"] for r in rows)
    by_model = Counter(r["metadata"]["model_source"] for r in rows)
    by_condition = Counter(r["metadata"]["condition"] for r in rows)
    score_buckets = Counter()
    for r in rows:
        s = r["metadata"]["judge_score"]
        if s >= 0.9:
            score_buckets["[0.9, 1.0]"] += 1
        elif s >= 0.7:
            score_buckets["[0.7, 0.9)"] += 1
        elif s >= 0.5:
            score_buckets["[0.5, 0.7)"] += 1
        else:
            score_buckets["[0.0, 0.5)"] += 1

    summary = {
        "n_total_episodes":   n_total,
        "n_after_filter":     n_after_filter,
        "n_after_dedup":      n_after_dedup,
        "n_rows":             len(rows),
        "filter_kept_pct":    round(n_after_filter / n_total * 100, 1) if n_total else 0.0,
        "filters_applied": {
            "require_passed":     require_passed,
            "conditions":         conditions,
            "min_score":          min_score,
            "min_response_chars": min_response_chars,
            "dedup":              dedup,
        },
        "rows_per_skill":     dict(by_skill),
        "rows_per_model":     dict(by_model),
        "rows_per_condition": dict(by_condition),
        "rows_per_score_bucket": dict(score_buckets),
    }

    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if verbose:
        print(f"Total episodes:    {n_total}")
        print(f"After filter:      {n_after_filter}")
        print(f"After dedup:       {n_after_dedup}")
        print(f"SFT rows written:  {len(rows)}")
        print(f"By skill:          {dict(by_skill)}")
        print(f"By model:          {dict(by_model)}")
        print(f"By condition:      {dict(by_condition)}")
        print(f"By score bucket:   {dict(score_buckets)}")
        print(f"-> {out_path}")
        print(f"   {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--episodes", type=Path, required=True,
                        help="Path to bench_traced/episodes.jsonl")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL path (one SFT row per line)")
    parser.add_argument("--conditions", default="curated",
                        help="Comma-separated conditions to keep "
                             "(default: curated; use 'curated,baseline' for both)")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include episodes where judge=failed (default: passed only)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Drop rows with judge_score below this (default: 0.0)")
    parser.add_argument("--min-response-chars", type=int, default=50,
                        help="Drop rows with response shorter than this many chars "
                             "(default: 50; filters one-line ANSWER-only responses)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip dedup by (task_uid, condition, model)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    summary = build_dataset(
        episodes_path=args.episodes,
        out_path=args.out,
        require_passed=not args.include_failed,
        conditions=conditions,
        min_score=args.min_score,
        min_response_chars=args.min_response_chars,
        dedup=not args.no_dedup,
        verbose=args.verbose,
    )

    print(
        f"\nWrote {summary['n_rows']} SFT rows -> {args.out}\n"
        f"Summary: {args.out.parent}/summary.json"
    )


if __name__ == "__main__":
    main()
