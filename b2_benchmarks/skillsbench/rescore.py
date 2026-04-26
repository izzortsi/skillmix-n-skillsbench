"""
b2_benchmarks.skillsbench.rescore

Post-hoc re-derive the `passed` flag for an existing `episodes.json` using the
new semantics (passed == conclusion_reached), without re-running bench.

Two strategies, in order of trust:

  1. lexical match: does `response` contain the expected `correct_conclusion`
     substring (case-insensitive, whitespace-normalized)? Works well when the
     expected answer is a short phrase like "no", "north", "47%", "promise".

  2. judge rationale parse: scan `judge_rationale` for phrases indicating the
     model reached the correct conclusion ("reaches the correct conclusion",
     "arrives at the correct", "correctly identifies", ...). Fallback when
     the lexical path is ambiguous (e.g., expected answer is a common word).

If neither heuristic is confident, the original `passed` is kept unchanged.

A full re-judge (actually calling Opus again) would be more accurate but
costs money; this tool is the zero-cost path. Re-run the bench for future
runs to get the updated judge semantics directly.

Usage:
    python -m b2_benchmarks.skillsbench.rescore \\
        --episodes data/pipeline-runs/default/bench/episodes.json \\
        --tasks    data/mini-tasks.json \\
        --out      data/pipeline-runs/default/bench/episodes.rescored.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_RATIONALE_PATTERNS = [
    r"reach(?:es|ed)\s+the\s+correct\s+conclusion",
    r"arriv(?:es|ed)\s+at\s+the\s+correct",
    r"correctly\s+identif(?:ies|ied)",
    r"correctly\s+conclud(?:es|ed)",
    r"identif(?:ies|ied)\s+.*?\s+as\s+the\s+correct",
    r"matches\s+the\s+expected",
]
_NEGATION_PATTERNS = [
    r"does\s+not\s+reach\s+the\s+correct",
    r"fails?\s+to\s+reach\s+the\s+correct",
    r"fails?\s+to\s+identify",
    r"wrong\s+conclusion",
    r"incorrect\s+conclusion",
    r"misinterprets?",
]


def _norm(s: str) -> str:
    """Collapse whitespace + lowercase for tolerant substring matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _lexical_match(response: str, expected: str) -> bool:
    """True iff expected (as a phrase) appears in response after normalization.

    Expected is short (<= 40 chars) and not a trivial token like single letters.
    """
    resp = _norm(response)
    exp = _norm(expected)
    if not exp or not resp:
        return False
    # Skip overly short expectations to avoid spurious hits (e.g. "a" inside "abandoned").
    if len(exp) < 2:
        return False
    return exp in resp


def _rationale_verdict(rationale: str) -> Tuple[bool, bool]:
    """Return (confident_pass, confident_fail) from judge_rationale text."""
    r = _norm(rationale)
    if not r:
        return False, False
    pass_hit = any(re.search(p, r) for p in _RATIONALE_PATTERNS)
    fail_hit = any(re.search(p, r) for p in _NEGATION_PATTERNS)
    # If a negation and a positive both hit, prefer negation (judge explicitly said "fails to reach").
    if fail_hit:
        return False, True
    if pass_hit:
        return True, False
    return False, False


def rescore_episodes(
    episodes: List[Dict[str, Any]],
    expected_by_task: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return (updated_episodes, stats). Mutates episodes in place shallowly."""
    stats = {
        "total": len(episodes),
        "unchanged": 0,
        "flipped_to_pass_lexical": 0,
        "flipped_to_pass_rationale": 0,
        "flipped_to_fail_rationale": 0,
        "missing_expected": 0,
    }

    for ep in episodes:
        original_passed = bool(ep.get("passed", False))
        expected = expected_by_task.get(ep.get("task_uid", ""), "")
        rationale = ep.get("judge_rationale", "") or ""
        response = ep.get("response", "") or ""

        if not expected:
            stats["missing_expected"] += 1
            ep["conclusion_reached"] = original_passed
            continue

        lex_pass = _lexical_match(response, expected)
        rat_pass, rat_fail = _rationale_verdict(rationale)

        # Confidence order: rationale fail > rationale pass > lexical pass > original.
        new_passed = original_passed
        source = "unchanged"
        if rat_fail:
            if original_passed:
                new_passed = False
                source = "flipped_to_fail_rationale"
        elif rat_pass and not original_passed:
            new_passed = True
            source = "flipped_to_pass_rationale"
        elif lex_pass and not original_passed:
            new_passed = True
            source = "flipped_to_pass_lexical"

        ep["passed"] = new_passed
        ep["conclusion_reached"] = new_passed
        if source != "unchanged":
            stats[source] += 1
        else:
            stats["unchanged"] += 1

    return episodes, stats


def _load_expected_by_task(tasks_path: Path) -> Dict[str, str]:
    with open(tasks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for t in raw:
        uid = t.get("task_uid") or t.get("task_id") or ""
        expected = t.get("output") or (t.get("acceptance_criteria") or {}).get("correct_conclusion") or ""
        if uid:
            out[uid] = str(expected)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--episodes", type=Path, required=True,
                        help="Existing episodes.json written by bench")
    parser.add_argument("--tasks", type=Path, required=True,
                        help="ExtractedTask JSON with expected answers (task.output)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Default: overwrites the input episodes.json in place.")
    args = parser.parse_args()

    with open(args.episodes, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    expected_by_task = _load_expected_by_task(args.tasks)
    print(f"loaded {len(episodes)} episodes; expected answers for {len(expected_by_task)} tasks")

    updated, stats = rescore_episodes(episodes, expected_by_task)
    out_path = args.out or args.episodes
    out_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {len(updated)} episodes -> {out_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
