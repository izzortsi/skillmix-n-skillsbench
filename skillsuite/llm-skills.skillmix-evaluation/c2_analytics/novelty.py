"""
novelty.py

PROJECT_SPECS section 2 Method 2: "Use probability calculations based on
estimated training data size to verify novelty". Given a k-skill tuple
and per-skill rarity estimates, compute the expected number of times
the tuple co-occurs in a training corpus and decide whether the
combination is "likely novel" (expected co-occurrences < 1).

The math is deliberately simple and well-documented: per-skill
frequency -> per-example co-occurrence probability -> expected
co-occurrences over a corpus of N examples. Defensible as a rough
estimate, not as a published claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List


# Rarity tiers assigned by the source/method of each ExtractedSkill.
# Values are base-10 logarithms of per-token frequency in a general
# web/literature corpus. Looser bounds than word-level word frequency
# because these are procedural-skill phrases, not single words.
#
#   TIER_NOVEL:     deliberately unnamed (PROJECT_SPECS 1.M2) -> ~1 in 1 billion tokens
#   TIER_CATALOG:   procedural snake_case (PROJECT_SPECS 1.M3) -> ~1 in 10 million tokens
#   TIER_WIKIPEDIA: established Wikipedia concepts             -> ~1 in 100 thousand tokens
#   TIER_UNKNOWN:   provenance unknown                         -> midway between cat/wiki
TIER_LOG10_FREQ_PER_TOKEN = {
    "novel": -9.0,
    "catalog": -7.0,
    "wikipedia": -5.0,
    "unknown": -6.0,
}


def classify_tier(skill: Dict[str, Any]) -> str:
    method = (skill.get("extraction_method") or "").lower()
    category = (skill.get("category") or "").lower()

    if method.startswith("frontier-novel") or category == "novel-root":
        return "novel"
    if method.startswith("frontier-catalog"):
        return "catalog"
    if "wikipedia" in method or "wikipedia" in category:
        return "wikipedia"
    return "unknown"


def estimate_novelty(
    tier_labels: List[str],
    corpus_examples: float = 1e10,
    tokens_per_example: float = 1000.0,
) -> Dict[str, Any]:
    if not tier_labels:
        return {
            "k": 0,
            "tier_labels": [],
            "log10_expected_cooccurrences": float("-inf"),
            "expected_cooccurrences": 0.0,
            "likely_novel": True,
        }

    log10_p_per_example = 0.0
    for tier in tier_labels:
        log10_freq = TIER_LOG10_FREQ_PER_TOKEN.get(tier, TIER_LOG10_FREQ_PER_TOKEN["unknown"])
        freq_per_token = 10.0 ** log10_freq
        p_skill_in_example = min(1.0, freq_per_token * tokens_per_example)
        if p_skill_in_example <= 0.0:
            return {
                "k": len(tier_labels),
                "tier_labels": tier_labels,
                "log10_expected_cooccurrences": float("-inf"),
                "expected_cooccurrences": 0.0,
                "likely_novel": True,
            }
        log10_p_per_example += math.log10(p_skill_in_example)

    log10_expected = math.log10(corpus_examples) + log10_p_per_example
    expected = 10.0 ** log10_expected
    return {
        "k": len(tier_labels),
        "tier_labels": tier_labels,
        "log10_expected_cooccurrences": round(log10_expected, 3),
        "expected_cooccurrences": expected,
        "likely_novel": expected < 1.0,
    }


def score_task(
    task_skill_uids: List[str],
    skill_by_uid: Dict[str, Dict[str, Any]],
    corpus_examples: float = 1e10,
    tokens_per_example: float = 1000.0,
) -> Dict[str, Any]:
    tiers: List[str] = []
    resolved_names: List[str] = []
    for uid in task_skill_uids:
        sk = skill_by_uid.get(uid)
        if sk is None:
            tiers.append("unknown")
            resolved_names.append(f"(unresolved: {uid})")
            continue
        tiers.append(classify_tier(sk))
        resolved_names.append(sk.get("name", uid))

    estimate = estimate_novelty(
        tiers, corpus_examples=corpus_examples, tokens_per_example=tokens_per_example
    )
    estimate["skill_names"] = resolved_names
    estimate["skill_uids"] = task_skill_uids
    return estimate


def score_tasks_file(
    tasks_path: Path,
    skills_path: Path,
    output_path: Path,
    corpus_examples: float = 1e10,
    tokens_per_example: float = 1000.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    skills = json.loads(skills_path.read_text(encoding="utf-8"))
    skill_by_uid = {s["skill_uid"]: s for s in skills}

    rows: List[Dict[str, Any]] = []
    n_novel = 0
    for task in tasks:
        ac = task.get("acceptance_criteria", {}) or {}
        uids = ac.get("skill_uids") or ([ac.get("skill_uid")] if ac.get("skill_uid") else [])
        uids = [u for u in uids if u]
        if not uids:
            continue
        row = score_task(
            uids, skill_by_uid,
            corpus_examples=corpus_examples,
            tokens_per_example=tokens_per_example,
        )
        row["task_uid"] = task.get("task_uid", "")
        row["task_title"] = task.get("title", "")
        rows.append(row)
        if row["likely_novel"]:
            n_novel += 1

    report = {
        "tasks_scored": len(rows),
        "likely_novel": n_novel,
        "novelty_rate": (n_novel / len(rows)) if rows else 0.0,
        "corpus_examples": corpus_examples,
        "tokens_per_example": tokens_per_example,
        "per_task": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"Scored {len(rows)} tasks, {n_novel} likely novel "
              f"({report['novelty_rate']:.1%}). Wrote {output_path}")
    return report


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Score novelty of k-skill combinations in a tasks.json (PROJECT_SPECS 2.M2).")
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--skills", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--corpus-examples", type=float, default=1e10)
    ap.add_argument("--tokens-per-example", type=float, default=1000.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    score_tasks_file(
        tasks_path=args.tasks, skills_path=args.skills,
        output_path=args.output,
        corpus_examples=args.corpus_examples,
        tokens_per_example=args.tokens_per_example,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
