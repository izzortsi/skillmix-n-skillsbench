"""
s3.m4 — Agentic Answer Verification (PROJECT_SPECS §3 Method 4).

Take SkillExamples produced by s3.m1 / s3.m2 / s3.m3 and loop a frontier LLM
agentically to verify and refine each answer, ensuring it correctly
demonstrates every listed skill.

Loop for each example:
    1. Verifier judges whether `answer` exhibits every skill in `skill_uids`.
    2. If any skill is not exhibited, ask a refiner to produce a revised
       `answer`; keep the same question.
    3. Repeat up to max_rounds.

Mark `example.verified=True` only when the verifier signs off on all skills.
Failed examples carry `verified=False` with the final verdict attached.

Ported from skillsuite/llm-skills.extraction-pipeline/c2_extraction/
    frontier_verifier.py  (verify_tasks / verify_one)
    frontier_reviser.py   (revise_tasks / revise_one)
with the semantic adjustment that we verify an ANSWER against skills, not a
TASK's requirement of skills (the frontier versions operated on
ExtractedTask; here we operate on SkillExample).

Output:
    verified-examples.json    accepted SkillExamples (verified=True)
    rejected-examples.jsonl   final verdicts for examples that never passed
    verify-transcripts.jsonl  per-example verification trail across rounds
    summary.json              counts + rounds taken

Usage:
    python -m cli s3.m4 \\
        --examples data/s3-m1-examples.json \\
        --catalog  data/wikipedia-seed/skills.json \\
        --out-dir  data/s3-m4-out \\
        --verifier anthropic:claude-opus-4-7 \\
        --refiner  anthropic:claude-sonnet-4-6 \\
        --max-rounds 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import (
    Skill,
    SkillExample,
    load_examples,
    load_skills,
    save_json,
)
from core.providers import create_provider


SOURCE_ID = "s3.m4.agentic-verified"
DEFAULT_MAX_ROUNDS = 2


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


VERIFIER_SYSTEM_PROMPT = (
    "You are a strict judge. You check whether a generated answer EXHIBITS "
    "every listed skill (in its technical sense). Decorative, colloquial, or "
    "name-only uses do not count."
)


VERIFY_PROMPT = """Verify that the answer below EXHIBITS every listed skill.

Required skills:
{skills_block}

Question:
{question}

Answer:
{answer}

For each skill, decide whether the ANSWER demonstrates the skill in its
technical sense. If a skill is only named, colloquially alluded to, or not
exercised at all, mark exhibits=false.

Return ONLY valid JSON:

{{
  "per_skill": [
    {{
      "skill_name": "<name exactly as given>",
      "exhibits":   true|false,
      "rationale":  "<one sentence>"
    }}
  ],
  "overall": {{
    "exhibits_all": true|false,
    "summary": "<one sentence>"
  }}
}}
"""


REFINER_SYSTEM_PROMPT = (
    "You revise an answer that failed verification. You read the judge's "
    "per-skill rationale and produce a revised answer that exhibits every "
    "required skill in its technical sense."
)


REFINE_PROMPT = """Revise the previous answer so it exhibits every required skill.

Required skills:
{skills_block}

Question (unchanged):
{question}

Previous answer (failed):
{answer}

Judge's verdict:
  Overall: {overall_summary}
  Per-skill findings:
{per_skill_block}

Produce a REVISED ANSWER that:
  - Directly addresses every "exhibits: false" rationale above.
  - Uses each skill in its TECHNICAL sense (not colloquial).
  - Does NOT explicitly name any of the skills.
  - Keeps the same question.
  - Is substantive (a few sentences to a short paragraph).

Return ONLY valid JSON with no markdown fences and no preface:

{{
  "answer": "<the revised answer>"
}}
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return text.strip()


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_markdown_fences(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Skill block rendering
# ---------------------------------------------------------------------------


def _resolve_skills(skill_uids: List[str], catalog: List[Skill]) -> List[Skill]:
    """Look up Skills by uid; skip uids not in the catalog (caller warns)."""
    by_uid = {s.skill_uid: s for s in catalog}
    return [by_uid[u] for u in skill_uids if u in by_uid]


def _render_skills_block(skills: List[Skill]) -> str:
    lines: List[str] = []
    for i, s in enumerate(skills, 1):
        cat = s.category or "unspecified"
        lines.append(f"Skill {i}: {s.name} ({cat})")
        if s.description:
            lines.append(f"  Definition: {s.description}")
        if s.example:
            lines.append(f"  Example:    {s.example}")
    return "\n".join(lines)


def _render_per_skill(verdict: Dict[str, Any]) -> str:
    per = verdict.get("per_skill", []) or []
    lines = []
    for item in per:
        if not isinstance(item, dict):
            continue
        name = item.get("skill_name", "?")
        exhibits = bool(item.get("exhibits", False))
        rationale = item.get("rationale", "")
        lines.append(f"  - {name}: exhibits={exhibits}. {rationale}")
    return "\n".join(lines) if lines else "  (no per-skill findings)"


# ---------------------------------------------------------------------------
# Single-example verify + refine
# ---------------------------------------------------------------------------


def _verify(example: SkillExample, skills: List[Skill], verifier) -> Tuple[bool, Dict[str, Any]]:
    """Returns (exhibits_all, verdict). verdict always has 'per_skill'+'overall' or an 'error'."""
    prompt = VERIFY_PROMPT.format(
        skills_block=_render_skills_block(skills),
        question=example.question,
        answer=example.answer,
    )
    result = verifier.chat(
        [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = _parse_json_object(result.text)
    if parsed is None:
        return False, {"error": "json_parse_failed", "final_text": result.text}

    per_skill = parsed.get("per_skill", []) or []
    overall = parsed.get("overall", {}) or {}
    exhibits_all = bool(overall.get("exhibits_all", False))
    if per_skill:
        exhibits_all = exhibits_all and all(
            bool(s.get("exhibits", False)) for s in per_skill
        )
    return exhibits_all, {
        "per_skill": per_skill,
        "overall": overall,
        "exhibits_all": exhibits_all,
        "usage": dict(result.usage),
    }


def _refine(
    example: SkillExample, skills: List[Skill], verdict: Dict[str, Any], refiner
) -> Optional[str]:
    """Ask refiner for a new answer; return revised text, or None on parse failure."""
    prompt = REFINE_PROMPT.format(
        skills_block=_render_skills_block(skills),
        question=example.question,
        answer=example.answer,
        overall_summary=(verdict.get("overall", {}) or {}).get("summary", ""),
        per_skill_block=_render_per_skill(verdict),
    )
    result = refiner.chat(
        [
            {"role": "system", "content": REFINER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = _parse_json_object(result.text)
    if parsed is None:
        return None
    answer = str(parsed.get("answer", "")).strip()
    return answer or None


def verify_example(
    example: SkillExample,
    skills: List[Skill],
    verifier,
    refiner,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Tuple[SkillExample, List[Dict[str, Any]]]:
    """Verify + optionally refine one example.

    Returns (updated_example, rounds) where rounds is the per-round log
    (verdict + refined_answer per round). updated_example.verified is True iff
    the final verdict had exhibits_all=True.
    """
    rounds_log: List[Dict[str, Any]] = []
    current = example

    for round_idx in range(1, max_rounds + 1):
        passed, verdict = _verify(current, skills, verifier)
        rounds_log.append({"round": round_idx, "verdict": verdict, "answer": current.answer})
        if passed:
            updated = SkillExample(
                example_uid=current.example_uid,
                skill_uids=current.skill_uids,
                topic_uid=current.topic_uid,
                question=current.question,
                answer=current.answer,
                verified=True,
                source=SOURCE_ID,
            )
            return updated, rounds_log

        if round_idx == max_rounds:
            break

        refined = _refine(current, skills, verdict, refiner)
        if refined is None or refined == current.answer:
            break
        current = SkillExample(
            example_uid=current.example_uid,
            skill_uids=current.skill_uids,
            topic_uid=current.topic_uid,
            question=current.question,
            answer=refined,
            verified=False,
            source=SOURCE_ID,
        )

    return current, rounds_log


# ---------------------------------------------------------------------------
# Dataset-level driver
# ---------------------------------------------------------------------------


def verify_dataset(
    examples: List[SkillExample],
    catalog: List[Skill],
    verifier,
    refiner,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    verbose: bool = False,
) -> Tuple[List[SkillExample], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Verify + refine every example. Returns (verified, rejected_records, transcripts).

    `rejected_records` entries carry the final verdict. `transcripts` contains
    one per-example log suitable for JSONL persistence.
    """
    verified: List[SkillExample] = []
    rejected_records: List[Dict[str, Any]] = []
    transcripts: List[Dict[str, Any]] = []

    for i, ex in enumerate(examples, 1):
        skills = _resolve_skills(ex.skill_uids, catalog)
        if not skills:
            if verbose:
                print(f"[{i}/{len(examples)}] {ex.example_uid}: SKIP (no skills resolved)")
            rejected_records.append({
                "example_uid": ex.example_uid,
                "error": "no_skills_resolved_from_catalog",
                "skill_uids": ex.skill_uids,
            })
            continue

        if verbose:
            names = [s.name for s in skills]
            print(f"[{i}/{len(examples)}] {ex.example_uid} k={len(skills)} skills={names}")

        try:
            updated, rounds_log = verify_example(
                ex, skills, verifier, refiner, max_rounds=max_rounds,
            )
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            rejected_records.append({
                "example_uid": ex.example_uid,
                "error": f"exception:{e}",
            })
            continue

        transcripts.append({
            "example_uid": ex.example_uid,
            "rounds": rounds_log,
            "final_verified": updated.verified,
        })

        if updated.verified:
            verified.append(updated)
            if verbose:
                print(f"  PASS in {len(rounds_log)} round(s)")
        else:
            rejected_records.append({
                "example_uid": ex.example_uid,
                "final_verdict": rounds_log[-1]["verdict"] if rounds_log else {},
                "rounds": len(rounds_log),
            })
            if verbose:
                last = rounds_log[-1]["verdict"] if rounds_log else {}
                summary = (last.get("overall") or {}).get("summary", last.get("error", ""))
                print(f"  REJECT after {len(rounds_log)} round(s): {str(summary)[:120]}")

    return verified, rejected_records, transcripts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--examples", type=Path, required=True,
                        help="SkillExample JSON produced by s3.m1 / s3.m2 / s3.m3")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="Skill catalog JSON (for skill_uid -> Skill lookup)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/s3-m4-out"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N examples (0 = all)")
    parser.add_argument("--verifier", default="anthropic:claude-opus-4-7",
                        help="Verifier provider spec")
    parser.add_argument("--refiner", default="anthropic:claude-opus-4-7",
                        help="Refiner provider spec (may equal verifier)")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                        help=f"Max verify/refine rounds per example (default: {DEFAULT_MAX_ROUNDS})")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    examples = load_examples(args.examples)
    if args.limit > 0:
        examples = examples[: args.limit]
    catalog = load_skills(args.catalog)
    print(f"Loaded {len(examples)} examples and {len(catalog)} catalog skills")

    v_name, v_model = _parse_provider_spec(args.verifier)
    r_name, r_model = _parse_provider_spec(args.refiner)
    verifier = create_provider(v_name, v_model)
    refiner = create_provider(r_name, r_model)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    verified, rejected, transcripts = verify_dataset(
        examples, catalog, verifier, refiner,
        max_rounds=args.max_rounds, verbose=args.verbose,
    )

    save_json(verified, args.out_dir / "verified-examples.json")
    with (args.out_dir / "rejected-examples.jsonl").open("w", encoding="utf-8") as fh:
        for rec in rejected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (args.out_dir / "verify-transcripts.jsonl").open("w", encoding="utf-8") as fh:
        for rec in transcripts:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "examples_attempted": len(examples),
        "verified": len(verified),
        "rejected": len(rejected),
        "max_rounds": args.max_rounds,
        "verifier_model": v_model,
        "refiner_model": r_model,
        "source": SOURCE_ID,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nverified={len(verified)}  rejected={len(rejected)}  -> {args.out_dir}/")


if __name__ == "__main__":
    main()
