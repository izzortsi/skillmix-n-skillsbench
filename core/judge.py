"""
core/judge.py

Two judge functions, named for what they grade — no "SkillMix" ambiguity:

  - per_skill_rubric_judge:  Yu et al. Skill-Mix k+3 rubric
                             (each skill illustrated + on_topic + coherent + <=N sentences)
  - task_acceptance_judge:   reading-comprehension style
                             (must_identify items + correct_conclusion)

Both return a dict-shaped result; the caller decides which fields to persist.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from core.schemas import Skill


# ---------------------------------------------------------------------------
# Yu et al. Skill-Mix rubric (paper, Appendix C.3)
# ---------------------------------------------------------------------------


RUBRIC_PROMPT = """You are grading a student's answer on a skill-composition task.

## Task
The student was asked:

> Give a single piece of text with up to {sentence_limit} sentences in the context
> of {topic}. This piece of text should illustrate all of the following skills:
> {skill_names}.

## Skill definitions
{skill_defs_block}

## Student's answer
{response}

## Grading criteria

Award 1 point or 0 points for each:
{rubric_items}

Deduct 1 point if the student explicitly names any skill in the text.
Deduct 1 point if the student uses a skill only in its colloquial English sense
(rather than the technical definition above).

Return ONLY valid JSON, no markdown:

{{
  "per_skill": {{{per_skill_keys}}},
  "on_topic": <0 or 1>,
  "coherent": <0 or 1>,
  "within_sentence_limit": <0 or 1>,
  "explicit_skill_name_penalty": <0 or 1>,
  "colloquial_use_penalty": <0 or 1>,
  "rationale": "<two sentences>"
}}
"""


def per_skill_rubric_judge(
    response: str,
    skills: List[Skill],
    topic_name: str,
    sentence_limit: int,
    judge_provider,
) -> Dict[str, Any]:
    """Grade a SkillMix trial against the k+3 rubric.

    Returns a dict with per-skill points, on_topic, coherent, within_sentence_limit,
    two penalty flags, total score, and rationale. Total = sum(per_skill) +
    on_topic + coherent + within_sentence_limit - penalties.
    """
    skill_names = ", ".join(s.name for s in skills)
    skill_defs_block = "\n".join(
        f"- {s.name}: {s.description}" + (f" Example: {s.example}" if s.example else "")
        for s in skills
    )
    rubric_items = "\n".join(
        [f"- Skill {i+1} ({s.name}) is illustrated" for i, s in enumerate(skills)]
        + ["- Text is on topic", "- Text is coherent / makes sense",
           f"- Text is within {sentence_limit} sentences"]
    )
    per_skill_keys = ", ".join(f'"skill_{i+1}": <0 or 1>' for i in range(len(skills)))

    prompt = RUBRIC_PROMPT.format(
        sentence_limit=sentence_limit,
        topic=topic_name,
        skill_names=skill_names,
        skill_defs_block=skill_defs_block,
        response=response,
        rubric_items=rubric_items,
        per_skill_keys=per_skill_keys,
    )

    result = judge_provider.chat([{"role": "user", "content": prompt}], temperature=0.0)
    data = _parse_json_response(result.text)

    per_skill = data.get("per_skill", {})
    skill_sum = sum(int(v) for v in per_skill.values())
    on_topic = int(data.get("on_topic", 0))
    coherent = int(data.get("coherent", 0))
    within = int(data.get("within_sentence_limit", 0))
    explicit_pen = int(data.get("explicit_skill_name_penalty", 0))
    colloq_pen = int(data.get("colloquial_use_penalty", 0))

    total_possible = len(skills) + 3
    raw_total = skill_sum + on_topic + coherent + within - explicit_pen - colloq_pen
    raw_total = max(0, raw_total)

    return {
        "per_skill": per_skill,
        "on_topic": on_topic,
        "coherent": coherent,
        "within_sentence_limit": within,
        "explicit_skill_name_penalty": explicit_pen,
        "colloquial_use_penalty": colloq_pen,
        "score": raw_total,
        "total_possible": total_possible,
        "full_points": raw_total == total_possible,
        "rationale": data.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Reading-comprehension task acceptance judge
# ---------------------------------------------------------------------------


TASK_ACCEPTANCE_PROMPT = """You are a precise evaluator for a reading-comprehension response.

## Passage
{passage}

## Challenge
{challenge}

## Acceptance criteria

Must identify:
{must_identify_block}

Correct conclusion: {correct_conclusion}

## Model response
{response}

Return ONLY valid JSON, no markdown:

{{
  "score": <float in [0.0, 1.0]>,
  "passed": <true if score >= 0.6>,
  "criteria_met": <integer>,
  "criteria_total": <integer>,
  "rationale": "<one or two sentences>"
}}
"""


def task_acceptance_judge(
    response: str,
    passage: str,
    challenge: str,
    acceptance_criteria: Dict[str, Any],
    judge_provider,
) -> Dict[str, Any]:
    """Grade a reading-comprehension response against task acceptance criteria."""
    must_identify = acceptance_criteria.get("must_identify", [])
    must_identify_block = "\n".join(f"- {item}" for item in must_identify) or "(none)"
    correct_conclusion = acceptance_criteria.get("correct_conclusion", "")

    prompt = TASK_ACCEPTANCE_PROMPT.format(
        passage=passage,
        challenge=challenge,
        must_identify_block=must_identify_block,
        correct_conclusion=correct_conclusion,
        response=response,
    )
    result = judge_provider.chat([{"role": "user", "content": prompt}], temperature=0.0)
    return _parse_json_response(result.text)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response.

    The judge prompt asks for JSON-only but models sometimes wrap it in fences or
    prose; this tolerates both.
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(1))
    return {}
