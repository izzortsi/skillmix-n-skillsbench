"""
b2_benchmarks.skillsbench.llm_judge

LLM-as-judge evaluator for reading-comprehension responses. Scores against
ExtractedTask.acceptance_criteria:
  - FREE_FORM:
        The judge returns BOTH a continuous `score` (0.0..1.0, rubric-weighted
        across must_identify + conclusion) AND a binary `conclusion_reached`.
        `passed` tracks conclusion_reached, NOT score >= 0.6 — we don't want to
        count a response that reaches the right answer with weak rationale as
        "failed" in the baseline/uplift heatmaps. Partial-credit nuance is
        available separately in the `score` field (see `score_heatmap`).
  - YES_NO / YES_NO_VERIFICATION / SINGLE_WORD / RANKING:
        deterministic exact-match, pass iff match

Adapted from skillsuite/llm-skills.skillsbench-evaluation/c2_evaluation/llm_judge.py,
rewired onto core.providers' ChatResult interface (`.text` field, not
`.message["content"]`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict


JUDGE_PROMPT_TEMPLATE = """You are a precise evaluator for a reading comprehension benchmark.

## Task Passage
{passage}

## Challenge
{challenge}

## Acceptance Criteria

Must identify:
{must_identify_formatted}

Correct conclusion: {correct_conclusion}

## Model Response
{response}

## Instructions

Evaluate whether the model's response adequately addresses the challenge.

1. For each "must identify" item, determine whether the response identifies or
   addresses it (even with different wording/synonyms).
2. Determine whether the response reaches the correct conclusion (even if
   phrased differently).

Return ONLY valid JSON (no markdown, no explanation):

{{
  "score": <float 0.0 to 1.0, rubric credit across must-identify items and conclusion>,
  "conclusion_reached": <true iff the response's final answer matches the correct conclusion, even if phrased differently>,
  "passed": <MUST equal conclusion_reached>,
  "criteria_met": <number of must-identify items addressed>,
  "criteria_total": <total must-identify items>,
  "rationale": "<1-2 sentence explanation>"
}}

Important: a response that reaches the correct conclusion with weak or
incomplete supporting rationale should have `conclusion_reached=true` and
`passed=true`, even if `score` is low. `score` captures rubric completeness;
`passed` captures whether the answer itself is correct.
"""


@dataclass
class JudgeResult:
    passed: bool
    score: float
    rationale: str
    criteria_met: int = 0
    criteria_total: int = 0
    conclusion_reached: bool = False
    raw_response: str = ""


class LLMJudgeEvaluator:
    """Score model responses against ExtractedTask acceptance criteria."""

    def __init__(self, provider):
        """`provider` must expose `.chat(messages) -> ChatResult(text=..., ...)`
        — any object that matches core.providers.AnthropicProvider / MockProvider."""
        self._provider = provider

    def evaluate(
        self,
        response: str,
        passage: str,
        challenge: str,
        acceptance_criteria: Dict[str, Any],
        query_type: str = "FREE_FORM",
    ) -> JudgeResult:
        """Grade `response`. Uses deterministic match for non-FREE_FORM query types."""
        if not acceptance_criteria:
            return JudgeResult(passed=True, score=1.0, rationale="no criteria to evaluate")

        if query_type in ("YES_NO", "YES_NO_VERIFICATION", "SINGLE_WORD", "RANKING"):
            return _score_deterministic(response, acceptance_criteria, query_type)

        must_identify = acceptance_criteria.get("must_identify", []) or []
        correct_conclusion = acceptance_criteria.get("correct_conclusion", "")

        must_identify_formatted = "\n".join(f"- {item}" for item in must_identify) or "(none specified)"

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            passage=passage,
            challenge=challenge,
            must_identify_formatted=must_identify_formatted,
            correct_conclusion=correct_conclusion or "(none specified)",
            response=response,
        )

        try:
            result = self._provider.chat([{"role": "user", "content": prompt}])
            return _parse_judge_response(getattr(result, "text", ""))
        except Exception as e:
            return JudgeResult(
                passed=False, score=0.0,
                rationale=f"judge evaluation failed: {e}",
                raw_response="",
            )


def _parse_judge_response(response_text: str) -> JudgeResult:
    """Parse a JSON judgment from a model response; tolerates fences."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()

    try:
        data = json.loads(text)
        # Primary signal is conclusion_reached; fall back to passed when absent
        # (older judge responses). Finally, treat score >= 0.6 as a last resort.
        score = float(data.get("score", 0.0))
        if "conclusion_reached" in data:
            conclusion_reached = bool(data["conclusion_reached"])
        elif "passed" in data:
            conclusion_reached = bool(data["passed"])
        else:
            conclusion_reached = score >= 0.6
        return JudgeResult(
            passed=conclusion_reached,
            score=score,
            rationale=str(data.get("rationale", "")),
            criteria_met=int(data.get("criteria_met", 0)),
            criteria_total=int(data.get("criteria_total", 0)),
            conclusion_reached=conclusion_reached,
            raw_response=response_text,
        )
    except (json.JSONDecodeError, ValueError):
        return JudgeResult(
            passed=False, score=0.0,
            rationale=f"failed to parse judge response: {response_text[:200]}",
            conclusion_reached=False,
            raw_response=response_text,
        )


_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_MARKER_RE = re.compile(
    r"(?:final answer|answer|conclusion|verdict)\s*(?:is|=|[:\-—])\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_MD_STRIP_RE = re.compile(r"^[\s\*_`#>\-]+|[\s\*_`'\"\.]+$")


def _normalize_answer(s: str) -> str:
    """Strip markdown formatting + trailing punctuation from an extracted answer."""
    return _MD_STRIP_RE.sub("", s).strip()


def _extract_answer_line(response: str) -> str:
    """Extract the conclusion from a free-form response.

    Order of preference:
      1. Last 'ANSWER:' line (explicit format from the s4.m4 solver prompt).
      2. Last 'Final answer:' / 'Conclusion:' / 'Verdict:' marker line.
      3. Last non-empty line, with markdown formatting stripped.

    Without normalization in (3), responses like '**yes**' fail exact-match
    against expected 'yes' — biasing the deterministic judge against models
    that produce richly-formatted prose.
    """
    matches = _ANSWER_RE.findall(response)
    if matches:
        return _normalize_answer(matches[-1])
    matches = _MARKER_RE.findall(response)
    if matches:
        return _normalize_answer(matches[-1])
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    return _normalize_answer(lines[-1]) if lines else ""


def _score_deterministic(
    response: str, acceptance_criteria: Dict[str, Any], query_type: str,
) -> JudgeResult:
    """Exact-match scoring for deterministic query types."""
    expected = str(acceptance_criteria.get("correct_conclusion", "")).strip().lower()
    answer = _extract_answer_line(response).lower()

    if query_type == "YES_NO":
        is_match = answer in ("yes", "no") and answer == expected
    elif query_type == "YES_NO_VERIFICATION":
        is_match = answer in ("correct", "incorrect") and answer == expected
    elif query_type == "SINGLE_WORD":
        first_word = answer.split()[0] if answer.split() else ""
        is_match = first_word == expected or answer == expected
    elif query_type == "RANKING":
        is_match = answer in ("a", "b") and answer == expected
    else:
        is_match = False

    score = 1.0 if is_match else 0.0
    rationale = (
        f"exact match: expected {expected!r}, got {answer!r}"
        if is_match
        else f"no match: expected {expected!r}, got {answer!r}"
    )
    return JudgeResult(
        passed=is_match,
        score=score,
        rationale=rationale,
        criteria_met=1 if is_match else 0,
        criteria_total=1,
        conclusion_reached=is_match,
    )
