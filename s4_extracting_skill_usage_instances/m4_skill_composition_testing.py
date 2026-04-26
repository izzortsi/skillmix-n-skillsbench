"""
s4.m4 — Skill Composition Testing, eval half (PROJECT_SPECS §4 Method 4).

§4.m4 has two halves:
  Training half (OOS): train on k-tuples drawn from train_skills, evaluate on
      k'-tuples drawn (at least partly) from holdout_skills. Requires SFT
      infrastructure we don't have here.
  Eval half (this module): given ExtractedTasks targeting held-out k'-tuples,
      solve them with a frontier model, and record exact-match verification
      per the spec: "Use exact-match verification for objective evaluation."

Ported from skillsuite/llm-skills.extraction-pipeline/c2_extraction/
    skill_solver.py          (solve_tasks + _skills_label + _primary_skill_metadata)
    frontier_trace_adapter.py (adapt_one + _split_into_steps + _concat_thinking)

Uses harness.LinearAgent (not core.providers) so native thinking blocks are
captured per turn — those are what populate ReasoningTrace.thinking for any
downstream analysis.

Typical pipeline:
  1. split the catalog with `partition_catalog` (by category or random fraction).
  2. s3.m1 on the holdout_skills catalog with --k {3,4,5} to generate tasks.
  3. s4.m4 solves the holdout tasks and emits {solutions.jsonl, traces.jsonl,
     summary.json} with pass rate.

Output:
    solutions.jsonl        per-task transcript + answer + matches_expected flag
    traces.jsonl           ReasoningTrace records (downstream-friendly)
    summary.json           tasks_attempted, pass_rate, per-k breakdown, model

Usage:
    python -m cli s4.m4 \\
        --tasks data/s3-m1-holdout-k3.json \\
        --out-dir data/s4-m4-eval-k3 \\
        --model claude-opus-4-7 --thinking-budget 4096
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import (
    ExtractedTask,
    Skill,
    load_extracted_tasks,
    load_skills,
    save_json,
)
from harness import LinearAgent, Transcript, record

from b2_benchmarks.skillsbench.skill_injection import format_skill_as_system
from b2_benchmarks.skillsbench.llm_judge import LLMJudgeEvaluator, JudgeResult


SOLVING_METHOD = "s4.m4.frontier-solver-v1"
TRACE_METHOD = "s4.m4.frontier-solver-trace-v1"
INJECTION_METHOD = "s4.m4.skill-injection-trace-v1"

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_MAX_TOKENS = 8000


# ---------------------------------------------------------------------------
# ReasoningTrace (kept local to s4.m4; not elevated to core.schemas because
# no other method consumes it yet)
# ---------------------------------------------------------------------------


@dataclass
class ReasoningTrace:
    """Procedural trace captured while solving an ExtractedTask."""

    task_uid: str
    model: str
    system_prompt: str
    user_prompt: str
    response: str
    procedural_steps: List[str]
    conclusion: str
    tokens: int
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking: str = ""
    raw_steps: List[dict] = field(default_factory=list)
    extraction_method: str = ""
    # Skill-injection mode (closes Gap A: COT trace + injected procedural skill).
    # condition = "baseline" (no skill in system) or "curated" (skill injected).
    condition: str = "baseline"
    injected_skill_name: str = ""
    injected_skill_uid: str = ""


# ---------------------------------------------------------------------------
# Catalog partitioning
# ---------------------------------------------------------------------------


def partition_catalog(
    catalog: List[Skill],
    holdout_categories: Optional[List[str]] = None,
    holdout_fraction: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[Skill], List[Skill]]:
    """Split the catalog into (train, holdout) skill lists.

    Two modes (mutually exclusive):
      - holdout_categories: every Skill whose category is in this list goes
        to holdout; the rest go to train.
      - holdout_fraction: a random fraction of skills (by seed) goes to
        holdout; the rest go to train.
    """
    if holdout_categories and holdout_fraction is not None:
        raise ValueError("specify holdout_categories OR holdout_fraction, not both")

    if holdout_categories:
        wanted = {c.strip() for c in holdout_categories if c.strip()}
        train = [s for s in catalog if (s.category or "") not in wanted]
        holdout = [s for s in catalog if (s.category or "") in wanted]
        return train, holdout

    if holdout_fraction is None:
        raise ValueError("specify either holdout_categories or holdout_fraction")
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(f"holdout_fraction must be in (0, 1); got {holdout_fraction}")

    rng = random.Random(seed)
    shuffled = list(catalog)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * holdout_fraction))
    holdout = shuffled[:cut]
    train = shuffled[cut:]
    return train, holdout


# ---------------------------------------------------------------------------
# Solver: LinearAgent-driven one episode per task, captures thinking blocks
# ---------------------------------------------------------------------------


SOLVER_SYSTEM_PROMPT = (
    "You are an expert problem solver. Work through the task step by step in "
    "writing — explain your reasoning explicitly, then state your conclusion. "
    "If a SKILL block appears in the system prompt, follow its procedure "
    "explicitly: as you apply each step, name the step from the procedure you "
    "are applying. End your response with a line beginning exactly 'ANSWER: ' "
    "followed by your final conclusion. The grader compares only that ANSWER "
    "line to a single correct conclusion string."
)


SOLVE_PROMPT = """Task title: {title}
Difficulty: {difficulty}{skills_line}

--- context ---
{input}
--- question ---
{question}

Give your final answer as ONE concise conclusion on the last line,
prefixed exactly with "ANSWER: ".
"""


def _skills_label(task: ExtractedTask) -> str:
    """Format skills as a comma-separated "name (category)" list, handling both
    k=1 (skill_name/skill_uid) and k>=2 (skill_names/skill_uids) shapes in
    acceptance_criteria."""
    ac = task.acceptance_criteria or {}
    names = ac.get("skill_names")
    cats = ac.get("skill_categories")
    if names:
        cats = cats or [""] * len(names)
        return ", ".join(
            f"{n} ({c or 'unspecified'})" for n, c in zip(names, cats)
        )
    name = ac.get("skill_name", "")
    cat = ac.get("skill_category", "")
    if name:
        return f"{name} ({cat or 'unspecified'})"
    return "(unspecified)"


def _primary_skill_metadata(task: ExtractedTask):
    """Return (primary_uid, primary_name, skill_uids, skill_names) handling
    k=1 and k>=2 uniformly for downstream join."""
    ac = task.acceptance_criteria or {}
    uids = ac.get("skill_uids")
    names = ac.get("skill_names")
    if uids and names:
        return uids[0], names[0], list(uids), list(names)
    uid = ac.get("skill_uid", "")
    name = ac.get("skill_name", "")
    return uid, name, [uid] if uid else [], [name] if name else []


def _build_solve_prompt(task: ExtractedTask, include_skill_label: bool = True) -> str:
    """Render the solver user prompt.

    `include_skill_label=False` drops the "Required skills:" line — used when
    a procedural Skill is already injected into the system prompt (curated
    condition), so we don't double-mention skill names.
    """
    skills_line = (
        f"\nRequired skills: {_skills_label(task)}" if include_skill_label else ""
    )
    return SOLVE_PROMPT.format(
        title=task.title,
        difficulty=task.difficulty,
        skills_line=skills_line,
        input=task.input,
        question=task.question,
    )


def _final_text_from_transcript(transcript: Transcript) -> str:
    parts: List[str] = []
    for turn in transcript.turns:
        for block in turn.text:
            parts.append(block.text)
    return "".join(parts)


def _aggregate_usage(transcript: Transcript) -> Dict[str, int]:
    total_in = 0
    total_out = 0
    for turn in transcript.turns:
        total_in += int(turn.usage.get("input_tokens", 0) or 0)
        total_out += int(turn.usage.get("output_tokens", 0) or 0)
    return {"input_tokens": total_in, "output_tokens": total_out}


def _extract_answer(text: str) -> str:
    """Pull the final "ANSWER: ..." line; fall back to the last non-empty line."""
    marker = "ANSWER:"
    idx = text.rfind(marker)
    if idx == -1:
        stripped = text.strip().splitlines()
        return stripped[-1].strip() if stripped else ""
    tail = text[idx + len(marker):].strip()
    return tail.splitlines()[0].strip() if tail else ""


def solve_one(
    agent: LinearAgent,
    task: ExtractedTask,
    skill: Optional[Skill] = None,
) -> Dict[str, Any]:
    """Run one solving episode and package the result + transcript.

    `skill` is metadata only — the caller is responsible for constructing the
    `agent` with the right system prompt (e.g. via `format_skill_as_system`
    for the curated condition). Passing `skill` here merely:
      - drops the redundant "Required skills:" line from the user prompt
      - tags the output with condition="curated" + skill_name/skill_uid
    """
    prompt = _build_solve_prompt(task, include_skill_label=skill is None)
    transcript = record(agent.run(prompt))
    final_text = _final_text_from_transcript(transcript)
    answer = _extract_answer(final_text)
    expected = (task.output or "").strip()
    matches = bool(answer) and answer.lower() == expected.lower()
    primary_uid, primary_name, skill_uids, skill_names = _primary_skill_metadata(task)
    return {
        "task_uid": task.task_uid,
        "skill_uid": primary_uid,
        "skill_name": primary_name,
        "skill_uids": skill_uids,
        "skill_names": skill_names,
        "k": len(skill_uids),
        "title": task.title,
        "question": task.question,
        "expected_output": expected,
        "answer": answer,
        "final_text": final_text,
        "matches_expected": matches,
        "usage": _aggregate_usage(transcript),
        "transcript": transcript.to_dict(),
        "solving_method": SOLVING_METHOD,
        "model": agent.model,
        "system_prompt": agent.system,
        "user_prompt": prompt,
        "condition": "curated" if skill is not None else "baseline",
        "injected_skill_name": skill.name if skill is not None else "",
        "injected_skill_uid": skill.skill_uid if skill is not None else "",
    }


def solve_tasks(
    tasks: List[ExtractedTask],
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Solve every task via a single shared LinearAgent (stateless across tasks).

    Returns (solutions, failures). A failure record carries task_uid + error;
    a solution record is the full payload from solve_one.
    """
    agent = LinearAgent(
        model=model,
        system=SOLVER_SYSTEM_PROMPT,
        tools=[],
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        max_turns=1,
    )

    solutions: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"[{i}/{len(tasks)}] {task.title!r}")
        try:
            sol = solve_one(agent, task)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            failures.append({"task_uid": task.task_uid, "title": task.title, "error": str(e)})
            continue
        solutions.append(sol)
        if verbose:
            tag = "MATCH" if sol["matches_expected"] else "DIFF "
            print(f"  {tag} answer={sol['answer'][:80]!r}")
    return solutions, failures


# ---------------------------------------------------------------------------
# Trace adapter — solutions[] -> ReasoningTrace[]
# ---------------------------------------------------------------------------


def _split_into_steps(thinking_text: str, min_len: int = 40, max_steps: int = 20) -> List[str]:
    """Split thinking into procedural steps by paragraph + sentence boundary.

    `min_len` filters out small cruft. `max_steps` caps long runs so downstream
    analysis stays tractable.
    """
    if not thinking_text.strip():
        return []
    chunks: List[str] = []
    for block in re.split(r"\n\s*\n", thinking_text.strip()):
        block = block.strip()
        if not block:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'`(])", block):
            s = sentence.strip()
            if len(s) >= min_len:
                chunks.append(s)
    if not chunks:
        chunks = [thinking_text.strip()[:500]]
    return chunks[:max_steps]


def _concat_thinking(transcript: Dict[str, Any]) -> str:
    parts: List[str] = []
    for turn in transcript.get("turns", []) or []:
        for tb in turn.get("thinking", []) or []:
            text = tb.get("text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _rebuild_user_prompt(task: Optional[ExtractedTask]) -> str:
    if task is None:
        return ""
    lines = [
        f"Task title: {task.title}",
        f"Difficulty: {task.difficulty}",
    ]
    ac = task.acceptance_criteria or {}
    names = ac.get("skill_names") or (
        [ac.get("skill_name")] if ac.get("skill_name") else []
    )
    if names:
        lines.append(f"Required skill(s): {', '.join(n for n in names if n)}")
    lines.append("")
    lines.append("--- context ---")
    lines.append(task.input or "")
    lines.append("--- question ---")
    lines.append(task.question or "")
    return "\n".join(lines)


def adapt_solution_to_trace(
    solution: Dict[str, Any], task: Optional[ExtractedTask]
) -> ReasoningTrace:
    """Native thinking blocks become procedural_steps (paragraph/sentence split).

    Order of preference for procedural_steps:
      1. Native API thinking blocks (when extended thinking was used).
      2. Response-text CoT (when the model writes reasoning before "ANSWER:").
         The trailing ANSWER line is stripped before splitting so it doesn't
         dilute the reasoning structure.
      3. Truncated response as a single chunk (last-resort fallback).
    """
    transcript = solution.get("transcript", {}) or {}
    thinking_text = _concat_thinking(transcript)
    response_text = solution.get("final_text", "")

    procedural_steps = _split_into_steps(thinking_text)
    if not procedural_steps and response_text:
        # Strip the trailing "ANSWER: ..." marker so it doesn't anchor a step.
        reasoning_text = re.split(r"\n\s*ANSWER\s*:", response_text, maxsplit=1)[0]
        procedural_steps = _split_into_steps(reasoning_text)
    if not procedural_steps:
        procedural_steps = [response_text[:500] or "(no reasoning recorded)"]

    conclusion = solution.get("answer", "") or solution.get("expected_output", "")
    usage = solution.get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    condition = solution.get("condition", "baseline")
    extraction_method = (
        INJECTION_METHOD if condition == "curated" else TRACE_METHOD
    )

    # Prefer the prompt actually sent (preserved by solve_one) over the
    # task-based rebuild — the rebuild is a backward-compat fallback for
    # legacy solutions and can drift from the real prompt when the curated
    # path drops the "Required skills:" line.
    user_prompt = solution.get("user_prompt") or _rebuild_user_prompt(task)

    return ReasoningTrace(
        task_uid=solution.get("task_uid", ""),
        model=solution.get("model", ""),
        system_prompt=solution.get("system_prompt", SOLVER_SYSTEM_PROMPT),
        user_prompt=user_prompt,
        response=response_text,
        procedural_steps=procedural_steps,
        conclusion=conclusion,
        tokens=input_tokens + output_tokens,
        elapsed_s=float(solution.get("elapsed_s", 0.0) or 0.0),
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        thinking=thinking_text,
        raw_steps=list(transcript.get("turns", []) or []),
        extraction_method=extraction_method,
        condition=condition,
        injected_skill_name=solution.get("injected_skill_name", ""),
        injected_skill_uid=solution.get("injected_skill_uid", ""),
    )


def adapt_solutions_to_traces(
    solutions: List[Dict[str, Any]],
    tasks: List[ExtractedTask],
) -> List[ReasoningTrace]:
    tasks_by_uid = {t.task_uid: t for t in tasks}
    return [adapt_solution_to_trace(sol, tasks_by_uid.get(sol.get("task_uid", ""))) for sol in solutions]


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def evaluate_holdout(
    tasks: List[ExtractedTask],
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solve holdout tasks and return {solutions, traces, failures, summary}.

    summary.pass_rate is over solved tasks (excluding API failures). per_k
    breaks down pass rate by the task's k value.
    """
    solutions, failures = solve_tasks(
        tasks, model=model, thinking_budget=thinking_budget,
        max_tokens=max_tokens, verbose=verbose,
    )
    traces = adapt_solutions_to_traces(solutions, tasks)

    matches = sum(1 for s in solutions if s["matches_expected"])
    per_k: Dict[int, Dict[str, int]] = {}
    for s in solutions:
        k = int(s.get("k", 0))
        bucket = per_k.setdefault(k, {"attempted": 0, "matches": 0})
        bucket["attempted"] += 1
        if s["matches_expected"]:
            bucket["matches"] += 1

    summary = {
        "tasks_attempted": len(tasks),
        "solutions_produced": len(solutions),
        "api_failures": len(failures),
        "answer_matches_expected": matches,
        "pass_rate": round(matches / len(solutions), 4) if solutions else 0.0,
        "per_k": {
            str(k): {
                "attempted": v["attempted"],
                "matches": v["matches"],
                "pass_rate": round(v["matches"] / v["attempted"], 4) if v["attempted"] else 0.0,
            }
            for k, v in sorted(per_k.items())
        },
        "model": model,
        "thinking_budget": thinking_budget,
        "solving_method": SOLVING_METHOD,
    }
    return {"solutions": solutions, "traces": traces, "failures": failures, "summary": summary}


# ---------------------------------------------------------------------------
# Skill-injection driver (closes Gap A)
#
# Per task: run a baseline episode (no skill in system) and, if the task is
# in `skill_map`, a curated episode (procedural Skill injected into the system
# prompt). Both episodes capture native thinking blocks via the same
# LinearAgent path used by the baseline solver, then optionally pass through
# an LLMJudgeEvaluator for pass/score/conclusion_reached.
#
# Output is one episode record per (task, condition) carrying the full trace
# AND the judge verdict — exactly what an SFT pipeline downstream needs in
# order to filter for (passed AND has_injected_skill) demonstrations.
# ---------------------------------------------------------------------------


def _build_solver_agent(
    system_prompt: str,
    model: str,
    thinking_budget: int,
    max_tokens: int,
) -> LinearAgent:
    return LinearAgent(
        model=model,
        system=system_prompt,
        tools=[],
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        max_turns=1,
    )


def _judge_solution(
    solution: Dict[str, Any],
    task: ExtractedTask,
    judge: Optional[LLMJudgeEvaluator],
) -> Dict[str, Any]:
    """Return judge fields (passed/score/conclusion_reached/rationale).

    With no judge: derive passed from exact-match against task.output so the
    summary is still meaningful for tasks with deterministic answers.
    """
    if judge is None:
        matches = bool(solution.get("matches_expected"))
        return {
            "passed": matches,
            "score": 1.0 if matches else 0.0,
            "conclusion_reached": matches,
            "judge_rationale": "exact-match (no judge)",
        }

    response_text = solution.get("final_text", "") or ""
    result: JudgeResult = judge.evaluate(
        response=response_text,
        passage=task.passage,
        challenge=task.challenge,
        acceptance_criteria=task.acceptance_criteria,
        query_type=getattr(task, "query_type", "FREE_FORM"),
    )
    return {
        "passed": result.passed,
        "score": result.score,
        "conclusion_reached": result.conclusion_reached,
        "judge_rationale": result.rationale,
    }


def _episode_record(
    solution: Dict[str, Any],
    trace: ReasoningTrace,
    judge_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """One serializable episode record: solution metadata + trace + judge."""
    usage = solution.get("usage") or {}
    tokens = int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
    return {
        "task_uid": solution.get("task_uid", ""),
        "model": solution.get("model", ""),
        "condition": solution.get("condition", "baseline"),
        "injected_skill_name": solution.get("injected_skill_name", ""),
        "injected_skill_uid": solution.get("injected_skill_uid", ""),
        "title": solution.get("title", ""),
        "expected_output": solution.get("expected_output", ""),
        "answer": solution.get("answer", ""),
        "matches_expected": solution.get("matches_expected", False),
        "passed": judge_fields["passed"],
        "score": judge_fields["score"],
        "conclusion_reached": judge_fields["conclusion_reached"],
        "judge_rationale": judge_fields["judge_rationale"],
        "tokens": tokens,
        "trace": asdict(trace),
        "extraction_method": INJECTION_METHOD,
    }


def _summarize_injection(
    episodes: List[Dict[str, Any]],
    model: str,
    tasks_attempted: int,
    api_failures: int,
    thinking_budget: int,
) -> Dict[str, Any]:
    """Aggregate per-condition pass/score and per-task uplift."""
    by_cond: Dict[str, Dict[str, float]] = {}
    by_task: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        cond = ep["condition"]
        bucket = by_cond.setdefault(cond, {"n": 0, "passed": 0, "score_sum": 0.0})
        bucket["n"] += 1
        bucket["passed"] += int(bool(ep["passed"]))
        bucket["score_sum"] += float(ep["score"])

        rec = by_task.setdefault(ep["task_uid"], {})
        rec[f"{cond}_passed"] = bool(ep["passed"])
        rec[f"{cond}_score"] = float(ep["score"])
        if cond == "curated":
            rec["skill"] = ep.get("injected_skill_name", "")

    per_condition = {
        cond: {
            "n": int(b["n"]),
            "passed": int(b["passed"]),
            "pass_rate": round(b["passed"] / b["n"], 4) if b["n"] else 0.0,
            "mean_score": round(b["score_sum"] / b["n"], 4) if b["n"] else 0.0,
        }
        for cond, b in by_cond.items()
    }

    uplifts: List[Dict[str, Any]] = []
    for task_uid, rec in by_task.items():
        if "baseline_passed" in rec and "curated_passed" in rec:
            uplifts.append({
                "task_uid": task_uid,
                "skill": rec.get("skill", ""),
                "baseline_passed": rec["baseline_passed"],
                "curated_passed": rec["curated_passed"],
                "baseline_score": rec["baseline_score"],
                "curated_score": rec["curated_score"],
                "delta_passed": int(rec["curated_passed"]) - int(rec["baseline_passed"]),
                "delta_score": round(rec["curated_score"] - rec["baseline_score"], 4),
            })

    return {
        "tasks_attempted": tasks_attempted,
        "episodes_produced": len(episodes),
        "api_failures": api_failures,
        "per_condition": per_condition,
        "per_task_uplift": uplifts,
        "model": model,
        "thinking_budget": thinking_budget,
        "extraction_method": INJECTION_METHOD,
    }


def solve_with_injection(
    tasks: List[ExtractedTask],
    skill_map: Dict[str, Skill],
    judge: Optional[LLMJudgeEvaluator] = None,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Per task, run baseline + (if mapped) curated episodes with skill injection.

    For every task:
      - baseline: SOLVER_SYSTEM_PROMPT, no skill in system, no skill label
                  in user prompt. Captures native thinking.
    For tasks with a Skill in `skill_map`:
      - curated:  format_skill_as_system(SOLVER_SYSTEM_PROMPT, skill) as system,
                  user prompt drops "Required skills:" line. Captures native
                  thinking AS THE MODEL APPLIES THE INJECTED PROCEDURE.

    Each episode is judged (or exact-matched if `judge=None`) and emitted as
    one episode record carrying both the trace and the judge verdict.
    """
    baseline_agent = _build_solver_agent(
        SOLVER_SYSTEM_PROMPT, model, thinking_budget, max_tokens,
    )

    episodes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        mapped_skill = skill_map.get(task.task_uid)
        if verbose:
            tag = (
                f"with skill={mapped_skill.name!r}" if mapped_skill else "baseline only"
            )
            print(f"[{i}/{len(tasks)}] {task.title!r} ({tag})")

        # ---- baseline ------------------------------------------------------
        try:
            sol_b = solve_one(baseline_agent, task, skill=None)
            jf_b = _judge_solution(sol_b, task, judge)
            tr_b = adapt_solution_to_trace(sol_b, task)
            episodes.append(_episode_record(sol_b, tr_b, jf_b))
            if verbose:
                print(
                    f"  baseline  passed={jf_b['passed']} "
                    f"score={jf_b['score']:.2f} "
                    f"answer={(sol_b.get('answer') or '')[:60]!r}"
                )
        except Exception as e:
            failures.append({
                "task_uid": task.task_uid, "condition": "baseline",
                "title": task.title, "error": str(e),
            })
            if verbose:
                print(f"  baseline ERROR: {e}")

        # ---- curated (only if mapped) -------------------------------------
        if mapped_skill is None:
            continue

        curated_system = format_skill_as_system(SOLVER_SYSTEM_PROMPT, mapped_skill)
        curated_agent = _build_solver_agent(
            curated_system, model, thinking_budget, max_tokens,
        )
        try:
            sol_c = solve_one(curated_agent, task, skill=mapped_skill)
            jf_c = _judge_solution(sol_c, task, judge)
            tr_c = adapt_solution_to_trace(sol_c, task)
            episodes.append(_episode_record(sol_c, tr_c, jf_c))
            if verbose:
                print(
                    f"  curated   passed={jf_c['passed']} "
                    f"score={jf_c['score']:.2f} "
                    f"answer={(sol_c.get('answer') or '')[:60]!r}"
                )
        except Exception as e:
            failures.append({
                "task_uid": task.task_uid, "condition": "curated",
                "title": task.title, "skill": mapped_skill.name, "error": str(e),
            })
            if verbose:
                print(f"  curated ERROR: {e}")

    summary = _summarize_injection(
        episodes, model=model,
        tasks_attempted=len(tasks),
        api_failures=len(failures),
        thinking_budget=thinking_budget,
    )
    return {"episodes": episodes, "failures": failures, "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_solutions_jsonl(path: Path, solutions: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sol in solutions:
            fh.write(json.dumps(sol, ensure_ascii=False) + "\n")


def _write_traces_jsonl(path: Path, traces: List[ReasoningTrace]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


def _write_failures_jsonl(path: Path, failures: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")


def _write_episodes_jsonl(path: Path, episodes: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")


def _load_task_skill_map(path: Path, catalog: List[Skill]) -> Dict[str, Skill]:
    """Resolve {task_uid: skill_name} JSON to {task_uid: Skill} via the catalog.

    Mirrors b2_benchmarks.skillsbench.corpus_harness._load_task_skill_map so
    the same map file works for both bench and bench_traced. Unknown skill
    names are dropped with a warning.
    """
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected JSON object {{task_uid: skill_name}}")

    by_name = {s.name: s for s in catalog}
    resolved: Dict[str, Skill] = {}
    missing: List[str] = []
    for task_uid, skill_name in raw.items():
        skill = by_name.get(str(skill_name))
        if skill is None:
            missing.append(f"{task_uid} -> {skill_name!r}")
            continue
        resolved[str(task_uid)] = skill
    if missing:
        print(
            f"warning: {len(missing)} map entries skipped (skill_name not in catalog): "
            + "; ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    return resolved


def _parse_provider_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--mode", choices=["solve", "inject"], default="solve",
                        help="solve: baseline holdout solver (legacy default). "
                             "inject: per-task baseline + curated with skill injected "
                             "into the system prompt (closes Gap A; emits episodes.jsonl).")
    parser.add_argument("--tasks", type=Path, required=True,
                        help="ExtractedTask JSON (typically holdout tasks from s3.m1).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory for output files. solve: solutions.jsonl, "
                             "traces.jsonl, summary.json, failures.jsonl. "
                             "inject: episodes.jsonl, summary.json, failures.jsonl.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only solve the first N tasks (0 = all).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET,
                        help="Set to 0 to disable thinking; traces will be populated from response text.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--verbose", "-v", action="store_true")
    # ---- inject-mode-only args -------------------------------------------
    parser.add_argument("--catalog", type=Path,
                        help="Skill catalog JSON (procedural form preferred — bench_catalog/skills.json). "
                             "Required for --mode inject.")
    parser.add_argument("--task-skill-map", type=Path,
                        help="JSON {task_uid: skill_name}. Each listed task gets a "
                             "curated episode with its mapped skill. Tasks not in "
                             "the map run baseline only. Required for --mode inject.")
    parser.add_argument("--judge", default="",
                        help="Judge provider spec, e.g. 'anthropic:claude-opus-4-7'. "
                             "Empty = exact-match fallback against task.output.")
    args = parser.parse_args()

    tasks = load_extracted_tasks(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "solve":
        _run_solve_mode(args, tasks)
    else:
        _run_inject_mode(args, tasks)


def _run_solve_mode(args, tasks: List[ExtractedTask]) -> None:
    result = evaluate_holdout(
        tasks,
        model=args.model,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    _write_solutions_jsonl(args.out_dir / "solutions.jsonl", result["solutions"])
    _write_traces_jsonl(args.out_dir / "traces.jsonl", result["traces"])
    _write_failures_jsonl(args.out_dir / "failures.jsonl", result["failures"])
    (args.out_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )

    s = result["summary"]
    print(
        f"\nsolved {s['solutions_produced']}/{s['tasks_attempted']} tasks  "
        f"pass_rate={s['pass_rate']}  per_k={s['per_k']}  -> {args.out_dir}/"
    )


def _run_inject_mode(args, tasks: List[ExtractedTask]) -> None:
    if not args.catalog:
        raise SystemExit("--mode inject requires --catalog (procedural skill catalog)")
    if not args.task_skill_map:
        raise SystemExit("--mode inject requires --task-skill-map")

    catalog = load_skills(args.catalog)
    skill_map = _load_task_skill_map(args.task_skill_map, catalog)
    print(f"Loaded {len(catalog)} skills + {len(skill_map)} task->skill mappings")

    judge: Optional[LLMJudgeEvaluator] = None
    if args.judge.strip():
        from core.providers import create_provider
        j_name, j_model = _parse_provider_spec(args.judge)
        judge = LLMJudgeEvaluator(create_provider(j_name, j_model))
        print(f"Judge: {args.judge}")
    else:
        print("Judge: (none — using exact-match fallback)")

    result = solve_with_injection(
        tasks,
        skill_map,
        judge=judge,
        model=args.model,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    _write_episodes_jsonl(args.out_dir / "episodes.jsonl", result["episodes"])
    _write_failures_jsonl(args.out_dir / "failures.jsonl", result["failures"])
    (args.out_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )

    s = result["summary"]
    per_cond_lines = [
        f"  {cond}: pass_rate={c['pass_rate']} mean_score={c['mean_score']} (n={c['n']})"
        for cond, c in s.get("per_condition", {}).items()
    ]
    print(
        f"\ninject  episodes={s['episodes_produced']} "
        f"api_failures={s['api_failures']}  -> {args.out_dir}/\n"
        + "\n".join(per_cond_lines)
    )


if __name__ == "__main__":
    main()
