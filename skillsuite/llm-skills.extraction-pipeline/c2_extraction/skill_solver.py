"""
skill_solver.py

Stage 2 (parallel path): solve each ExtractedTask using a frontier
Anthropic model with extended thinking, capturing the full reasoning
transcript.

Pure reasoning - no tools. One LinearAgent episode per task.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.skill_solver \\
        --tasks data/pipeline-runs/frontier-wikipedia/stage1-extraction/tasks.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage2-solving \\
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, Transcript, record  # noqa: E402

from c1_types.extracted_task import ExtractedTask, load_extracted_tasks  # noqa: E402


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_MAX_TOKENS = 8000
SOLVING_METHOD = "frontier-solver-v1"


SYSTEM_PROMPT = (
    "You are an expert problem solver. Think carefully about the task, "
    "then give your final answer as one concise conclusion. The grader "
    "compares your final answer to a single correct conclusion string."
)


SOLVE_PROMPT = """Task title: {title}
Difficulty: {difficulty}
Required skills: {skills}

--- context ---
{input}
--- question ---
{question}

Give your final answer as ONE concise conclusion on the last line,
prefixed exactly with "ANSWER: ".
"""


def _skills_label(task: ExtractedTask) -> str:
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
    ac = task.acceptance_criteria or {}
    uids = ac.get("skill_uids")
    names = ac.get("skill_names")
    if uids and names:
        return uids[0], names[0], uids, names
    uid = ac.get("skill_uid", "")
    name = ac.get("skill_name", "")
    return uid, name, [uid] if uid else [], [name] if name else []


def _build_prompt(task: ExtractedTask) -> str:
    return SOLVE_PROMPT.format(
        title=task.title,
        difficulty=task.difficulty,
        skills=_skills_label(task),
        input=task.input,
        question=task.question,
    )


def _final_text(transcript: Transcript) -> str:
    parts: List[str] = []
    for turn in transcript.turns:
        for block in turn.text:
            parts.append(block.text)
    return "".join(parts)


def _extract_answer(text: str) -> str:
    marker = "ANSWER:"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip().splitlines()[-1].strip() if text.strip() else ""
    return text[idx + len(marker) :].strip().splitlines()[0].strip() if text[idx + len(marker) :].strip() else ""


def _aggregate_usage(transcript: Transcript) -> Dict[str, int]:
    total_in = 0
    total_out = 0
    for turn in transcript.turns:
        total_in += int(turn.usage.get("input_tokens", 0) or 0)
        total_out += int(turn.usage.get("output_tokens", 0) or 0)
    return {"input_tokens": total_in, "output_tokens": total_out}


def solve_one(
    agent: LinearAgent,
    task: ExtractedTask,
) -> Dict[str, Any]:
    prompt = _build_prompt(task)
    transcript = record(agent.run(prompt))
    final_text = _final_text(transcript)
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
    }


def solve_tasks(
    tasks_path: Path,
    output_dir: Path,
    limit: int = 0,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    tasks = load_extracted_tasks(tasks_path)
    if limit > 0:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = output_dir / "solutions.jsonl"
    failures_path = output_dir / "solving-failures.jsonl"
    summary_path = output_dir / "summary.json"

    agent = LinearAgent(
        model=model,
        system=SYSTEM_PROMPT,
        tools=[],
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        max_turns=1,
    )

    solutions: List[Dict[str, Any]] = []
    matches = 0
    with solutions_path.open("w", encoding="utf-8") as out, \
         failures_path.open("w", encoding="utf-8") as f_out:
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"[{i}/{len(tasks)}] {task.title}")

            try:
                sol = solve_one(agent, task)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                f_out.write(json.dumps({
                    "task_uid": task.task_uid,
                    "title": task.title,
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
                continue

            solutions.append(sol)
            if sol["matches_expected"]:
                matches += 1
            out.write(json.dumps(sol, ensure_ascii=False) + "\n")
            if verbose:
                tag = "MATCH" if sol["matches_expected"] else "DIFF "
                print(f"  {tag} answer={sol['answer'][:80]!r}")

    summary = {
        "tasks_attempted": len(tasks),
        "solutions_produced": len(solutions),
        "answer_matches_expected": matches,
        "model": model,
        "thinking_budget": thinking_budget,
        "solving_method": SOLVING_METHOD,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nWrote {len(solutions)} solutions -> {solutions_path}")
        print(f"Matches: {matches}/{len(solutions)}  Summary -> {summary_path}")

    return solutions


def main() -> int:
    ap = argparse.ArgumentParser(description="Solve ExtractedTasks with a frontier model, capturing reasoning transcripts.")
    ap.add_argument("--tasks", required=True, type=Path, help="Path to tasks.json produced by frontier_extractor.")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output directory for solutions.jsonl.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N tasks (0 = all).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    solutions = solve_tasks(
        tasks_path=args.tasks,
        output_dir=args.output_dir,
        limit=args.limit,
        model=args.model,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )
    return 0 if solutions else 1


if __name__ == "__main__":
    sys.exit(main())
