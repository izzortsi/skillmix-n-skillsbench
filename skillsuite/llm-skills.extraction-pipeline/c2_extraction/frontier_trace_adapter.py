"""
frontier_trace_adapter.py

Adapter: convert frontier solutions.jsonl (from skill_solver.py) into
ReasoningTrace JSONL compatible with the existing stage-3+ pipeline
(skill_extractor.py, compose-skills, skillmix-evaluation).

Key upgrade vs the legacy text-regex extract_thinking path: we already
have NATIVE Anthropic thinking blocks (with signatures) in each
transcript, so we populate ReasoningTrace.thinking from structured
data rather than regex-scanning response text. Procedural steps are
derived from the thinking text by paragraph boundary.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_trace_adapter \\
        --solutions data/pipeline-runs/frontier-wikipedia/stage2-solving/solutions.jsonl \\
        --tasks data/pipeline-runs/frontier-wikipedia/stage1-extraction/tasks.json \\
        --output data/pipeline-runs/frontier-wikipedia/stage2-solving/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from c1_types.extracted_task import ExtractedTask, load_extracted_tasks  # noqa: E402
from c2_extraction.trace_capturer import (  # noqa: E402
    ReasoningTrace,
    _parse_structured_trace,
    save_traces,
)


TRACE_METHOD = "frontier-solver-trace-v1"


def _split_into_steps(thinking_text: str, min_len: int = 40, max_steps: int = 20) -> List[str]:
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
        chunks = [thinking_text.strip()[: 500]]
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
    names = ac.get("skill_names") or ([ac.get("skill_name")] if ac.get("skill_name") else [])
    if names:
        lines.append(f"Required skill(s): {', '.join(n for n in names if n)}")
    lines.append("")
    lines.append("--- context ---")
    lines.append(task.input or "")
    lines.append("--- question ---")
    lines.append(task.question or "")
    return "\n".join(lines)


def adapt_one(
    solution: Dict[str, Any],
    task: Optional[ExtractedTask],
) -> ReasoningTrace:
    transcript = solution.get("transcript", {}) or {}
    thinking_text = _concat_thinking(transcript)
    response_text = solution.get("final_text", "")

    procedural_steps = _split_into_steps(thinking_text)
    if not procedural_steps:
        parsed_steps, _parsed_conclusion = _parse_structured_trace(response_text)
        procedural_steps = parsed_steps

    if not procedural_steps:
        procedural_steps = [response_text[:500] or "(no reasoning recorded)"]

    conclusion = solution.get("answer", "") or solution.get("expected_output", "")
    usage = solution.get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    raw_turns = transcript.get("turns", []) or []

    return ReasoningTrace(
        task_uid=solution.get("task_uid", ""),
        model=solution.get("model", ""),
        system_prompt="",
        user_prompt=_rebuild_user_prompt(task),
        response=response_text,
        procedural_steps=procedural_steps,
        conclusion=conclusion,
        tokens=input_tokens + output_tokens,
        elapsed_s=0.0,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        thinking=thinking_text,
        raw_steps=raw_turns,
        extraction_method=TRACE_METHOD,
    )


def adapt_solutions(
    solutions_path: Path,
    tasks_path: Optional[Path],
    output_path: Path,
    verbose: bool = True,
) -> List[ReasoningTrace]:
    tasks_by_uid: Dict[str, ExtractedTask] = {}
    if tasks_path and tasks_path.exists():
        tasks_by_uid = {t.task_uid: t for t in load_extracted_tasks(tasks_path)}

    traces: List[ReasoningTrace] = []
    with solutions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = tasks_by_uid.get(entry.get("task_uid", ""))
            traces.append(adapt_one(entry, task))

    save_traces(traces, output_path)

    if verbose:
        with_thinking = sum(1 for t in traces if t.thinking.strip())
        print(f"Wrote {len(traces)} traces -> {output_path}")
        print(f"  {with_thinking}/{len(traces)} traces carry native thinking content")
        print(f"  task metadata joined: {len(tasks_by_uid)} tasks indexed")

    return traces


def main() -> int:
    ap = argparse.ArgumentParser(description="Adapt frontier solutions.jsonl into stage-3-compatible ReasoningTrace JSONL.")
    ap.add_argument("--solutions", required=True, type=Path, help="solutions.jsonl from skill_solver.")
    ap.add_argument("--tasks", type=Path, default=None, help="Optional tasks.json to recover full user_prompt context.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    traces = adapt_solutions(
        solutions_path=args.solutions,
        tasks_path=args.tasks,
        output_path=args.output,
        verbose=not args.quiet,
    )
    return 0 if traces else 1


if __name__ == "__main__":
    sys.exit(main())
