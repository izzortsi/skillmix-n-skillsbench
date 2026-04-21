"""
frontier_labeler.py

PROJECT_SPECS section 1 Method 1 ("Task-Based Skill Labeling"): given a
set of existing tasks, ask a frontier model to label each task with the
2-5 procedural skills required to solve it. Labels use snake_case of
2-4 words, matching the spec's constrained format.

Input: any JSON file that load_extracted_tasks can parse (the output of
frontier_extractor, frontier_verifier, frontier_pair_extractor, or the
legacy task_extractor).

Output:
    labeled-tasks.jsonl        - one record per task: task_uid, title,
                                  skill_labels[], transcript, usage
    skill-frequency.json       - {label: count} across all tasks
                                  (input for a downstream clustering pass,
                                  which spec 1.M1 describes as step 2)
    labeling-failures.jsonl    - tasks that failed to parse or errored

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_labeler \\
        --tasks data/pipeline-runs/frontier-wikipedia/stage1b-verification/verified-tasks.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage0-labeling \\
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, Transcript, record  # noqa: E402

from c0_utils.text_utils import strip_markdown_fences  # noqa: E402
from c1_types.extracted_task import ExtractedTask, load_extracted_tasks  # noqa: E402


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_MAX_TOKENS = 3000
LABELING_METHOD = "frontier-labeler-v1"
MIN_LABELS = 2
MAX_LABELS = 5


SYSTEM_PROMPT = (
    "You label evaluation tasks with the specific procedural skills a "
    "solver must exercise to reach the correct answer. Labels are "
    "snake_case of 2-4 words. You prefer specific procedural moves "
    "over vague topic words."
)


LABEL_PROMPT = """Label this task with the {min_labels}-{max_labels} discrete skills a solver must exercise.

Task title:  {title}
Difficulty:  {difficulty}
Question:    {question}
Context/input:
{input}

Expected answer:
{output}

Rules for labels:
  - Format: snake_case, 2-4 words (e.g. self_serving_bias_identification)
  - Prefer specific procedural moves: "causal_attribution_analysis",
    "loaded_question_detection", "loop_invariant_derivation".
  - Reject vague labels: "reasoning", "thinking", "math", "writing_well".
  - A skill names a procedure or cognitive move, not a topic.
  - Distinct labels; no synonyms of each other.

Return ONLY valid JSON:

{{
  "skills": [
    {{"label": "<snake_case_label>", "rationale": "<one sentence on why the task requires this skill>"}}
  ]
}}
"""


def _build_prompt(task: ExtractedTask) -> str:
    return LABEL_PROMPT.format(
        min_labels=MIN_LABELS,
        max_labels=MAX_LABELS,
        title=task.title,
        difficulty=task.difficulty or "unspecified",
        question=task.question,
        input=task.input,
        output=task.output or "(not given)",
    )


def _parse_json(text: str):
    raw = strip_markdown_fences(text).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{"); end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None


def _final_text(transcript: Transcript) -> str:
    return "".join(b.text for turn in transcript.turns for b in turn.text)


def _aggregate_usage(transcript: Transcript) -> Dict[str, int]:
    total_in = 0
    total_out = 0
    for turn in transcript.turns:
        total_in += int(turn.usage.get("input_tokens", 0) or 0)
        total_out += int(turn.usage.get("output_tokens", 0) or 0)
    return {"input_tokens": total_in, "output_tokens": total_out}


def _sanitize_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    cleaned = []
    for ch in label:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        elif ch in (" ", "-", "/"):
            cleaned.append("_")
    collapsed = "".join(cleaned).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed


def label_one(agent: LinearAgent, task: ExtractedTask) -> Dict[str, Any]:
    transcript = record(agent.run(_build_prompt(task)))
    text = _final_text(transcript)
    parsed = _parse_json(text)

    skill_labels: List[Dict[str, str]] = []
    parse_error = None

    if parsed is None:
        parse_error = "json_parse_failed"
    else:
        raw_skills = parsed.get("skills", []) or []
        for item in raw_skills:
            if not isinstance(item, dict):
                continue
            raw_label = item.get("label", "")
            if not raw_label:
                continue
            label = _sanitize_label(str(raw_label))
            if not label:
                continue
            skill_labels.append({
                "label": label,
                "rationale": str(item.get("rationale", "")).strip(),
            })
        if not skill_labels:
            parse_error = "no_valid_skill_labels"

    return {
        "task_uid": task.task_uid,
        "title": task.title,
        "difficulty": task.difficulty,
        "domain": task.domain,
        "skill_labels": skill_labels,
        "final_text": text,
        "parse_error": parse_error,
        "usage": _aggregate_usage(transcript),
        "transcript": transcript.to_dict(),
        "labeling_method": LABELING_METHOD,
        "model": agent.model,
    }


def label_tasks(
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
    labeled_path = output_dir / "labeled-tasks.jsonl"
    failures_path = output_dir / "labeling-failures.jsonl"
    freq_path = output_dir / "skill-frequency.json"
    summary_path = output_dir / "summary.json"

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    results: List[Dict[str, Any]] = []
    label_counter: Counter = Counter()
    errors = 0

    with labeled_path.open("w", encoding="utf-8") as l_out, \
         failures_path.open("w", encoding="utf-8") as f_out:
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"[{i}/{len(tasks)}] {task.title}")

            try:
                result = label_one(agent, task)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                errors += 1
                f_out.write(json.dumps({
                    "task_uid": task.task_uid,
                    "title": task.title,
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
                continue

            l_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if result["parse_error"]:
                if verbose:
                    print(f"  PARSE_FAIL: {result['parse_error']}")
                f_out.write(json.dumps({
                    "task_uid": task.task_uid,
                    "title": task.title,
                    "error": result["parse_error"],
                    "final_text": result["final_text"],
                }, ensure_ascii=False) + "\n")
                continue

            results.append(result)
            for s in result["skill_labels"]:
                label_counter[s["label"]] += 1
            if verbose:
                labels = [s["label"] for s in result["skill_labels"]]
                print(f"  labels={labels}")

    frequency = dict(sorted(label_counter.items(), key=lambda kv: (-kv[1], kv[0])))
    freq_path.write_text(json.dumps(frequency, indent=2), encoding="utf-8")

    summary = {
        "tasks_attempted": len(tasks),
        "tasks_labeled": len(results),
        "parse_errors_or_api_errors": len(tasks) - len(results),
        "api_errors": errors,
        "unique_skill_labels": len(frequency),
        "total_label_assignments": sum(frequency.values()),
        "model": model,
        "labeling_method": LABELING_METHOD,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n{len(results)}/{len(tasks)} labeled -> {labeled_path}")
        print(f"Unique labels: {len(frequency)}  -> {freq_path}")
        print(f"Summary -> {summary_path}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Label tasks with 2-5 snake_case skill names using a frontier model (PROJECT_SPECS 1.M1).")
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results = label_tasks(
        tasks_path=args.tasks, output_dir=args.output_dir,
        limit=args.limit, model=args.model,
        thinking_budget=args.thinking_budget, max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
