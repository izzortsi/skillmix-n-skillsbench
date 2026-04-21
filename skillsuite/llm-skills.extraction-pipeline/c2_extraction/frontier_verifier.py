"""
frontier_verifier.py

Between stage 1 (task generation) and stage 2 (task solving): use a
frontier model as an agentic judge to verify that each generated task
actually REQUIRES every named skill to solve.

This closes PROJECT_SPECS section 3 Method 4 ("Agentic Answer
Verification") for the task-generation side: ensure generated tasks
genuinely exercise their labelled skills before we spend compute
solving them.

Handles both k=1 tasks (single skill_uid/skill_name in
acceptance_criteria) and k=2 tasks (skill_uids/skill_names lists) by
normalizing to the plural form.

Output:
    verified-tasks.json       - tasks where every skill passes
    rejected-tasks.jsonl      - tasks that failed, with per-skill rationale
    verification-transcripts.jsonl - one Transcript per task

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_verifier \\
        --tasks data/pipeline-runs/frontier-wikipedia/stage1-extraction/tasks.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage1b-verification \\
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, Transcript, record  # noqa: E402

from c0_utils.text_utils import strip_markdown_fences  # noqa: E402
from c1_types.extracted_task import (  # noqa: E402
    ExtractedTask,
    load_extracted_tasks,
    save_extracted_tasks,
)


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_MAX_TOKENS = 4000
VERIFICATION_METHOD = "frontier-verifier-v1"


SYSTEM_PROMPT = (
    "You are a strict judge for an LLM evaluation benchmark. You check "
    "whether a candidate task genuinely REQUIRES every named skill. If a "
    "competent solver could reach the correct answer while skipping one "
    "of the skills, the task fails for that skill."
)


VERIFY_PROMPT = """Verify that this task requires every listed skill.

Title:       {title}
Difficulty:  {difficulty}
Question:    {question}
Context/input:
{input}

Correct conclusion (single answer):
{correct_conclusion}

Required skills to verify:
{skills_block}

For each skill, decide whether a solver CANNOT reach the correct
conclusion without exercising that skill. If the skill is decorative or
optional, mark exercises=false.

Return ONLY valid JSON:

{{
  "per_skill": [
    {{
      "skill_name": "<name exactly as given>",
      "exercises":  true|false,
      "rationale":  "<one sentence>"
    }}
  ],
  "overall": {{
    "exercises_all": true|false,
    "summary": "<one sentence>"
  }}
}}
"""


def _skills_of(task: ExtractedTask) -> List[Dict[str, str]]:
    ac = task.acceptance_criteria or {}
    names = ac.get("skill_names")
    uids = ac.get("skill_uids")
    cats = ac.get("skill_categories")
    if names and uids:
        cats = cats or [""] * len(names)
        return [
            {"skill_uid": u, "skill_name": n, "skill_category": c or ""}
            for u, n, c in zip(uids, names, cats)
        ]
    name = ac.get("skill_name", "")
    uid = ac.get("skill_uid", "")
    cat = ac.get("skill_category", "")
    if name and uid:
        return [{"skill_uid": uid, "skill_name": name, "skill_category": cat}]
    return []


def _skills_block(skills: List[Dict[str, str]]) -> str:
    lines = []
    for i, s in enumerate(skills, 1):
        cat = s["skill_category"] or "unspecified"
        lines.append(f"  {i}. {s['skill_name']} ({cat})")
    return "\n".join(lines)


def _build_prompt(task: ExtractedTask, skills: List[Dict[str, str]]) -> str:
    return VERIFY_PROMPT.format(
        title=task.title,
        difficulty=task.difficulty,
        question=task.question,
        input=task.input,
        correct_conclusion=task.output,
        skills_block=_skills_block(skills),
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


def verify_one(
    agent: LinearAgent, task: ExtractedTask
) -> Tuple[bool, Dict[str, Any], Transcript]:
    skills = _skills_of(task)
    if not skills:
        return False, {"error": "no_skills_found_in_acceptance_criteria"}, Transcript()

    transcript = record(agent.run(_build_prompt(task, skills)))
    text = _final_text(transcript)
    parsed = _parse_json(text)
    if parsed is None:
        return False, {"error": "json_parse_failed", "final_text": text}, transcript

    per_skill = parsed.get("per_skill", []) or []
    overall = parsed.get("overall", {}) or {}
    exercises_all = bool(overall.get("exercises_all", False))
    if per_skill:
        exercises_all = exercises_all and all(
            bool(s.get("exercises", False)) for s in per_skill
        )

    verdict = {
        "per_skill": per_skill,
        "overall": overall,
        "exercises_all": exercises_all,
    }
    return exercises_all, verdict, transcript


def verify_tasks(
    tasks_path: Path,
    output_dir: Path,
    limit: int = 0,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> List[ExtractedTask]:
    tasks = load_extracted_tasks(tasks_path)
    if limit > 0:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    v_path = output_dir / "verification-transcripts.jsonl"
    r_path = output_dir / "rejected-tasks.jsonl"
    s_path = output_dir / "summary.json"

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    verified: List[ExtractedTask] = []
    rejected_count = 0
    error_count = 0

    with v_path.open("w", encoding="utf-8") as v_out, \
         r_path.open("w", encoding="utf-8") as r_out:
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"[{i}/{len(tasks)}] {task.title}")

            try:
                passed, verdict, transcript = verify_one(agent, task)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                error_count += 1
                r_out.write(json.dumps({
                    "task_uid": task.task_uid,
                    "title": task.title,
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
                continue

            v_out.write(json.dumps({
                "task_uid": task.task_uid,
                "title": task.title,
                "verdict": verdict,
                "transcript": transcript.to_dict(),
            }, ensure_ascii=False) + "\n")

            if passed:
                verified.append(task)
                if verbose:
                    print("  PASS")
            else:
                rejected_count += 1
                if verbose:
                    print(f"  REJECT: {verdict.get('overall', {}).get('summary', verdict.get('error', ''))[:120]}")
                r_out.write(json.dumps({
                    "task_uid": task.task_uid,
                    "title": task.title,
                    "verdict": verdict,
                }, ensure_ascii=False) + "\n")

    out_path = output_dir / "verified-tasks.json"
    save_extracted_tasks(verified, out_path)

    summary = {
        "tasks_attempted": len(tasks),
        "verified": len(verified),
        "rejected": rejected_count,
        "errors": error_count,
        "model": model,
        "verification_method": VERIFICATION_METHOD,
    }
    s_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n{len(verified)}/{len(tasks)} verified -> {out_path}")
        print(f"Summary -> {s_path}")

    return verified


def main() -> int:
    ap = argparse.ArgumentParser(description="Agentic verification: check that each task genuinely requires its labelled skills.")
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verified = verify_tasks(
        tasks_path=args.tasks, output_dir=args.output_dir,
        limit=args.limit, model=args.model,
        thinking_budget=args.thinking_budget, max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
