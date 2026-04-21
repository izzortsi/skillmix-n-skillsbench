"""
frontier_extractor.py

Stage 1 (parallel path): generate ExtractedTask records from Wikipedia-seed
skills using a frontier Anthropic model with extended thinking.

For each seed skill (name + definition + example) we ask the model to
invent one FREE_FORM evaluation task that exercises the skill. The full
streaming transcript (thinking blocks with signatures + final text) is
persisted alongside the task, so the reasoning behind each extraction
is auditable.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_extractor \\
        --seed /workspace/llm-skills/skillsuite/llm-skills.shared-data/260420.skillmix-wikipedia-seed/skills.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage1-extraction \\
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, Transcript, record  # noqa: E402

from c0_utils.text_utils import strip_markdown_fences  # noqa: E402
from c0_utils.uid import generate_uid  # noqa: E402
from c1_types.extracted_skill import load_extracted_skills  # noqa: E402
from c1_types.extracted_task import ExtractedTask, save_extracted_tasks  # noqa: E402


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_MAX_TOKENS = 8000
DEFAULT_DOMAIN = "wikipedia-seed"
DEFAULT_SOURCE_ARTIFACT = "260420.skillmix-wikipedia-seed"
EXTRACTION_METHOD = "frontier-wikipedia-v1"


SYSTEM_PROMPT = (
    "You are a task designer for an LLM evaluation benchmark. You design "
    "tasks that test whether a model can demonstrate a specific named skill "
    "when answering a free-form question."
)


TASK_PROMPT = """Design ONE evaluation task that requires the skill below.

Skill name: {skill_name}
Category: {category}
Definition: {definition}
Illustrative example: {example}

Constraints on the task:
  - query_type = FREE_FORM (open-ended, exactly ONE correct conclusion)
  - The task must force the solver to exercise the skill above, not just
    describe it.
  - Difficulty: basic, intermediate, or advanced.
  - "input" is the context/passage the solver reads; it may quote the
    example but should set up a concrete situation.
  - "question" is what the solver must answer.
  - "correct_conclusion" is the one-line correct answer a grader would
    accept as exactly right.
  - "must_identify" lists 2-4 supporting facts the solver must mention.

Return ONLY valid JSON (no markdown, no prose) with this exact shape:

{{
  "title": "<short descriptive title>",
  "difficulty": "<basic|intermediate|advanced>",
  "question": "<the question the solver must answer>",
  "input": "<the passage/context the solver reads>",
  "acceptance_criteria": {{
    "must_identify": ["<fact 1>", "<fact 2>"],
    "correct_conclusion": "<the single correct conclusion>"
  }}
}}
"""


def _build_prompt(skill: Dict[str, Any]) -> str:
    return TASK_PROMPT.format(
        skill_name=skill.get("name", ""),
        category=skill.get("category", "") or "unspecified",
        definition=skill.get("description", "") or skill.get("definition", ""),
        example=skill.get("example", "") or "(no example provided)",
    )


def _parse_task_json(text: str) -> Optional[Dict[str, Any]]:
    raw = strip_markdown_fences(text).strip()
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


def _build_task(
    skill_uid: str,
    skill_name: str,
    skill_category: str,
    parsed: Dict[str, Any],
    domain: str,
    source_artifact: str,
) -> ExtractedTask:
    title = str(parsed.get("title", "")).strip() or skill_name
    question = str(parsed.get("question", "")).strip()
    input_text = str(parsed.get("input", "")).strip()
    difficulty = str(parsed.get("difficulty", "intermediate")).strip() or "intermediate"

    ac = parsed.get("acceptance_criteria", {}) or {}
    must_identify = ac.get("must_identify", []) or []
    correct_conclusion = str(ac.get("correct_conclusion", "")).strip()

    ac_out: Dict[str, Any] = {
        "must_identify": [str(x) for x in must_identify],
        "correct_conclusion": correct_conclusion,
        "skill_uid": skill_uid,
        "skill_name": skill_name,
        "skill_category": skill_category,
    }

    task_uid = generate_uid(f"{EXTRACTION_METHOD}|{skill_uid}|{title}")
    source_document_uid = skill_uid

    return ExtractedTask(
        task_uid=task_uid,
        title=title,
        domain=domain,
        source_artifact=source_artifact,
        source_document_uid=source_document_uid,
        question=question,
        input=input_text,
        output=correct_conclusion,
        difficulty=difficulty,
        acceptance_criteria=ac_out,
        query_type="FREE_FORM",
        extraction_method=EXTRACTION_METHOD,
    )


def _final_text(transcript: Transcript) -> str:
    parts: List[str] = []
    for turn in transcript.turns:
        for block in turn.text:
            parts.append(block.text)
    return "".join(parts)


def extract_one(
    agent: LinearAgent,
    skill: Dict[str, Any],
    domain: str,
    source_artifact: str,
):
    prompt = _build_prompt(skill)
    transcript = record(agent.run(prompt))
    text = _final_text(transcript)
    parsed = _parse_task_json(text)
    if parsed is None:
        return None, transcript, text
    task = _build_task(
        skill_uid=skill["skill_uid"],
        skill_name=skill["name"],
        skill_category=skill.get("category", ""),
        parsed=parsed,
        domain=domain,
        source_artifact=source_artifact,
    )
    return task, transcript, text


def extract_tasks_from_seeds(
    seed_path: Path,
    output_dir: Path,
    limit: int = 0,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    domain: str = DEFAULT_DOMAIN,
    source_artifact: str = DEFAULT_SOURCE_ARTIFACT,
    verbose: bool = True,
) -> List[ExtractedTask]:
    skills = load_extracted_skills(seed_path)
    seed_dicts: List[Dict[str, Any]] = [
        {
            "skill_uid": s.skill_uid,
            "name": s.name,
            "category": s.category,
            "description": s.description,
            "example": s.example,
        }
        for s in skills
    ]
    if limit > 0:
        seed_dicts = seed_dicts[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = output_dir / "extraction-transcripts.jsonl"
    failures_path = output_dir / "extraction-failures.jsonl"

    agent = LinearAgent(
        model=model,
        system=SYSTEM_PROMPT,
        tools=[],
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        max_turns=1,
    )

    tasks: List[ExtractedTask] = []
    with transcripts_path.open("w", encoding="utf-8") as t_out, \
         failures_path.open("w", encoding="utf-8") as f_out:
        for i, seed in enumerate(seed_dicts, 1):
            if verbose:
                print(f"[{i}/{len(seed_dicts)}] {seed['name']}")

            try:
                task, transcript, text = extract_one(agent, seed, domain, source_artifact)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                f_out.write(json.dumps({
                    "skill_uid": seed["skill_uid"],
                    "skill_name": seed["name"],
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
                continue

            record_line: Dict[str, Any] = {
                "skill_uid": seed["skill_uid"],
                "skill_name": seed["name"],
                "task_uid": task.task_uid if task else None,
                "final_text": text,
                "transcript": transcript.to_dict(),
            }
            t_out.write(json.dumps(record_line, ensure_ascii=False) + "\n")

            if task is None:
                if verbose:
                    print(f"  PARSE_FAIL: final_text={text[:120]!r}")
                f_out.write(json.dumps({
                    "skill_uid": seed["skill_uid"],
                    "skill_name": seed["name"],
                    "error": "json_parse_failed",
                    "final_text": text,
                }, ensure_ascii=False) + "\n")
                continue

            tasks.append(task)
            if verbose:
                print(f"  task_uid={task.task_uid} title={task.title!r}")

    tasks_path = output_dir / "tasks.json"
    save_extracted_tasks(tasks, tasks_path)
    if verbose:
        print(f"\nWrote {len(tasks)} tasks -> {tasks_path}")
        print(f"Wrote transcripts -> {transcripts_path}")

    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate ExtractedTasks from Wikipedia-seed skills using a frontier model.")
    ap.add_argument("--seed", required=True, type=Path, help="Path to skills.json (Wikipedia seed).")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output directory for tasks.json and transcripts.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N seed skills (0 = all).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tasks = extract_tasks_from_seeds(
        seed_path=args.seed,
        output_dir=args.output_dir,
        limit=args.limit,
        model=args.model,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        domain=args.domain,
        source_artifact=args.source_artifact,
        verbose=not args.quiet,
    )
    return 0 if tasks else 1


if __name__ == "__main__":
    sys.exit(main())
