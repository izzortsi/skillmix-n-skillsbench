"""
frontier_reviser.py

Close the loop in PROJECT_SPECS section 3 Method 4 ("Agentic Answer
Verification"): take tasks the verifier rejected, feed the judge's
rationale back to a frontier model, and ask it to revise the task so
that every listed skill is genuinely required.

Inputs:
  --tasks       the ORIGINAL tasks.json (produces by any frontier
                extractor). We need the full context to revise.
  --rejections  rejected-tasks.jsonl from frontier_verifier.

Output:
  revised-tasks.json         - ExtractedTask list (re-verify separately)
  revision-transcripts.jsonl - full Transcript per revision attempt
  revision-failures.jsonl    - entries that errored or failed to parse
  summary.json

Typical loop:
  1. extract   -> tasks.json
  2. verify    -> verified-tasks.json + rejected-tasks.jsonl
  3. revise    -> revised-tasks.json                    (this module)
  4. verify    -> verified-revised-tasks.json
  5. union (2) and (4) for the final set.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_reviser \\
        --tasks data/pipeline-runs/frontier-wikipedia/stage1-extraction-k3/tasks.json \\
        --rejections data/pipeline-runs/frontier-wikipedia/stage1b-verification-k3/rejected-tasks.jsonl \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage1c-revision-k3
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

from c0_utils.text_utils import strip_markdown_fences  # noqa: E402
from c0_utils.uid import generate_uid  # noqa: E402
from c1_types.extracted_task import (  # noqa: E402
    ExtractedTask,
    load_extracted_tasks,
    save_extracted_tasks,
)


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_MAX_TOKENS = 4000
REVISION_METHOD_SUFFIX = "-revised-v1"


SYSTEM_PROMPT = (
    "You are a task designer revising a task that failed a judge's "
    "verification. You read the judge's per-skill rationale and produce "
    "a revised task in which every listed skill is genuinely required."
)


REVISE_PROMPT = """Your previous task failed verification. Revise it.

Required skills (all must be genuinely needed by the solver):
{skills_block}

--- your previous attempt ---
Title:    {title}
Question: {question}
Input:    {input}
Correct conclusion: {output}

--- judge's verdict ---
Overall: {overall_summary}
Per-skill findings:
{per_skill_block}

Produce a REVISED task that:
  - Forces the solver to exercise EVERY listed skill. Directly address
    each per-skill "exercises: false" rationale above.
  - Keeps query_type = FREE_FORM with exactly ONE correct conclusion.
  - Has concrete context (not abstract) so skills cannot be skipped.
  - Is not a minor reword of the failed attempt; restructure if needed.

Return ONLY valid JSON:

{{
  "title": "<short descriptive title>",
  "difficulty": "<basic|intermediate|advanced>",
  "question": "<the question>",
  "input": "<the passage/context>",
  "acceptance_criteria": {{
    "must_identify": ["<fact 1>", "<fact 2>"],
    "correct_conclusion": "<single correct conclusion>"
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


def _render_skills_block(skills: List[Dict[str, str]]) -> str:
    lines = []
    for i, s in enumerate(skills, 1):
        cat = s["skill_category"] or "unspecified"
        lines.append(f"  {i}. {s['skill_name']} ({cat})")
    return "\n".join(lines)


def _render_per_skill(verdict: Dict[str, Any]) -> str:
    per = verdict.get("per_skill", []) or []
    lines = []
    for item in per:
        if not isinstance(item, dict):
            continue
        name = item.get("skill_name", "?")
        exercises = item.get("exercises", False)
        rationale = item.get("rationale", "")
        lines.append(f"  - {name}: exercises={bool(exercises)}. {rationale}")
    return "\n".join(lines) if lines else "  (no per-skill findings recorded)"


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


def _build_revised_task(
    original: ExtractedTask,
    parsed: Dict[str, Any],
) -> ExtractedTask:
    title = str(parsed.get("title", "")).strip() or original.title
    question = str(parsed.get("question", "")).strip()
    input_text = str(parsed.get("input", "")).strip()
    difficulty = str(parsed.get("difficulty", original.difficulty or "intermediate")).strip() or "intermediate"

    ac = parsed.get("acceptance_criteria", {}) or {}
    must_identify = [str(x) for x in (ac.get("must_identify", []) or [])]
    correct_conclusion = str(ac.get("correct_conclusion", "")).strip()

    new_ac = dict(original.acceptance_criteria or {})
    new_ac["must_identify"] = must_identify
    new_ac["correct_conclusion"] = correct_conclusion
    new_ac["revised_from_task_uid"] = original.task_uid

    new_method = (original.extraction_method or "") + REVISION_METHOD_SUFFIX
    new_task_uid = generate_uid(f"{new_method}|{original.task_uid}|{title}")

    return ExtractedTask(
        task_uid=new_task_uid,
        title=title,
        domain=original.domain,
        source_artifact=original.source_artifact,
        source_document_uid=original.source_document_uid,
        question=question,
        input=input_text,
        output=correct_conclusion,
        difficulty=difficulty,
        acceptance_criteria=new_ac,
        query_type=original.query_type,
        extraction_method=new_method,
    )


def _load_rejections(rejections_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with rejections_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def revise_one(
    agent: LinearAgent,
    task: ExtractedTask,
    verdict: Dict[str, Any],
) -> (Optional[ExtractedTask], Transcript, str):
    skills = _skills_of(task)
    if not skills:
        return None, Transcript(), "no_skills_found"

    prompt = REVISE_PROMPT.format(
        skills_block=_render_skills_block(skills),
        title=task.title,
        question=task.question,
        input=task.input,
        output=task.output or "(none)",
        overall_summary=(verdict.get("overall", {}) or {}).get("summary", ""),
        per_skill_block=_render_per_skill(verdict),
    )
    transcript = record(agent.run(prompt))
    text = _final_text(transcript)
    parsed = _parse_json(text)
    if parsed is None:
        return None, transcript, text
    return _build_revised_task(task, parsed), transcript, text


def revise_tasks(
    tasks_path: Path,
    rejections_path: Path,
    output_dir: Path,
    limit: int = 0,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> List[ExtractedTask]:
    original_tasks = {t.task_uid: t for t in load_extracted_tasks(tasks_path)}
    rejections = _load_rejections(rejections_path)
    if limit > 0:
        rejections = rejections[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    t_path = output_dir / "revision-transcripts.jsonl"
    f_path = output_dir / "revision-failures.jsonl"
    s_path = output_dir / "summary.json"

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    revised: List[ExtractedTask] = []
    errors = 0
    parse_fails = 0
    not_found = 0

    with t_path.open("w", encoding="utf-8") as t_out, \
         f_path.open("w", encoding="utf-8") as f_out:
        for i, entry in enumerate(rejections, 1):
            task_uid = entry.get("task_uid", "")
            verdict = entry.get("verdict", {}) or {}
            original = original_tasks.get(task_uid)
            if original is None:
                not_found += 1
                if verbose:
                    print(f"[{i}/{len(rejections)}] {task_uid}: NOT FOUND in tasks.json")
                f_out.write(json.dumps({
                    "task_uid": task_uid, "error": "original_task_not_found"
                }, ensure_ascii=False) + "\n")
                continue

            if verbose:
                print(f"[{i}/{len(rejections)}] {original.title}")

            try:
                new_task, transcript, text = revise_one(agent, original, verdict)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  ERROR: {e}")
                f_out.write(json.dumps({
                    "task_uid": task_uid, "error": str(e)
                }, ensure_ascii=False) + "\n")
                continue

            t_out.write(json.dumps({
                "original_task_uid": task_uid,
                "revised_task_uid": new_task.task_uid if new_task else None,
                "final_text": text,
                "transcript": transcript.to_dict(),
            }, ensure_ascii=False) + "\n")

            if new_task is None:
                parse_fails += 1
                if verbose:
                    print(f"  PARSE_FAIL: {text[:100]!r}")
                f_out.write(json.dumps({
                    "task_uid": task_uid,
                    "error": "json_parse_failed",
                    "final_text": text,
                }, ensure_ascii=False) + "\n")
                continue

            revised.append(new_task)
            if verbose:
                print(f"  new_task_uid={new_task.task_uid} title={new_task.title!r}")

    out_path = output_dir / "revised-tasks.json"
    save_extracted_tasks(revised, out_path)
    summary = {
        "rejections_loaded": len(rejections),
        "original_not_found": not_found,
        "parse_failures": parse_fails,
        "api_errors": errors,
        "revised_tasks": len(revised),
        "model": model,
    }
    s_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n{len(revised)}/{len(rejections)} revised -> {out_path}")

    return revised


def main() -> int:
    ap = argparse.ArgumentParser(description="Revise verifier-rejected tasks using the judge's rationale (PROJECT_SPECS 3.M4 close-the-loop).")
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--rejections", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    revised = revise_tasks(
        tasks_path=args.tasks, rejections_path=args.rejections,
        output_dir=args.output_dir, limit=args.limit,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, verbose=not args.quiet,
    )
    return 0 if revised else 1


if __name__ == "__main__":
    sys.exit(main())
