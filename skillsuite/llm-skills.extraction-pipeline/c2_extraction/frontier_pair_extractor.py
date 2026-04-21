"""
frontier_pair_extractor.py

Stage 1b (parallel path, k=2): generate ExtractedTask records that
require a PAIR of Wikipedia-seed skills to solve.

This closes PROJECT_SPECS section 3 Method 1 ("Random Skill Pair
Composition"): randomly sample skill pairs, prompt a frontier model to
create one plausible task exhibiting both skills, capture the full
thinking transcript.

Pair-task schema reuses ExtractedTask; acceptance_criteria carries:
    skill_uids:       [uid_a, uid_b]
    skill_names:      [name_a, name_b]
    skill_categories: [cat_a, cat_b]
    k:                2

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_pair_extractor \\
        --seed /workspace/llm-skills/skillsuite/llm-skills.shared-data/260420.skillmix-wikipedia-seed/skills.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage1-extraction-k2 \\
        --max-pairs 3 \\
        --random-seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
EXTRACTION_METHOD = "frontier-wikipedia-k2-v1"


SYSTEM_PROMPT = (
    "You are a task designer for an LLM evaluation benchmark. You design "
    "tasks that force a solver to exercise TWO named skills simultaneously "
    "while producing one free-form answer."
)


PAIR_PROMPT = """Design ONE evaluation task that REQUIRES both skills below.

Skill A: {name_a}  ({cat_a})
  Definition: {def_a}
  Example:    {ex_a}

Skill B: {name_b}  ({cat_b})
  Definition: {def_b}
  Example:    {ex_b}

Constraints:
  - query_type = FREE_FORM, exactly ONE correct conclusion.
  - A correct solver cannot avoid exercising EITHER skill. If the task can
    be answered well while skipping one of the skills, you have not met
    the constraint; redesign.
  - Difficulty: basic, intermediate, or advanced.
  - "input" sets up a concrete situation.
  - "must_identify" lists 2-4 supporting facts naming BOTH skills.
  - "correct_conclusion" is the single one-line correct answer.

Return ONLY valid JSON (no markdown, no prose):

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


def _build_prompt(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    return PAIR_PROMPT.format(
        name_a=a["name"], cat_a=a.get("category", "") or "unspecified",
        def_a=a.get("description", ""), ex_a=a.get("example", "") or "(no example)",
        name_b=b["name"], cat_b=b.get("category", "") or "unspecified",
        def_b=b.get("description", ""), ex_b=b.get("example", "") or "(no example)",
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


def _build_pair_task(
    a: Dict[str, Any],
    b: Dict[str, Any],
    parsed: Dict[str, Any],
    domain: str,
    source_artifact: str,
) -> ExtractedTask:
    title = str(parsed.get("title", "")).strip() or f"{a['name']} + {b['name']}"
    question = str(parsed.get("question", "")).strip()
    input_text = str(parsed.get("input", "")).strip()
    difficulty = str(parsed.get("difficulty", "intermediate")).strip() or "intermediate"

    ac = parsed.get("acceptance_criteria", {}) or {}
    must_identify = [str(x) for x in (ac.get("must_identify", []) or [])]
    correct_conclusion = str(ac.get("correct_conclusion", "")).strip()

    pair_key = "+".join(sorted([a["skill_uid"], b["skill_uid"]]))
    task_uid = generate_uid(f"{EXTRACTION_METHOD}|{pair_key}|{title}")

    ac_out: Dict[str, Any] = {
        "must_identify": must_identify,
        "correct_conclusion": correct_conclusion,
        "skill_uids": [a["skill_uid"], b["skill_uid"]],
        "skill_names": [a["name"], b["name"]],
        "skill_categories": [a.get("category", ""), b.get("category", "")],
        "k": 2,
    }

    return ExtractedTask(
        task_uid=task_uid,
        title=title,
        domain=domain,
        source_artifact=source_artifact,
        source_document_uid=pair_key,
        question=question,
        input=input_text,
        output=correct_conclusion,
        difficulty=difficulty,
        acceptance_criteria=ac_out,
        query_type="FREE_FORM",
        extraction_method=EXTRACTION_METHOD,
    )


def _sample_pairs(
    seeds: List[Dict[str, Any]],
    max_pairs: int,
    random_seed: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    all_pairs = list(itertools.combinations(seeds, 2))
    if max_pairs <= 0 or max_pairs >= len(all_pairs):
        return all_pairs
    rng = random.Random(random_seed)
    return rng.sample(all_pairs, max_pairs)


def extract_pair_tasks(
    seed_path: Path,
    output_dir: Path,
    max_pairs: int = 10,
    random_seed: int = 42,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    domain: str = DEFAULT_DOMAIN,
    source_artifact: str = DEFAULT_SOURCE_ARTIFACT,
    verbose: bool = True,
) -> List[ExtractedTask]:
    skills = load_extracted_skills(seed_path)
    seed_dicts = [
        {
            "skill_uid": s.skill_uid,
            "name": s.name,
            "category": s.category,
            "description": s.description,
            "example": s.example,
        }
        for s in skills
    ]
    pairs = _sample_pairs(seed_dicts, max_pairs, random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    t_path = output_dir / "extraction-transcripts.jsonl"
    f_path = output_dir / "extraction-failures.jsonl"

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    tasks: List[ExtractedTask] = []
    with t_path.open("w", encoding="utf-8") as t_out, \
         f_path.open("w", encoding="utf-8") as f_out:
        for i, (a, b) in enumerate(pairs, 1):
            if verbose:
                print(f"[{i}/{len(pairs)}] {a['name']}  +  {b['name']}")

            try:
                transcript = record(agent.run(_build_prompt(a, b)))
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                f_out.write(json.dumps({
                    "skill_uids": [a["skill_uid"], b["skill_uid"]],
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
                continue

            text = _final_text(transcript)
            parsed = _parse_json(text)

            t_out.write(json.dumps({
                "skill_uids": [a["skill_uid"], b["skill_uid"]],
                "skill_names": [a["name"], b["name"]],
                "task_uid": None,
                "final_text": text,
                "transcript": transcript.to_dict(),
            }, ensure_ascii=False) + "\n")

            if parsed is None:
                if verbose:
                    print(f"  PARSE_FAIL: {text[:100]!r}")
                f_out.write(json.dumps({
                    "skill_uids": [a["skill_uid"], b["skill_uid"]],
                    "error": "json_parse_failed",
                    "final_text": text,
                }, ensure_ascii=False) + "\n")
                continue

            task = _build_pair_task(a, b, parsed, domain, source_artifact)
            tasks.append(task)
            if verbose:
                print(f"  task_uid={task.task_uid} title={task.title!r}")

    out_path = output_dir / "tasks.json"
    save_extracted_tasks(tasks, out_path)
    if verbose:
        print(f"\nWrote {len(tasks)} pair-tasks -> {out_path}")

    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate k=2 ExtractedTasks requiring pairs of Wikipedia-seed skills.")
    ap.add_argument("--seed", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-pairs", type=int, default=10)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tasks = extract_pair_tasks(
        seed_path=args.seed, output_dir=args.output_dir,
        max_pairs=args.max_pairs, random_seed=args.random_seed,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, domain=args.domain,
        source_artifact=args.source_artifact, verbose=not args.quiet,
    )
    return 0 if tasks else 1


if __name__ == "__main__":
    sys.exit(main())
