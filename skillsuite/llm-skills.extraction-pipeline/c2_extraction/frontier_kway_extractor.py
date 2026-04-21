"""
frontier_kway_extractor.py

Generalize k=2 pair composition (PROJECT_SPECS section 3 Method 1) to
k=2..5 skill-tuple composition. Given a set of seed skills and a value
k, randomly sample k-tuples and ask a frontier model to design one task
that REQUIRES every skill in the tuple.

For small k on small seed sets we enumerate combinations. For larger k
or larger seed sets (where C(n,k) explodes), we sample k-tuples by
drawing k-element samples directly and deduplicating by sorted uid
tuple.

Schema: same ExtractedTask shape as frontier_pair_extractor, with
acceptance_criteria carrying skill_uids/skill_names/skill_categories
lists of length k, and k=<N>.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_kway_extractor \\
        --seed /workspace/llm-skills/skillsuite/llm-skills.shared-data/260420.skillmix-wikipedia-seed/skills.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage1-extraction-k3 \\
        --k 3 --max-tuples 3 --random-seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from math import comb
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

MIN_K = 2
MAX_K = 5
# Enumerate all C(n,k) combinations only when it's cheap. Above this we
# fall back to random k-tuple sampling with dedup, to avoid materializing
# e.g. C(100,5)=75_287_520 tuples in memory.
ENUMERATION_THRESHOLD = 50_000


def _extraction_method(k: int) -> str:
    return f"frontier-wikipedia-k{k}-v1"


SYSTEM_PROMPT = (
    "You are a task designer for an LLM evaluation benchmark. You design "
    "tasks that force a solver to exercise EVERY listed skill while "
    "producing one free-form answer."
)


KWAY_PROMPT = """Design ONE evaluation task that REQUIRES all {k} skills below.

{skills_block}

Constraints:
  - query_type = FREE_FORM, exactly ONE correct conclusion.
  - A correct solver cannot reach the correct conclusion while skipping
    ANY of the {k} skills. If any skill is decorative or optional,
    redesign until the task genuinely needs all of them.
  - Difficulty: basic, intermediate, or advanced.
  - "input" sets up a concrete situation.
  - "must_identify" lists {k}-{must_upper} supporting facts that
    collectively reference every skill.
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


def _render_skills_block(skills: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, s in enumerate(skills, 1):
        cat = s.get("category", "") or "unspecified"
        lines.append(f"Skill {i}: {s['name']}  ({cat})")
        lines.append(f"  Definition: {s.get('description', '')}")
        lines.append(f"  Example:    {s.get('example', '') or '(no example)'}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_prompt(tuple_: List[Dict[str, Any]]) -> str:
    k = len(tuple_)
    return KWAY_PROMPT.format(
        k=k,
        must_upper=max(k, 4),
        skills_block=_render_skills_block(tuple_),
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


def _build_task(
    tuple_: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    domain: str,
    source_artifact: str,
    extraction_method: str,
) -> ExtractedTask:
    k = len(tuple_)
    names = [s["name"] for s in tuple_]
    title = str(parsed.get("title", "")).strip() or " + ".join(names)
    question = str(parsed.get("question", "")).strip()
    input_text = str(parsed.get("input", "")).strip()
    difficulty = str(parsed.get("difficulty", "intermediate")).strip() or "intermediate"

    ac = parsed.get("acceptance_criteria", {}) or {}
    must_identify = [str(x) for x in (ac.get("must_identify", []) or [])]
    correct_conclusion = str(ac.get("correct_conclusion", "")).strip()

    tuple_key = "+".join(sorted(s["skill_uid"] for s in tuple_))
    task_uid = generate_uid(f"{extraction_method}|{tuple_key}|{title}")

    ac_out: Dict[str, Any] = {
        "must_identify": must_identify,
        "correct_conclusion": correct_conclusion,
        "skill_uids": [s["skill_uid"] for s in tuple_],
        "skill_names": names,
        "skill_categories": [s.get("category", "") for s in tuple_],
        "k": k,
    }

    return ExtractedTask(
        task_uid=task_uid,
        title=title,
        domain=domain,
        source_artifact=source_artifact,
        source_document_uid=tuple_key,
        question=question,
        input=input_text,
        output=correct_conclusion,
        difficulty=difficulty,
        acceptance_criteria=ac_out,
        query_type="FREE_FORM",
        extraction_method=extraction_method,
    )


def _filter_seeds_by_category(
    seeds: List[Dict[str, Any]],
    allowed_categories: List[str],
) -> List[Dict[str, Any]]:
    if not allowed_categories:
        return seeds
    allowed = {c.strip() for c in allowed_categories if c.strip()}
    return [s for s in seeds if (s.get("category", "") or "") in allowed]


def _same_category_tuples(
    seeds: List[Dict[str, Any]],
    k: int,
) -> List[Tuple[Dict[str, Any], ...]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for s in seeds:
        cat = s.get("category", "") or ""
        by_cat.setdefault(cat, []).append(s)
    tuples: List[Tuple[Dict[str, Any], ...]] = []
    for members in by_cat.values():
        if len(members) < k:
            continue
        tuples.extend(itertools.combinations(members, k))
    return tuples


def _sample_tuples(
    seeds: List[Dict[str, Any]],
    k: int,
    max_tuples: int,
    random_seed: int,
    same_category: bool = False,
) -> List[Tuple[Dict[str, Any], ...]]:
    n = len(seeds)
    if n < k:
        raise ValueError(f"need at least {k} seed skills, got {n}")

    rng = random.Random(random_seed)

    if same_category:
        pool = _same_category_tuples(seeds, k)
        if not pool:
            raise ValueError(
                f"same-category sampling: no category has >= {k} members"
            )
        if max_tuples <= 0 or max_tuples >= len(pool):
            return pool
        return rng.sample(pool, max_tuples)

    total = comb(n, k)
    if total <= ENUMERATION_THRESHOLD:
        all_tuples = list(itertools.combinations(seeds, k))
        if max_tuples <= 0 or max_tuples >= len(all_tuples):
            return all_tuples
        return rng.sample(all_tuples, max_tuples)

    target = max_tuples if max_tuples > 0 else min(total, 10_000)
    seen: set = set()
    out: List[Tuple[Dict[str, Any], ...]] = []
    attempts = 0
    max_attempts = target * 20
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        sample = tuple(rng.sample(seeds, k))
        key = tuple(sorted(s["skill_uid"] for s in sample))
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    return out


def extract_kway_tasks(
    seed_path: Path,
    output_dir: Path,
    k: int,
    max_tuples: int = 10,
    random_seed: int = 42,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    domain: str = DEFAULT_DOMAIN,
    source_artifact: str = DEFAULT_SOURCE_ARTIFACT,
    categories: List[str] = None,
    same_category: bool = False,
    verbose: bool = True,
) -> List[ExtractedTask]:
    if k < MIN_K or k > MAX_K:
        raise ValueError(f"k must be in [{MIN_K}, {MAX_K}], got {k}")

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

    if categories:
        before = len(seed_dicts)
        seed_dicts = _filter_seeds_by_category(seed_dicts, categories)
        if verbose:
            print(f"category filter: {before} -> {len(seed_dicts)} seeds "
                  f"(allowed: {sorted(set(categories))})")

    tuples = _sample_tuples(seed_dicts, k, max_tuples, random_seed, same_category=same_category)
    extraction_method = _extraction_method(k)
    if same_category:
        extraction_method = extraction_method + "-samecat"

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
        for i, tpl in enumerate(tuples, 1):
            names = [s["name"] for s in tpl]
            if verbose:
                print(f"[{i}/{len(tuples)}] k={k}  " + " + ".join(names))

            try:
                transcript = record(agent.run(_build_prompt(list(tpl))))
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                f_out.write(json.dumps({
                    "skill_uids": [s["skill_uid"] for s in tpl],
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
                continue

            text = _final_text(transcript)
            parsed = _parse_json(text)
            t_out.write(json.dumps({
                "skill_uids": [s["skill_uid"] for s in tpl],
                "skill_names": names,
                "k": k,
                "task_uid": None,
                "final_text": text,
                "transcript": transcript.to_dict(),
            }, ensure_ascii=False) + "\n")

            if parsed is None:
                if verbose:
                    print(f"  PARSE_FAIL: {text[:100]!r}")
                f_out.write(json.dumps({
                    "skill_uids": [s["skill_uid"] for s in tpl],
                    "error": "json_parse_failed",
                    "final_text": text,
                }, ensure_ascii=False) + "\n")
                continue

            task = _build_task(list(tpl), parsed, domain, source_artifact, extraction_method)
            tasks.append(task)
            if verbose:
                print(f"  task_uid={task.task_uid} title={task.title!r}")

    out_path = output_dir / "tasks.json"
    save_extracted_tasks(tasks, out_path)
    if verbose:
        print(f"\nWrote {len(tasks)} k={k} tasks -> {out_path}")

    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate k-way ExtractedTasks requiring a tuple of Wikipedia-seed skills (PROJECT_SPECS 3.M1, k=2..5).")
    ap.add_argument("--seed", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k", type=int, required=True, help=f"Tuple size, {MIN_K}..{MAX_K}")
    ap.add_argument("--max-tuples", type=int, default=10)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT)
    ap.add_argument("--categories", default="", help="Comma-separated category filter (e.g. 'rhetorical,logical').")
    ap.add_argument("--same-category", action="store_true", help="Every k-tuple must come from a single category.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    tasks = extract_kway_tasks(
        seed_path=args.seed, output_dir=args.output_dir,
        k=args.k, max_tuples=args.max_tuples, random_seed=args.random_seed,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, domain=args.domain,
        source_artifact=args.source_artifact,
        categories=cats, same_category=args.same_category,
        verbose=not args.quiet,
    )
    return 0 if tasks else 1


if __name__ == "__main__":
    sys.exit(main())
