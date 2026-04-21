"""
frontier_catalog_generator.py

PROJECT_SPECS section 1 Method 3 ("Comprehensive Skill Catalog
Generation"). No dataset input: generate a flat, dataset-free catalog
of ~N skills grouped by top-level category, via batched frontier-model
calls with running dedup.

Emits skills.json in the same schema as the Wikipedia seed, so it is a
drop-in seed for frontier_kway_extractor (fueling same-category k=2..5
composition at scale).

Output:
  skills.json               - ExtractedSkill list (drop-in seed)
  catalog.json              - hierarchy {category: [name, ...]}
  generation-transcripts.jsonl - one Transcript per batch call
  summary.json

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_catalog_generator \\
        --output-dir data/pipeline-runs/frontier-catalog/seed-v1 \\
        --target 60 --batch-size 15 --max-batches-per-category 2
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, Transcript, record  # noqa: E402

from c0_utils.text_utils import strip_markdown_fences  # noqa: E402
from c0_utils.uid import generate_uid  # noqa: E402
from c1_types.extracted_skill import (  # noqa: E402
    ExtractedSkill,
    save_extracted_skills,
)


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 3072
DEFAULT_MAX_TOKENS = 4000
GENERATION_METHOD = "frontier-catalog-v1"


DEFAULT_CATEGORIES: Dict[str, str] = {
    "reasoning": "cognitive moves that evaluate evidence, infer causes, and judge plausibility",
    "rhetorical": "moves used to persuade, frame, or redirect an argument or exchange",
    "literary": "devices writers use to shape meaning, imagery, and tone in text",
    "logical": "formal-inference moves that derive conclusions from premises",
    "mathematical": "procedural moves involved in setting up and solving math problems",
    "linguistic": "operations on language structure (parsing, paraphrase, register shift)",
    "scientific": "moves used in scientific inquiry: hypothesis, control, measurement, critique",
    "instruction_following": "moves that parse a user's request and obey its structural constraints",
    "creative_writing": "moves used to invent characters, scenes, plots, and voice",
    "analytical": "moves that decompose a problem into parts and compare alternatives",
}

MIN_NEW_RATE_TO_CONTINUE = 0.25


SYSTEM_PROMPT = (
    "You build an evaluation catalog of procedural skills. A skill is a "
    "specific cognitive or linguistic move, not a topic. Names are "
    "snake_case of 2-4 words. You never repeat or paraphrase skills "
    "already in the catalog."
)


BATCH_PROMPT = """Generate {batch_size} NEW skills in the category below.

Category: {category_name}
Brief:    {category_brief}

Already generated in this category (DO NOT repeat or paraphrase):
{existing_block}

Rules:
  - Each skill is a procedural move (something a solver DOES), not a
    topic (e.g. prefer "counterexample_construction" over "proofs").
  - Name: snake_case, 2-4 words.
  - Distinct from every existing skill above.
  - description: one sentence.
  - example: one concrete instance of the skill exercised in text.

Return ONLY valid JSON:

{{
  "skills": [
    {{"name": "<snake_case>", "description": "<one sentence>", "example": "<one instance>"}}
  ]
}}
"""


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


def _sanitize_name(raw_name: str) -> str:
    label = raw_name.strip().lower()
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


def _to_extracted_skill(
    category: str, name: str, description: str, example: str
) -> ExtractedSkill:
    skill_uid = generate_uid(f"catalog|{category}|{name}")
    return ExtractedSkill(
        skill_uid=skill_uid,
        name=name,
        description=description.strip(),
        procedure=[],
        when_to_use=f"producing text that illustrates {name.replace('_', ' ')}",
        constraints=[],
        source_task_uids=[],
        source_trace_uids=[],
        extraction_method=GENERATION_METHOD,
        category=category,
        example=example.strip(),
    )


def _existing_block(names: List[str]) -> str:
    if not names:
        return "  (none yet)"
    return "\n".join(f"  - {n}" for n in names)


def generate_one_batch(
    agent: LinearAgent,
    category_name: str,
    category_brief: str,
    existing_names: List[str],
    batch_size: int,
):
    prompt = BATCH_PROMPT.format(
        batch_size=batch_size,
        category_name=category_name,
        category_brief=category_brief,
        existing_block=_existing_block(sorted(existing_names)),
    )
    transcript = record(agent.run(prompt))
    text = _final_text(transcript)
    parsed = _parse_json(text)
    return parsed, transcript, text


def _process_category(
    agent: LinearAgent,
    cat_name: str,
    cat_brief: str,
    per_category_target: int,
    batch_size: int,
    max_batches: int,
    transcript_lock: threading.Lock,
    transcript_file,
    verbose: bool,
) -> Tuple[str, List[ExtractedSkill], int]:
    names_seen: Dict[str, ExtractedSkill] = {}
    batches_run = 0

    for batch_idx in range(max_batches):
        if len(names_seen) >= per_category_target:
            break
        existing = sorted(names_seen.keys())
        remaining = max(1, per_category_target - len(existing))
        this_batch = min(batch_size, remaining)

        if verbose:
            print(f"[{cat_name}] batch {batch_idx+1}: "
                  f"have {len(existing)} / target {per_category_target}, requesting {this_batch}")

        try:
            parsed, transcript, text = generate_one_batch(
                agent, cat_name, cat_brief, existing, this_batch
            )
        except Exception as e:
            if verbose:
                print(f"[{cat_name}]   API ERROR: {e}")
            with transcript_lock:
                transcript_file.write(json.dumps({
                    "category": cat_name, "batch": batch_idx + 1, "error": str(e)
                }, ensure_ascii=False) + "\n")
            break

        with transcript_lock:
            transcript_file.write(json.dumps({
                "category": cat_name,
                "batch": batch_idx + 1,
                "existing_count": len(existing),
                "final_text": text,
                "transcript": transcript.to_dict(),
            }, ensure_ascii=False) + "\n")
        batches_run += 1

        if parsed is None:
            if verbose:
                print(f"[{cat_name}]   PARSE_FAIL: {text[:100]!r}")
            break

        new_this_batch = 0
        for item in parsed.get("skills", []) or []:
            if not isinstance(item, dict):
                continue
            raw_name = item.get("name", "")
            if not raw_name:
                continue
            name = _sanitize_name(str(raw_name))
            if not name or name in names_seen:
                continue
            description = str(item.get("description", "")).strip()
            example = str(item.get("example", "")).strip()
            skill = _to_extracted_skill(cat_name, name, description, example)
            names_seen[name] = skill
            new_this_batch += 1

        if verbose:
            print(f"[{cat_name}]   +{new_this_batch} new (total {len(names_seen)})")

        if this_batch > 0 and (new_this_batch / this_batch) < MIN_NEW_RATE_TO_CONTINUE:
            if verbose:
                print(f"[{cat_name}]   low new-rate ({new_this_batch}/{this_batch}), stopping category early")
            break

    return cat_name, list(names_seen.values()), batches_run


def generate_catalog(
    output_dir: Path,
    categories: Dict[str, str] = None,
    target: int = 100,
    batch_size: int = 15,
    max_batches_per_category: int = 5,
    concurrency: int = 1,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> List[ExtractedSkill]:
    categories = categories or DEFAULT_CATEGORIES
    output_dir.mkdir(parents=True, exist_ok=True)
    t_path = output_dir / "generation-transcripts.jsonl"
    summary_path = output_dir / "summary.json"

    per_category_target = max(1, target // len(categories))

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    catalog: Dict[str, List[ExtractedSkill]] = {c: [] for c in categories}
    batches_run = 0

    with t_path.open("w", encoding="utf-8") as t_out:
        transcript_lock = threading.Lock()
        effective_concurrency = max(1, min(concurrency, len(categories)))

        if effective_concurrency == 1:
            for cat_name, cat_brief in categories.items():
                _, skills, b = _process_category(
                    agent, cat_name, cat_brief, per_category_target,
                    batch_size, max_batches_per_category,
                    transcript_lock, t_out, verbose,
                )
                catalog[cat_name] = skills
                batches_run += b
        else:
            if verbose:
                print(f"Running {len(categories)} categories with concurrency={effective_concurrency}")
            with ThreadPoolExecutor(max_workers=effective_concurrency) as ex:
                futures = {
                    ex.submit(
                        _process_category, agent, cat_name, cat_brief,
                        per_category_target, batch_size,
                        max_batches_per_category,
                        transcript_lock, t_out, verbose,
                    ): cat_name
                    for cat_name, cat_brief in categories.items()
                }
                for future in as_completed(futures):
                    cat_name, skills, b = future.result()
                    catalog[cat_name] = skills
                    batches_run += b

    all_skills: List[ExtractedSkill] = []
    hierarchy: Dict[str, List[str]] = {}
    for cat_name, skills in catalog.items():
        hierarchy[cat_name] = [s.name for s in skills]
        all_skills.extend(skills)

    skills_path = output_dir / "skills.json"
    save_extracted_skills(all_skills, skills_path)
    (output_dir / "catalog.json").write_text(
        json.dumps(hierarchy, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    summary = {
        "total_skills": len(all_skills),
        "target": target,
        "per_category_target": per_category_target,
        "per_category_count": {k: len(v) for k, v in catalog.items()},
        "categories": list(categories.keys()),
        "batches_run": batches_run,
        "model": model,
        "generation_method": GENERATION_METHOD,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nTotal {len(all_skills)} skills across {len(categories)} categories")
        for cat, skills in catalog.items():
            print(f"  {cat:24s} {len(skills):3d}")
        print(f"Wrote {skills_path}")
        print(f"Wrote {output_dir / 'catalog.json'}")

    return all_skills


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a dataset-free skill catalog (PROJECT_SPECS 1.M3).")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--target", type=int, default=100, help="Target total skills across categories.")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-batches-per-category", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1, help="Number of categories to process in parallel (default 1, serial).")
    ap.add_argument("--categories", default="", help="Comma-separated category names; default uses built-in 10 categories.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.categories.strip():
        requested = [c.strip() for c in args.categories.split(",") if c.strip()]
        categories = {c: DEFAULT_CATEGORIES.get(c, f"skills in category {c}") for c in requested}
    else:
        categories = DEFAULT_CATEGORIES

    generate_catalog(
        output_dir=args.output_dir, categories=categories,
        target=args.target, batch_size=args.batch_size,
        max_batches_per_category=args.max_batches_per_category,
        concurrency=args.concurrency,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
