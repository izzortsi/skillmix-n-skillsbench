"""
s1.m3 — Comprehensive Skill Catalog Generation (PROJECT_SPECS §1 Method 3).

Prompt: "You are a great chat agent. Give us instruction-following skills in
this format." Request a hierarchical organization. Target: ~1000 skills.

Strategy (faithful to the paper's pipeline, scaled to batched inference):
  1. Elicit N top-level categories.
  2. For each category, request batches of M skills until the per-category
     quota is filled (with an avoid-list so the model does not repeat itself).
  3. Dedupe globally by kebab name; category fills its own parent slot so
     the Skill hierarchy has exactly two levels (category-as-root,
     skill-as-child).

Output: Skill JSON with source="s1.m3.catalog" and hierarchy via
parent_skill_uid (category UIDs are themselves stored as Skills so a flat
JSON file is enough).

Usage:
    python -m cli s1.m3 --target-size 200 --categories 10 \\
                        --provider anthropic:claude-opus-4-7 \\
                        --out data/s1-m3-catalog.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set

from core.schemas import Skill, stable_uid, to_kebab, save_json
from core.providers import create_provider


SOURCE_ID = "s1.m3.catalog"


CATEGORY_PROMPT = """You are a great chat agent. List {n_categories} broad top-level
categories of instruction-following skills that a helpful assistant exercises
across all requests it receives. Categories should be MECE-ish -- minimal
overlap -- and cover breadth (e.g. writing, reasoning, coding, math, social,
research, planning, formatting, etc.).

Return ONLY valid JSON in exactly this shape, no markdown fences, no preface:

{{
  "categories": [
    {{
      "name":        "<2-4 words, short category label>",
      "description": "<one-sentence definition of the category>"
    }}
  ]
}}
"""


SKILLS_PROMPT = """You are a great chat agent. List {batch_size} instruction-following
skills in the category "{category}" ({category_description}).

Each skill should be a concrete capability that can be named and demonstrated
(e.g. "paraphrasing", "step-by-step reasoning", "code review", "unit
conversion"). Prefer skills that would plausibly be tested in isolation --
atomic, composable, recognizable.

{avoid_block}

Return ONLY valid JSON in exactly this shape, no markdown fences, no preface:

{{
  "skills": [
    {{
      "name":        "<2-4 words>",
      "description": "<one-sentence definition>",
      "example":     "<one short illustrative use>",
      "when_to_use": "<the circumstance that prompts the skill>"
    }}
  ]
}}
"""


def _parse_json_object(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else ""
    if not candidate:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else ""
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _build_avoid_block(already: List[str]) -> str:
    if not already:
        return ""
    # Keep the avoid-list bounded so we do not blow the prompt on large runs.
    shown = already[-80:]
    bullet = "\n".join(f"  - {n}" for n in shown)
    return ("Do NOT propose any of the following (already collected in this "
            "category); pick genuinely different skills:\n" + bullet + "\n")


def elicit_categories(provider, n_categories: int) -> List[Skill]:
    """Ask for n_categories top-level categories. Stored as parent Skills."""
    prompt = CATEGORY_PROMPT.format(n_categories=n_categories)
    result = provider.chat([{"role": "user", "content": prompt}])
    data = _parse_json_object(result.text)

    categories: List[Skill] = []
    seen: set[str] = set()
    for entry in data.get("categories", []) or []:
        display = str(entry.get("name", "")).strip()
        if not display:
            continue
        kebab = to_kebab(display)
        if kebab in seen:
            continue
        seen.add(kebab)
        categories.append(Skill(
            skill_uid=stable_uid(f"{SOURCE_ID}|category|{kebab}"),
            name=kebab,
            description=str(entry.get("description", "")).strip(),
            category="",
            example="",
            procedure=[],
            when_to_use="",
            constraints=[],
            source=SOURCE_ID,
            parent_skill_uid="",
        ))
    return categories


def elicit_skills_in_category(
    provider,
    category: Skill,
    batch_size: int,
    avoid: List[str] = None,
) -> List[Skill]:
    """One batch call for skills under `category`. Caller dedupes globally."""
    prompt = SKILLS_PROMPT.format(
        category=category.name,
        category_description=category.description or "(no description)",
        batch_size=batch_size,
        avoid_block=_build_avoid_block(avoid or []),
    )
    result = provider.chat([{"role": "user", "content": prompt}])
    data = _parse_json_object(result.text)

    skills: List[Skill] = []
    seen: set[str] = set()
    for entry in data.get("skills", []) or []:
        display = str(entry.get("name", "")).strip()
        if not display:
            continue
        kebab = to_kebab(display)
        if kebab in seen or kebab == category.name:
            continue
        seen.add(kebab)
        skills.append(Skill(
            skill_uid=stable_uid(f"{SOURCE_ID}|skill|{category.name}|{kebab}"),
            name=kebab,
            description=str(entry.get("description", "")).strip(),
            category=category.name,
            example=str(entry.get("example", "")).strip(),
            procedure=[],
            when_to_use=str(entry.get("when_to_use", "")).strip(),
            constraints=[],
            source=SOURCE_ID,
            parent_skill_uid=category.skill_uid,
        ))
    return skills


def _process_category(
    provider,
    category: Skill,
    per_category_target: int,
    batch_size: int,
    max_batches_per_category: int,
    verbose: bool,
) -> List[Skill]:
    """Generate up to `per_category_target` skills under `category`. Caller dedupes
    globally; this function only dedupes within the category (by name).

    Thread-safe as long as `provider` itself is thread-safe (anthropic.Anthropic
    is). No shared mutable state is touched here.
    """
    collected: List[Skill] = []
    collected_names: Set[str] = set()
    for batch_idx in range(max_batches_per_category):
        remaining = per_category_target - len(collected)
        if remaining <= 0:
            break
        ask = min(batch_size, remaining)
        batch = elicit_skills_in_category(
            provider, category, batch_size=ask, avoid=list(collected_names),
        )
        added = 0
        for s in batch:
            if s.name in collected_names:
                continue
            collected_names.add(s.name)
            collected.append(s)
            added += 1
        if verbose:
            print(f"[{category.name}] batch {batch_idx+1}: asked {ask}, added {added} "
                  f"(dedup dropped {len(batch)-added})")
        if added == 0:
            # Model is saturated on this category; stop re-asking.
            break
    return collected


def generate_catalog(
    provider,
    target_size: int = 200,
    n_categories: int = 10,
    batch_size: int = 20,
    max_batches_per_category: int = 5,
    concurrency: int = 1,
    verbose: bool = False,
) -> List[Skill]:
    """Build a catalog of ~target_size skills across n_categories.

    Per-category batches run with ThreadPoolExecutor when concurrency > 1.
    The report in b1.reports/260421 measured 3.23x wall-clock speedup at
    concurrency=3 on a 3-category, 30-skill run.

    Returns categories first (as parent Skills), then all child skills.
    Globally unique by kebab name.
    """
    if target_size < n_categories:
        raise ValueError(
            f"target_size ({target_size}) must be >= n_categories ({n_categories})"
        )

    if verbose:
        print(f"Eliciting {n_categories} categories...")
    categories = elicit_categories(provider, n_categories)
    if not categories:
        if verbose:
            print("  (no categories returned; aborting)")
        return []
    if verbose:
        for c in categories:
            print(f"  - {c.name}: {c.description[:60]}")

    per_category_quota = math.ceil((target_size - len(categories)) / len(categories))
    effective_concurrency = max(1, min(concurrency, len(categories)))

    per_category_skills: dict[str, List[Skill]] = {}
    if effective_concurrency == 1:
        for c_idx, category in enumerate(categories, 1):
            if verbose:
                print(f"\n[category {c_idx}/{len(categories)}] {category.name} "
                      f"(target {per_category_quota} skills)")
            per_category_skills[category.name] = _process_category(
                provider, category, per_category_quota, batch_size,
                max_batches_per_category, verbose,
            )
    else:
        if verbose:
            print(f"\nRunning {len(categories)} categories with concurrency={effective_concurrency}")
        with ThreadPoolExecutor(max_workers=effective_concurrency) as ex:
            futures = {
                ex.submit(
                    _process_category, provider, category, per_category_quota,
                    batch_size, max_batches_per_category, verbose,
                ): category.name
                for category in categories
            }
            for fut in as_completed(futures):
                cat_name = futures[fut]
                per_category_skills[cat_name] = fut.result()

    # Apply global dedup in deterministic (category) order so the output
    # ordering does not depend on thread scheduling.
    seen_global: Set[str] = {c.name for c in categories}
    child_skills: List[Skill] = []
    for category in categories:
        for s in per_category_skills.get(category.name, []):
            if s.name in seen_global:
                continue
            seen_global.add(s.name)
            child_skills.append(s)

    return categories + child_skills


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target-size", type=int, default=200,
                        help="Target total skill count incl. categories (default: 200)")
    parser.add_argument("--categories", type=int, default=10,
                        help="Number of top-level categories (default: 10)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Skills requested per call (default: 20)")
    parser.add_argument("--max-batches-per-category", type=int, default=5,
                        help="Cap on retry batches per category (default: 5)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Categories processed in parallel (default: 1, serial). "
                             "3-4 is a good choice on multi-category runs.")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7",
                        help="Provider spec (default: anthropic:claude-opus-4-7)")
    parser.add_argument("--out", type=Path, default=Path("data/s1-m3-catalog.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    catalog = generate_catalog(
        provider=provider,
        target_size=args.target_size,
        n_categories=args.categories,
        batch_size=args.batch_size,
        max_batches_per_category=args.max_batches_per_category,
        concurrency=args.concurrency,
        verbose=args.verbose,
    )

    save_json(catalog, args.out)
    n_cat = sum(1 for s in catalog if not s.parent_skill_uid)
    n_sk = len(catalog) - n_cat
    print(f"\nWrote {len(catalog)} skills -> {args.out}  ({n_cat} categories, {n_sk} skills)")


if __name__ == "__main__":
    main()
