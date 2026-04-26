"""
s1.m2 — Direct Elicitation of Novel Skills (PROJECT_SPECS §1 Method 2).

Prompt the LLM: "Give me a broad skill that has no existing name but which
many humans will recognize." Then follow up to elicit subskills:
  - "What are some subskills of [skill name]?"
  - "Give me finer-grained skills associated with [skill name]"

Cited example: "linguistic exorcism" (reword text to be less offensive/more
useful) -> subskills "synonym_substitution_technique", "tone_moderation_adjustment".

Output: Skill JSON with source="s1.m2.direct-elicitation", hierarchical via
parent_skill_uid.

Usage:
    python -m cli s1.m2 --n-roots 5 --max-subskills 4 \\
                        --provider anthropic:claude-opus-4-7 \\
                        --out data/s1-m2-skills.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

from core.schemas import Skill, stable_uid, to_kebab, save_json
from core.providers import create_provider


SOURCE_ID = "s1.m2.direct-elicitation"


ROOT_BATCH_PROMPT = """You are a careful observer of human cognition.

Give me {count} broad skills that have no widely-used conventional name but
which many humans will recognize once pointed out. A classic example is
"linguistic exorcism": rewording a piece of text to be less offensive or more
useful -- everyone does it, but no standard term exists.

Each skill must be genuinely distinct from the others. Do NOT repeat, rename,
or paraphrase any skill.

{avoid_block}

Return ONLY valid JSON in exactly this shape, no markdown fences, no preface:

{{
  "skills": [
    {{
      "name":        "<2-4 words, a novel label for the skill>",
      "description": "<one-sentence definition>",
      "category":    "<rhetorical | cognitive | linguistic | social | other>",
      "example":     "<one concrete situation in which the skill is applied>",
      "when_to_use": "<the circumstance that prompts the skill>"
    }}
  ]
}}
"""


SUBSKILL_PROMPT = """We are building a hierarchy under the parent skill below.

Parent skill:
  name:        {name}
  description: {description}
  example:     {example}

Give up to {max_subskills} finer-grained subskills of this parent. Each
subskill should be a strictly narrower capability that a person exercising
the parent skill will recognize as one of the moves they make.

Return ONLY valid JSON in exactly this shape, no markdown fences, no preface:

{{
  "subskills": [
    {{
      "name":        "<2-4 words>",
      "description": "<one-sentence definition>",
      "example":     "<one concrete situation>",
      "when_to_use": "<the circumstance that prompts the subskill>"
    }}
  ]
}}
"""


def _parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response. Returns {} on failure."""
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
    bullet = "\n".join(f"  - {n}" for n in already)
    return ("Do NOT propose any of the following (already collected); "
            "pick genuinely different skills:\n" + bullet + "\n")


def _entry_to_skill(entry: dict) -> Optional[Skill]:
    """Convert one model-returned entry to a root Skill; return None if the
    entry is missing a name or the description is empty."""
    display = str(entry.get("name", "")).strip()
    if not display:
        return None
    description = str(entry.get("description", "")).strip()
    if not description:
        return None
    kebab = to_kebab(display)
    return Skill(
        skill_uid=stable_uid(f"{SOURCE_ID}|root|{kebab}"),
        name=kebab,
        description=description,
        category=str(entry.get("category", "")).strip(),
        example=str(entry.get("example", "")).strip(),
        procedure=[],
        when_to_use=str(entry.get("when_to_use", "")).strip(),
        constraints=[],
        source=SOURCE_ID,
        parent_skill_uid="",
    )


def elicit_root_batch(provider, count: int, avoid: Optional[List[str]] = None) -> List[Skill]:
    """Single LLM call returning up to `count` novel, unnamed-but-recognizable skills.

    Efficient vs. per-call elicitation: one API round-trip instead of N. Caller
    still runs multiple batches when the model drops duplicates or falls short.
    """
    avoid = avoid or []
    prompt = ROOT_BATCH_PROMPT.format(count=count, avoid_block=_build_avoid_block(avoid))
    result = provider.chat([{"role": "user", "content": prompt}])
    data = _parse_json_object(result.text)

    raw_list = data.get("skills", []) or []
    out: List[Skill] = []
    seen_in_batch: set = set()
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        s = _entry_to_skill(entry)
        if s is None or s.name in seen_in_batch:
            continue
        seen_in_batch.add(s.name)
        out.append(s)
    return out


def elicit_root_skill(provider, avoid: Optional[List[str]] = None) -> Optional[Skill]:
    """Backward-compatible single-root helper. Returns the first root from a
    1-element batch; returns None if the model produced nothing usable."""
    batch = elicit_root_batch(provider, count=1, avoid=avoid)
    return batch[0] if batch else None


def elicit_subskills(root: Skill, provider, max_subskills: int = 5) -> List[Skill]:
    """Follow-up call to extract subskills of `root`. parent_skill_uid links each."""
    prompt = SUBSKILL_PROMPT.format(
        name=root.name,
        description=root.description,
        example=root.example or "(none)",
        max_subskills=max_subskills,
    )
    result = provider.chat([{"role": "user", "content": prompt}])
    data = _parse_json_object(result.text)

    raw_list = data.get("subskills", []) or []
    subskills: List[Skill] = []
    seen: set[str] = set()
    for entry in raw_list[:max_subskills]:
        display = str(entry.get("name", "")).strip()
        if not display:
            continue
        kebab = to_kebab(display)
        if kebab in seen or kebab == root.name:
            continue
        seen.add(kebab)
        subskills.append(Skill(
            skill_uid=stable_uid(f"{SOURCE_ID}|sub|{root.name}|{kebab}"),
            name=kebab,
            description=str(entry.get("description", "")).strip(),
            category=root.category,
            example=str(entry.get("example", "")).strip(),
            procedure=[],
            when_to_use=str(entry.get("when_to_use", "")).strip(),
            constraints=[],
            source=SOURCE_ID,
            parent_skill_uid=root.skill_uid,
        ))
    return subskills


def elicit_skill_tree(
    provider,
    n_roots: int,
    max_subskills: int = 4,
    root_batch_size: int = 5,
    verbose: bool = False,
) -> List[Skill]:
    """Elicit n_roots root skills and up to max_subskills subskills of each.

    Roots are elicited in batches (root_batch_size per API call) with a growing
    avoid-list, so the model drops fewer duplicates and we make O(n_roots /
    root_batch_size) calls instead of O(n_roots). Duplicates across batches are
    filtered silently; the loop stops early once a batch returns no new roots.
    """
    all_skills: List[Skill] = []
    roots: List[Skill] = []
    seen_roots: set = set()

    # A small retry cap per batch: if a batch returns all duplicates, try once
    # or twice more before giving up on the frontier saturating.
    empty_batches_in_a_row = 0
    max_empty_batches = 2

    while len(roots) < n_roots and empty_batches_in_a_row < max_empty_batches:
        want = min(root_batch_size, n_roots - len(roots))
        batch = elicit_root_batch(provider, count=want, avoid=[r.name for r in roots])

        added = 0
        for root in batch:
            if root.name in seen_roots or not root.description:
                if verbose:
                    print(f"  [skip duplicate/empty root: {root.name!r}]")
                continue
            seen_roots.add(root.name)
            roots.append(root)
            all_skills.append(root)
            added += 1
            if verbose:
                print(f"[root {len(roots)}/{n_roots}] {root.name} — {root.description[:70]}")

            subs = elicit_subskills(root, provider, max_subskills=max_subskills)
            all_skills.extend(subs)
            if verbose:
                for s in subs:
                    print(f"    - {s.name}: {s.description[:60]}")

            if len(roots) >= n_roots:
                break

        if added == 0:
            empty_batches_in_a_row += 1
        else:
            empty_batches_in_a_row = 0

    return all_skills


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
    parser.add_argument("--n-roots", type=int, default=5,
                        help="Number of root skills to elicit (default: 5)")
    parser.add_argument("--max-subskills", type=int, default=4,
                        help="Max subskills per root (default: 4)")
    parser.add_argument("--root-batch-size", type=int, default=5,
                        help="Roots requested per API call (default: 5). Larger "
                             "batches reduce round-trips; set to 1 to restore "
                             "one-root-per-call behavior.")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7",
                        help="Provider spec (default: anthropic:claude-opus-4-7)")
    parser.add_argument("--out", type=Path, default=Path("data/s1-m2-skills.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    skills = elicit_skill_tree(
        provider=provider,
        n_roots=args.n_roots,
        max_subskills=args.max_subskills,
        root_batch_size=args.root_batch_size,
        verbose=args.verbose,
    )

    save_json(skills, args.out)
    n_roots = sum(1 for s in skills if not s.parent_skill_uid)
    n_subs = len(skills) - n_roots
    print(f"\nWrote {len(skills)} skills -> {args.out}  ({n_roots} roots, {n_subs} subskills)")


if __name__ == "__main__":
    main()
