"""
s2.m1 — Wikipedia Skill Extraction (PROJECT_SPECS §2 Method 1).

Seed a canonical Skill catalog from a hand-curated list of Wikipedia-style
skills. The source list is transcribed from the published skill catalog
(e.g. Yu et al. 2023 Skill-Mix, Table 5, 10 skills released of 101).

Input JSON schema (hand-curated):
    [
        {
            "category": "rhetorical",
            "name": "red herring",
            "definition": "Introducing irrelevant points ...",
            "example": "A member of the press asks the president ..."
        },
        ...
    ]

Output: Skill JSON with declarative fields populated; procedural fields empty.
UIDs are deterministic (SHA-256 of source + kebab name).

Topics (Topic JSON) can be loaded via load_topics() for use in s2.m2.

Usage:
    python -m scaffold.s2_extracting_skills_from_text.m1_wikipedia_seeder \\
        --source data/wikipedia-seed/source.json \\
        --out data/wikipedia-seed/skills.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from core.schemas import Skill, Topic, stable_uid, to_kebab, save_json, load_topics


SOURCE_ID = "s2.m1.wikipedia-seed"
DEFAULT_SOURCE = Path("data/wikipedia-seed/source.json")
DEFAULT_OUT = Path("data/wikipedia-seed/skills.json")


def seed_skill(entry: dict) -> Skill:
    display = entry["name"].strip()
    kebab = to_kebab(display)
    definition = entry["definition"].strip()
    example = entry.get("example", "").strip()
    category = entry.get("category", "").strip()

    return Skill(
        skill_uid=stable_uid(f"{SOURCE_ID}|{kebab}"),
        name=kebab,
        description=definition,
        category=category,
        example=example,
        procedure=[],
        when_to_use=f"producing text that illustrates {display}",
        constraints=[],
        source=SOURCE_ID,
    )


def seed_from_file(source_path: Path) -> List[Skill]:
    with open(source_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"{source_path}: expected a JSON list")
    skills = [seed_skill(e) for e in entries]
    _assert_unique_names(skills, source_path)
    return skills


def seed_topics_from_file(topics_path: Path) -> List[Topic]:
    """Load topics and populate missing topic_uids."""
    with open(topics_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    topics: List[Topic] = []
    for e in entries:
        name = e["name"].strip()
        tid = e.get("topic_uid") or stable_uid(f"topic|{name}")
        topics.append(Topic(topic_uid=tid, name=name, source=SOURCE_ID))
    return topics


def _assert_unique_names(skills: List[Skill], path: Path) -> None:
    seen: dict[str, str] = {}
    for s in skills:
        if s.name in seen:
            raise ValueError(
                f"{path}: duplicate kebab name {s.name!r} — two display names "
                f"collapsed to the same slug"
            )
        seen[s.name] = s.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"Source JSON list (default: {DEFAULT_SOURCE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output skills.json (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    skills = seed_from_file(args.source)
    save_json(skills, args.out)
    print(f"Seeded {len(skills)} skills: {args.source} -> {args.out}")
    for s in skills:
        cat = f"[{s.category}] " if s.category else ""
        print(f"  {cat}{s.name}")


if __name__ == "__main__":
    main()
