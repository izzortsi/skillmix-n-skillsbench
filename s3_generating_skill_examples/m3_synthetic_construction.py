"""
s3.m3 — Synthetic Construction Testing (PROJECT_SPECS §3 Method 3).

Prompt template: "Give me an example of a piece of text with two sentences,
which could appear in a fiction about [topic] and has these two skills".

Used to test composition in novel contexts (e.g. "sushi" + "ad_hominem_attack").
Conceptually a generator variant of s2.m2's Skill-Mix eval, but the output is
a held-out dataset (SkillExample) rather than a graded trial.

SkillExample layout:
  question = the generation instruction (reproducible constraint)
  answer   = the generated passage
  topic_uid populated with the sampled topic

Usage:
    python -m cli s3.m3 --catalog data/wikipedia-seed/skills.json \\
                        --topics data/topics/topics.json \\
                        --n 20 --k 2 --sentence-limit 2 \\
                        --provider anthropic:claude-opus-4-7 \\
                        --out data/s3-m3-examples.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import List

from core.schemas import (
    Skill, SkillExample, Topic,
    stable_uid, save_json, load_skills,
)
from core.providers import create_provider


SOURCE_ID = "s3.m3.synthetic-construction"


GENERATION_PROMPT = """Give me a piece of text, up to {sentence_limit} sentences long,
which could appear in a fiction about "{topic}" and which illustrates ALL of
the following skills at once. The passage should be natural prose (not a
list) and should not explicitly name any of the skills.

Skills to illustrate (with definitions and examples for reference):
{skill_block}

Constraints:
- Stay within {sentence_limit} sentences.
- Stay in the fictional context of "{topic}".
- Use each skill in its technical sense, not a colloquial one.
- Do NOT name any of the skills in the text.

Return ONLY valid JSON in exactly this shape, no markdown fences, no preface:

{{
  "passage": "<the text>"
}}
"""


def _format_skill_block(skills: List[Skill]) -> str:
    lines = []
    for i, s in enumerate(skills, 1):
        lines.append(f"Skill {i}: {s.name}")
        lines.append(f"  Definition: {s.description}")
        if s.example:
            lines.append(f"  Example: {s.example}")
    return "\n".join(lines)


def build_generation_prompt(skills: List[Skill], topic: Topic, sentence_limit: int) -> str:
    return GENERATION_PROMPT.format(
        sentence_limit=sentence_limit,
        topic=topic.name,
        skill_block=_format_skill_block(skills),
    )


def _parse_passage(text: str) -> str:
    """Extract {"passage": ...} from a model response, tolerating non-JSON."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else ""
    if not candidate:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else ""
    if candidate:
        try:
            data = json.loads(candidate)
            passage = data.get("passage", "")
            if passage:
                return str(passage).strip()
        except json.JSONDecodeError:
            pass
    return text.strip()


def compose_fiction_example(
    skills: List[Skill],
    topic: Topic,
    provider,
    sentence_limit: int = 2,
) -> SkillExample:
    """Generate one fiction-framed passage illustrating all given skills."""
    prompt = build_generation_prompt(skills, topic, sentence_limit)
    result = provider.chat([{"role": "user", "content": prompt}])
    passage = _parse_passage(result.text)

    model_name = getattr(provider, "model_name", "unknown")
    skill_ids = [s.skill_uid for s in skills]
    example_uid = stable_uid(
        f"{SOURCE_ID}|{model_name}|{topic.topic_uid}|{','.join(skill_ids)}|{time.time_ns()}"
    )

    return SkillExample(
        example_uid=example_uid,
        skill_uids=skill_ids,
        topic_uid=topic.topic_uid,
        question=prompt,
        answer=passage,
        verified=False,
        source=SOURCE_ID,
    )


def generate_fiction_dataset(
    catalog: List[Skill],
    topics: List[Topic],
    provider,
    n_examples: int,
    k: int = 2,
    sentence_limit: int = 2,
    seed: int = 42,
    verbose: bool = False,
) -> List[SkillExample]:
    """Sample n (k-tuple, topic) pairs uniformly, generate one example each."""
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    if k > len(catalog):
        raise ValueError(f"k={k} exceeds catalog size ({len(catalog)})")
    if not topics:
        raise ValueError("topics list is empty")

    rng = random.Random(seed)
    examples: List[SkillExample] = []

    for i in range(n_examples):
        sampled = rng.sample(catalog, k)
        topic = rng.choice(topics)
        if verbose:
            names = ", ".join(s.name for s in sampled)
            print(f"[{i+1}/{n_examples}] k={k} topic='{topic.name}' skills=({names})")

        example = compose_fiction_example(sampled, topic, provider, sentence_limit)
        examples.append(example)

        if verbose:
            preview = example.answer.replace("\n", " ")[:80]
            print(f"    passage='{preview}...' len={len(example.answer)}")

    return examples


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
    parser.add_argument("--catalog", type=Path,
                        default=Path("data/wikipedia-seed/skills.json"))
    parser.add_argument("--topics", type=Path,
                        default=Path("data/topics/topics.json"))
    parser.add_argument("--n", type=int, default=10,
                        help="Number of examples to generate (default: 10)")
    parser.add_argument("--k", type=int, default=2,
                        help="Skills per example (default: 2)")
    parser.add_argument("--sentence-limit", type=int, default=2,
                        help="Max sentences in each passage (default: 2)")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7",
                        help="Provider spec (default: anthropic:claude-opus-4-7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/s3-m3-examples.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    catalog = load_skills(args.catalog)
    # topics.json uses {"name": ...} entries without UIDs; rebuild via seeder
    from s2_extracting_skills_from_text.m1_wikipedia_seeder import seed_topics_from_file
    topics = seed_topics_from_file(args.topics)

    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    examples = generate_fiction_dataset(
        catalog=catalog,
        topics=topics,
        provider=provider,
        n_examples=args.n,
        k=args.k,
        sentence_limit=args.sentence_limit,
        seed=args.seed,
        verbose=args.verbose,
    )

    save_json(examples, args.out)
    parsed_ok = sum(1 for e in examples if e.answer and e.topic_uid)
    print(f"\nWrote {len(examples)} examples -> {args.out}")
    print(f"  k={args.k}  catalog={len(catalog)}  topics={len(topics)}  "
          f"with_passage={parsed_ok}/{len(examples)}")


if __name__ == "__main__":
    main()
