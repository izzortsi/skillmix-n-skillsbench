"""
s3.m2 — Domain-Specific Skill Combination (PROJECT_SPECS §3 Method 2).

A thin domain-scoped specialization of s3.m1: restrict sampling to skills
sharing a single category (== domain), and optionally adjust the generation
prompt's framing to match the domain style. Two cited spec templates:

  - math:                 "Create a question requiring [math_skill1] and [math_skill2]"
  - instruction-following: "Create a detailed question with many moving parts
                             requiring [skill1] and [skill2]"

Produces questions that combine skills within a single domain — e.g. algebra
+ probability for math, or critical_thinking + communication for IF — for
cross-topic evaluation within the domain.

The domain constraint is enforced by (1) filtering the catalog to skills with
`category == domain`, and (2) requiring every sampled k-tuple to come from that
filtered pool. The generation prompt is the same as s3.m1's with a one-line
domain preamble.

Output: SkillExample JSON with source="s3.m2.domain-specific". The domain
string is saved as `topic_uid` for downstream traceability.

Usage:
    python -m cli s3.m2 --catalog data/wikipedia-seed/skills.json \\
                         --domain rhetorical \\
                         --n 5 --k 2 \\
                         --provider anthropic:claude-opus-4-7 \\
                         --out data/s3-m2-rhetorical.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from core.schemas import Skill, SkillExample, stable_uid, save_json, load_skills
from core.providers import create_provider

from s3_generating_skill_examples.m1_random_pair_composition import (
    MAX_K, MIN_K,
    _parse_qa_json,
    _sample_tuples,
    _format_skill_block,
)


SOURCE_ID = "s3.m2.domain-specific"


# Domain-specific preambles that steer the model toward the domain's
# typical shape. The body of the prompt is shared with s3.m1.
DOMAIN_PREAMBLES = {
    "math": (
        "You are building math problems. Create ONE math question that "
        "requires the student to exercise ALL of the following skills at "
        "once; the answer should be a worked-out solution (not a list)."
    ),
    "instruction-following": (
        "You are building instruction-following evaluation items. Create ONE "
        "detailed user instruction with many moving parts that REQUIRES a "
        "helpful assistant to exercise ALL of the following skills; the "
        "answer should be the high-quality response."
    ),
    "code": (
        "You are building coding tasks. Create ONE programming question that "
        "requires all of the following skills; the answer should include "
        "working code plus a brief rationale."
    ),
}

DEFAULT_PREAMBLE = (
    "You are building a skill-composition item in the {domain} domain. "
    "Create ONE plausible user query plus a high-quality answer. Both the "
    "query and the answer together must naturally exhibit ALL of the "
    "following skills at once."
)


DOMAIN_PROMPT_BODY = """Skills to exhibit (with definitions and examples for reference):
{skill_block}

Requirements:
- The user query should have several moving parts that require the listed
  skills to answer well. It does not need to explicitly name the skills.
- The answer should be substantive (a few sentences to a short paragraph).
- Use each skill in its technical sense, not a colloquial one.
- Do NOT mention the names of the skills in the query or the answer.

Return ONLY valid JSON in exactly this shape, with no markdown fences and no
preface:

{{
  "question": "<the user query>",
  "answer":   "<the assistant's answer>"
}}
"""


def _build_domain_prompt(skills: List[Skill], domain: str) -> str:
    preamble = DOMAIN_PREAMBLES.get(domain, DEFAULT_PREAMBLE.format(domain=domain))
    return preamble + "\n\n" + DOMAIN_PROMPT_BODY.format(skill_block=_format_skill_block(skills))


def compose_domain_example(skills: List[Skill], domain: str, provider) -> SkillExample:
    """Generate one (question, answer) pair in the given domain."""
    prompt = _build_domain_prompt(skills, domain)
    result = provider.chat([{"role": "user", "content": prompt}])
    question, answer = _parse_qa_json(result.text)

    model_name = getattr(provider, "model_name", "unknown")
    skill_ids = [s.skill_uid for s in skills]
    example_uid = stable_uid(
        f"{SOURCE_ID}|{domain}|{model_name}|{','.join(skill_ids)}|{time.time_ns()}"
    )

    # Store the domain in topic_uid for downstream traceability. Not a true
    # Topic record — the domain is the constraint, not a narrative setting.
    return SkillExample(
        example_uid=example_uid,
        skill_uids=skill_ids,
        topic_uid=f"domain:{domain}",
        question=question,
        answer=answer,
        verified=False,
        source=SOURCE_ID,
    )


def generate_domain_dataset(
    catalog: List[Skill],
    domain: str,
    provider,
    n_examples: int,
    k: int = 2,
    seed: int = 42,
    verbose: bool = False,
) -> List[SkillExample]:
    """Restrict catalog to `domain`, sample k-tuples within, compose one example per tuple."""
    if k < MIN_K or k > MAX_K:
        raise ValueError(f"k must be in [{MIN_K}, {MAX_K}] (got {k})")
    if not domain.strip():
        raise ValueError("domain must be a non-empty string")

    # s3.m1's sampler enforces the filter + single-category constraint for us.
    tuples = _sample_tuples(
        catalog, k=k, n_examples=n_examples, random_seed=seed,
        same_category=True, allowed_categories=[domain],
    )

    examples: List[SkillExample] = []
    for i, tup in enumerate(tuples):
        skills_in = list(tup)
        if verbose:
            names = ", ".join(s.name for s in skills_in)
            print(f"[{i+1}/{len(tuples)}] k={k} domain={domain!r} skills=({names})")

        example = compose_domain_example(skills_in, domain, provider)
        examples.append(example)

        if verbose:
            preview = example.question.replace("\n", " ")[:80]
            print(f"    q='{preview}...' answer_len={len(example.answer)}")

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
                        default=Path("data/wikipedia-seed/skills.json"),
                        help="Skill catalog JSON")
    parser.add_argument("--domain", required=True,
                        help="Domain (== Skill.category) to restrict to. Known "
                             f"preambles: {sorted(DOMAIN_PREAMBLES.keys())}; "
                             "any other value falls through to a generic preamble.")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of examples to generate (default: 10)")
    parser.add_argument("--k", type=int, default=2,
                        help=f"Skills per example, in [{MIN_K}, {MAX_K}] (default: 2)")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: data/s3-m2-{domain}-examples.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    catalog = load_skills(args.catalog)
    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    out_path = args.out or Path(f"data/s3-m2-{args.domain}-examples.json")

    examples = generate_domain_dataset(
        catalog=catalog,
        domain=args.domain,
        provider=provider,
        n_examples=args.n,
        k=args.k,
        seed=args.seed,
        verbose=args.verbose,
    )

    save_json(examples, out_path)
    parsed_ok = sum(1 for e in examples if e.question)
    print(f"\nWrote {len(examples)} examples -> {out_path}")
    print(f"  domain={args.domain!r}  k={args.k}  catalog={len(catalog)}  "
          f"parsed_question={parsed_ok}/{len(examples)}")


if __name__ == "__main__":
    main()
