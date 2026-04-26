"""
s3.m1 — Random Skill-Tuple Composition (PROJECT_SPECS §3 Method 1).

Using a skill catalog, randomly select k-tuples of skills (k in [1, 5]) and
prompt the model to produce ONE user query + answer that exhibits every skill
in the tuple.

Cited result: 4,000 such Q&A pairs trained Llama-3 8B past Opus and 405B on
AlpacaEval -- vs Alpaca's 50K or UltraChat's 1M. The mechanism: the questions
have many moving parts, forcing retrieval + composition.

Sampling (from the frontier_kway_extractor port):
  - If C(n, k) <= ENUMERATION_THRESHOLD, enumerate all combinations and random.sample.
  - Otherwise, draw random k-subsets with dedup by sorted skill_uid tuple.
  - --same-category restricts every k-tuple to a single category. The b1 report
    found this dramatically improves verifier pass rates (k=2: 33% -> 100%,
    k=3: 0% -> 50%) when random cross-category tuples mix incompatible domains.
  - --categories filters the seed pool before sampling.

Output: SkillExample JSON with source="s3.m1.random-tuple".

Usage:
    python -m cli s3.m1 --catalog data/wikipedia-seed/skills.json \\
                        --n 20 --k 2 \\
                        --provider anthropic:claude-opus-4-7 \\
                        --out data/s3-m1-examples.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import time
from math import comb
from pathlib import Path
from typing import List, Optional, Tuple

from core.schemas import Skill, SkillExample, stable_uid, save_json, load_skills
from core.providers import create_provider


SOURCE_ID = "s3.m1.random-tuple"
MIN_K = 1
MAX_K = 5
# Enumerate C(n, k) combinations when the count stays under this threshold.
# Above it we sample random k-subsets to avoid materialising a huge list.
ENUMERATION_THRESHOLD = 50_000


GENERATION_PROMPT = """You are helping build a skill-composition training set.

Produce ONE user query together with a high-quality answer. The query should
be a plausible request a real user might make. Both the query and the answer
together must naturally exhibit ALL of the following skills at once -- the
question is where retrieval is forced, the answer is where composition is
forced.

Skills to exhibit (with definitions and examples for reference):
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


def _format_skill_block(skills: List[Skill]) -> str:
    lines = []
    for i, s in enumerate(skills, 1):
        lines.append(f"Skill {i}: {s.name}")
        lines.append(f"  Definition: {s.description}")
        if s.example:
            lines.append(f"  Example: {s.example}")
    return "\n".join(lines)


def build_generation_prompt(skills: List[Skill]) -> str:
    return GENERATION_PROMPT.format(skill_block=_format_skill_block(skills))


def _parse_qa_json(text: str) -> tuple[str, str]:
    """Extract {"question", "answer"} from a model response.

    Tolerates optional markdown fences and leading/trailing prose. Returns
    ("", response) if no JSON object parses — caller records the raw text as
    `answer` and leaves the example unverified.
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else ""

    if candidate:
        try:
            data = json.loads(candidate)
            return (
                str(data.get("question", "")).strip(),
                str(data.get("answer", "")).strip(),
            )
        except json.JSONDecodeError:
            pass
    return "", text.strip()


def compose_skill_example(skills: List[Skill], provider) -> SkillExample:
    """Generate one (question, answer) pair that exhibits all given skills."""
    prompt = build_generation_prompt(skills)
    result = provider.chat([{"role": "user", "content": prompt}])
    question, answer = _parse_qa_json(result.text)

    model_name = getattr(provider, "model_name", "unknown")
    skill_ids = [s.skill_uid for s in skills]
    example_uid = stable_uid(
        f"{SOURCE_ID}|{model_name}|{','.join(skill_ids)}|{time.time_ns()}"
    )

    return SkillExample(
        example_uid=example_uid,
        skill_uids=skill_ids,
        topic_uid="",
        question=question,
        answer=answer,
        verified=False,
        source=SOURCE_ID,
    )


def _filter_by_category(catalog: List[Skill], allowed: List[str]) -> List[Skill]:
    if not allowed:
        return catalog
    wanted = {c.strip() for c in allowed if c.strip()}
    return [s for s in catalog if (s.category or "") in wanted]


def _same_category_tuples(catalog: List[Skill], k: int) -> List[Tuple[Skill, ...]]:
    """All k-combinations within each category (cross-category tuples dropped)."""
    by_cat: dict[str, List[Skill]] = {}
    for s in catalog:
        by_cat.setdefault(s.category or "", []).append(s)
    out: List[Tuple[Skill, ...]] = []
    for members in by_cat.values():
        if len(members) < k:
            continue
        out.extend(itertools.combinations(members, k))
    return out


def _sample_tuples(
    catalog: List[Skill],
    k: int,
    n_examples: int,
    random_seed: int,
    same_category: bool = False,
    allowed_categories: Optional[List[str]] = None,
) -> List[Tuple[Skill, ...]]:
    """Deterministic k-tuple sampling with combinatorial / random strategies.

    For same_category=True: enumerate every within-category k-tuple, then sample.
    For cross-category with C(n, k) <= ENUMERATION_THRESHOLD: enumerate + sample.
    Above that threshold: draw k-subsets directly, dedup by sorted skill_uid tuple.
    """
    pool = _filter_by_category(catalog, allowed_categories or [])
    n = len(pool)
    if n < k:
        raise ValueError(
            f"need at least {k} skills in the pool; got {n} after filters"
        )

    rng = random.Random(random_seed)

    if same_category:
        tuples = _same_category_tuples(pool, k)
        if not tuples:
            raise ValueError(
                f"same-category sampling: no category has >= {k} members"
            )
        if n_examples <= 0 or n_examples >= len(tuples):
            return tuples
        return rng.sample(tuples, n_examples)

    total = comb(n, k)
    if total <= ENUMERATION_THRESHOLD:
        all_tuples = list(itertools.combinations(pool, k))
        if n_examples <= 0 or n_examples >= len(all_tuples):
            return all_tuples
        return rng.sample(all_tuples, n_examples)

    target = n_examples if n_examples > 0 else min(total, 10_000)
    seen: set = set()
    out: List[Tuple[Skill, ...]] = []
    attempts = 0
    max_attempts = target * 10
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        picked = tuple(sorted(rng.sample(pool, k), key=lambda s: s.skill_uid))
        key = tuple(s.skill_uid for s in picked)
        if key in seen:
            continue
        seen.add(key)
        out.append(picked)
    return out


def generate_dataset(
    catalog: List[Skill],
    provider,
    n_examples: int,
    k: int = 2,
    seed: int = 42,
    same_category: bool = False,
    allowed_categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[SkillExample]:
    """Sample k-tuples from catalog, generate one example per tuple.

    Default is random cross-category sampling. Set same_category=True to
    restrict tuples to a single category, which in our experiments produced
    much higher verifier pass rates (see module docstring).
    """
    if k < MIN_K or k > MAX_K:
        raise ValueError(f"k must be in [{MIN_K}, {MAX_K}] (got {k})")
    if k > len(catalog):
        raise ValueError(f"k={k} exceeds catalog size ({len(catalog)})")

    tuples = _sample_tuples(
        catalog, k, n_examples, seed,
        same_category=same_category, allowed_categories=allowed_categories,
    )

    examples: List[SkillExample] = []
    for i, tup in enumerate(tuples):
        skills_in = list(tup)
        if verbose:
            names = ", ".join(s.name for s in skills_in)
            cats = {s.category or "" for s in skills_in}
            tag = "same-cat" if same_category else ("multi-cat" if len(cats) > 1 else "one-cat")
            print(f"[{i+1}/{len(tuples)}] k={k} [{tag}] skills=({names})")

        example = compose_skill_example(skills_in, provider)
        examples.append(example)

        if verbose:
            q_preview = example.question.replace("\n", " ")[:80]
            print(f"    q='{q_preview}...' answer_len={len(example.answer)}")

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    """'anthropic:claude-opus-4-7' -> (provider_name, model)."""
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--catalog", type=Path,
                        default=Path("data/wikipedia-seed/skills.json"),
                        help="Skill catalog JSON (default: data/wikipedia-seed/skills.json)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of examples to generate (default: 10)")
    parser.add_argument("--k", type=int, default=2,
                        help=f"Skills per example, in [{MIN_K}, {MAX_K}] (default: 2)")
    parser.add_argument("--same-category", action="store_true",
                        help="Restrict every k-tuple to a single category "
                             "(verifier pass rates are much higher; see docstring)")
    parser.add_argument("--categories", default="",
                        help="Comma-separated category names to restrict the pool.")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7",
                        help="Provider spec (default: anthropic:claude-opus-4-7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/s3-m1-examples.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    catalog = load_skills(args.catalog)
    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    allowed_categories: Optional[List[str]] = None
    if args.categories.strip():
        allowed_categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    examples = generate_dataset(
        catalog=catalog,
        provider=provider,
        n_examples=args.n,
        k=args.k,
        seed=args.seed,
        same_category=args.same_category,
        allowed_categories=allowed_categories,
        verbose=args.verbose,
    )

    save_json(examples, args.out)
    parsed_ok = sum(1 for e in examples if e.question)
    print(f"\nWrote {len(examples)} examples -> {args.out}")
    print(f"  k={args.k}  catalog={len(catalog)}  parsed_question={parsed_ok}/{len(examples)}")


if __name__ == "__main__":
    main()
