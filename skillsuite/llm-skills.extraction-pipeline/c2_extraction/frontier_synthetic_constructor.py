"""
frontier_synthetic_constructor.py

PROJECT_SPECS section 3 Method 3 ("Synthetic Construction Testing").
Given a skill seed and a list of fiction topics, sample (topic, k-skill
tuple) pairs and ask the frontier model to write a short passage (e.g.,
2 sentences) that could appear in fiction about the topic AND exhibits
every named skill. Full thinking transcripts captured.

Spec example: ("sushi", "ad_hominem_attack")  ->  two-sentence passage
in a sushi-themed story that exhibits an ad hominem attack.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_synthetic_constructor \\
        --seed data/pipeline-runs/frontier-catalog/seed-parallel/skills.json \\
        --output-dir data/pipeline-runs/frontier-synthetic/seed-smoke \\
        --k 2 --max-pairs 5 --sentences 2 --random-seed 42
"""

from __future__ import annotations

import argparse
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


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 3072
DEFAULT_MAX_TOKENS = 3000
CONSTRUCTION_METHOD = "frontier-synthetic-v1"


# Fiction topics kept deliberately concrete and heterogeneous - the
# spec's point is novelty of context, so a thinly-populated pool helps
# force genuine composition rather than interpolation from training data.
DEFAULT_TOPICS: List[str] = [
    "sushi",
    "pirates",
    "dragons",
    "a noir detective story",
    "a dystopian megacity",
    "a medieval monastery",
    "space travel",
    "high school",
    "a haunted mansion",
    "a chess tournament",
    "a kitchen during a dinner rush",
    "a rural small town",
]


SYSTEM_PROMPT = (
    "You write short passages of fiction that simultaneously exhibit a "
    "specified set of named skills. Every named skill must be genuinely "
    "present in the passage, not merely alluded to."
)


PROMPT = """Write a {sentences}-sentence passage that could appear in a piece of
fiction about: {topic}

The passage MUST exhibit every one of these skills:

{skills_block}

Constraints:
  - The setting is the fiction topic above; the passage must read as a
    plausible excerpt from such a story.
  - Every listed skill must be clearly present. If a skill is merely
    referenced without being exercised, redraft.
  - Output is prose only, no lists, no headings.

Return ONLY valid JSON:

{{
  "passage": "<the prose passage>",
  "skill_evidence": [
    {{"skill_name": "<name from list>", "evidence": "<1 short quote or paraphrase from the passage that exhibits the skill>"}}
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


def _skills_block(tuple_: List[Dict[str, Any]]) -> str:
    lines = []
    for i, s in enumerate(tuple_, 1):
        cat = s.get("category", "") or "unspecified"
        lines.append(f"Skill {i}: {s['name']}  ({cat})")
        lines.append(f"  Definition: {s.get('description', '')}")
        lines.append(f"  Example:    {s.get('example', '') or '(no example)'}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _aggregate_usage(transcript: Transcript) -> Dict[str, int]:
    tin = tout = 0
    for turn in transcript.turns:
        tin += int(turn.usage.get("input_tokens", 0) or 0)
        tout += int(turn.usage.get("output_tokens", 0) or 0)
    return {"input_tokens": tin, "output_tokens": tout}


def _sample_pairs(
    seeds: List[Dict[str, Any]],
    topics: List[str],
    k: int,
    max_pairs: int,
    random_seed: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    if len(seeds) < k:
        raise ValueError(f"need at least {k} seed skills, got {len(seeds)}")
    rng = random.Random(random_seed)
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    seen: set = set()
    attempts = 0
    while len(out) < max_pairs and attempts < max_pairs * 20:
        attempts += 1
        topic = rng.choice(topics)
        tuple_ = rng.sample(seeds, k)
        key = (topic, tuple(sorted(s["skill_uid"] for s in tuple_)))
        if key in seen:
            continue
        seen.add(key)
        out.append((topic, tuple_))
    return out


def construct_one(
    agent: LinearAgent,
    topic: str,
    tuple_: List[Dict[str, Any]],
    sentences: int,
) -> Dict[str, Any]:
    prompt = PROMPT.format(
        topic=topic,
        sentences=sentences,
        skills_block=_skills_block(tuple_),
    )
    transcript = record(agent.run(prompt))
    text = _final_text(transcript)
    parsed = _parse_json(text)

    passage = ""
    skill_evidence: List[Dict[str, str]] = []
    parse_error = None
    if parsed is None:
        parse_error = "json_parse_failed"
    else:
        passage = str(parsed.get("passage", "")).strip()
        for item in parsed.get("skill_evidence", []) or []:
            if isinstance(item, dict):
                skill_evidence.append({
                    "skill_name": str(item.get("skill_name", "")).strip(),
                    "evidence": str(item.get("evidence", "")).strip(),
                })
        if not passage:
            parse_error = "empty_passage"

    tuple_key = "+".join(sorted(s["skill_uid"] for s in tuple_))
    synthesis_uid = generate_uid(f"{CONSTRUCTION_METHOD}|{topic}|{tuple_key}")

    return {
        "synthesis_uid": synthesis_uid,
        "topic": topic,
        "skill_uids": [s["skill_uid"] for s in tuple_],
        "skill_names": [s["name"] for s in tuple_],
        "skill_categories": [s.get("category", "") for s in tuple_],
        "sentences_target": sentences,
        "passage": passage,
        "skill_evidence": skill_evidence,
        "parse_error": parse_error,
        "final_text": text,
        "usage": _aggregate_usage(transcript),
        "transcript": transcript.to_dict(),
        "construction_method": CONSTRUCTION_METHOD,
        "model": agent.model,
    }


def construct_synthetic(
    seed_path: Path,
    output_dir: Path,
    k: int = 2,
    max_pairs: int = 10,
    sentences: int = 2,
    random_seed: int = 42,
    topics: List[str] = None,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
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
    topics = topics or DEFAULT_TOPICS
    pairs = _sample_pairs(seed_dicts, topics, k, max_pairs, random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "syntheses.jsonl"
    fail_path = output_dir / "construction-failures.jsonl"

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    results: List[Dict[str, Any]] = []
    parse_fails = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as out, \
         fail_path.open("w", encoding="utf-8") as f_out:
        for i, (topic, tuple_) in enumerate(pairs, 1):
            names = [s["name"] for s in tuple_]
            if verbose:
                print(f"[{i}/{len(pairs)}] topic={topic!r}  skills={names}")

            try:
                result = construct_one(agent, topic, tuple_, sentences)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  ERROR: {e}")
                f_out.write(json.dumps({
                    "topic": topic, "skill_names": names, "error": str(e)
                }, ensure_ascii=False) + "\n")
                continue

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["parse_error"]:
                parse_fails += 1
                if verbose:
                    print(f"  PARSE_FAIL: {result['parse_error']}")
                f_out.write(json.dumps({
                    "topic": topic, "skill_names": names,
                    "error": result["parse_error"],
                    "final_text": result["final_text"],
                }, ensure_ascii=False) + "\n")
                continue

            results.append(result)
            if verbose:
                print(f"  passage: {result['passage'][:120]!r}")

    (output_dir / "summary.json").write_text(json.dumps({
        "pairs_attempted": len(pairs),
        "passages_produced": len(results),
        "parse_failures": parse_fails,
        "api_errors": errors,
        "k": k,
        "sentences_target": sentences,
        "model": model,
        "construction_method": CONSTRUCTION_METHOD,
    }, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n{len(results)}/{len(pairs)} passages -> {out_path}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate short fiction passages that exhibit k skills in a given topic (PROJECT_SPECS 3.M3).")
    ap.add_argument("--seed", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=10)
    ap.add_argument("--sentences", type=int, default=2)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--topics", default="", help="Comma-separated topics; empty uses built-in default list.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    topics = [t.strip() for t in args.topics.split(",") if t.strip()] if args.topics.strip() else None

    construct_synthetic(
        seed_path=args.seed, output_dir=args.output_dir,
        k=args.k, max_pairs=args.max_pairs,
        sentences=args.sentences, random_seed=args.random_seed,
        topics=topics,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
