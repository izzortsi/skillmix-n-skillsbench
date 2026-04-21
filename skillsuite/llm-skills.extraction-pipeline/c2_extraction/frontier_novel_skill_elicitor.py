"""
frontier_novel_skill_elicitor.py

PROJECT_SPECS section 1 Method 2 ("Direct Elicitation of Novel Skills").
Two stages:

  1. Root generation: ask the frontier model for broad skills that have
     NO existing conventional name but which many humans will recognize
     (spec's example: "linguistic exorcism" - rewording text to be less
     offensive). Batched with running dedup.

  2. Subskill expansion: for each accepted root, ask the model for
     finer-grained subskills (spec: "What are some subskills of X? Give
     me finer-grained skills associated with X").

Emits a flat ExtractedSkill list (drop-in seed for frontier_kway_extractor)
plus a hierarchical view for inspection.

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_novel_skill_elicitor \\
        --output-dir data/pipeline-runs/frontier-novel/seed-v1 \\
        --root-count 6 --subskills-per-root 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
ELICITATION_METHOD = "frontier-novel-v1"

ROOT_CATEGORY = "novel-root"


ROOT_SYSTEM_PROMPT = (
    "You invent names for broad cognitive or communicative skills that "
    "humans practise routinely but have no established noun phrase for. "
    "You are playful but precise: each proposed skill must be a genuine "
    "procedural move, recognizable to many people once named, not a "
    "tautology or joke. Names are snake_case of 2-4 words."
)


ROOT_PROMPT = """Generate {batch_size} broad skills that humans exercise but for which
there is NO widely-used conventional name. Example: "linguistic
exorcism" - rewording text to be less offensive or more useful. The
skill must be RECOGNIZABLE once pointed out, not esoteric.

Already proposed in this session (DO NOT repeat or paraphrase):
{existing_block}

For each skill provide:
  name        - snake_case, 2-4 words, NOT an established term
  description - one sentence defining the skill
  example     - one concrete everyday situation where the skill is used

Return ONLY valid JSON:

{{
  "skills": [
    {{"name": "<snake_case>", "description": "<one sentence>", "example": "<concrete situation>"}}
  ]
}}
"""


SUBSKILL_SYSTEM_PROMPT = (
    "You decompose a broad skill into its finer-grained component "
    "subskills. Each subskill is a procedural move that contributes to "
    "the parent skill. Names are snake_case of 2-4 words."
)


SUBSKILL_PROMPT = """Parent skill: {parent_name}
Definition: {parent_description}
Example:    {parent_example}

Generate {count} distinct SUBSKILLS of the parent. Each subskill:
  - is a narrower procedural move used when exercising the parent skill
  - names a specific step, not a synonym of the parent
  - snake_case, 2-4 words

Return ONLY valid JSON:

{{
  "subskills": [
    {{"name": "<snake_case>", "description": "<one sentence>", "example": "<concrete instance>"}}
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


def _sanitize_name(raw: str) -> str:
    label = raw.strip().lower()
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


def _to_skill(category: str, name: str, description: str, example: str) -> ExtractedSkill:
    skill_uid = generate_uid(f"novel|{category}|{name}")
    return ExtractedSkill(
        skill_uid=skill_uid,
        name=name,
        description=description.strip(),
        procedure=[],
        when_to_use=f"producing text that illustrates {name.replace('_', ' ')}",
        constraints=[],
        source_task_uids=[],
        source_trace_uids=[],
        extraction_method=ELICITATION_METHOD,
        category=category,
        example=example.strip(),
    )


def _existing_block(names: List[str]) -> str:
    if not names:
        return "  (none yet)"
    return "\n".join(f"  - {n}" for n in sorted(names))


def elicit_roots(
    agent: LinearAgent,
    target: int,
    batch_size: int,
    max_batches: int,
    transcripts_out,
    verbose: bool,
) -> Dict[str, ExtractedSkill]:
    names_seen: Dict[str, ExtractedSkill] = {}

    for batch_idx in range(max_batches):
        if len(names_seen) >= target:
            break
        remaining = max(1, target - len(names_seen))
        this_batch = min(batch_size, remaining)

        if verbose:
            print(f"[roots] batch {batch_idx+1}: have {len(names_seen)}/{target}, requesting {this_batch}")

        prompt = ROOT_PROMPT.format(
            batch_size=this_batch,
            existing_block=_existing_block(list(names_seen.keys())),
        )
        try:
            transcript = record(agent.run(prompt))
        except Exception as e:
            if verbose:
                print(f"  API ERROR: {e}")
            transcripts_out.write(json.dumps({
                "stage": "roots", "batch": batch_idx + 1, "error": str(e)
            }, ensure_ascii=False) + "\n")
            break

        text = _final_text(transcript)
        transcripts_out.write(json.dumps({
            "stage": "roots", "batch": batch_idx + 1,
            "existing_count": len(names_seen),
            "final_text": text,
            "transcript": transcript.to_dict(),
        }, ensure_ascii=False) + "\n")

        parsed = _parse_json(text)
        if parsed is None:
            if verbose:
                print(f"  PARSE_FAIL: {text[:100]!r}")
            break

        new_in_batch = 0
        for item in parsed.get("skills", []) or []:
            if not isinstance(item, dict):
                continue
            name = _sanitize_name(str(item.get("name", "")))
            if not name or name in names_seen:
                continue
            skill = _to_skill(
                ROOT_CATEGORY,
                name,
                str(item.get("description", "")),
                str(item.get("example", "")),
            )
            names_seen[name] = skill
            new_in_batch += 1

        if verbose:
            print(f"  +{new_in_batch} new (total {len(names_seen)})")

        if new_in_batch == 0:
            break

    return names_seen


def expand_subskills(
    agent: LinearAgent,
    parent: ExtractedSkill,
    count: int,
    transcripts_out,
    verbose: bool,
) -> List[ExtractedSkill]:
    prompt = SUBSKILL_PROMPT.format(
        parent_name=parent.name,
        parent_description=parent.description,
        parent_example=parent.example or "(no example)",
        count=count,
    )

    if verbose:
        print(f"[{parent.name}] expanding into {count} subskills")

    try:
        transcript = record(agent.run(prompt))
    except Exception as e:
        if verbose:
            print(f"  API ERROR: {e}")
        transcripts_out.write(json.dumps({
            "stage": "subskills", "parent": parent.name, "error": str(e)
        }, ensure_ascii=False) + "\n")
        return []

    text = _final_text(transcript)
    transcripts_out.write(json.dumps({
        "stage": "subskills", "parent": parent.name,
        "final_text": text, "transcript": transcript.to_dict(),
    }, ensure_ascii=False) + "\n")

    parsed = _parse_json(text)
    if parsed is None:
        if verbose:
            print(f"  PARSE_FAIL: {text[:100]!r}")
        return []

    subs: List[ExtractedSkill] = []
    seen: set = set()
    for item in parsed.get("subskills", []) or []:
        if not isinstance(item, dict):
            continue
        name = _sanitize_name(str(item.get("name", "")))
        if not name or name == parent.name or name in seen:
            continue
        seen.add(name)
        skill = _to_skill(
            parent.name,
            name,
            str(item.get("description", "")),
            str(item.get("example", "")),
        )
        subs.append(skill)

    if verbose:
        print(f"  +{len(subs)} subskills")
    return subs


def elicit_novel_catalog(
    output_dir: Path,
    root_count: int = 8,
    root_batch_size: int = 6,
    max_root_batches: int = 3,
    subskills_per_root: int = 5,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> List[ExtractedSkill]:
    output_dir.mkdir(parents=True, exist_ok=True)
    t_path = output_dir / "generation-transcripts.jsonl"

    root_agent = LinearAgent(
        model=model, system=ROOT_SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )
    sub_agent = LinearAgent(
        model=model, system=SUBSKILL_SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    hierarchy: Dict[str, Any] = {}
    all_skills: List[ExtractedSkill] = []

    with t_path.open("w", encoding="utf-8") as t_out:
        roots = elicit_roots(
            root_agent, root_count, root_batch_size, max_root_batches, t_out, verbose
        )
        all_skills.extend(roots.values())

        for root in roots.values():
            subs = expand_subskills(sub_agent, root, subskills_per_root, t_out, verbose)
            all_skills.extend(subs)
            hierarchy[root.name] = {
                "description": root.description,
                "example": root.example,
                "skill_uid": root.skill_uid,
                "subskills": [
                    {"name": s.name, "description": s.description, "example": s.example, "skill_uid": s.skill_uid}
                    for s in subs
                ],
            }

    skills_path = output_dir / "skills.json"
    save_extracted_skills(all_skills, skills_path)
    (output_dir / "novel-hierarchy.json").write_text(
        json.dumps(hierarchy, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({
            "roots": len(hierarchy),
            "total_skills": len(all_skills),
            "subskills": len(all_skills) - len(hierarchy),
            "model": model,
            "elicitation_method": ELICITATION_METHOD,
        }, indent=2), encoding="utf-8",
    )

    if verbose:
        print(f"\n{len(hierarchy)} roots / {len(all_skills)} total skills")
        print(f"Wrote {skills_path}")
        print(f"Wrote {output_dir / 'novel-hierarchy.json'}")

    return all_skills


def main() -> int:
    ap = argparse.ArgumentParser(description="Elicit novel (unnamed) skills and subskills via a frontier model (PROJECT_SPECS 1.M2).")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--root-count", type=int, default=8)
    ap.add_argument("--root-batch-size", type=int, default=6)
    ap.add_argument("--max-root-batches", type=int, default=3)
    ap.add_argument("--subskills-per-root", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    elicit_novel_catalog(
        output_dir=args.output_dir,
        root_count=args.root_count,
        root_batch_size=args.root_batch_size,
        max_root_batches=args.max_root_batches,
        subskills_per_root=args.subskills_per_root,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
