"""
b2_benchmarks.skillsbench.procedural_catalog

Transform a declarative seed catalog (skills.json from s2.m1 / s1.m3) into a
procedural Skill catalog suitable for the bench skill-injection evaluation.

Why this exists:
  The `bench` stage injects a Skill into the student's system prompt and
  measures whether it helps the student solve a task. The SkillsBench paper
  (Li et al. 2026) defines a Skill as a *procedural package*: numbered steps
  + constraints + examples, applied to a CLASS of tasks. It explicitly
  excludes few-shot-style declarative definitions (which is what our
  Wikipedia-seed catalog is). Injecting declarative "skills" into a detection
  task causes mid-tier models to over-match on the positive example (see
  regression pattern in the bench experiments).

  This module rewrites each declarative entry into a detection-oriented
  procedural Skill by calling Opus. It preserves `name` + `skill_uid` +
  `category` so task_skill_map mappings keep resolving after the rewrite.

Input fields that matter (from scaffold Skill):
    name, description, category, example

Output fields (enriched):
    procedure:    3-6 numbered steps (how to detect or apply, as imperatives)
    when_to_use:  single sentence phrased for detection/application
    constraints:  2-4 guardrails (anti-overmatch: "do not flag unless...")
    example:      POSITIVE + NEGATIVE example (dual_example)
    source:       "bench_catalog.procedural-v1"

Usage:
    python -m b2_benchmarks.skillsbench.procedural_catalog \\
        --input-catalog data/pipeline-runs/default/catalog/skills.json \\
        --out           data/pipeline-runs/default/bench_catalog/skills.json \\
        --provider      anthropic:claude-opus-4-7
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import Skill, load_skills, save_json
from core.providers import create_provider


SOURCE_TAG = "bench_catalog.procedural-v1"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You rewrite declarative skill definitions into procedural Skills suitable "
    "for LLM agent system-prompt injection. A procedural Skill names concrete "
    "detection or application steps a student model should follow. It must "
    "apply to a CLASS of tasks, not a single instance. It avoids declarative "
    "answers ('what to output') and instead gives imperative guidance ('how "
    "to approach')."
)


REWRITE_PROMPT = """Rewrite the declarative skill below into a procedural Skill.

Source skill:
  name:        {name}
  category:    {category}
  description: {description}
  example:     {example}

The output Skill will be injected into a student LLM's system prompt. The
student is then asked to solve a reading-comprehension-style task (passage +
challenge + acceptance criteria). Your procedure should help the student
DETECT this skill or APPLY it to the challenge -- not produce prose that
illustrates the skill.

Requirements:

1. PROCEDURE -- 3 to 6 numbered imperative steps. Each is a concrete action the
   student performs against the given passage/challenge, in order. Avoid
   abstract definitions. Examples of good steps:
     "Identify the sentence that states the challenge's question."
     "Check whether each candidate sentence addresses that question."
     "Flag any sentence that praises unrelated positive outcomes."

2. WHEN_TO_USE -- one sentence, framed for DETECTION or APPLICATION. Must NOT
   begin with 'producing text'. Good: 'when asked to identify the loaded
   presupposition in a question'. Bad: 'producing text that illustrates loaded
   questions'.

3. CONSTRAINTS -- 2 to 4 guardrails, each a short sentence. Every constraint
   reduces over-matching. Typical shapes:
     "Do not flag an on-topic item even if its tone is unusual."
     "Require that the full pattern be present; one matching piece is not
      sufficient."

4. DUAL_EXAMPLE -- one POSITIVE case (skill present) AND one NEGATIVE case
   (skill absent despite surface resemblance). Format exactly:

     Positive: <scenario>. Why: <one-line reason>.
     Negative: <scenario that could be mistaken for the skill>. Why: <one-line reason it does not qualify>.

Constraints on the whole output:
  - Do NOT mention the skill's own name inside constraints or dual_example.
  - Do NOT embed answers or outputs the student should produce.
  - Do NOT include task-specific content (no specific task ids, file paths,
    or benchmark fixtures).
  - Keep the procedure general enough to apply to any instance of this
    class of tasks.

Return ONLY valid JSON (no markdown fences, no preface):

{{
  "procedure":    ["<step 1>", "<step 2>", "..."],
  "when_to_use":  "<single sentence>",
  "constraints":  ["<guardrail 1>", "<guardrail 2>"],
  "dual_example": "<multiline string in the Positive: / Negative: format above>"
}}
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response. Tolerates fences."""
    if not text:
        return None
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", stripped, re.DOTALL)
        candidate = brace.group(0) if brace else ""
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


# ---------------------------------------------------------------------------
# Per-skill rewriter
# ---------------------------------------------------------------------------


def _build_prompt(skill: Skill) -> str:
    return REWRITE_PROMPT.format(
        name=skill.name,
        category=skill.category or "(none)",
        description=skill.description or "(none given)",
        example=skill.example or "(none given)",
    )


def rewrite_skill(
    skill: Skill,
    provider,
    verbose: bool = False,
) -> Tuple[Optional[Skill], Dict[str, Any]]:
    """Call the LLM to transform a declarative Skill into a procedural one.

    Returns (new_skill_or_none, diagnostics).  new_skill is None when the
    model's response cannot be parsed into the expected shape; the caller
    decides whether to drop or keep the original entry.
    """
    prompt = _build_prompt(skill)
    result = provider.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    raw_text = getattr(result, "text", "") or ""
    parsed = _parse_json_object(raw_text)

    if parsed is None:
        return None, {"error": "json_parse_failed", "raw_text": raw_text[:400]}

    procedure = _coerce_str_list(parsed.get("procedure", []))
    constraints = _coerce_str_list(parsed.get("constraints", []))
    when_to_use = str(parsed.get("when_to_use", "")).strip()
    dual_example = str(parsed.get("dual_example", "")).strip()

    if not procedure:
        return None, {"error": "empty_procedure", "raw_text": raw_text[:400]}
    if not when_to_use:
        return None, {"error": "empty_when_to_use", "raw_text": raw_text[:400]}

    new_skill = Skill(
        skill_uid=skill.skill_uid,                  # preserve so task_skill_map still matches
        name=skill.name,                            # preserve
        description=skill.description,              # declarative definition still useful
        category=skill.category,                    # preserve
        example=dual_example or skill.example,
        procedure=procedure[:6],                    # cap per the paper's 2-3-modules finding (keep up to 6)
        when_to_use=when_to_use,
        constraints=constraints[:4],
        source=SOURCE_TAG,
        parent_skill_uid=skill.parent_skill_uid,
    )
    diag: Dict[str, Any] = {
        "steps": len(new_skill.procedure),
        "constraints": len(new_skill.constraints),
        "usage": dict(result.usage) if getattr(result, "usage", None) else {},
    }
    if verbose:
        print(
            f"  [{skill.name}]  steps={diag['steps']}  "
            f"constraints={diag['constraints']}  "
            f"tokens={diag['usage'].get('total_tokens', '?')}"
        )
    return new_skill, diag


# ---------------------------------------------------------------------------
# Whole-catalog driver
# ---------------------------------------------------------------------------


def build_procedural_catalog(
    input_catalog_path: Path,
    output_path: Path,
    provider,
    limit: int = 0,
    only_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[List[Skill], List[Dict[str, Any]]]:
    """Rewrite every skill in `input_catalog_path` and write to `output_path`.

    Returns (written_skills, failures). Failures keep skills that the model
    couldn't rewrite cleanly; the written catalog excludes them entirely so
    downstream bench won't inject broken content.
    """
    source = load_skills(input_catalog_path)
    subset = source
    if only_names:
        wanted = {n.strip() for n in only_names if n.strip()}
        subset = [s for s in subset if s.name in wanted]
        if verbose:
            missing = sorted(wanted - {s.name for s in subset})
            if missing:
                print(f"  filter: {len(missing)} requested name(s) not in catalog: {missing}")
    if limit > 0:
        subset = subset[:limit]

    print(f"Loaded {len(source)} seed skills; rewriting {len(subset)}")

    written: List[Skill] = []
    failures: List[Dict[str, Any]] = []

    for i, skill in enumerate(subset, 1):
        started = time.time()
        if verbose:
            print(f"\n[{i}/{len(subset)}] {skill.name}")
        try:
            new_skill, diag = rewrite_skill(skill, provider, verbose=verbose)
        except Exception as e:
            failures.append({"name": skill.name, "error": f"exception:{e}"})
            if verbose:
                print(f"  ERROR: {e}")
            continue
        elapsed = time.time() - started
        if new_skill is None:
            failures.append({"name": skill.name, **diag})
            if verbose:
                print(f"  FAIL: {diag.get('error')} ({elapsed:.1f}s)")
            continue
        written.append(new_skill)
        if verbose:
            print(f"  OK ({elapsed:.1f}s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(written, output_path)
    if failures:
        fail_path = output_path.with_name(output_path.stem + ".failures.jsonl")
        with fail_path.open("w", encoding="utf-8") as fh:
            for rec in failures:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nFailures ({len(failures)}) -> {fail_path}")
    print(f"Wrote {len(written)} procedural skills -> {output_path}")
    return written, failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--input-catalog", type=Path, required=True,
                        help="Declarative skills.json (from catalog stage)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output procedural skills.json path")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7",
                        help="LLM provider spec (default: Opus for quality)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only rewrite the first N skills (0 = all)")
    parser.add_argument("--skills", default="",
                        help="Comma-separated list of skill names to rewrite; "
                             "others are skipped (useful for probes before full regen)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    only_names: Optional[List[str]] = None
    if args.skills.strip():
        only_names = [s.strip() for s in args.skills.split(",") if s.strip()]

    build_procedural_catalog(
        input_catalog_path=args.input_catalog,
        output_path=args.out,
        provider=provider,
        limit=args.limit,
        only_names=only_names,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
