"""
s3.m5 — ExtractedTask Synthesis from Procedural Skills.

Closes the schema gap between s3.m1 (which produces SkillExample) and
bench_traced (which consumes ExtractedTask). Given a procedural Skill
(when_to_use + procedure + constraints + example), generate N tasks per
skill in the ExtractedTask schema, each tagged with the source skill_uid
in acceptance_criteria.skill_uids/skill_names/skill_categories.

This is NOT a PROJECT_SPECS-numbered method — it is infrastructure that
bridges s3 (example generation) and bench_traced (skill-injected eval +
trace capture for SFT). Lives under s3 because it is task-emission, but
its contract is "skill -> ExtractedTask" not "skill-tuple -> SkillExample".

Output: a single tasks.json (ExtractedTask list) directly consumable by
bench_traced via --tasks. Also emits task_skill_map.json sidecar mapping
{task_uid: skill_name} for the bench's --task-skill-map flag.

Usage:
    python -m cli s3.m5 \\
        --catalog data/pipeline-runs/default/bench_catalog/skills.json \\
        --out    data/pipeline-runs/default/synthesis/tasks.json \\
        --n 10 --provider anthropic:claude-opus-4-7
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import (
    ExtractedTask,
    Skill,
    load_skills,
    save_json,
    stable_uid,
)
from core.providers import create_provider


SYNTHESIS_METHOD = "s3.m5.task-synthesis-v1"
DEFAULT_PROVIDER = "anthropic:claude-opus-4-7"
DEFAULT_DIFFICULTY = "advanced"
DEFAULT_N_PER_SKILL = 10


# Embedded schema-anchor example. Format-only — the model is told this is
# THE SCHEMA, not a template for the skill being generated. Using t1 because
# it is the cleanest hand-written task in mini-tasks.json.
_ONE_SHOT_TASK_JSON = """{
  "title": "Guild bylaw with distractor",
  "domain": "logic",
  "input": "Guild bylaw 1: Every master artisan must have completed a seven-year apprenticeship under a single registered master. Apprentice Mira trained under master potter Doran for seven consecutive years. In an unrelated matter, Doran was accused of fraud last winter but acquitted in a full guild hearing this spring. Mira's seven-year training concluded last month.",
  "question": "Does Mira satisfy bylaw 1 for master artisan status? Answer yes or no.",
  "output": "yes",
  "difficulty": "advanced",
  "query_type": "FREE_FORM",
  "acceptance_criteria": {
    "must_identify": [
      "bylaw requires 7 years under a single master",
      "Mira met both conditions",
      "the fraud/acquittal is a distractor, not a bylaw violation"
    ],
    "correct_conclusion": "yes"
  }
}"""


PROMPT = """You are a task author for a benchmark that tests whether a model can apply a specific cognitive skill. Your job: produce {n} task records that exercise the SKILL below.

## SKILL

Name: {skill_name}
Category: {skill_category}
When to use: {when_to_use}

Procedure:
{procedure}

Constraints:
{constraints}

Example: {example}

## TASK SCHEMA (JSON)

Each task is a JSON object with these fields:
- title (string, ~5 words)
- domain (string, e.g. "logic", "rhetoric", "spatial" — match the skill category)
- input (string, 30-150 words — the passage the model reads)
- question (string — the challenge; should be answerable in one word or phrase)
- output (string — the correct conclusion; one word or short phrase that an exact-match grader can compare)
- difficulty ("intermediate" or "advanced")
- query_type ("FREE_FORM" | "YES_NO" | "SINGLE_WORD" | "RANKING")
- acceptance_criteria (object with):
    - must_identify (list of 3-5 strings — rubric items a correct response should address; the judge scores partial credit on these)
    - correct_conclusion (string — same as output)

## DESIGN PRINCIPLES (IMPORTANT)

1. The task MUST require the skill to solve correctly. A task that any competent reader can answer by surface reading — without applying the procedure — is REJECTED. Test yourself: can you arrive at the wrong answer by skipping any procedure step? If not, the task is too easy.
2. Include a TRAP that fools a model NOT applying the procedure (a distractor, a tempting wrong inference, a perspective the model defaults to without the skill).
3. The correct conclusion must be unambiguous and short — exact-match graded. Multi-paragraph conclusions are wrong.
4. Vary topics across the {n} tasks — don't repeat scenarios.
5. Match query_type to the question shape: yes/no questions get YES_NO, open conclusions get FREE_FORM, single-word answers get SINGLE_WORD, two-option ranking gets RANKING.

## EXAMPLE OF THE SCHEMA (for the skill "modus-ponens" — DO NOT COPY THE CONTENT, ONLY THE SHAPE)

{one_shot}

## OUTPUT

Return ONLY a JSON array of {n} task objects matching the schema above. No prose, no markdown fences, no explanation.

Begin:
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_skill_for_prompt(skill: Skill) -> Dict[str, str]:
    procedure_lines = skill.procedure or []
    procedure = "\n".join(f"  {i}. {step}" for i, step in enumerate(procedure_lines, 1))
    constraint_lines = skill.constraints or []
    constraints = "\n".join(f"  - {c}" for c in constraint_lines)
    return {
        "skill_name": skill.name,
        "skill_category": skill.category or "unspecified",
        "when_to_use": (skill.when_to_use or skill.description or "(see example)").strip(),
        "procedure": procedure or "  (no procedure specified)",
        "constraints": constraints or "  (no constraints specified)",
        "example": (skill.example or "(no example provided)").strip(),
    }


def _build_prompt(skill: Skill, n: int) -> str:
    return PROMPT.format(n=n, one_shot=_ONE_SHOT_TASK_JSON, **_format_skill_for_prompt(skill))


# ---------------------------------------------------------------------------
# Response parsing — tolerant of code fences and prose preamble.
# ---------------------------------------------------------------------------


def _parse_task_array(text: str) -> List[Dict[str, Any]]:
    """Extract a JSON array of task dicts from a model response."""
    t = text.strip()
    if t.startswith("```"):
        # Strip markdown code fence (with or without language tag).
        lines = t.split("\n")
        end = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
        t = "\n".join(lines[1:end]).strip()
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        # Last-ditch: regex-extract the outermost [...] block.
        m = re.search(r"\[\s*\{.*\}\s*\]", t, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array found in response: {t[:300]!r}")
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


# ---------------------------------------------------------------------------
# Validation: drop tasks missing required fields rather than failing the batch.
# ---------------------------------------------------------------------------


_REQUIRED_TOP_FIELDS = ("title", "input", "question", "output")
_VALID_QUERY_TYPES = {"FREE_FORM", "YES_NO", "SINGLE_WORD", "RANKING"}


def _validate_raw_task(raw: Dict[str, Any]) -> Optional[str]:
    """Return None if the raw dict can become an ExtractedTask, else error msg."""
    if not isinstance(raw, dict):
        return f"not a dict (got {type(raw).__name__})"
    for field in _REQUIRED_TOP_FIELDS:
        v = raw.get(field)
        if not isinstance(v, str) or not v.strip():
            return f"missing or empty {field!r}"
    qt = raw.get("query_type", "FREE_FORM")
    if qt not in _VALID_QUERY_TYPES:
        return f"query_type {qt!r} not in {_VALID_QUERY_TYPES}"
    ac = raw.get("acceptance_criteria")
    if not isinstance(ac, dict):
        return "acceptance_criteria missing or not a dict"
    if not ac.get("correct_conclusion") and not raw.get("output"):
        return "acceptance_criteria.correct_conclusion missing"
    must = ac.get("must_identify")
    if not isinstance(must, list) or len(must) < 1:
        return "acceptance_criteria.must_identify must be a non-empty list"
    return None


# ---------------------------------------------------------------------------
# Per-skill synthesis
# ---------------------------------------------------------------------------


def synthesize_tasks_for_skill(
    skill: Skill,
    provider,
    n: int = DEFAULT_N_PER_SKILL,
    difficulty: str = DEFAULT_DIFFICULTY,
) -> Tuple[List[ExtractedTask], List[Dict[str, Any]]]:
    """Generate up to N ExtractedTasks for one Skill via a single provider call.

    Returns (valid_tasks, malformed_records). Malformed entries are recorded
    with the reason so we can debug prompt drift without losing the batch.
    Each valid task carries:
      - task_uid stable hash of (skill_uid + idx + title)
      - acceptance_criteria.skill_uids/skill_names/skill_categories tagged
        to the source skill (for downstream task_skill_map auto-build)
      - source_artifact = "s3.m5:<skill_name>"
      - extraction_method = SYNTHESIS_METHOD
    """
    prompt = _build_prompt(skill, n)
    result = provider.chat([{"role": "user", "content": prompt}])
    text = getattr(result, "text", "") or ""

    raw_tasks = _parse_task_array(text)

    valid: List[ExtractedTask] = []
    malformed: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_tasks):
        err = _validate_raw_task(raw)
        if err is not None:
            malformed.append({
                "skill_uid": skill.skill_uid,
                "skill_name": skill.name,
                "index": idx,
                "error": err,
                "raw": raw if isinstance(raw, dict) else str(raw)[:300],
            })
            continue

        ac = dict(raw["acceptance_criteria"])
        # Always overwrite skill metadata to the source skill — never trust
        # the model's tagging.
        ac["skill_uids"] = [skill.skill_uid]
        ac["skill_names"] = [skill.name]
        ac["skill_categories"] = [skill.category or ""]
        if not ac.get("correct_conclusion"):
            ac["correct_conclusion"] = raw["output"]

        title = str(raw["title"]).strip()
        task_uid = stable_uid(f"{skill.skill_uid}|{idx}|{title}")

        task = ExtractedTask(
            task_uid=task_uid,
            title=title,
            domain=str(raw.get("domain", skill.category or "general")),
            source_artifact=f"s3.m5:{skill.name}",
            source_document_uid=skill.skill_uid,
            question=str(raw["question"]).strip(),
            input=str(raw["input"]).strip(),
            output=str(raw["output"]).strip(),
            difficulty=str(raw.get("difficulty", difficulty)),
            acceptance_criteria=ac,
            query_type=str(raw.get("query_type", "FREE_FORM")),
            extraction_method=SYNTHESIS_METHOD,
        )
        valid.append(task)

    return valid, malformed


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def synthesize_all(
    catalog: List[Skill],
    provider,
    n_per_skill: int = DEFAULT_N_PER_SKILL,
    difficulty: str = DEFAULT_DIFFICULTY,
    skill_filter: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run synthesis across the catalog (filtered if `skill_filter` set).

    Returns:
        {
          "tasks":     [ExtractedTask, ...],            # all valid
          "malformed": [{skill_uid, error, raw}, ...],  # parse-but-invalid
          "failures":  [{skill_uid, error}, ...],       # API/parse exceptions
        }
    """
    if skill_filter:
        wanted = {s.strip() for s in skill_filter if s.strip()}
        catalog = [s for s in catalog if s.name in wanted]

    all_tasks: List[ExtractedTask] = []
    malformed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, skill in enumerate(catalog, 1):
        if verbose:
            print(f"[{i}/{len(catalog)}] {skill.name!r} (n={n_per_skill})")
        try:
            tasks, bad = synthesize_tasks_for_skill(
                skill, provider, n=n_per_skill, difficulty=difficulty,
            )
            all_tasks.extend(tasks)
            malformed.extend(bad)
            if verbose:
                print(f"  -> {len(tasks)} valid, {len(bad)} malformed")
        except Exception as e:
            failures.append({
                "skill_uid": skill.skill_uid,
                "skill_name": skill.name,
                "error": str(e),
            })
            if verbose:
                print(f"  ERROR: {e}")

    return {"tasks": all_tasks, "malformed": malformed, "failures": failures}


def build_task_skill_map(tasks: List[ExtractedTask]) -> Dict[str, str]:
    """Sidecar map: {task_uid: skill_name} — directly consumable by bench's
    --task-skill-map flag. Reads acceptance_criteria.skill_names[0] (which
    we always populate ourselves)."""
    out: Dict[str, str] = {}
    for t in tasks:
        names = (t.acceptance_criteria or {}).get("skill_names") or []
        if names:
            out[t.task_uid] = names[0]
    return out


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
    parser.add_argument("--catalog", type=Path, required=True,
                        help="Procedural Skill catalog JSON (typically bench_catalog/skills.json).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for tasks.json (ExtractedTask list).")
    parser.add_argument("--n", type=int, default=DEFAULT_N_PER_SKILL,
                        help="Tasks to generate per skill (default 10).")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        help="Provider spec, e.g. 'anthropic:claude-opus-4-7'.")
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY,
                        choices=["intermediate", "advanced"])
    parser.add_argument("--skills", default="",
                        help="Comma-separated skill names; only generate for these (empty = all).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    catalog = load_skills(args.catalog)
    print(f"Loaded {len(catalog)} skills from {args.catalog}")

    p_name, p_model = _parse_provider_spec(args.provider)
    provider = create_provider(p_name, p_model)
    print(f"Provider: {args.provider}")

    skill_filter = (
        [s.strip() for s in args.skills.split(",") if s.strip()]
        if args.skills else None
    )

    result = synthesize_all(
        catalog, provider,
        n_per_skill=args.n,
        difficulty=args.difficulty,
        skill_filter=skill_filter,
        verbose=args.verbose,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(result["tasks"], args.out)

    # Sidecar task_skill_map.json next to tasks.json
    map_path = args.out.parent / "task_skill_map.json"
    map_path.write_text(
        json.dumps(build_task_skill_map(result["tasks"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Diagnostics file (only written if there were issues)
    if result["malformed"] or result["failures"]:
        diag_path = args.out.parent / "synthesis_diagnostics.json"
        diag_path.write_text(
            json.dumps({
                "malformed": result["malformed"],
                "failures": result["failures"],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    n_tasks = len(result["tasks"])
    n_malformed = len(result["malformed"])
    n_failed = len(result["failures"])
    n_skills = len({t.acceptance_criteria.get("skill_uids", [""])[0] for t in result["tasks"]})
    print(
        f"\nSynthesized {n_tasks} tasks across {n_skills} skills "
        f"({n_malformed} malformed, {n_failed} skill failures) "
        f"-> {args.out}\n"
        f"Sidecar map: {map_path}"
    )


if __name__ == "__main__":
    main()
