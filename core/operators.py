"""
core/operators.py

Composition operators for skills — a non-PROJECT_SPECS utility used by
higher-level methods that want to compose atomic Skill records into
multi-skill tasks.

Four operators:
  - seq  : sequential  (output of skill_i feeds skill_{i+1})
  - par  : parallel    (all skills applied to the same input, results merged)
  - cond : conditional (first skill's outcome chooses then/else branch)
  - sem  : semantic    (LLM-based fusion — see SemanticCompositor)

Ported from skillsuite/llm-skills.extraction-pipeline/c2_composition/operators.py,
rewired onto core.schemas.Skill (scaffold's canonical Skill dataclass) and
core.providers.ChatResult. SkillRegistry / `.md` discovery are dropped —
callers pass a plain List[Skill]; load via core.schemas.load_skills or
core.formatters.skills_dir_to_json.

Usage:
    from core.schemas import load_skills
    from core.operators import compose_seq, generate_all_compositions

    skills = load_skills(Path("data/wikipedia-seed/skills.json"))
    composed = compose_seq(skills[:2])
    print(composed.to_markdown())

    all_comps = generate_all_compositions(skills, max_k=3)
    for op, entries in all_comps.items():
        print(op, len(entries))
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.schemas import Skill


# ---------------------------------------------------------------------------
# ComposedSkill — the output type of every operator
# ---------------------------------------------------------------------------


@dataclass
class ComposedSkill:
    """A skill composed from atomic Skill records.

    Carries structured examples (input/process/output) and composition metadata
    that the atomic Skill dataclass does not track.
    """

    name: str
    description: str
    when_to_use: List[str]
    procedure: List[str]
    constraints: List[str]
    examples: List[Dict[str, str]]
    related_skills: List[str]
    composition_type: str                         # "seq" | "par" | "cond" | "sem"
    source_skills: List[str]                      # atomic Skill.name values used
    k_value: int                                  # number of atomic skills composed

    def to_markdown(self) -> str:
        """Render in skill-creator template format (same shape as core.formatters)."""
        lines: List[str] = []
        lines.append("---")
        lines.append(f"name: {self.name}")
        lines.append(f"description: {self.description}")
        lines.append(f"composition_type: {self.composition_type}")
        lines.append(f"k_value: {self.k_value}")
        lines.append(f"source_skills: {self.source_skills}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {format_title(self.name)}")
        lines.append("")

        lines.append("## When to Use")
        lines.append("")
        for trigger in self.when_to_use:
            lines.append(f"- {trigger}")
        lines.append("")

        lines.append("## Procedure")
        lines.append("")
        for i, step in enumerate(self.procedure, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        if self.constraints:
            lines.append("## Constraints")
            lines.append("")
            for c in self.constraints:
                lines.append(f"- {c}")
            lines.append("")

        for i, ex in enumerate(self.examples, 1):
            title = ex.get("title", f"Example {i}")
            lines.append(f"### Example {i}: {title}")
            lines.append("")
            if ex.get("input"):
                lines.append(f"**Input:** {ex['input']}")
                lines.append("")
            if ex.get("process"):
                lines.append(f"**Process:** {ex['process']}")
                lines.append("")
            if ex.get("output"):
                lines.append(f"**Output:** {ex['output']}")
                lines.append("")

        if self.related_skills:
            lines.append("## Related Skills")
            lines.append("")
            for n in self.related_skills:
                lines.append(f"- {n}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter: scaffold Skill -> internal view used by the operators
#
# Scaffold Skill carries `when_to_use: str` and `example: str`; operators want
# `when_to_use: List[str]` and `examples: List[Dict[str,str]]`. We convert at
# the boundary so the operator code stays readable.
# ---------------------------------------------------------------------------


@dataclass
class _SkillView:
    name: str
    description: str
    when_to_use: List[str]
    procedure: List[str]
    constraints: List[str]
    examples: List[Dict[str, str]]
    related_skills: List[str] = field(default_factory=list)


def _view(skill: Skill) -> _SkillView:
    when_list = [skill.when_to_use] if skill.when_to_use else []
    if skill.example:
        examples = [{
            "title": format_title(skill.name),
            "input": skill.example,
            "process": "",
            "output": "",
        }]
    else:
        examples = []
    return _SkillView(
        name=skill.name,
        description=skill.description,
        when_to_use=when_list,
        procedure=list(skill.procedure),
        constraints=list(skill.constraints),
        examples=examples,
        related_skills=[],
    )


def format_title(name: str) -> str:
    """Convert kebab-case name to Title Case."""
    return name.replace("-", " ").title()


# ---------------------------------------------------------------------------
# Example builders for each operator type
# ---------------------------------------------------------------------------


def _sequential_examples(views: List[_SkillView]) -> List[Dict[str, str]]:
    if not views:
        return []

    heads: List[Dict[str, str]] = []
    for v in views:
        if v.examples:
            heads.append(v.examples[0])
        else:
            heads.append({
                "title": f"Using {format_title(v.name)}",
                "input": "[input requiring this skill]",
                "process": f"follow the procedure for {format_title(v.name)}",
                "output": f"[output from {format_title(v.name)}]",
            })

    if len(heads) == 1:
        return [dict(heads[0])]

    titles = " -> ".join(format_title(v.name) for v in views)
    process_lines = ["Apply the following skills in sequence:"]
    for i, (v, head) in enumerate(zip(views, heads), 1):
        step = head.get("process") or f"Apply {format_title(v.name)}"
        process_lines.append(f"{i}. {format_title(v.name)}: {step[:100]}")

    return [{
        "title": f"Sequential application of {titles}",
        "input": heads[0].get("input", "[input]"),
        "process": "\n".join(process_lines),
        "output": heads[-1].get("output", f"[output from {format_title(views[-1].name)}]"),
    }]


def _parallel_examples(views: List[_SkillView]) -> List[Dict[str, str]]:
    if not views:
        return []
    bullets = []
    for v in views:
        tag = format_title(v.name)
        if v.examples and v.examples[0].get("input"):
            bullets.append(f"- {tag}: {v.examples[0]['input'][:50]}")
        else:
            bullets.append(f"- {tag}")
    return [{
        "title": "Parallel skill application",
        "input": "Input requiring analysis from multiple skill perspectives simultaneously.",
        "process": "Apply the following skills in parallel to the same input:\n"
                   + "\n".join(bullets)
                   + "\n\nThen integrate the results from all skills.",
        "output": "Combined analysis integrating outputs from all parallel skills.",
    }]


def _conditional_examples(
    condition: _SkillView,
    then_views: List[_SkillView],
    else_views: Optional[List[_SkillView]],
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    cond_title = format_title(condition.name)

    then_desc = " -> ".join(format_title(v.name) for v in then_views)
    examples.append({
        "title": "Condition is true",
        "input": f"Input where {cond_title} evaluates to TRUE",
        "process": f"1. {cond_title} evaluates to TRUE\n2. Apply then-skills: {then_desc}",
        "output": f"Output from applying {then_desc}",
    })

    if else_views:
        else_desc = " -> ".join(format_title(v.name) for v in else_views)
        examples.append({
            "title": "Condition is false",
            "input": f"Input where {cond_title} evaluates to FALSE",
            "process": f"1. {cond_title} evaluates to FALSE\n2. Apply else-skills: {else_desc}",
            "output": f"Output from applying {else_desc}",
        })
    return examples


# ---------------------------------------------------------------------------
# Operators: seq, par, cond, single_skill_wrapper
# ---------------------------------------------------------------------------


def compose_seq(skills: List[Skill]) -> ComposedSkill:
    """Sequential composition: alpha_seq(S1, ..., Sn) = S1 -> S2 -> ... -> Sn."""
    if not skills:
        raise ValueError("compose_seq: empty skill list")

    views = [_view(s) for s in skills]
    if len(views) == 1:
        return _single_wrapper(views[0], "seq")

    names = [v.name for v in views]
    name = f"seq-{'-then-'.join(names)}"
    description = (
        f"Apply skills in sequence: {' -> '.join(v.description for v in views)}. "
        "Use when multiple capabilities are needed in order."
    )

    seen: set = set()
    when_to_use: List[str] = []
    for v in views:
        for trigger in v.when_to_use:
            if trigger not in seen:
                seen.add(trigger)
                when_to_use.append(trigger)

    procedure: List[str] = []
    for i, v in enumerate(views):
        lead = f"Apply {format_title(v.name)}"
        if v.procedure:
            for step in v.procedure:
                procedure.append(step if i == 0 else f"{lead}: {step}")
        else:
            # Declarative skill (Wikipedia-seeded / elicited): no procedure
            # steps on file. Emit the lead as a single step so the composition
            # is still usable.
            procedure.append(lead)

    constraints: List[str] = []
    for v in views:
        constraints.extend(v.constraints)

    related: List[str] = []
    seen_names = set(names)
    for v in views:
        for ref in v.related_skills:
            if ref not in seen_names:
                seen_names.add(ref)
                related.append(ref)

    return ComposedSkill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=_sequential_examples(views),
        related_skills=related,
        composition_type="seq",
        source_skills=names,
        k_value=len(views),
    )


def compose_par(skills: List[Skill]) -> ComposedSkill:
    """Parallel composition: alpha_par(S1, ..., Sn) = S1 & S2 & ... & Sn."""
    if not skills:
        raise ValueError("compose_par: empty skill list")

    views = [_view(s) for s in skills]
    if len(views) == 1:
        return _single_wrapper(views[0], "par")

    names = [v.name for v in views]
    name = f"par-{'-and-'.join(names)}"
    description = (
        f"Apply skills in parallel: {' & '.join(v.description for v in views)}. "
        "Use when multiple capabilities are needed simultaneously."
    )

    when_to_use: List[str] = [
        "Multiple skills are needed simultaneously: "
        + ", ".join(format_title(v.name) for v in views)
    ]
    for v in views:
        when_to_use.extend(v.when_to_use)

    procedure: List[str] = ["Apply the following skills in parallel:"]
    for v in views:
        procedure.append(f"  - {format_title(v.name)}: {v.description}")
    procedure.extend([
        "Integrate the outputs from all parallel skills:",
        "  - Identify complementary results across skills",
        "  - Resolve any conflicts between skill outputs",
        "  - Synthesize a combined result",
    ])

    constraints: List[str] = ["All parallel skills must receive the same input context."]
    for v in views:
        if v.constraints:
            constraints.append(v.constraints[0])

    related: List[str] = []
    seen_names = set(names)
    for v in views:
        for ref in v.related_skills:
            if ref not in seen_names:
                seen_names.add(ref)
                related.append(ref)

    return ComposedSkill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=_parallel_examples(views),
        related_skills=related,
        composition_type="par",
        source_skills=names,
        k_value=len(views),
    )


def compose_cond(
    condition_skill: Skill,
    then_skills: List[Skill],
    else_skills: Optional[List[Skill]] = None,
) -> ComposedSkill:
    """Conditional composition: if condition then then_skills else else_skills."""
    if not then_skills:
        raise ValueError("compose_cond: empty then_skills list")

    cond_view = _view(condition_skill)
    then_views = [_view(s) for s in then_skills]
    else_views = [_view(s) for s in (else_skills or [])]
    all_views = [cond_view] + then_views + else_views

    then_names = [v.name for v in then_views]
    else_names = [v.name for v in else_views]

    if else_names:
        name = (
            f"cond-{cond_view.name}-then-{'-and-'.join(then_names)}"
            f"-else-{'-and-'.join(else_names)}"
        )
    else:
        name = f"cond-{cond_view.name}-then-{'-and-'.join(then_names)}"

    description = (
        f"Conditional skill: if {format_title(cond_view.name)}, "
        f"then apply {' -> '.join(format_title(v.name) for v in then_views)}"
    )
    if else_views:
        description += (
            f", else apply {' -> '.join(format_title(v.name) for v in else_views)}"
        )
    description += ". Use when skill application depends on a condition."

    when_to_use = [f"When {format_title(cond_view.name)} determines the path"]
    when_to_use.extend(cond_view.when_to_use)

    procedure: List[str] = ["Evaluate the condition:"]
    procedure.append(f"1. Check {format_title(cond_view.name)}")
    procedure.append("2. If condition is TRUE:")
    for i, v in enumerate(then_views, 1):
        procedure.append(f"   {i}. Apply {format_title(v.name)}")
    procedure.append("3. If condition is FALSE:")
    if else_views:
        for i, v in enumerate(else_views, 1):
            procedure.append(f"   {i}. Apply {format_title(v.name)}")
    else:
        procedure.append("   No action (or apply default behavior)")

    constraints: List[str] = [
        f"Condition must be evaluable: {format_title(cond_view.name)}",
        "Only one branch (then or else) is executed.",
    ]
    constraints.extend(cond_view.constraints)

    all_names = [v.name for v in all_views]
    related: List[str] = []
    seen_names = set(all_names)
    for v in all_views:
        for ref in v.related_skills:
            if ref not in seen_names:
                seen_names.add(ref)
                related.append(ref)

    return ComposedSkill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=_conditional_examples(cond_view, then_views, else_views or None),
        related_skills=related,
        composition_type="cond",
        source_skills=all_names,
        k_value=len(all_views),
    )


def _single_wrapper(view: _SkillView, composition_type: str) -> ComposedSkill:
    """k=1 edge case: wrap a single skill as a composition."""
    return ComposedSkill(
        name=f"{composition_type}-{view.name}",
        description=view.description,
        when_to_use=view.when_to_use,
        procedure=view.procedure,
        constraints=view.constraints,
        examples=view.examples,
        related_skills=view.related_skills,
        composition_type=composition_type,
        source_skills=[view.name],
        k_value=1,
    )


# ---------------------------------------------------------------------------
# Batch generator (combinatorial)
# ---------------------------------------------------------------------------


def generate_all_compositions(
    skills: List[Skill],
    max_k: int = 5,
    min_k: int = 2,
) -> Dict[str, List[ComposedSkill]]:
    """Generate mechanical seq / par / cond compositions for all k-tuples.

    Uses itertools.combinations so every C(n, k) tuple is covered (not the
    sliding-window subset the skillsuite source produced).

    Returns {"seq": [...], "par": [...], "cond": [...]}.
    """
    n = len(skills)
    if n == 0:
        return {"seq": [], "par": [], "cond": []}

    result: Dict[str, List[ComposedSkill]] = {"seq": [], "par": [], "cond": []}

    for k in range(min_k, max_k + 1):
        if k > n:
            break
        for combo in itertools.combinations(skills, k):
            combo_list = list(combo)
            result["seq"].append(compose_seq(combo_list))
            result["par"].append(compose_par(combo_list))
            result["cond"].append(
                compose_cond(combo_list[0], combo_list[1:], else_skills=None)
            )

    return result


# ---------------------------------------------------------------------------
# Semantic composition (LLM-based fusion)
#
# Adapted from skillsuite's SemanticCompositor. Uses core.providers provider
# protocol: `provider.chat(messages) -> ChatResult(text, usage, raw)`.
# ---------------------------------------------------------------------------


@dataclass
class SemanticCompositionConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-4-7"
    fusion_type: str = "auto"                     # "auto" | "sequential" | "parallel" | "conditional"


FUSION_GUIDANCE = {
    "auto": "automatically determine the best way to combine these skills",
    "sequential": "combine these skills so the output of each feeds into the next",
    "parallel": "combine these skills to be applied simultaneously to the same input",
    "conditional": "combine these skills where one acts as a condition for others",
}


SEMANTIC_PROMPT = """You are a skill composition expert. Your task is to create a NEW composite skill by semantically fusing the following input skills.

# Input Skills
{skill_block}

# Your Task
Create a composite skill that {guidance}.

The composite skill should:
1. Have a clear, descriptive name in hyphen-case
2. Include a comprehensive description explaining when and why to use it
3. Specify "When to Use" triggers
4. Provide a step-by-step Procedure
5. List Constraints or limitations
6. Include 2-3 concrete Examples with Input / Process / Output
7. Reference Related Skills

Return ONLY valid JSON with this exact structure:

{{
  "name": "composite-skill-name",
  "description": "Clear description and when to use it",
  "when_to_use": ["trigger 1", "trigger 2"],
  "procedure": ["step 1", "step 2"],
  "constraints": ["constraint 1"],
  "examples": [
    {{"title": "...", "input": "...", "process": "...", "output": "..."}}
  ],
  "fusion_rationale": "how and why these skills were combined"
}}
"""


def compose_sem(
    skills: List[Skill],
    provider=None,
    config: Optional[SemanticCompositionConfig] = None,
) -> ComposedSkill:
    """LLM-fuse skills into a composite skill.

    If `provider` is None, build one from `config` via core.providers.create_provider.
    Falls back to a mechanical default composition when the LLM response can't
    be parsed, so callers always get a ComposedSkill back.
    """
    if not skills:
        raise ValueError("compose_sem: empty skill list")

    views = [_view(s) for s in skills]
    if len(views) == 1:
        return _single_wrapper(views[0], "sem")

    cfg = config or SemanticCompositionConfig()

    if provider is None:
        from core.providers import create_provider
        provider = create_provider(cfg.provider, cfg.model)

    prompt = _build_sem_prompt(views, cfg.fusion_type)
    result = provider.chat([{"role": "user", "content": prompt}])
    text = getattr(result, "text", "")

    data = _parse_sem_json(text)

    names = [v.name for v in views]
    related: List[str] = []
    seen_names = set(names)
    for v in views:
        for ref in v.related_skills:
            if ref not in seen_names:
                seen_names.add(ref)
                related.append(ref)

    if not data:
        return ComposedSkill(
            name=f"sem-{'-'.join(names)}",
            description=f"Semantic fusion of {len(views)} skills.",
            when_to_use=["When integrated skill application is needed."],
            procedure=["Apply skills with semantic understanding."],
            constraints=[],
            examples=[],
            related_skills=related,
            composition_type="sem",
            source_skills=names,
            k_value=len(views),
        )

    examples: List[Dict[str, str]] = []
    for ex in data.get("examples", []) or []:
        examples.append({
            "title": ex.get("title", "Example"),
            "input": ex.get("input", ""),
            "process": ex.get("process", ""),
            "output": ex.get("output", ""),
        })

    return ComposedSkill(
        name=data.get("name", f"sem-{'-'.join(names)}"),
        description=data.get("description", ""),
        when_to_use=list(data.get("when_to_use", []) or []),
        procedure=list(data.get("procedure", []) or []),
        constraints=list(data.get("constraints", []) or []),
        examples=examples,
        related_skills=related,
        composition_type="sem",
        source_skills=names,
        k_value=len(views),
    )


def _build_sem_prompt(views: List[_SkillView], fusion_type: str) -> str:
    blocks: List[str] = []
    for i, v in enumerate(views, 1):
        when = "\n".join(f"- {t}" for t in v.when_to_use[:5]) or "- (none)"
        procedure = "\n".join(f"{j+1}. {s}" for j, s in enumerate(v.procedure[:5])) or "(none)"
        constraints = "\n".join(f"- {c}" for c in v.constraints[:3]) or "- (none)"
        blocks.append(
            f"## Skill {i}: {format_title(v.name)}\n\n"
            f"**Description:** {v.description}\n\n"
            f"**When to Use:**\n{when}\n\n"
            f"**Procedure (summary):**\n{procedure}\n\n"
            f"**Key Constraints:**\n{constraints}"
        )
    return SEMANTIC_PROMPT.format(
        skill_block="\n".join(blocks),
        guidance=FUSION_GUIDANCE.get(fusion_type, FUSION_GUIDANCE["auto"]),
    )


def _parse_sem_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response. Tolerates fences."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else ""
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def generate_semantic_compositions(
    skills: List[Skill],
    max_k: int = 5,
    min_k: int = 2,
    provider=None,
    config: Optional[SemanticCompositionConfig] = None,
) -> List[ComposedSkill]:
    """Apply `compose_sem` to every k-tuple in [min_k, max_k]."""
    n = len(skills)
    if n == 0:
        return []

    if provider is None and config is not None:
        from core.providers import create_provider
        provider = create_provider(config.provider, config.model)

    out: List[ComposedSkill] = []
    for k in range(min_k, max_k + 1):
        if k > n:
            break
        for combo in itertools.combinations(skills, k):
            out.append(compose_sem(list(combo), provider=provider, config=config))
    return out
