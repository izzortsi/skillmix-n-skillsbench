"""
b2_benchmarks.skillsbench.skill_injection

Format Skill content for system-prompt injection, matching the SkillsBench
paper's approach: "Skills are provided as system-level context preceding the
task instruction."

Produces a `--- SKILL: name --- ... --- END SKILL ---` block appended to the
base system prompt.
"""

from __future__ import annotations

from core.schemas import Skill


DEFAULT_READING_COMPREHENSION_PROMPT = (
    "You are an expert in reading comprehension and analysis. Analyze the given "
    "passage and challenge carefully. Provide a thorough, well-structured answer "
    "that addresses all aspects of the challenge. Support your analysis with "
    "specific evidence from the passage."
)


DEFAULT_CODING_PROMPT = (
    "You are an expert software developer. Solve the given programming task step "
    "by step. You have access to a bash tool to execute commands."
)


def get_default_system_prompt(domain: str) -> str:
    """Return the default base system prompt for a task domain."""
    if domain in ("coding", "code_bugfix"):
        return DEFAULT_CODING_PROMPT
    return DEFAULT_READING_COMPREHENSION_PROMPT


def _render_skill_body(skill: Skill) -> str:
    """Markdown body for a Skill: description + when_to_use + procedure + constraints + example."""
    parts = []
    if skill.description:
        parts.append(skill.description)
        parts.append("")
    if skill.when_to_use:
        parts.append("## When to Use")
        parts.append(skill.when_to_use)
        parts.append("")
    if skill.procedure:
        parts.append("## Procedure")
        for i, step in enumerate(skill.procedure, 1):
            parts.append(f"{i}. {step}")
        parts.append("")
    if skill.constraints:
        parts.append("## Constraints")
        for c in skill.constraints:
            parts.append(f"- {c}")
        parts.append("")
    if skill.example:
        parts.append("## Example")
        parts.append(skill.example)
    return "\n".join(parts).rstrip()


def format_skill_as_system(base_system_prompt: str, skill: Skill) -> str:
    """Append a `--- SKILL: name --- ... --- END SKILL ---` block to a base prompt."""
    body = _render_skill_body(skill)
    block = (
        f"\n\n--- SKILL: {skill.name} ---\n"
        f"{body}\n"
        f"--- END SKILL ---"
    )
    return base_system_prompt.rstrip() + block
