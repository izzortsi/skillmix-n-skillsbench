"""
core/formatters.py

Bidirectional converters between JSON pipeline format and markdown
skill-creator / task template format.

Public functions:
    skill_to_markdown      -- Skill -> .md string
    markdown_to_skill      -- .md string -> Skill
    task_to_markdown       -- ExtractedTask -> .md string
    markdown_to_task       -- .md string -> ExtractedTask
    skills_json_to_dir     -- skills.json -> directory of .md files
    skills_dir_to_json     -- directory of .md files -> skills.json
    tasks_json_to_dir      -- tasks.json -> directory of .md files
    tasks_dir_to_json      -- directory of .md files -> tasks.json

Markdown format is the one the skill-creator template and registry use;
it round-trips all Skill / ExtractedTask fields with the exceptions noted
in each converter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

from core.schemas import (
    ExtractedTask,
    Skill,
    load_extracted_tasks,
    load_skills,
    save_json,
)


# ---------------------------------------------------------------------------
# Skill: JSON <-> Markdown
# ---------------------------------------------------------------------------


def skill_to_markdown(skill: Skill) -> str:
    """Convert a Skill to markdown skill-creator template format.

    Layout:
        ---
        name: kebab-case-name
        description: "Use when ... one-line description"
        skill_uid: xxxx-xxxx-xxxx-xxxx
        category: rhetorical
        source: s2.m1.wikipedia-seed
        parent_skill_uid: ....
        ---
        # name

        description

        ## When to Use
        - when_to_use

        ## Procedure
        1. step one
        2. step two

        ## Constraints
        - constraint

        ## Example
        example text
    """
    frontmatter = {
        "name": skill.name,
        "description": (
            skill.when_to_use
            if skill.when_to_use.startswith("Use when")
            else f"Use when {skill.when_to_use}"
        ),
    }
    if skill.skill_uid:
        frontmatter["skill_uid"] = skill.skill_uid
    if skill.category:
        frontmatter["category"] = skill.category
    if skill.source:
        frontmatter["source"] = skill.source
    if skill.parent_skill_uid:
        frontmatter["parent_skill_uid"] = skill.parent_skill_uid

    lines: List[str] = []
    lines.append("---")
    lines.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).rstrip())
    lines.append("---")
    lines.append("")

    lines.append(f"# {skill.name}")
    lines.append("")
    if skill.description:
        lines.append(skill.description)
        lines.append("")

    lines.append("## When to Use")
    lines.append("")
    lines.append(f"- {skill.when_to_use}")
    lines.append("")

    if skill.procedure:
        lines.append("## Procedure")
        lines.append("")
        for i, step in enumerate(skill.procedure, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if skill.constraints:
        lines.append("## Constraints")
        lines.append("")
        for c in skill.constraints:
            lines.append(f"- {c}")
        lines.append("")

    if skill.example:
        lines.append("## Example")
        lines.append("")
        lines.append(skill.example)
        lines.append("")

    return "\n".join(lines)


def markdown_to_skill(content: str) -> Skill:
    """Parse a markdown skill-creator template into a Skill."""
    fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not fm_match:
        raise ValueError("no YAML frontmatter found in skill markdown")
    frontmatter = yaml.safe_load(fm_match.group(1)) or {}
    body = content[fm_match.end():]

    description_raw = str(frontmatter.get("description", ""))
    when_to_use = ""
    if description_raw.startswith("Use when "):
        when_to_use = description_raw[len("Use when "):]
    else:
        bullets = _extract_bullets(body, "When to Use")
        when_to_use = bullets[0] if bullets else description_raw

    description = description_raw
    desc_match = re.search(r"^# .+\n\n(.+?)(?:\n\n|\n## )", body, re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()

    return Skill(
        skill_uid=frontmatter.get("skill_uid", ""),
        name=frontmatter.get("name", ""),
        description=description,
        category=frontmatter.get("category", ""),
        example=_extract_section_text(body, "Example"),
        procedure=_extract_numbered(body, "Procedure"),
        when_to_use=when_to_use,
        constraints=_extract_bullets(body, "Constraints"),
        source=frontmatter.get("source", ""),
        parent_skill_uid=frontmatter.get("parent_skill_uid", ""),
    )


# ---------------------------------------------------------------------------
# ExtractedTask: JSON <-> Markdown
# ---------------------------------------------------------------------------


def task_to_markdown(task: ExtractedTask) -> str:
    """Convert an ExtractedTask to markdown with YAML frontmatter."""
    frontmatter = {
        "task_uid": task.task_uid,
        "title": task.title,
        "domain": task.domain,
        "query_type": task.query_type,
        "difficulty": task.difficulty,
        "source_artifact": task.source_artifact,
        "source_document_uid": task.source_document_uid,
    }
    if task.extraction_method:
        frontmatter["extraction_method"] = task.extraction_method

    lines: List[str] = []
    lines.append("---")
    lines.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).rstrip())
    lines.append("---")
    lines.append("")
    lines.append(f"# {task.title}")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(task.question)
    lines.append("")
    lines.append("## Passage")
    lines.append("")
    lines.append(task.input)
    lines.append("")
    lines.append("## Expected Output")
    lines.append("")
    lines.append(task.output)
    lines.append("")

    ac = task.acceptance_criteria or {}
    if ac:
        lines.append("## Acceptance Criteria")
        lines.append("")
        must_identify = ac.get("must_identify") or []
        if must_identify:
            lines.append("### Must Identify")
            lines.append("")
            for item in must_identify:
                lines.append(f"- {item}")
            lines.append("")
        conclusion = ac.get("correct_conclusion", "")
        if conclusion:
            lines.append("### Correct Conclusion")
            lines.append("")
            lines.append(conclusion)
            lines.append("")

    return "\n".join(lines)


def markdown_to_task(content: str) -> ExtractedTask:
    """Parse a markdown task file into an ExtractedTask."""
    fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not fm_match:
        raise ValueError("no YAML frontmatter found in task markdown")
    frontmatter = yaml.safe_load(fm_match.group(1)) or {}
    body = content[fm_match.end():]

    must_identify = _extract_bullets(body, "Must Identify")
    conclusion = _extract_section_text(body, "Correct Conclusion")
    acceptance_criteria: dict = {}
    if must_identify:
        acceptance_criteria["must_identify"] = must_identify
    if conclusion:
        acceptance_criteria["correct_conclusion"] = conclusion

    return ExtractedTask(
        task_uid=frontmatter.get("task_uid", ""),
        title=frontmatter.get("title", ""),
        domain=frontmatter.get("domain", ""),
        source_artifact=frontmatter.get("source_artifact", ""),
        source_document_uid=frontmatter.get("source_document_uid", ""),
        question=_extract_section_text(body, "Question"),
        input=_extract_section_text(body, "Passage"),
        output=_extract_section_text(body, "Expected Output"),
        difficulty=frontmatter.get("difficulty", ""),
        acceptance_criteria=acceptance_criteria,
        query_type=frontmatter.get("query_type", "FREE_FORM"),
        extraction_method=frontmatter.get("extraction_method", ""),
    )


# ---------------------------------------------------------------------------
# Batch: JSON file <-> directory of .md files
# ---------------------------------------------------------------------------


def skills_json_to_dir(json_path: Path, output_dir: Path) -> int:
    """Convert a skills JSON file to a directory of {name}.md files."""
    skills = load_skills(json_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for skill in skills:
        (output_dir / f"{skill.name}.md").write_text(
            skill_to_markdown(skill), encoding="utf-8"
        )
    return len(skills)


def skills_dir_to_json(input_dir: Path, output_path: Path) -> int:
    """Convert a directory of skill .md files to a skills JSON file."""
    skills = [
        markdown_to_skill(p.read_text(encoding="utf-8"))
        for p in sorted(input_dir.glob("*.md"))
    ]
    save_json(skills, output_path)
    return len(skills)


def tasks_json_to_dir(json_path: Path, output_dir: Path) -> int:
    """Convert a tasks JSON file to a directory of {task_uid}.md files."""
    tasks = load_extracted_tasks(json_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        (output_dir / f"{task.task_uid}.md").write_text(
            task_to_markdown(task), encoding="utf-8"
        )
    return len(tasks)


def tasks_dir_to_json(input_dir: Path, output_path: Path) -> int:
    """Convert a directory of task .md files to a tasks JSON file."""
    tasks = [
        markdown_to_task(p.read_text(encoding="utf-8"))
        for p in sorted(input_dir.glob("*.md"))
    ]
    save_json(tasks, output_path)
    return len(tasks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_bullets(content: str, section_name: str) -> List[str]:
    pattern = rf"##+ {re.escape(section_name)}\n(.*?)(?=\n##+ |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    return re.findall(r"^\s*[-*]\s+(.+)$", match.group(1), re.MULTILINE)


def _extract_numbered(content: str, section_name: str) -> List[str]:
    pattern = rf"##+ {re.escape(section_name)}\n(.*?)(?=\n##+ |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    return re.findall(r"^\s*\d+\.\s+(.+)$", match.group(1), re.MULTILINE)


def _extract_section_text(content: str, section_name: str) -> str:
    pattern = rf"##+ {re.escape(section_name)}\n\n(.*?)(?=\n##+ |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert between JSON and markdown for skills and tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m core.formatters skills-to-md   --input skills.json --output-dir skills-md/
  python -m core.formatters skills-to-json --input-dir skills-md/ --output skills.json
  python -m core.formatters tasks-to-md    --input tasks.json  --output-dir tasks-md/
  python -m core.formatters tasks-to-json  --input-dir tasks-md/ --output tasks.json
""",
    )
    parser.add_argument(
        "command",
        choices=["skills-to-md", "skills-to-json", "tasks-to-md", "tasks-to-json"],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.command == "skills-to-md":
        n = skills_json_to_dir(args.input, args.output_dir)
        print(f"converted {n} skills -> {args.output_dir}/")
    elif args.command == "skills-to-json":
        n = skills_dir_to_json(args.input_dir, args.output)
        print(f"converted {n} skills -> {args.output}")
    elif args.command == "tasks-to-md":
        n = tasks_json_to_dir(args.input, args.output_dir)
        print(f"converted {n} tasks -> {args.output_dir}/")
    elif args.command == "tasks-to-json":
        n = tasks_dir_to_json(args.input_dir, args.output)
        print(f"converted {n} tasks -> {args.output}")


if __name__ == "__main__":
    main()
