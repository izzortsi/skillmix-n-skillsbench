"""
core/schemas.py

Single canonical data types for the scaffold. Every module in s1/s2/s3/s4
imports from here; there are no parallel dataclass definitions elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Skill:
    """A skill — either declarative (Wikipedia-style definition + example) or
    procedural (extracted from traces with a numbered procedure).

    `source` identifies the PROJECT_SPECS method that produced this skill, e.g.
    "s2.m1.wikipedia-seed", "s1.m2.direct-elicitation", "s1.m1.task-labeling".
    """

    skill_uid: str
    name: str                                   # kebab-case
    description: str                            # one-sentence definition
    category: str = ""                          # rhetorical / logical / literary / ...
    example: str = ""                           # representative use of the skill
    procedure: List[str] = field(default_factory=list)    # empty for declarative
    when_to_use: str = ""
    constraints: List[str] = field(default_factory=list)
    source: str = ""                            # PROJECT_SPECS method id
    parent_skill_uid: str = ""                  # for hierarchy (subskills)


@dataclass
class Topic:
    """A topic — a domain context used as constraint in skill-composition tasks."""

    topic_uid: str
    name: str                                   # human-readable
    source: str = ""


@dataclass
class SkillExample:
    """A generated (question, answer) pair that exhibits one or more skills.

    Produced by s3 methods. If `verified` is true, an agentic verification loop
    (s3.m4) confirmed the answer correctly demonstrates each listed skill.
    """

    example_uid: str
    skill_uids: List[str]                       # which skills this example exhibits
    topic_uid: str = ""                         # optional topic constraint
    question: str = ""
    answer: str = ""
    verified: bool = False
    source: str = ""                            # s3.m1 / s3.m2 / s3.m3 / s3.m4


@dataclass
class SkillMixTrial:
    """One trial of s2.m2 (Skill-Mix evaluation) for a single model.

    `rubric_points` is the k+3 rubric from Yu et al.: one point per skill
    illustrated + on_topic + coherent + within_sentence_limit. `score` is the
    raw sum; `rescaled` is the paper's rescaled score (requires ratio of
    full-points trials).
    """

    trial_uid: str
    model: str
    k: int
    sentence_limit: int
    skill_uids: List[str]
    topic_uid: str
    prompt: str = ""
    response: str = ""
    rubric_points: dict = field(default_factory=dict)   # {"skill_1": 1, ..., "on_topic": 1, ...}
    score: float = 0.0
    full_points: bool = False
    rationale: str = ""


VALID_QUERY_TYPES = ("YES_NO", "YES_NO_VERIFICATION", "SINGLE_WORD", "RANKING", "FREE_FORM")


@dataclass
class ExtractedTask:
    """A task generated from a text artifact — input to solving/evaluation.

    Distinct from SkillExample:
      - SkillExample is a (question, answer) PAIR generated TO illustrate skills
        (s3 methods produce these as training/eval data).
      - ExtractedTask is a TASK to be solved, sourced from a text artifact.
        A downstream solver runs a model on it and compares to `output`
        (s4.m4 eval half, b2.benchmarks/skillsbench/).

    Canonical query structure:
        question:    the query itself (legacy alias: challenge)
        input:       the corpus/passage the query refers to (legacy alias: passage)
        output:      the expected answer
        query_type:  one of VALID_QUERY_TYPES
    """

    task_uid: str
    title: str
    domain: str
    source_artifact: str
    source_document_uid: str
    question: str
    input: str
    output: str
    difficulty: str
    acceptance_criteria: dict
    query_type: str = "FREE_FORM"
    extraction_method: str = ""

    @property
    def passage(self) -> str:
        """Backward-compatible alias for `input`."""
        return self.input

    @property
    def challenge(self) -> str:
        """Backward-compatible alias for `question`."""
        return self.question


def validate_free_form_single_answer(task: ExtractedTask) -> None:
    """FREE_FORM tasks must have exactly one `output`; raise otherwise.

    Why: earlier pipelines packed multiple answers into `output` with separators
    like ' OR ', ' / ', ' | '. Solving against a multi-answer output silently
    biases scoring, so we fail loud on load.
    """
    if task.query_type != "FREE_FORM":
        return
    if not task.output.strip():
        raise ValueError(f"FREE_FORM task {task.task_uid} has empty output")
    conclusion = task.output.strip()
    for sep in (" OR ", " / ", " | "):
        if sep in conclusion:
            raise ValueError(
                f"FREE_FORM task {task.task_uid} has multiple answers separated "
                f"by {sep.strip()!r} in output: {conclusion[:100]!r}"
            )


# ---------------------------------------------------------------------------
# UID + kebab helpers — used by every method
# ---------------------------------------------------------------------------


def stable_uid(seed: str) -> str:
    """Deterministic UID from a seed string. Format: xxxx-xxxx-xxxx-xxxx."""
    h = hashlib.sha256(seed.encode()).digest()
    n = int.from_bytes(h[:8], "big")
    hx = f"{n:016x}"
    return f"{hx[0:4]}-{hx[4:8]}-{hx[8:12]}-{hx[12:16]}"


def to_kebab(name: str) -> str:
    """Normalize a display name to kebab-case. Strips punctuation, preserves words."""
    cleaned = re.sub(r"[()]", "", name)
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return "-".join(cleaned.lower().split())


# ---------------------------------------------------------------------------
# Generic JSON I/O — Skill / Topic / SkillExample / SkillMixTrial
# ---------------------------------------------------------------------------


def save_json(items: list, output_path: Path) -> None:
    """Save a list of dataclass instances to a pretty-printed JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(x) for x in items]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_skills(path: Path) -> List[Skill]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Skill(**_filter_fields(entry, Skill)) for entry in data]


def load_topics(path: Path) -> List[Topic]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Topic(**_filter_fields(entry, Topic)) for entry in data]


def load_examples(path: Path) -> List[SkillExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SkillExample(**_filter_fields(entry, SkillExample)) for entry in data]


def load_trials(path: Path) -> List[SkillMixTrial]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SkillMixTrial(**_filter_fields(entry, SkillMixTrial)) for entry in data]


def load_extracted_tasks(path: Path) -> List[ExtractedTask]:
    """Load ExtractedTasks with back-compat for legacy field names.

    Reads legacy aliases (task_id -> task_uid, challenge -> question,
    passage -> input, acceptance_criteria.correct_conclusion -> output)
    so archived runs deserialize. Every loaded task is validated.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks: List[ExtractedTask] = []
    for entry in data:
        ac = entry.get("acceptance_criteria", {}) or {}
        tasks.append(ExtractedTask(
            task_uid=entry.get("task_uid", entry.get("task_id", "")),
            title=entry.get("title", ""),
            domain=entry.get("domain", ""),
            source_artifact=entry.get("source_artifact", ""),
            source_document_uid=entry.get("source_document_uid", ""),
            question=entry.get("question", entry.get("challenge", "")),
            input=entry.get("input", entry.get("passage", "")),
            output=entry.get("output", ac.get("correct_conclusion", "")),
            difficulty=entry.get("difficulty", "intermediate"),
            acceptance_criteria=ac,
            query_type=entry.get("query_type", "FREE_FORM"),
            extraction_method=entry.get("extraction_method", ""),
        ))
    for t in tasks:
        validate_free_form_single_answer(t)
    return tasks


def _filter_fields(entry: dict, dataclass_type) -> dict:
    """Drop unknown keys so we stay forward-compatible with new fields on disk."""
    known = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    return {k: v for k, v in entry.items() if k in known}
