"""
s2.m2 — Skill-Mix Evaluation (PROJECT_SPECS §2 Method 2, Yu et al. 2023).

The true Skill-Mix evaluation, as described in the paper (not the misnamed
skill-injection eval in skillsuite/llm-skills.skillmix-evaluation):

  1. Sample a random k-tuple of atomic skills + 1 topic.
  2. Prompt the model to produce a short piece of text (<= N sentences) in
     the context of the topic, illustrating all k skills at once.
  3. Grade with the per-skill rubric (k+3 points: each skill illustrated,
     on-topic, coherent, within sentence limit) via core.judge.

Scoring (paper, Appendix C.1):
  - Total Score:  avg raw points (out of k+3) across trials
  - Ratio Full Marks: fraction of trials scoring all k+3 points
  - Skill Fraction: avg number of skills illustrated / k

Usage:
    python -m scaffold.s2_extracting_skills_from_text.m2_skill_mix_evaluation \\
        --skills data/wikipedia-seed/skills.json \\
        --topics data/topics/topics.json \\
        --k 2 --trials 20 --sentence-limit 2 \\
        --student anthropic:claude-sonnet-4-6 \\
        --judge anthropic:claude-opus-4-6 \\
        --out data/skill-mix-trials.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Optional

from core.schemas import Skill, Topic, SkillMixTrial, stable_uid, save_json, load_skills, load_topics
from core.providers import create_provider
from core.judge import per_skill_rubric_judge


GENERATION_PROMPT = """I am studying natural-language skill composition. Please write a \
single piece of text, up to {sentence_limit} sentences long, in the context of \
"{topic}", that illustrates ALL of the following skills at once. The text should be \
minimal and natural -- a short passage, not a list.

Skills to illustrate (with definitions and examples for reference):
{skill_block}

Constraints:
- Do NOT explicitly name any of the skills in the text.
- Use each skill in its technical sense, not a colloquial one.
- Stay within {sentence_limit} sentences.
- Stay on the topic "{topic}".

Return ONLY the text itself -- no explanation, no preface.
"""


def _format_skill_block(skills: List[Skill]) -> str:
    lines = []
    for i, s in enumerate(skills, 1):
        lines.append(f"Skill {i}: {s.name}")
        lines.append(f"  Definition: {s.description}")
        if s.example:
            lines.append(f"  Example: {s.example}")
    return "\n".join(lines)


def build_generation_prompt(skills: List[Skill], topic: Topic, sentence_limit: int) -> str:
    return GENERATION_PROMPT.format(
        sentence_limit=sentence_limit,
        topic=topic.name,
        skill_block=_format_skill_block(skills),
    )


def run_trial(
    skills: List[Skill],
    topic: Topic,
    sentence_limit: int,
    student,
    judge,
) -> SkillMixTrial:
    """Run one Skill-Mix trial: sample k skills + topic, generate, grade."""
    prompt = build_generation_prompt(skills, topic, sentence_limit)
    gen = student.chat([{"role": "user", "content": prompt}])
    response = gen.text.strip()

    rubric = per_skill_rubric_judge(
        response=response,
        skills=skills,
        topic_name=topic.name,
        sentence_limit=sentence_limit,
        judge_provider=judge,
    )

    model_name = getattr(student, "model_name", "unknown")
    trial_uid = stable_uid(
        f"{model_name}|{topic.topic_uid}|{','.join(s.skill_uid for s in skills)}|{time.time_ns()}"
    )

    return SkillMixTrial(
        trial_uid=trial_uid,
        model=model_name,
        k=len(skills),
        sentence_limit=sentence_limit,
        skill_uids=[s.skill_uid for s in skills],
        topic_uid=topic.topic_uid,
        prompt=prompt,
        response=response,
        rubric_points={
            "per_skill": rubric.get("per_skill", {}),
            "on_topic": rubric.get("on_topic", 0),
            "coherent": rubric.get("coherent", 0),
            "within_sentence_limit": rubric.get("within_sentence_limit", 0),
            "explicit_skill_name_penalty": rubric.get("explicit_skill_name_penalty", 0),
            "colloquial_use_penalty": rubric.get("colloquial_use_penalty", 0),
        },
        score=float(rubric.get("score", 0)),
        full_points=bool(rubric.get("full_points", False)),
        rationale=rubric.get("rationale", ""),
    )


def run_evaluation(
    skills: List[Skill],
    topics: List[Topic],
    k: int,
    trials: int,
    sentence_limit: int,
    student,
    judge,
    seed: int = 42,
    verbose: bool = False,
) -> List[SkillMixTrial]:
    """Run `trials` trials: each samples k unique skills + 1 topic uniformly."""
    if k > len(skills):
        raise ValueError(f"k={k} exceeds available skills ({len(skills)})")
    if not topics:
        raise ValueError("topics list is empty")

    rng = random.Random(seed)
    results: List[SkillMixTrial] = []

    for t in range(trials):
        sampled = rng.sample(skills, k)
        topic = rng.choice(topics)

        if verbose:
            names = ", ".join(s.name for s in sampled)
            print(f"[{t+1}/{trials}] k={k} topic='{topic.name}' skills=({names})")

        trial = run_trial(sampled, topic, sentence_limit, student, judge)
        results.append(trial)

        if verbose:
            max_pts = k + 3
            print(f"    score={trial.score}/{max_pts}"
                  f"  full_points={trial.full_points}  {trial.rationale[:80]}")

    return results


def summarize(trials: List[SkillMixTrial], k: int) -> dict:
    """Paper-style aggregate metrics across trials."""
    if not trials:
        return {"trials": 0}
    max_pts = k + 3
    total = sum(t.score for t in trials)
    full = sum(1 for t in trials if t.full_points)
    skill_fraction = 0.0
    for t in trials:
        pts = t.rubric_points.get("per_skill", {})
        skill_fraction += sum(int(v) for v in pts.values()) / max(k, 1)
    return {
        "trials": len(trials),
        "k": k,
        "max_points_per_trial": max_pts,
        "total_score_avg": round(total / len(trials), 3),
        "ratio_full_marks": round(full / len(trials), 3),
        "skill_fraction_avg": round(skill_fraction / len(trials), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str):
    """'anthropic:claude-opus-4-6' -> (provider_name, model)."""
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--skills", type=Path, default=Path("data/wikipedia-seed/skills.json"))
    parser.add_argument("--topics", type=Path, default=Path("data/topics/topics.json"))
    parser.add_argument("--k", type=int, default=2, help="skills per trial (default: 2)")
    parser.add_argument("--trials", type=int, default=10, help="number of trials (default: 10)")
    parser.add_argument("--sentence-limit", type=int, default=2, help="max sentences (default: 2)")
    parser.add_argument("--student", default="anthropic:claude-sonnet-4-6",
                        help="student provider spec (default: anthropic:claude-sonnet-4-6)")
    parser.add_argument("--judge", default="anthropic:claude-opus-4-7",
                        help="judge provider spec (default: anthropic:claude-opus-4-7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/skill-mix-trials.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    skills = load_skills(args.skills)
    # topics.json uses {"name": ...} entries without UIDs; rebuild via seeder
    from s2_extracting_skills_from_text.m1_wikipedia_seeder import seed_topics_from_file
    topics = seed_topics_from_file(args.topics)

    student_name, student_model = _parse_provider_spec(args.student)
    judge_name, judge_model = _parse_provider_spec(args.judge)
    student = create_provider(student_name, student_model)
    judge = create_provider(judge_name, judge_model)

    trials = run_evaluation(
        skills=skills, topics=topics, k=args.k, trials=args.trials,
        sentence_limit=args.sentence_limit,
        student=student, judge=judge, seed=args.seed, verbose=args.verbose,
    )

    save_json(trials, args.out)
    summary = summarize(trials, args.k)
    print(f"\nWrote {len(trials)} trials -> {args.out}")
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
