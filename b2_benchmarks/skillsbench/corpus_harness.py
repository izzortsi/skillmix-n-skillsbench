"""
b2_benchmarks.skillsbench.corpus_harness

Run ExtractedTasks through a provider under two conditions:
  - baseline: default system prompt only
  - curated:  default system prompt + a Skill block injected

Two modes:
  - singlecall: one chat() per episode (baseline or curated)
  - guided:     multi-turn, one turn per Skill.procedure step + a synthesis turn

Judge (core.providers-compatible) scores responses via LLMJudgeEvaluator.

Ported-and-trimmed from skillsuite/llm-skills.skillsbench-evaluation/
c3_skillsbench/corpus_harness.py. Key simplifications vs the skillsuite
version:
  - No build_task_skill_map (scaffold Skill has no source_task_uids field).
    Caller passes an optional per-task skill list.
  - Single-provider-per-call; no run_multi_model_evaluation. Callers iterate
    models externally.
  - Reads ChatResult.text directly (no message-content shape juggling).

Usage:
    python -m b2_benchmarks.skillsbench.corpus_harness \\
        --tasks data/holdout-tasks.json --catalog data/wikipedia-seed/skills.json \\
        --out-dir data/skillsbench-out \\
        --provider anthropic:claude-haiku-4-5-20251001 \\
        --judge    anthropic:claude-opus-4-7 \\
        --mode singlecall --cross-task
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schemas import (
    ExtractedTask,
    Skill,
    load_extracted_tasks,
    load_skills,
)
from core.providers import create_provider

from b2_benchmarks.skillsbench.llm_judge import LLMJudgeEvaluator
from b2_benchmarks.skillsbench.skill_injection import (
    format_skill_as_system,
    get_default_system_prompt,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepTrace:
    """One turn of a guided-mode episode."""

    step_index: int
    user_prompt: str
    raw_response: str
    parsed_action: str
    tokens: int = 0
    elapsed_s: float = 0.0
    selected_skill: str = ""


@dataclass
class CorpusEpisode:
    """One evaluation episode: (task, model, condition, skill) -> judge result."""

    task_uid: str
    model: str
    condition: str                          # "baseline" | "curated"
    skill_name: str
    mode: str                               # "singlecall" | "guided"
    response: str
    passed: bool
    score: float
    tokens: int
    elapsed_s: float
    judge_rationale: str = ""
    steps: List[StepTrace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-call mode
# ---------------------------------------------------------------------------


def run_singlecall_episode(
    task: ExtractedTask,
    provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    skill: Optional[Skill] = None,
    verbose: bool = False,
) -> CorpusEpisode:
    """One chat(): baseline if skill is None, else curated with the skill injected."""
    base_system = get_default_system_prompt(task.domain)
    user_prompt = f"Passage: {task.passage}\n\nChallenge: {task.challenge}"

    if skill is not None:
        system_prompt = format_skill_as_system(base_system, skill)
        condition = "curated"
        skill_name = skill.name
    else:
        system_prompt = base_system
        condition = "baseline"
        skill_name = ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    started = time.time()
    tokens = 0
    response_text = ""
    try:
        result = provider.chat(messages)
        tokens = int((result.usage or {}).get("total_tokens", 0) or 0)
        response_text = getattr(result, "text", "") or ""
    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")
    elapsed = time.time() - started

    if verbose:
        _print_answer("answer", response_text)

    judge_result = judge.evaluate(
        response=response_text,
        passage=task.passage,
        challenge=task.challenge,
        acceptance_criteria=task.acceptance_criteria,
        query_type=getattr(task, "query_type", "FREE_FORM"),
    )

    if verbose:
        _print_verdict(judge_result, task.output)

    resolved = model_name or getattr(provider, "model_name", getattr(provider, "model", "unknown"))
    return CorpusEpisode(
        task_uid=task.task_uid,
        model=resolved,
        condition=condition,
        skill_name=skill_name,
        mode="singlecall",
        response=response_text,
        passed=judge_result.passed,
        score=judge_result.score,
        tokens=tokens,
        elapsed_s=round(elapsed, 2),
        judge_rationale=judge_result.rationale,
    )


# ---------------------------------------------------------------------------
# Verbose output helpers
# ---------------------------------------------------------------------------


_ANSWER_PREVIEW_CHARS = 240


def _print_answer(label: str, text: str) -> None:
    """Stream-friendly one-line preview of a model response."""
    if not text:
        print(f"    {label}: (empty)")
        return
    collapsed = " ".join(text.split())
    if len(collapsed) <= _ANSWER_PREVIEW_CHARS:
        print(f"    {label}: {collapsed}")
    else:
        head = collapsed[: _ANSWER_PREVIEW_CHARS].rstrip()
        print(f"    {label}: {head}... [{len(collapsed)} chars total]")


def _print_verdict(judge_result, expected: str) -> None:
    tag = "PASS" if judge_result.passed else "FAIL"
    rationale = " ".join(judge_result.rationale.split())[:180] if judge_result.rationale else ""
    exp = (expected or "").strip()
    exp_preview = exp if len(exp) <= 80 else exp[:77] + "..."
    print(
        f"    judge: [{tag}] score={judge_result.score:.2f}  "
        f"expected={exp_preview!r}"
    )
    if rationale:
        print(f"           {rationale}")


# ---------------------------------------------------------------------------
# Guided mode (one turn per skill.procedure step + synthesis)
# ---------------------------------------------------------------------------


_GUIDED_SYSTEM = (
    "You are an expert analyst. You will solve the challenge by following a "
    "procedure one step at a time.\n\n"
    "Each turn you receive a specific instruction. Carry it out using evidence "
    "from the passage and write your finding in 2-4 sentences. Reference "
    "specific words, phrases, or claims from the passage.\n\n"
    "On the final turn you will be asked to synthesize your findings into a "
    "conclusion. Write a concise paragraph that directly answers the challenge."
)


def _format_guided_user(
    passage: str, challenge: str, procedure_step: str,
    step_index: int, total_steps: int, findings_so_far: List[str],
) -> str:
    parts = [f"Passage: {passage}", f"\nChallenge: {challenge}"]
    if findings_so_far:
        parts.append("\nFindings so far:")
        for i, finding in enumerate(findings_so_far, 1):
            parts.append(f"  {i}. {finding}")
    parts.append(f"\nStep {step_index} of {total_steps}: {procedure_step}")
    return "\n".join(parts)


def run_guided_episode(
    task: ExtractedTask,
    provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    skill: Optional[Skill] = None,
    verbose: bool = False,
) -> CorpusEpisode:
    """One chat() per skill.procedure step, plus a synthesis turn; falls back to
    run_singlecall_episode when the skill has no procedure (declarative skill)."""
    if skill is None or not skill.procedure:
        return run_singlecall_episode(task, provider, judge, model_name=model_name,
                                       skill=skill, verbose=verbose)

    started = time.time()
    procedure = list(skill.procedure)
    procedure.append(
        "Synthesize your findings into a single paragraph that directly "
        "answers the challenge."
    )
    total_steps = len(procedure)
    system_prompt = format_skill_as_system(_GUIDED_SYSTEM, skill)

    findings: List[str] = []
    step_traces: List[StepTrace] = []
    total_tokens = 0

    for step_idx, step_instruction in enumerate(procedure, 1):
        step_started = time.time()
        user_msg = _format_guided_user(
            task.passage, task.challenge, step_instruction,
            step_idx, total_steps, findings,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        step_tokens = 0
        response_text = ""
        try:
            result = provider.chat(messages)
            step_tokens = int((result.usage or {}).get("total_tokens", 0) or 0)
            total_tokens += step_tokens
            response_text = getattr(result, "text", "") or ""
        except Exception as e:
            if verbose:
                print(f"      step {step_idx} ERROR: {e}")
            break

        step_elapsed = time.time() - step_started
        finding = response_text.strip()

        if verbose:
            print(f"      step {step_idx}/{total_steps}: {finding[:100]}")

        step_traces.append(StepTrace(
            step_index=step_idx,
            user_prompt=user_msg,
            raw_response=response_text,
            parsed_action=finding,
            tokens=step_tokens,
            elapsed_s=round(step_elapsed, 2),
            selected_skill=skill.name,
        ))
        findings.append(finding)

    elapsed = time.time() - started
    full_response = "\n".join(f"{i}. {s}" for i, s in enumerate(findings, 1))

    if verbose:
        _print_answer("synthesized", full_response)

    judge_result = judge.evaluate(
        response=full_response,
        passage=task.passage,
        challenge=task.challenge,
        acceptance_criteria=task.acceptance_criteria,
        query_type=getattr(task, "query_type", "FREE_FORM"),
    )

    if verbose:
        _print_verdict(judge_result, task.output)

    resolved = model_name or getattr(provider, "model_name", getattr(provider, "model", "unknown"))
    return CorpusEpisode(
        task_uid=task.task_uid,
        model=resolved,
        condition="curated",
        skill_name=skill.name,
        mode="guided",
        response=full_response,
        passed=judge_result.passed,
        score=judge_result.score,
        tokens=total_tokens,
        elapsed_s=round(elapsed, 2),
        judge_rationale=judge_result.rationale,
        steps=step_traces,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_corpus_evaluation(
    tasks: List[ExtractedTask],
    skills: List[Skill],
    provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    mode: str = "singlecall",
    cross_task: bool = False,
    task_skill_map: Optional[Dict[str, Skill]] = None,
    verbose: bool = False,
) -> List[CorpusEpisode]:
    """For each task: always run a baseline episode, then run zero or more curated episodes.

    cross_task=True       -> baseline + one episode per skill in `skills`.
    task_skill_map given  -> baseline + one episode with task_skill_map[task.task_uid] (if present).
    both None/false       -> baseline only for every task.
    """
    episode_fn = run_guided_episode if mode == "guided" else run_singlecall_episode
    episodes: List[CorpusEpisode] = []

    total = len(tasks) * (1 + (len(skills) if cross_task else 1))
    count = 0

    for task in tasks:
        count += 1
        if verbose:
            print(f"[{count}/{total}] {task.task_uid} baseline")
        episodes.append(
            episode_fn(task, provider, judge, model_name=model_name, verbose=verbose)
        )

        if cross_task:
            for skill in skills:
                count += 1
                if verbose:
                    print(f"[{count}/{total}] {task.task_uid} + {skill.name}")
                episodes.append(
                    episode_fn(task, provider, judge, model_name=model_name,
                               skill=skill, verbose=verbose)
                )
        elif task_skill_map is not None:
            matching = task_skill_map.get(task.task_uid)
            if matching is not None:
                count += 1
                if verbose:
                    print(f"[{count}/{total}] {task.task_uid} + {matching.name}")
                episodes.append(
                    episode_fn(task, provider, judge, model_name=model_name,
                               skill=matching, verbose=verbose)
                )

    return episodes


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_episodes(episodes: List[CorpusEpisode], output_path: Path) -> None:
    """Write episodes to JSON, keeping step_traces compactly (no raw_response)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for ep in episodes:
        entry = asdict(ep)
        if ep.steps:
            entry["steps"] = [
                {
                    "step_index": s.step_index,
                    "parsed_action": s.parsed_action,
                    "selected_skill": s.selected_skill,
                    "tokens": s.tokens,
                    "elapsed_s": s.elapsed_s,
                }
                for s in ep.steps
            ]
        else:
            entry.pop("steps", None)
        data.append(entry)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def _load_task_skill_map(path: Path, catalog: List[Skill]) -> Dict[str, Skill]:
    """Load {task_uid: skill_name} JSON and resolve to {task_uid: Skill}.

    Unknown skill names are dropped with a warning; unknown task_uids are
    harmless (the harness only injects when task_skill_map.get(task.task_uid)
    returns a Skill).
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object {{task_uid: skill_name}}")

    by_name = {s.name: s for s in catalog}
    resolved: Dict[str, Skill] = {}
    missing: List[str] = []
    for task_uid, skill_name in raw.items():
        skill = by_name.get(str(skill_name))
        if skill is None:
            missing.append(f"{task_uid} -> {skill_name!r}")
            continue
        resolved[str(task_uid)] = skill
    if missing:
        print(
            f"warning: {len(missing)} map entries skipped (skill_name not in catalog): "
            + "; ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--tasks", type=Path, required=True,
                        help="ExtractedTask JSON")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="Skill catalog JSON (optional at runtime; always loaded so "
                             "cross_task/task_skill_map work)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory for episodes.json + summary.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate the first N tasks (0 = all)")
    parser.add_argument("--provider", default="",
                        help="Single student provider spec (e.g. 'anthropic:claude-haiku-4-5-20251001'). "
                             "Superseded by --models when both are given.")
    parser.add_argument("--models", default="",
                        help="Comma-separated list of 'provider:model' specs. "
                             "Runs the full corpus against each student and merges episodes. "
                             "Episodes are tagged with .model so downstream visualization "
                             "automatically breaks down per student.")
    parser.add_argument("--judge", default="anthropic:claude-opus-4-7",
                        help="Judge provider spec")
    parser.add_argument("--mode", choices=["singlecall", "guided"], default="singlecall")
    parser.add_argument("--cross-task", action="store_true",
                        help="Inject every skill in --catalog into every task (N*(1+M) episodes)")
    parser.add_argument("--task-skill-map", type=Path, default=None,
                        help="JSON {task_uid: skill_name} — each listed task gets a "
                             "baseline episode AND one curated episode with the mapped "
                             "skill injected. Ignored when --cross-task is set.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve the list of student specs. --models wins; --provider is a 1-element shortcut.
    if args.models.strip():
        model_specs = [s.strip() for s in args.models.split(",") if s.strip()]
    elif args.provider.strip():
        model_specs = [args.provider.strip()]
    else:
        parser.error("provide --provider <spec> or --models <spec1,spec2,...>")

    tasks = load_extracted_tasks(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    skills = load_skills(args.catalog) if args.catalog else []

    # Resolve the task -> Skill mapping when requested.
    task_skill_map: Optional[Dict[str, Skill]] = None
    if args.task_skill_map is not None and not args.cross_task:
        task_skill_map = _load_task_skill_map(args.task_skill_map, skills)

    j_name, j_model = _parse_provider_spec(args.judge)
    judge_provider = create_provider(j_name, j_model)
    judge = LLMJudgeEvaluator(judge_provider)

    print(f"Loaded {len(tasks)} tasks, {len(skills)} skills")
    mode_tag = "cross_task" if args.cross_task else (
        f"task_skill_map ({len(task_skill_map)} mapped)"
        if task_skill_map else "baseline-only"
    )
    print(
        f"mode={args.mode}  skill-injection={mode_tag}  judge={args.judge}  "
        f"students={len(model_specs)}"
    )

    all_episodes: List[CorpusEpisode] = []
    for idx, spec in enumerate(model_specs, 1):
        p_name, p_model = _parse_provider_spec(spec)
        provider = create_provider(p_name, p_model)
        print(f"\n=== [{idx}/{len(model_specs)}] student: {spec} ===")
        episodes = run_corpus_evaluation(
            tasks=tasks, skills=skills, provider=provider, judge=judge,
            model_name=p_model, mode=args.mode, cross_task=args.cross_task,
            task_skill_map=task_skill_map,
            verbose=args.verbose,
        )
        all_episodes.extend(episodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_episodes(all_episodes, args.out_dir / "episodes.json")

    # lightweight summary (import effectiveness here to avoid top-level import
    # cycle if someone extends the module)
    from b2_benchmarks.skillsbench.effectiveness import compute_overall_summary
    summary = compute_overall_summary(all_episodes)
    summary["mode"] = args.mode
    summary["cross_task"] = args.cross_task
    summary["n_tasks"] = len(tasks)
    summary["n_skills"] = len(skills)
    summary["students"] = model_specs
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(all_episodes)} episodes across {len(model_specs)} student(s) -> {args.out_dir}/episodes.json")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
