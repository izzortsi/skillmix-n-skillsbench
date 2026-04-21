"""
frontier_skillmix.py

Frontier-native SkillMix runner. Parallels c3_skillmix/runner.py but
uses the harness/LinearAgent directly (no lmproxy, no OpenAI adapter)
and consumes the outputs of c2_extraction/frontier_* modules.

For each task: runs baseline (no skill prompt) and skill-injected
(skill description/example in system prompt) against a subject model,
then scores each response with a judge model against the task's
acceptance_criteria. Attaches a novelty score per episode via
c2_analytics.novelty.

Output (compatible with existing c2_analytics/summary.py and
c4_cli/visualize.py):
    episodes.json       - per-episode {task_uid, skill_name, model,
                          condition, response, score, novelty, ...}
    summary.json        - per-model baseline/skill means + delta

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.skillmix-evaluation
    python -m c3_skillmix.frontier_skillmix \\
        --tasks ../llm-skills.extraction-pipeline/data/pipeline-runs/frontier-wikipedia/stage1-extraction/tasks.json \\
        --skills /workspace/llm-skills/skillsuite/llm-skills.shared-data/260420.skillmix-wikipedia-seed/skills.json \\
        --output-dir results/frontier-wiki-smoke \\
        --subject-model claude-haiku-4-5-20251001 \\
        --judge-model  claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, OllamaAgent, Transcript, record  # noqa: E402
from harness.ollama_agent import MODEL_PREFIX as OLLAMA_MODEL_PREFIX  # noqa: E402

from c2_analytics.summary import compute_summary  # noqa: E402
from c2_analytics.novelty import classify_tier, estimate_novelty  # noqa: E402


DEFAULT_SUBJECT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_THINKING_BUDGET = 1024
DEFAULT_MAX_TOKENS = 2000


SUBJECT_SYSTEM_BASELINE = (
    "You are answering an evaluation question. Think about the passage, "
    "then produce one concise conclusion. Prefix your final answer with "
    "'ANSWER: ' on its own line."
)


SUBJECT_SYSTEM_SKILL_TEMPLATE = """You are answering an evaluation question. You must demonstrate the skill(s) listed below.

{skill_block}

Think about the passage, then produce one concise conclusion that
exercises every listed skill. Prefix your final answer with
'ANSWER: ' on its own line.
"""


JUDGE_SYSTEM = (
    "You are a strict grader for an LLM evaluation benchmark. You decide "
    "whether a candidate answer matches the task's correct conclusion and "
    "exercises every required skill the task lists. You return a JSON "
    "verdict only."
)


JUDGE_PROMPT = """Task title: {title}

Passage/context:
{input}

Question:
{question}

Expected correct conclusion:
{correct_conclusion}

Required skills: {skills}

Candidate answer:
{answer}

Grade the candidate. Award 1 point for correct conclusion (semantically
matches expected), and 1 point per skill genuinely exercised in the
answer. The maximum total is 1 + len(required_skills). Report the raw
score and the normalized score in [0, 1].

Return ONLY valid JSON:

{{
  "correct": true|false,
  "skills_exercised": ["<name>", ...],
  "raw_score": <int>,
  "max_score": <int>,
  "score": <float 0..1>,
  "rationale": "<one sentence>"
}}
"""


def _format_skill_block(skill: Dict[str, Any]) -> str:
    return (
        f"Skill: {skill['name']}  ({skill.get('category', 'unspecified')})\n"
        f"  Definition: {skill.get('description', '')}\n"
        f"  Example:    {skill.get('example') or '(none)'}"
    )


def _final_text(transcript: Transcript) -> str:
    return "".join(b.text for turn in transcript.turns for b in turn.text)


def _extract_answer(text: str) -> str:
    marker = "ANSWER:"
    idx = text.rfind(marker)
    if idx == -1:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""
    tail = text[idx + len(marker):].strip()
    return tail.splitlines()[0].strip() if tail else ""


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text)


def _parse_json(text: str):
    raw = _strip_fences(text).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("{"); e = raw.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        try:
            return json.loads(raw[s:e+1])
        except json.JSONDecodeError:
            return None


def _build_agent(model: str, system: str, thinking_budget: int, max_tokens: int):
    if model.startswith(OLLAMA_MODEL_PREFIX):
        return OllamaAgent(
            model=model, system=system,
            max_tokens=max_tokens, max_turns=1,
        )
    return LinearAgent(
        model=model, system=system, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )


def _build_judge_agent(model: str, thinking_budget: int, max_tokens: int):
    return _build_agent(model, JUDGE_SYSTEM, thinking_budget, max_tokens)


def _build_subject_agent(model: str, system: str, thinking_budget: int, max_tokens: int):
    return _build_agent(model, system, thinking_budget, max_tokens)


def _run_episode(
    task: Dict[str, Any],
    skills_for_task: List[Dict[str, Any]],
    condition: str,
    subject_model: str,
    judge_agent: LinearAgent,
    thinking_budget: int,
    max_tokens: int,
) -> Dict[str, Any]:
    ac = task.get("acceptance_criteria", {}) or {}
    correct_conclusion = ac.get("correct_conclusion", task.get("output", ""))
    skill_names = [s["name"] for s in skills_for_task]

    if condition == "skill_injected" and skills_for_task:
        block = "\n\n".join(_format_skill_block(s) for s in skills_for_task)
        system_prompt = SUBJECT_SYSTEM_SKILL_TEMPLATE.format(skill_block=block)
    else:
        system_prompt = SUBJECT_SYSTEM_BASELINE

    subject_agent = _build_subject_agent(
        subject_model, system_prompt, thinking_budget, max_tokens
    )

    user_prompt = (
        f"Task title: {task.get('title', '')}\n\n"
        f"Passage/context:\n{task.get('input', task.get('passage', ''))}\n\n"
        f"Question:\n{task.get('question', task.get('challenge', ''))}"
    )

    start = time.time()
    subject_transcript = record(subject_agent.run(user_prompt))
    subject_text = _final_text(subject_transcript)
    answer = _extract_answer(subject_text) or subject_text.strip().splitlines()[-1] if subject_text.strip() else ""
    elapsed_subject = time.time() - start

    judge_prompt = JUDGE_PROMPT.format(
        title=task.get("title", ""),
        input=task.get("input", task.get("passage", "")),
        question=task.get("question", task.get("challenge", "")),
        correct_conclusion=correct_conclusion,
        skills=", ".join(skill_names) if skill_names else "(none)",
        answer=answer,
    )
    judge_transcript = record(judge_agent.run(judge_prompt))
    judge_text = _final_text(judge_transcript)
    verdict = _parse_json(judge_text) or {}
    score = float(verdict.get("score", 0.0) or 0.0)

    ac = task.get("acceptance_criteria", {}) or {}
    k_from_ac = ac.get("k")
    k_value = int(k_from_ac) if isinstance(k_from_ac, int) else len(skill_names)

    return {
        "task_uid": task.get("task_uid", ""),
        "task_title": task.get("title", ""),
        "skill_name": ",".join(skill_names) if skill_names else "(baseline)",
        "skill_uids": [s["skill_uid"] for s in skills_for_task],
        "skill_names": skill_names,
        "k": k_value,
        "operator": "atomic" if k_value == 1 else "par",
        "model": subject_model,
        "condition": condition,
        "answer": answer,
        "response": subject_text,
        "score": score,
        "judge_verdict": verdict,
        "elapsed_s": round(elapsed_subject, 3),
        "subject_usage": sum(
            (t.usage.get("output_tokens", 0) or 0) for t in subject_transcript.turns
        ),
        "judge_usage": sum(
            (t.usage.get("output_tokens", 0) or 0) for t in judge_transcript.turns
        ),
    }


def _attach_novelty(
    episode: Dict[str, Any],
    skill_by_uid: Dict[str, Dict[str, Any]],
    corpus_examples: float,
    tokens_per_example: float,
) -> None:
    uids = episode.get("skill_uids") or []
    if not uids:
        episode["novelty"] = None
        return
    tiers = [
        classify_tier(skill_by_uid[u]) if u in skill_by_uid else "unknown"
        for u in uids
    ]
    estimate = estimate_novelty(
        tiers, corpus_examples=corpus_examples, tokens_per_example=tokens_per_example
    )
    episode["novelty"] = estimate


def run_frontier_skillmix(
    tasks_path,
    skills_path: Path,
    output_dir: Path,
    subject_models: List[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: int = 0,
    corpus_examples: float = 1e10,
    tokens_per_example: float = 1000.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    tasks_paths = [tasks_path] if isinstance(tasks_path, Path) else list(tasks_path)
    tasks: List[Dict[str, Any]] = []
    for p in tasks_paths:
        tasks.extend(json.loads(Path(p).read_text(encoding="utf-8")))
    skills = json.loads(skills_path.read_text(encoding="utf-8"))
    skill_by_uid = {s["skill_uid"]: s for s in skills}

    if limit > 0:
        tasks = tasks[:limit]

    if not subject_models:
        subject_models = [DEFAULT_SUBJECT_MODEL]

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.json"
    summary_path = output_dir / "summary.json"

    judge_agent = _build_judge_agent(judge_model, thinking_budget, max_tokens)

    episodes: List[Dict[str, Any]] = []
    for subject_model in subject_models:
        if verbose:
            print(f"\n=== subject model: {subject_model} ===")
        for i, task in enumerate(tasks, 1):
            ac = task.get("acceptance_criteria", {}) or {}
            uids = ac.get("skill_uids") or (
                [ac.get("skill_uid")] if ac.get("skill_uid") else []
            )
            uids = [u for u in uids if u]
            skills_for_task = [skill_by_uid[u] for u in uids if u in skill_by_uid]

            if verbose:
                names = [s["name"] for s in skills_for_task]
                print(f"[{i}/{len(tasks)}] {task.get('title', '')!r:60s} skills={names}")

            for condition in ("baseline", "skill_injected"):
                try:
                    ep = _run_episode(
                        task, skills_for_task, condition,
                        subject_model, judge_agent,
                        thinking_budget, max_tokens,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  {condition} ERROR: {e}")
                    continue

                _attach_novelty(ep, skill_by_uid, corpus_examples, tokens_per_example)
                episodes.append(ep)
                if verbose:
                    nov = ep.get("novelty") or {}
                    print(f"  {condition:14s} score={ep['score']:.2f}  "
                          f"novelty_log10={nov.get('log10_expected_cooccurrences', '?')}  "
                          f"likely_novel={nov.get('likely_novel', '?')}")

    episodes_path.write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    summary = compute_summary(episodes)
    skill_eps = [e for e in episodes if e["condition"] == "skill_injected" and e.get("novelty")]
    novelty_block = {
        "tasks_with_novelty_score": len(skill_eps),
        "likely_novel_tasks": sum(1 for e in skill_eps if e["novelty"]["likely_novel"]),
        "novelty_rate": (
            sum(1 for e in skill_eps if e["novelty"]["likely_novel"]) / len(skill_eps)
            if skill_eps else 0.0
        ),
        "median_log10_expected_cooccurrences": _median(
            [e["novelty"]["log10_expected_cooccurrences"] for e in skill_eps]
        ),
    }
    final_summary = {
        "per_model": summary,
        "novelty": novelty_block,
        "subject_models": subject_models,
        "judge_model": judge_model,
        "tasks_evaluated": len(tasks),
        "episodes": len(episodes),
    }
    summary_path.write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    if verbose:
        print(f"\nWrote {episodes_path}")
        print(f"Wrote {summary_path}")
        print(f"Novelty: {novelty_block['likely_novel_tasks']}/"
              f"{novelty_block['tasks_with_novelty_score']} likely novel "
              f"({novelty_block['novelty_rate']:.1%})")

    return final_summary


def _median(values: List[float]) -> Optional[float]:
    vs = sorted(v for v in values if v is not None and v != float("-inf"))
    if not vs:
        return None
    n = len(vs)
    return vs[n // 2] if n % 2 == 1 else 0.5 * (vs[n // 2 - 1] + vs[n // 2])


def main() -> int:
    ap = argparse.ArgumentParser(description="Frontier-native SkillMix runner (PROJECT_SPECS 2.M2).")
    ap.add_argument("--tasks", required=True, type=Path, nargs="+",
                    help="One or more tasks.json files (concatenated).")
    ap.add_argument("--skills", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--subject-model", default=DEFAULT_SUBJECT_MODEL,
                    help="Single subject model (deprecated, use --subject-models).")
    ap.add_argument("--subject-models", default="",
                    help="Comma-separated list of subject models. Overrides --subject-model if set.")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corpus-examples", type=float, default=1e10)
    ap.add_argument("--tokens-per-example", type=float, default=1000.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    subject_models = [m.strip() for m in args.subject_models.split(",") if m.strip()]
    if not subject_models:
        subject_models = [args.subject_model]

    run_frontier_skillmix(
        tasks_path=args.tasks, skills_path=args.skills,
        output_dir=args.output_dir,
        subject_models=subject_models,
        judge_model=args.judge_model,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        limit=args.limit,
        corpus_examples=args.corpus_examples,
        tokens_per_example=args.tokens_per_example,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
