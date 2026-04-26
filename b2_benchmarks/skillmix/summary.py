"""
b2_benchmarks.skillmix.summary

Aggregate an episodes.json produced by a composition experiment into the
summary.json shape `b2_benchmarks.skillmix.visualizer.generate_baseline_vs_skill_bar`
consumes: {model: {baseline_mean_score, skill_mean_score, delta, n_baseline, n_skill}}.

Ported from skillsuite/llm-skills.skillmix-evaluation/c2_analytics/summary.py
(unchanged except for formatting / type hints).
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_summary(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group episodes by model + condition, compute mean scores and delta.

    Episodes are expected to use condition == "baseline" or "skill_injected"
    and carry a numeric `score` field. Episodes without `score` are skipped.
    """
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for ep in episodes:
        model = ep.get("model", "unknown")
        condition = ep.get("condition", "")
        if "score" not in ep:
            continue
        by_model.setdefault(model, {"baseline": [], "skill_injected": []})
        by_model[model].setdefault(condition, []).append(float(ep["score"]))

    summary: Dict[str, Dict[str, Any]] = {}
    for model, conditions in by_model.items():
        b = conditions.get("baseline", [])
        s = conditions.get("skill_injected", [])
        baseline_mean = sum(b) / len(b) if b else 0.0
        skill_mean = sum(s) / len(s) if s else 0.0
        summary[model] = {
            "baseline_mean_score": round(baseline_mean, 4),
            "skill_mean_score": round(skill_mean, 4),
            "delta": round(skill_mean - baseline_mean, 4),
            "n_baseline": len(b),
            "n_skill": len(s),
        }
    return summary
