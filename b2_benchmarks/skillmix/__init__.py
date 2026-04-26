"""
b2_benchmarks.skillmix

SkillMix-style composition evaluation: measure whether injecting a k-way
composed skill (seq / par / cond / atomic) into an LLM's system prompt beats
the atomic baseline. Complementary to `b2_benchmarks.skillsbench`, which
measures atomic-skill injection against baseline. Naming is kept consistent
with the legacy skillmix-evaluation package (note: unrelated to Yu et al.
Skill-Mix; true Yu Skill-Mix lives at PROJECT_SPECS s2.m2).

Currently this package contains only the visualization surface. Callers
produce episodes.json + summary.json externally (harness not yet ported).

Public surface:
    compute_summary              — episodes.json -> summary.json shape
    generate_all                 — all six charts
    generate_score_by_k          — line: mean score vs k, one line per model
    generate_operator_heatmap    — heatmap: operator type x model
    generate_uplift_heatmap      — heatmap: skill x model, delta from baseline
    generate_k_operator_heatmap  — heatmap: (k, operator) x model
    generate_baseline_vs_skill_bar — grouped bar: baseline vs skill-injected mean
    generate_win_loss_bar        — stacked bar: wins / ties / losses per model
    parse_skill_name             — (operator, k) from composed skill name
"""

from b2_benchmarks.skillmix.summary import compute_summary
from b2_benchmarks.skillmix.visualizer import (
    generate_all,
    generate_baseline_vs_skill_bar,
    generate_k_operator_heatmap,
    generate_operator_heatmap,
    generate_score_by_k,
    generate_uplift_heatmap,
    generate_win_loss_bar,
    parse_skill_name,
)

__all__ = [
    "compute_summary",
    "generate_all",
    "generate_baseline_vs_skill_bar",
    "generate_k_operator_heatmap",
    "generate_operator_heatmap",
    "generate_score_by_k",
    "generate_uplift_heatmap",
    "generate_win_loss_bar",
    "parse_skill_name",
]
