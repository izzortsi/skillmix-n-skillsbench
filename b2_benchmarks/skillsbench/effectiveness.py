"""
b2_benchmarks.skillsbench.effectiveness

Pure-Python statistical utilities + effectiveness aggregation over
CorpusEpisode lists. Inlined from skillsuite's c0_utils.stat_utils (pass_rate,
bootstrap_ci, permutation_test) + c2_evaluation.effectiveness, so this module
stands alone without the skillsuite dependency tree.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pure stats
# ---------------------------------------------------------------------------


def mean(values: List[float]) -> float:
    """Arithmetic mean; 0.0 on empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def pass_rate(outcomes: List[bool]) -> float:
    """Fraction of outcomes that are True (in [0, 1]); 0.0 on empty input."""
    if not outcomes:
        return 0.0
    return sum(1.0 for o in outcomes if o) / len(outcomes)


def pass_rate_delta_pp(
    baseline_outcomes: List[bool], treatment_outcomes: List[bool],
) -> float:
    """(treatment_rate - baseline_rate) * 100 — percentage-point delta."""
    return (pass_rate(treatment_outcomes) - pass_rate(baseline_outcomes)) * 100.0


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (point, lo, hi)."""
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        v = values[0]
        return (v, v, v)

    rng = random.Random(seed)
    n = len(values)
    point = mean(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(mean(sample))
    bootstrap_means.sort()

    alpha = 1.0 - confidence
    lo_idx = max(0, min(int(math.floor((alpha / 2.0) * n_bootstrap)), n_bootstrap - 1))
    hi_idx = max(0, min(int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1, n_bootstrap - 1))
    return (point, bootstrap_means[lo_idx], bootstrap_means[hi_idx])


def permutation_test(
    group_a: List[float],
    group_b: List[float],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Two-sided permutation test on (mean(a) - mean(b)). Returns (observed_diff, p_value)."""
    if not group_a or not group_b:
        return (0.0, 1.0)

    rng = random.Random(seed)
    observed_diff = mean(group_a) - mean(group_b)
    abs_observed = abs(observed_diff)

    combined = list(group_a) + list(group_b)
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = abs(mean(perm_a) - mean(perm_b))
        if perm_diff >= abs_observed:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return (observed_diff, p_value)


# ---------------------------------------------------------------------------
# Aggregation over CorpusEpisode lists
#
# The functions below accept anything with .passed (bool), .score (float),
# .condition (str), .skill_name (str), .task_uid (str), .model (str). Both
# the CorpusEpisode dataclass from corpus_harness.py and dict records from
# archived JSONL runs work via attribute-or-key lookup.
# ---------------------------------------------------------------------------


def _get(entry, name: str, default=None):
    if hasattr(entry, name):
        return getattr(entry, name, default)
    if isinstance(entry, dict):
        return entry.get(name, default)
    return default


def compute_pass_rate_delta(
    baseline_entries: List[Any], treatment_entries: List[Any],
) -> Dict[str, Any]:
    """delta_pp + bootstrap CI on treatment rate + permutation p on scores."""
    baseline_outcomes = [bool(_get(r, "passed", False)) for r in baseline_entries]
    treatment_outcomes = [bool(_get(r, "passed", False)) for r in treatment_entries]

    baseline_rate = pass_rate(baseline_outcomes)
    treatment_rate = pass_rate(treatment_outcomes)
    delta_pp = pass_rate_delta_pp(baseline_outcomes, treatment_outcomes)

    baseline_scores = [float(_get(r, "score", 0.0)) for r in baseline_entries]
    treatment_scores = [float(_get(r, "score", 0.0)) for r in treatment_entries]
    _, p_value = permutation_test(treatment_scores, baseline_scores)

    treatment_floats = [1.0 if o else 0.0 for o in treatment_outcomes]
    _, ci_lo, ci_hi = bootstrap_ci(treatment_floats)

    return {
        "delta_pp": round(delta_pp, 2),
        "baseline_rate": round(baseline_rate, 4),
        "treatment_rate": round(treatment_rate, 4),
        "p_value": round(p_value, 4),
        "treatment_ci_lower": round(ci_lo, 4),
        "treatment_ci_upper": round(ci_hi, 4),
        "n_baseline": len(baseline_entries),
        "n_treatment": len(treatment_entries),
    }


def _group_by(entries: List[Any], key_fn) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = defaultdict(list)
    for r in entries:
        groups[key_fn(r)].append(r)
    return dict(groups)


def aggregate_by_skill(entries: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Per-skill effectiveness metrics, matching baseline on shared task_uids."""
    baseline = [r for r in entries if _get(r, "condition") == "baseline"]
    curated = [r for r in entries if _get(r, "condition") == "curated"]
    by_skill = _group_by(curated, lambda r: _get(r, "skill_name", ""))

    out: Dict[str, Dict[str, Any]] = {}
    for skill_name, skill_records in by_skill.items():
        task_uids = {_get(r, "task_uid", "") for r in skill_records}
        matching = [r for r in baseline if _get(r, "task_uid", "") in task_uids]
        if matching:
            out[skill_name] = compute_pass_rate_delta(matching, skill_records)
        else:
            out[skill_name] = {
                "delta_pp": 0.0,
                "baseline_rate": 0.0,
                "treatment_rate": pass_rate([bool(_get(r, "passed", False)) for r in skill_records]),
                "p_value": 1.0,
                "n_baseline": 0,
                "n_treatment": len(skill_records),
            }
    return out


def aggregate_by_model(entries: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Per-model baseline vs curated effectiveness."""
    baseline = [r for r in entries if _get(r, "condition") == "baseline"]
    curated = [r for r in entries if _get(r, "condition") == "curated"]
    by_model_b = _group_by(baseline, lambda r: _get(r, "model", ""))
    by_model_c = _group_by(curated, lambda r: _get(r, "model", ""))

    out: Dict[str, Dict[str, Any]] = {}
    for model in sorted(set(by_model_b) | set(by_model_c)):
        b = by_model_b.get(model, [])
        t = by_model_c.get(model, [])
        if b and t:
            out[model] = compute_pass_rate_delta(b, t)
        else:
            out[model] = {"delta_pp": 0.0, "n_baseline": len(b), "n_treatment": len(t)}
    return out


def compute_overall_summary(entries: List[Any]) -> Dict[str, Any]:
    """Roll-up: total trials + baseline-vs-curated delta + p-value."""
    baseline = [r for r in entries if _get(r, "condition") == "baseline"]
    curated = [r for r in entries if _get(r, "condition") == "curated"]
    summary: Dict[str, Any] = {"total_trials": len(entries)}
    if baseline and curated:
        res = compute_pass_rate_delta(baseline, curated)
        summary["curated_delta_pp"] = res["delta_pp"]
        summary["curated_p_value"] = res["p_value"]
        summary["baseline_rate"] = res["baseline_rate"]
        summary["treatment_rate"] = res["treatment_rate"]
    else:
        summary["curated_delta_pp"] = 0.0
        summary["curated_p_value"] = 1.0
    return summary
