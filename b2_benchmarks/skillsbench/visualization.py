"""
b2_benchmarks.skillsbench.visualization

Heatmaps and summary charts for SkillsBench corpus evaluation results.
Consumes episodes.json as produced by `b2_benchmarks.skillsbench.corpus_harness`
(list of CorpusEpisode-shaped dicts with task_uid, model, condition in
{"baseline", "curated"}, passed, score, skill_name, mode).

Charts produced by `generate_all`:

    uplift_heatmap.png            task x model, mean(curated) - mean(baseline)
    baseline_pass_rate.png        task x model, baseline pass rate
    combined_heatmap.png          baseline_pass_rate (left) + uplift (right)
    win_loss_bar.png              wins / ties / losses per model x mode
    delta_by_mode_bar.png         grouped bar: delta per mode per model
    baseline_vs_curated.png       scatter: baseline score vs curated score per episode

Legacy schemas tolerated on load:
  - "task_id" / "problem_id" treated as aliases of "task_uid"
  - condition "skill_injected" treated as "curated" (skillmix-evaluation style)

Ported-and-trimmed from skillsuite/llm-skills.skillsbench-evaluation/
c3_skillsbench/visualization.py. Dropped: token_vs_delta_scatter (needs a
`tokens` column our harness does not always emit) and skill_radar (heavy,
not wired into our output schema).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PALETTE = (
    "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
    "#64B5CD", "#DD8452", "#A1C9F4",
)


# ---------------------------------------------------------------------------
# Loaders + episode normalization
# ---------------------------------------------------------------------------


def _normalize(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied episode with canonical keys.

    - Maps task_id / problem_id -> task_uid
    - Maps condition=skill_injected -> curated
    """
    out = dict(ep)
    if "task_uid" not in out:
        out["task_uid"] = out.get("task_id") or out.get("problem_id") or ""
    if out.get("condition") == "skill_injected":
        out["condition"] = "curated"
    return out


def load_episodes(
    results_file: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    mode_filter: str = "",
) -> List[Dict[str, Any]]:
    """Load episodes from either a corpus JSON file or a directory containing episodes.json.

    Args:
        results_file: Path to a JSON file whose top level is a list of episodes.
        results_dir:  Path to a directory containing `episodes.json`.
        mode_filter:  If non-empty, keep only episodes whose `mode` matches (case-sensitive).

    Returns:
        List of normalized episode dicts.
    """
    episodes: List[Dict[str, Any]] = []
    if results_file is not None:
        with open(results_file, "r", encoding="utf-8") as f:
            episodes.extend(json.load(f))
    if results_dir is not None:
        p = results_dir / "episodes.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                episodes.extend(json.load(f))

    episodes = [_normalize(ep) for ep in episodes]
    if mode_filter:
        episodes = [ep for ep in episodes if ep.get("mode", "singlecall") == mode_filter]
    return episodes


def _shorten_task_uid(task_uid: str) -> str:
    """Shorten task UIDs for axis labels."""
    if task_uid.startswith(("ext-", "ipc-")):
        task_uid = task_uid[4:]
    if len(task_uid) > 25:
        return task_uid[:22] + "..."
    return task_uid


def _score(ep: Dict[str, Any]) -> float:
    """Prefer .score; fall back to .passed as {0.0, 1.0}."""
    if "score" in ep:
        return float(ep["score"])
    return 1.0 if ep.get("passed") else 0.0


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def build_task_model_uplift_matrix(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """{task_uid: {model: mean(curated) - mean(baseline)}}, for pairs with both."""
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        grouped[(ep["task_uid"], ep.get("model", ""))][ep.get("condition", "")].append(_score(ep))

    matrix: Dict[str, Dict[str, float]] = {}
    for (task_uid, model), conditions in grouped.items():
        b = conditions.get("baseline", [])
        c = conditions.get("curated", [])
        if not b or not c:
            continue
        delta = sum(c) / len(c) - sum(b) / len(b)
        matrix.setdefault(task_uid, {})[model] = round(delta, 4)
    return matrix


def _build_pass_rate_matrix(
    episodes: List[Dict[str, Any]], condition: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (data, tasks, models) for the pass-rate heatmap."""
    return _build_matrix_by_field(episodes, condition, field="passed", bool_to_float=True)


def _build_score_matrix(
    episodes: List[Dict[str, Any]], condition: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (data, tasks, models) for the continuous-score heatmap."""
    return _build_matrix_by_field(episodes, condition, field="score", bool_to_float=False)


def _build_matrix_by_field(
    episodes: List[Dict[str, Any]],
    condition: str,
    field: str,
    bool_to_float: bool,
) -> Tuple[np.ndarray, List[str], List[str]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for ep in episodes:
        if ep.get("condition") != condition:
            continue
        raw = ep.get(field, 0)
        if bool_to_float:
            val = 1.0 if raw else 0.0
        else:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = 0.0
        grouped[(ep["task_uid"], ep.get("model", ""))].append(val)
    tasks = sorted({k[0] for k in grouped})
    models = sorted({k[1] for k in grouped})
    data = np.zeros((len(tasks), len(models)))
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            vals = grouped.get((task, model), [])
            data[i, j] = sum(vals) / len(vals) if vals else 0.0
    return data, tasks, models


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def generate_uplift_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Skill Uplift per Task (curated − baseline)",
    dpi: int = 150,
) -> Optional[Path]:
    matrix = build_task_model_uplift_matrix(episodes)
    if not matrix:
        return None

    all_models: set = set()
    for deltas in matrix.values():
        all_models.update(deltas.keys())

    # rank models by mean uplift (descending) for readability
    model_avg = {
        m: (sum(matrix[t][m] for t in matrix if m in matrix[t]) /
            max(1, sum(1 for t in matrix if m in matrix[t])))
        for m in all_models
    }
    models = sorted(all_models, key=lambda m: model_avg[m], reverse=True)
    tasks = sorted(matrix.keys())

    data = np.zeros((len(tasks), len(models)))
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            data[i, j] = matrix.get(t, {}).get(m, 0.0)

    vmax = max(abs(data.min()), abs(data.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figsize = (max(6, len(models) * 1.5 + 3), max(4, len(tasks) * 0.4 + 2))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=plt.cm.RdBu, norm=norm, interpolation="nearest")

    for i in range(len(tasks)):
        for j in range(len(models)):
            val = data[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([_shorten_task_uid(t) for t in tasks], fontsize=8)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Task", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score Delta", fontsize=10)

    return _save(fig, output_path, dpi)


def generate_pass_rate_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    condition: str = "baseline",
    title: str = "",
    dpi: int = 150,
) -> Optional[Path]:
    data, tasks, models = _build_pass_rate_matrix(episodes, condition)
    if not tasks or not models:
        return None

    figsize = (max(6, len(models) * 1.5 + 3), max(4, len(tasks) * 0.4 + 2))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                   interpolation="nearest")

    for i in range(len(tasks)):
        for j in range(len(models)):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([_shorten_task_uid(t) for t in tasks], fontsize=8)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Task", fontsize=11)
    ax.set_title(title or f"{condition.title()} Pass Rate per Task x Model",
                 fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pass Rate", fontsize=10)

    return _save(fig, output_path, dpi)


def generate_score_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    condition: str = "baseline",
    title: str = "",
    dpi: int = 150,
) -> Optional[Path]:
    """Heatmap of the continuous judge score (rubric credit). Surfaces the
    difference between 'correct but with weak rationale' and 'completely wrong'
    that the binary pass_rate heatmap flattens to 0/1."""
    data, tasks, models = _build_score_matrix(episodes, condition)
    if not tasks or not models:
        return None

    figsize = (max(6, len(models) * 1.5 + 3), max(4, len(tasks) * 0.4 + 2))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                   interpolation="nearest")

    for i in range(len(tasks)):
        for j in range(len(models)):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([_shorten_task_uid(t) for t in tasks], fontsize=8)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Task", fontsize=11)
    ax.set_title(title or f"{condition.title()} Rubric Score per Task x Model",
                 fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Score (0..1)", fontsize=10)

    return _save(fig, output_path, dpi)


def generate_combined_score_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Side-by-side score view: baseline mean score (left) + score uplift (right).

    Parallel to `generate_combined_heatmap` but continuous throughout: the left
    pane uses the judge's rubric score (0..1) per (task, model); the right
    pane uses `build_task_model_uplift_matrix`, which is already score-based
    (mean(curated.score) − mean(baseline.score)). Together they show both
    absolute rationale quality and the skill-injection delta without the
    binary pass_rate collapse.
    """
    sc_data, tasks, models = _build_score_matrix(episodes, "baseline")
    if not tasks or not models:
        return None

    uplift = build_task_model_uplift_matrix(episodes)
    up_data = np.zeros_like(sc_data)
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            up_data[i, j] = uplift.get(t, {}).get(m, 0.0)

    n_tasks, n_models = len(tasks), len(models)
    pane_w = max(5, n_models * 1.2 + 2)
    pane_h = max(4, n_tasks * 0.45 + 1.5)
    fig, (ax_sc, ax_up) = plt.subplots(
        1, 2, figsize=(pane_w * 2 + 1.5, pane_h), sharey=True,
    )
    task_labels = [_shorten_task_uid(t) for t in tasks]

    # left pane: baseline rubric score (continuous 0..1)
    im_sc = ax_sc.imshow(sc_data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                         interpolation="nearest")
    for i in range(n_tasks):
        for j in range(n_models):
            val = sc_data[i, j]
            ax_sc.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=8, color=("white" if val > 0.6 else "black"))
    ax_sc.set_xticks(range(n_models))
    ax_sc.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax_sc.set_yticks(range(n_tasks))
    ax_sc.set_yticklabels(task_labels, fontsize=8)
    ax_sc.set_xlabel("Model", fontsize=11)
    ax_sc.set_ylabel("Task", fontsize=11)
    ax_sc.set_title("Baseline Rubric Score", fontsize=12, pad=10)
    fig.colorbar(im_sc, ax=ax_sc, shrink=0.8, pad=0.02).set_label("Mean Score (0..1)", fontsize=9)

    # right pane: score uplift (continuous delta)
    vmax = max(abs(up_data.min()), abs(up_data.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im_up = ax_up.imshow(up_data, aspect="auto", cmap=plt.cm.RdBu, norm=norm,
                         interpolation="nearest")
    for i in range(n_tasks):
        for j in range(n_models):
            val = up_data[i, j]
            ax_up.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                       color=("white" if abs(val) > vmax * 0.6 else "black"))
    ax_up.set_xticks(range(n_models))
    ax_up.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax_up.set_xlabel("Model", fontsize=11)
    ax_up.set_title("Score Uplift (curated − baseline)", fontsize=12, pad=10)
    fig.colorbar(im_up, ax=ax_up, shrink=0.8, pad=0.02).set_label("Score Delta", fontsize=9)

    return _save(fig, output_path, dpi)


def generate_score_comparison_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Side-by-side score comparison: baseline (left) + curated (right), same scale.

    Unlike `generate_combined_score_heatmap` (baseline + uplift-delta), this
    renders both absolute conditions on the same YlGnBu scale so you can read
    "how each student performs WITHOUT injection" next to "how they perform
    WITH injection" in a single glance. Cells with no episodes under a
    condition (e.g. tasks outside a sparse task_skill_map) are masked and
    drawn in gray rather than 0.0 — a black 0 would misread as a fail.
    """
    if not episodes:
        return None

    # Union of (task, model) across both conditions so the two panes stay
    # axis-aligned even when curated is sparse under a task_skill_map.
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"baseline": [], "curated": []}
    )
    for ep in episodes:
        cond = ep.get("condition", "")
        if cond not in ("baseline", "curated"):
            continue
        grouped[(ep["task_uid"], ep.get("model", ""))][cond].append(float(ep.get("score", 0.0)))

    tasks = sorted({k[0] for k in grouped})
    models = sorted({k[1] for k in grouped})
    if not tasks or not models:
        return None

    bl = np.full((len(tasks), len(models)), np.nan)
    cu = np.full((len(tasks), len(models)), np.nan)
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            bscores = grouped[(t, m)]["baseline"]
            cscores = grouped[(t, m)]["curated"]
            if bscores:
                bl[i, j] = sum(bscores) / len(bscores)
            if cscores:
                cu[i, j] = sum(cscores) / len(cscores)

    bl_m = np.ma.masked_invalid(bl)
    cu_m = np.ma.masked_invalid(cu)

    n_tasks, n_models = len(tasks), len(models)
    pane_w = max(5, n_models * 1.2 + 2)
    pane_h = max(4, n_tasks * 0.45 + 1.5)
    fig, (ax_b, ax_c) = plt.subplots(
        1, 2, figsize=(pane_w * 2 + 1.5, pane_h), sharey=True,
    )
    task_labels = [_shorten_task_uid(t) for t in tasks]

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#dddddd")                    # masked cells = no data

    def _draw(ax, data_m, data_raw, title):
        im = ax.imshow(data_m, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        for i in range(n_tasks):
            for j in range(n_models):
                val = data_raw[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#666666")
                else:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=("white" if val > 0.6 else "black"))
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(task_labels, fontsize=8)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        return im

    im_b = _draw(ax_b, bl_m, bl, "Baseline Score")
    ax_b.set_ylabel("Task", fontsize=11)
    fig.colorbar(im_b, ax=ax_b, shrink=0.8, pad=0.02).set_label("Mean Score (0..1)", fontsize=9)

    im_c = _draw(ax_c, cu_m, cu, "Curated Score (skill injected)")
    fig.colorbar(im_c, ax=ax_c, shrink=0.8, pad=0.02).set_label("Mean Score (0..1)", fontsize=9)

    return _save(fig, output_path, dpi)


def generate_combined_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Side-by-side: baseline pass rate (left) + uplift (right)."""
    pr_data, tasks, models = _build_pass_rate_matrix(episodes, "baseline")
    if not tasks or not models:
        return None

    uplift = build_task_model_uplift_matrix(episodes)
    up_data = np.zeros_like(pr_data)
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            up_data[i, j] = uplift.get(t, {}).get(m, 0.0)

    n_tasks, n_models = len(tasks), len(models)
    pane_w = max(5, n_models * 1.2 + 2)
    pane_h = max(4, n_tasks * 0.45 + 1.5)
    fig, (ax_pr, ax_up) = plt.subplots(
        1, 2, figsize=(pane_w * 2 + 1.5, pane_h), sharey=True,
    )
    task_labels = [_shorten_task_uid(t) for t in tasks]

    # left pane: baseline pass rate
    im_pr = ax_pr.imshow(pr_data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                         interpolation="nearest")
    for i in range(n_tasks):
        for j in range(n_models):
            val = pr_data[i, j]
            ax_pr.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=8, color=("white" if val > 0.6 else "black"))
    ax_pr.set_xticks(range(n_models))
    ax_pr.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax_pr.set_yticks(range(n_tasks))
    ax_pr.set_yticklabels(task_labels, fontsize=8)
    ax_pr.set_xlabel("Model", fontsize=11)
    ax_pr.set_ylabel("Task", fontsize=11)
    ax_pr.set_title("Baseline Pass Rate", fontsize=12, pad=10)
    fig.colorbar(im_pr, ax=ax_pr, shrink=0.8, pad=0.02).set_label("Pass Rate", fontsize=9)

    # right pane: uplift
    vmax = max(abs(up_data.min()), abs(up_data.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im_up = ax_up.imshow(up_data, aspect="auto", cmap=plt.cm.RdBu, norm=norm,
                         interpolation="nearest")
    for i in range(n_tasks):
        for j in range(n_models):
            val = up_data[i, j]
            ax_up.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                       color=("white" if abs(val) > vmax * 0.6 else "black"))
    ax_up.set_xticks(range(n_models))
    ax_up.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax_up.set_xlabel("Model", fontsize=11)
    ax_up.set_title("Skill Uplift (curated − baseline)", fontsize=12, pad=10)
    fig.colorbar(im_up, ax=ax_up, shrink=0.8, pad=0.02).set_label("Score Delta", fontsize=9)

    return _save(fig, output_path, dpi)


def generate_win_loss_bar(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    threshold: float = 0.01,
    dpi: int = 150,
) -> Optional[Path]:
    """Stacked bar: wins (curated > baseline + threshold) / ties / losses per model x mode."""
    modes = sorted({ep.get("mode", "singlecall") for ep in episodes})
    models = sorted({ep.get("model", "") for ep in episodes})

    cells: Dict[Tuple[str, str], List[int]] = {}
    for mode in modes:
        for model in models:
            task_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
            for ep in episodes:
                if ep.get("model") != model:
                    continue
                ep_mode = ep.get("mode", "singlecall")
                cond = ep.get("condition", "")
                # baselines are assumed mode-agnostic (always singlecall);
                # accept them into every mode's bucket.
                if ep_mode != mode and cond != "baseline":
                    continue
                task_scores[ep["task_uid"]][cond] = _score(ep)

            w = t = l = 0
            for task, sc in task_scores.items():
                bl, cu = sc.get("baseline"), sc.get("curated")
                if bl is None or cu is None:
                    continue
                delta = cu - bl
                if delta > threshold:
                    w += 1
                elif delta < -threshold:
                    l += 1
                else:
                    t += 1
            cells[(mode, model)] = [w, t, l]

    labels: List[str] = []
    wins_list: List[int] = []
    ties_list: List[int] = []
    losses_list: List[int] = []
    for mode in modes:
        for model in models:
            labels.append(f"{model}\n({mode})")
            w, t, l = cells.get((mode, model), [0, 0, 0])
            wins_list.append(w)
            ties_list.append(t)
            losses_list.append(l)

    if not any(wins_list) and not any(ties_list) and not any(losses_list):
        return None

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x, wins_list, color="#55A868", label="Win", edgecolor="white")
    ax.bar(x, ties_list, bottom=wins_list, color="#CCCCCC", label="Tie", edgecolor="white")
    bottoms = [w + t for w, t in zip(wins_list, ties_list)]
    ax.bar(x, losses_list, bottom=bottoms, color="#C44E52", label="Loss", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of Tasks", fontsize=10)
    ax.set_title("Win / Tie / Loss per Model x Mode", fontsize=13, pad=12)
    ax.legend(fontsize=9)

    return _save(fig, output_path, dpi)


def generate_delta_by_mode_bar(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    modes = sorted({ep.get("mode", "singlecall") for ep in episodes})
    models = sorted({ep.get("model", "") for ep in episodes})
    if not models:
        return None

    deltas: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        deltas[mode] = {}
        for model in models:
            bl = [_score(ep) for ep in episodes
                  if ep.get("model") == model and ep.get("condition") == "baseline"
                  and ep.get("mode", "singlecall") == mode]
            cu = [_score(ep) for ep in episodes
                  if ep.get("model") == model and ep.get("condition") == "curated"
                  and ep.get("mode") == mode]
            bl_avg = sum(bl) / len(bl) if bl else 0.0
            cu_avg = sum(cu) / len(cu) if cu else 0.0
            deltas[mode][model] = cu_avg - bl_avg

    x = np.arange(len(models))
    bar_width = 0.8 / max(1, len(modes))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
    for i, mode in enumerate(modes):
        vals = [deltas[mode].get(m, 0.0) for m in models]
        offset = (i - len(modes) / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width, label=mode,
            color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:+.3f}",
                    ha="center", va=("bottom" if y >= 0 else "top"), fontsize=7)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score Delta (curated − baseline)", fontsize=10)
    ax.set_title("Skill Delta by Model and Mode", fontsize=13, pad=12)
    ax.legend(title="Mode", fontsize=9, title_fontsize=10)

    return _save(fig, output_path, dpi)


def generate_baseline_vs_curated_scatter(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Scatter: baseline score (x) vs curated score (y), color-coded by model."""
    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for ep in episodes:
        key = (ep["task_uid"], ep.get("model", ""), ep.get("mode", "singlecall"))
        grouped.setdefault(key, {})[ep.get("condition", "")] = _score(ep)

    pairs = [
        (scores["baseline"], scores["curated"], key[1])
        for key, scores in grouped.items()
        if "baseline" in scores and "curated" in scores
    ]
    if not pairs:
        return None

    models = sorted({p[2] for p in pairs})
    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(7, 7))
    for bl, cu, model in pairs:
        ax.scatter(bl, cu, c=color_map[model], s=50, alpha=0.7,
                   edgecolors="white", linewidths=0.5)
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--", alpha=0.6)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[m],
                   markersize=8, label=m)
        for m in models
    ]
    ax.legend(handles=handles, title="Model", fontsize=8, title_fontsize=9,
              loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Baseline Score", fontsize=11)
    ax.set_ylabel("Curated Score", fontsize=11)
    ax.set_title("Baseline vs Curated Score per Task", fontsize=13, pad=12)
    ax.set_aspect("equal")

    return _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Batch generator + CLI
# ---------------------------------------------------------------------------


def _save(fig, output_path: Path, dpi: int) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all(
    episodes: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str = "",
    dpi: int = 150,
) -> List[str]:
    """Generate every supported chart into `output_dir`; returns list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    charts = [
        (f"uplift_heatmap{suffix}.png",          lambda p: generate_uplift_heatmap(episodes, p, dpi=dpi)),
        (f"baseline_pass_rate{suffix}.png",      lambda p: generate_pass_rate_heatmap(episodes, p, condition="baseline", dpi=dpi)),
        (f"baseline_score_heatmap{suffix}.png",  lambda p: generate_score_heatmap(episodes, p, condition="baseline", dpi=dpi)),
        (f"curated_score_heatmap{suffix}.png",   lambda p: generate_score_heatmap(episodes, p, condition="curated", dpi=dpi)),
        (f"combined_heatmap{suffix}.png",        lambda p: generate_combined_heatmap(episodes, p, dpi=dpi)),
        (f"combined_score_heatmap{suffix}.png",  lambda p: generate_combined_score_heatmap(episodes, p, dpi=dpi)),
        (f"score_comparison_heatmap{suffix}.png", lambda p: generate_score_comparison_heatmap(episodes, p, dpi=dpi)),
        (f"win_loss_bar{suffix}.png",            lambda p: generate_win_loss_bar(episodes, p, dpi=dpi)),
        (f"delta_by_mode_bar{suffix}.png",       lambda p: generate_delta_by_mode_bar(episodes, p, dpi=dpi)),
        (f"baseline_vs_curated{suffix}.png",     lambda p: generate_baseline_vs_curated_scatter(episodes, p, dpi=dpi)),
    ]
    for filename, gen in charts:
        path = output_dir / filename
        result = gen(path)
        if result is not None and path.exists():
            generated.append(str(path))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SkillsBench heatmaps + summary charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # All charts, all modes in one run
  python -m b2_benchmarks.skillsbench.visualization \\
      --results data/skillsbench-out/episodes.json -o data/skillsbench-out/heatmaps/

  # Only the combined heatmap
  python -m b2_benchmarks.skillsbench.visualization \\
      --results episodes.json -o heatmaps/ --type combined

  # Separate charts per mode (auto-detected)
  python -m b2_benchmarks.skillsbench.visualization \\
      --results episodes.json -o heatmaps/ --mode per-mode
""",
    )
    parser.add_argument("--results", type=Path, nargs="+",
                        help="One or more episodes.json files to merge")
    parser.add_argument("--results-dir", type=Path,
                        help="Directory containing episodes.json")
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--type", choices=["all", "uplift", "baseline", "score",
                                             "combined", "combined-score",
                                             "score-comparison",
                                             "win-loss", "delta", "scatter"],
                        default="all")
    parser.add_argument("--mode", type=str, default="",
                        help="singlecall | guided | per-mode (separate chart sets per mode)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.results and not args.results_dir:
        parser.error("provide --results or --results-dir")

    all_episodes: List[Dict[str, Any]] = []
    if args.results:
        for f in args.results:
            all_episodes.extend(load_episodes(results_file=f))
    if args.results_dir:
        all_episodes.extend(load_episodes(results_dir=args.results_dir))
    if not all_episodes:
        print("no episodes loaded; check input paths")
        return

    modes_present = sorted({ep.get("mode", "singlecall") for ep in all_episodes})
    print(f"loaded {len(all_episodes)} episodes, modes: {modes_present}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baselines = [ep for ep in all_episodes if ep.get("condition") == "baseline"]

    def run(eps, suffix=""):
        if args.type == "all":
            generated = generate_all(eps, args.output_dir, suffix=suffix, dpi=args.dpi)
        else:
            dispatch = {
                "uplift":         lambda p: generate_uplift_heatmap(eps, p, dpi=args.dpi),
                "baseline":       lambda p: generate_pass_rate_heatmap(eps, p, condition="baseline", dpi=args.dpi),
                "score":          lambda p: generate_score_heatmap(eps, p, condition="baseline", dpi=args.dpi),
                "combined":       lambda p: generate_combined_heatmap(eps, p, dpi=args.dpi),
                "combined-score": lambda p: generate_combined_score_heatmap(eps, p, dpi=args.dpi),
                "score-comparison": lambda p: generate_score_comparison_heatmap(eps, p, dpi=args.dpi),
                "win-loss":       lambda p: generate_win_loss_bar(eps, p, dpi=args.dpi),
                "delta":          lambda p: generate_delta_by_mode_bar(eps, p, dpi=args.dpi),
                "scatter":        lambda p: generate_baseline_vs_curated_scatter(eps, p, dpi=args.dpi),
            }
            fname = f"{args.type}{suffix}.png"
            path = args.output_dir / fname
            out = dispatch[args.type](path)
            generated = [str(path)] if out is not None and path.exists() else []
        for p in generated:
            print(f"  wrote: {p}")

    if args.mode == "per-mode":
        treatment_modes = [m for m in modes_present if m != "singlecall"] or modes_present
        for m in treatment_modes:
            mode_eps = [ep for ep in all_episodes if ep.get("mode", "singlecall") == m]
            combined = baselines + [ep for ep in mode_eps if ep.get("condition") != "baseline"]
            if not combined:
                continue
            print(f"\n--- mode: {m} ---")
            run(combined, suffix=f"_{m}")
    elif args.mode:
        mode_eps = [ep for ep in all_episodes if ep.get("mode", "singlecall") == args.mode]
        combined = baselines + [ep for ep in mode_eps if ep.get("condition") != "baseline"]
        run(combined, suffix=f"_{args.mode}")
    else:
        run(all_episodes)


if __name__ == "__main__":
    main()
