"""
visualizer.py

Generate charts from SkillMix experiment results. Reads episodes.json
and summary.json produced by run-skillmix and generates PNG visualizations.

SkillMix-specific dimensions parsed from skill_name:
    operator    seq / par / cond / atomic (no prefix = atomic)
    k           composition size (number of atomic skills combined)

Charts:
    score_by_k.png              line: mean score by k-value, one line per model
    operator_heatmap.png        heatmap: operator x model, mean score
    uplift_heatmap.png          heatmap: skill x model, delta from baseline (diverging)
    k_operator_heatmap.png      heatmap: (k, operator) x model, mean score
    baseline_vs_skill.png       grouped bar: baseline vs skill-injected per model
    win_loss.png                stacked bar: win/tie/loss per model
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


PALETTE = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
           "#64B5CD", "#DD8452", "#A1C9F4")

OPERATOR_COLORS = {"atomic": "#4C72B0", "seq": "#55A868", "par": "#DD8452", "cond": "#C44E52"}


def _load_data(results_dir: Path) -> tuple:
    """Load episodes.json and summary.json from a results directory.

    The summary.json produced by c3_skillmix.frontier_skillmix nests
    model stats under a "per_model" key (alongside "novelty" and run
    metadata). Legacy c3_skillmix.runner puts model stats at the top
    level. Normalize to the legacy flat shape for downstream charts.
    """
    with open(results_dir / "episodes.json", "r", encoding="utf-8") as f:
        episodes = json.load(f)
    with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    if isinstance(summary, dict) and "per_model" in summary and isinstance(
        summary["per_model"], dict
    ):
        summary = summary["per_model"]
    return episodes, summary


def _parse_skill_name(skill_name: str) -> Tuple[str, int]:
    """Extract operator type and k-value from a composed skill name.

    Naming conventions:
        atomic skill (no prefix):  "extract-parallel-claims"         -> ("atomic", 1)
        seq composition:           "seq-skill1-then-skill2"          -> ("seq", 2)
        par composition:           "par-skill1-and-skill2-and-s3"    -> ("par", 3)
        cond composition:          "cond-skill1-then-skill2-and-s3"  -> ("cond", 3)

    Returns:
        (operator, k) tuple
    """
    if not skill_name:
        return ("baseline", 0)

    for prefix in ("seq-", "par-", "cond-"):
        if skill_name.startswith(prefix):
            operator = prefix[:-1]
            body = skill_name[len(prefix):]
            separators = len(re.findall(r"-(?:then|and)-", body))
            k = separators + 1
            return (operator, k)

    return ("atomic", 1)


def generate_score_by_k(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Line chart: mean score by k-value, one line per model."""
    # group by (model, k) -> scores
    by_model_k = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        _, k = _parse_skill_name(ep.get("skill_name", ""))
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        by_model_k[model][k].append(score)

    if not by_model_k:
        return

    models = sorted(by_model_k.keys())
    all_k = sorted(set(k for m in models for k in by_model_k[m].keys()))

    # also compute baseline per model for reference line
    baseline_by_model = defaultdict(list)
    for ep in episodes:
        if ep.get("condition") == "baseline":
            model = ep.get("model", "")
            score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
            baseline_by_model[model].append(score)

    fig, ax = plt.subplots(figsize=(max(6, len(all_k) * 1.5 + 2), 5))

    for idx, model in enumerate(models):
        k_values = []
        means = []
        for k in all_k:
            vals = by_model_k[model].get(k, [])
            if vals:
                k_values.append(k)
                means.append(sum(vals) / len(vals))
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(k_values, means, marker="o", label=model, color=color, linewidth=2)

        # baseline reference
        b_vals = baseline_by_model.get(model, [])
        if b_vals:
            b_mean = sum(b_vals) / len(b_vals)
            ax.axhline(y=b_mean, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Composition Size (k)", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Score by Composition Size (k)", fontsize=13, pad=12)
    ax.set_xticks(all_k)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_operator_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: operator type x model, cell = mean score (skill-injected only)."""
    scores = defaultdict(lambda: defaultdict(list))
    models_set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        operator, _ = _parse_skill_name(ep.get("skill_name", ""))
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        scores[operator][model].append(score)
        models_set.add(model)

    if not scores:
        return

    operators = sorted(scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(operators), len(models)))
    for i, op in enumerate(operators):
        for j, m in enumerate(models):
            vals = scores[op][m]
            data[i, j] = sum(vals) / len(vals) if vals else 0.0

    figsize = (max(6, len(models) * 1.5 + 3), max(3, len(operators) * 0.8 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(operators)))
    ax.set_yticklabels(operators, fontsize=10)
    ax.set_title("Mean Score by Operator Type x Model", fontsize=13, pad=12)

    for i in range(len(operators)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Score", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _pair_by_task(episodes: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Group episodes by task_uid and pair baseline vs skill_injected per model.

    Returns a dict mapping task_uid -> {
        "title":      task title,
        "k":          composition size or 1,
        "novelty":    episode novelty dict (None if absent),
        "by_model":   {model: {"baseline": mean, "skill": mean, "delta": delta}},
        "skill_name": the skill name used in skill_injected condition,
    }. Only tasks with both baseline and skill_injected scores are kept.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        task_uid = ep.get("task_uid", ep.get("task_id", ""))
        if not task_uid:
            continue
        model = ep.get("model", "")
        cond = ep.get("condition", "")
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)

        entry = groups.setdefault(task_uid, {
            "title": ep.get("task_title", task_uid),
            "k": ep.get("novelty", {}).get("k") if ep.get("novelty") else None,
            "novelty": ep.get("novelty"),
            "by_model": {},
            "skill_name": "",
        })
        if not entry["title"]:
            entry["title"] = ep.get("task_title", task_uid)
        if entry["novelty"] is None and ep.get("novelty"):
            entry["novelty"] = ep["novelty"]
        if cond == "skill_injected" and ep.get("skill_name"):
            entry["skill_name"] = ep["skill_name"]

        model_entry = entry["by_model"].setdefault(
            model, {"baseline_scores": [], "skill_scores": []}
        )
        if cond == "baseline":
            model_entry["baseline_scores"].append(score)
        elif cond == "skill_injected":
            model_entry["skill_scores"].append(score)

    paired: Dict[str, Dict[str, Any]] = {}
    for task_uid, entry in groups.items():
        model_results: Dict[str, Dict[str, float]] = {}
        for model, rec in entry["by_model"].items():
            b = rec["baseline_scores"]
            s = rec["skill_scores"]
            if not b or not s:
                continue
            b_mean = sum(b) / len(b)
            s_mean = sum(s) / len(s)
            model_results[model] = {
                "baseline": b_mean,
                "skill": s_mean,
                "delta": s_mean - b_mean,
            }
        if not model_results:
            continue
        entry["by_model"] = model_results
        paired[task_uid] = entry
    return paired


def _truncate_label(text: str, width: int = 48) -> str:
    text = text or ""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def generate_uplift_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Per-task uplift heatmap. Rows = tasks (sorted by mean delta desc),
    columns = models, cell = per-task delta (skill - baseline) using
    correctly PAIRED scores. Includes a trailing strip that marks tasks
    flagged likely_novel by the novelty scorer (if present).
    """
    paired = _pair_by_task(episodes)
    if not paired:
        return

    models_set = set()
    for entry in paired.values():
        models_set.update(entry["by_model"].keys())
    models = sorted(models_set)

    def mean_delta(entry):
        vals = [entry["by_model"][m]["delta"] for m in entry["by_model"]]
        return sum(vals) / len(vals) if vals else 0.0

    task_items = sorted(paired.items(), key=lambda kv: -mean_delta(kv[1]))
    task_uids = [tu for tu, _ in task_items]

    data = np.zeros((len(task_uids), len(models)))
    for i, tu in enumerate(task_uids):
        entry = paired[tu]
        for j, m in enumerate(models):
            row = entry["by_model"].get(m)
            data[i, j] = row["delta"] if row else 0.0

    novelty_strip = np.zeros((len(task_uids), 1))
    has_novelty = False
    for i, tu in enumerate(task_uids):
        nov = paired[tu].get("novelty") or {}
        if nov:
            has_novelty = True
            novelty_strip[i, 0] = 1.0 if nov.get("likely_novel") else 0.0

    labels = []
    for tu in task_uids:
        entry = paired[tu]
        k = entry.get("k") or 1
        title = _truncate_label(entry.get("title") or tu, 56)
        labels.append(f"[k={k}] {title}")

    vmax = max(abs(data.min()), abs(data.max()), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ncols = len(models) + (1 if has_novelty else 0)
    figsize = (max(7, ncols * 1.2 + 5), max(4, len(task_uids) * 0.45 + 2))
    fig, axes = plt.subplots(
        1, 2 if has_novelty else 1,
        figsize=figsize,
        gridspec_kw={"width_ratios": [len(models), 0.35]} if has_novelty else None,
    )
    ax_main = axes[0] if has_novelty else axes

    im = ax_main.imshow(data, cmap="RdBu", norm=norm, aspect="auto")
    ax_main.set_xticks(range(len(models)))
    ax_main.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax_main.set_yticks(range(len(task_uids)))
    ax_main.set_yticklabels(labels, fontsize=8)
    ax_main.set_title(
        "Per-task Uplift: Score Delta (Skill-Injected - Baseline)",
        fontsize=13, pad=12,
    )

    for i in range(len(task_uids)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if abs(val) > vmax * 0.55 else "black"
            ax_main.text(
                j, i, f"{val:+.2f}",
                ha="center", va="center", fontsize=8, color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax_main, shrink=0.8, pad=0.02)
    cbar.set_label("Score Delta", fontsize=10)

    if has_novelty:
        ax_nov = axes[1]
        ax_nov.imshow(novelty_strip, cmap="Purples", aspect="auto", vmin=0.0, vmax=1.0)
        ax_nov.set_xticks([0])
        ax_nov.set_xticklabels(["novel?"], rotation=30, ha="right", fontsize=9)
        ax_nov.set_yticks([])
        for i in range(len(task_uids)):
            v = novelty_strip[i, 0]
            glyph = "★" if v > 0.5 else "·"
            ax_nov.text(
                0, i, glyph,
                ha="center", va="center", fontsize=14,
                color=("white" if v > 0.5 else "#555"),
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_comparison_matrix(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Tasks x [baseline | skill | delta] matrix, one block per model.

    For single-model runs this is the most informative comparison view:
    each row shows baseline score, skill-injected score, and their
    difference side-by-side, sorted by descending delta.
    """
    paired = _pair_by_task(episodes)
    if not paired:
        return

    models_set = set()
    for entry in paired.values():
        models_set.update(entry["by_model"].keys())
    models = sorted(models_set)

    def mean_delta(entry):
        vals = [entry["by_model"][m]["delta"] for m in entry["by_model"]]
        return sum(vals) / len(vals) if vals else 0.0

    task_items = sorted(paired.items(), key=lambda kv: -mean_delta(kv[1]))
    task_uids = [tu for tu, _ in task_items]

    n_rows = len(task_uids)
    block_cols = 3  # baseline, skill, delta
    n_models = len(models)
    n_cols = n_models * block_cols

    score_mat = np.zeros((n_rows, n_cols))
    delta_mat = np.zeros((n_rows, n_cols))
    is_delta = np.zeros((n_rows, n_cols), dtype=bool)

    for i, tu in enumerate(task_uids):
        entry = paired[tu]
        for j, m in enumerate(models):
            rec = entry["by_model"].get(m)
            if not rec:
                continue
            score_mat[i, j * block_cols + 0] = rec["baseline"]
            score_mat[i, j * block_cols + 1] = rec["skill"]
            delta_mat[i, j * block_cols + 2] = rec["delta"]
            is_delta[i, j * block_cols + 2] = True

    labels = []
    for tu in task_uids:
        entry = paired[tu]
        k = entry.get("k") or 1
        nov = entry.get("novelty") or {}
        star = "★ " if nov.get("likely_novel") else ""
        labels.append(f"{star}[k={k}] {_truncate_label(entry.get('title') or tu, 52)}")

    delta_vmax = max(abs(delta_mat.min()), abs(delta_mat.max()), 0.05)
    delta_norm = mcolors.TwoSlopeNorm(vmin=-delta_vmax, vcenter=0.0, vmax=delta_vmax)

    fig, ax = plt.subplots(figsize=(max(7, n_cols * 1.1 + 4), max(4, n_rows * 0.45 + 2)))

    score_bg = np.ma.masked_where(is_delta, score_mat)
    im_score = ax.imshow(score_bg, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    delta_bg = np.ma.masked_where(~is_delta, delta_mat)
    im_delta = ax.imshow(delta_bg, cmap="RdBu", norm=delta_norm, aspect="auto")

    sub_labels = ["baseline", "skill", "Δ"] * n_models
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sub_labels, fontsize=9)
    for m_idx, m in enumerate(models):
        center = m_idx * block_cols + 1
        ax.annotate(
            m,
            xy=(center, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 22), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Baseline vs Skill-Injected (paired per task, ★ = likely novel)",
                 fontsize=13, pad=36)

    for i in range(n_rows):
        for j in range(n_cols):
            if is_delta[i, j]:
                v = delta_mat[i, j]
                color = "white" if abs(v) > delta_vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color=color)
            else:
                v = score_mat[i, j]
                color = "white" if v > 0.60 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    for m_idx in range(n_models):
        for offset in (1, 2):
            ax.axvline(x=m_idx * block_cols + offset - 0.5, color="#cccccc",
                       linewidth=0.6, zorder=2)
        if m_idx < n_models - 1:
            ax.axvline(x=(m_idx + 1) * block_cols - 0.5, color="#222222",
                       linewidth=1.2, zorder=3)

    cbar_score = fig.colorbar(im_score, ax=ax, shrink=0.4, pad=0.02, location="right")
    cbar_score.set_label("Score (baseline / skill)", fontsize=9)
    cbar_delta = fig.colorbar(im_delta, ax=ax, shrink=0.4, pad=0.08, location="right")
    cbar_delta.set_label("Delta (skill - baseline)", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_uplift_by_novelty(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Aggregate uplift by (novelty bucket, k). Answers: does skill
    injection help more when the k-combination is likely novel?
    """
    paired = _pair_by_task(episodes)
    if not paired:
        return

    by_bucket: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for entry in paired.values():
        nov = entry.get("novelty") or {}
        bucket = "likely novel" if nov.get("likely_novel") else "not novel"
        k = entry.get("k") or 1
        for rec in entry["by_model"].values():
            by_bucket[(bucket, k)].append(rec["delta"])

    if not by_bucket:
        return

    buckets = ["likely novel", "not novel"]
    ks = sorted({k for _, k in by_bucket.keys()})
    if not ks:
        return

    data = np.full((len(buckets), len(ks)), np.nan)
    counts = np.zeros_like(data, dtype=int)
    for i, b in enumerate(buckets):
        for j, k in enumerate(ks):
            vals = by_bucket.get((b, k), [])
            if vals:
                data[i, j] = sum(vals) / len(vals)
                counts[i, j] = len(vals)

    vmax = max(0.05, np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(5, len(ks) * 1.2 + 3), 3))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=10)
    ax.set_yticks(range(len(buckets)))
    ax.set_yticklabels(buckets, fontsize=10)
    ax.set_title("Mean Uplift by Novelty Bucket x k", fontsize=13, pad=12)

    for i in range(len(buckets)):
        for j in range(len(ks)):
            v = data[i, j]
            n = counts[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#999")
            else:
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.2f}\n(n={n})",
                        ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean uplift (Δ = skill - baseline)", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_k_operator_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: rows = (k, operator) combinations, columns = models."""
    scores = defaultdict(lambda: defaultdict(list))
    models_set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        operator, k = _parse_skill_name(ep.get("skill_name", ""))
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        row_key = f"k={k} {operator}"
        scores[row_key][model].append(score)
        models_set.add(model)

    if not scores:
        return

    row_keys = sorted(scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(row_keys), len(models)))
    for i, rk in enumerate(row_keys):
        for j, m in enumerate(models):
            vals = scores[rk][m]
            data[i, j] = sum(vals) / len(vals) if vals else 0.0

    figsize = (max(6, len(models) * 1.5 + 3), max(3, len(row_keys) * 0.6 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_keys, fontsize=9)
    ax.set_title("Mean Score by (k, Operator) x Model", fontsize=13, pad=12)

    for i in range(len(row_keys)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Score", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_baseline_vs_skill_bar(
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Grouped bar chart: baseline vs skill-injected mean score per model."""
    models = sorted(summary.keys())
    baselines = [summary[m]["baseline_mean_score"] for m in models]
    skills = [summary[m]["skill_mean_score"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    bars_b = ax.bar(x - width / 2, baselines, width, label="Baseline", color=PALETTE[0])
    bars_s = ax.bar(x + width / 2, skills, width, label="Skill-Injected", color=PALETTE[1])

    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Baseline vs Skill-Injected Score by Model", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    for bar in list(bars_b) + list(bars_s):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_win_loss_bar(
    episodes: List[Dict],
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Stacked bar: wins/ties/losses per model.

    For each task, a win occurs when skill_injected score > baseline score,
    a loss when skill_injected < baseline, and a tie when equal.
    """
    by_model_task = defaultdict(lambda: defaultdict(lambda: {"baseline": [], "skill_injected": []}))
    for ep in episodes:
        model = ep.get("model", "")
        task = ep.get("task_uid", ep.get("task_id", ""))
        cond = ep.get("condition", "")
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        by_model_task[model][task][cond].append(score)

    models = sorted(by_model_task.keys())
    wins = []
    ties = []
    losses = []

    for m in models:
        w, t, l = 0, 0, 0
        for task, conds in by_model_task[m].items():
            b_scores = conds.get("baseline", [])
            s_scores = conds.get("skill_injected", [])
            if not b_scores or not s_scores:
                continue
            b_mean = sum(b_scores) / len(b_scores)
            s_mean = sum(s_scores) / len(s_scores)
            if s_mean > b_mean + 0.001:
                w += 1
            elif s_mean < b_mean - 0.001:
                l += 1
            else:
                t += 1
        wins.append(w)
        ties.append(t)
        losses.append(l)

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))

    ax.bar(x, wins, label="Win", color=PALETTE[1])
    ax.bar(x, ties, bottom=wins, label="Tie", color="#999999")
    bottoms = [w + t for w, t in zip(wins, ties)]
    ax.bar(x, losses, bottom=bottoms, label="Loss", color=PALETTE[2])

    ax.set_ylabel("Number of Tasks", fontsize=11)
    ax.set_title("Win / Tie / Loss per Model (Skill vs Baseline)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_all(
    results_dir: Path,
    output_dir: Path,
    dpi: int = 150,
) -> List[str]:
    """Generate all SkillMix visualization charts.

    Args:
        results_dir: directory containing episodes.json and summary.json
        output_dir: directory to write PNG files into
        dpi: output image resolution

    Returns:
        list of generated file paths
    """
    episodes, summary = _load_data(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    charts = [
        ("score_by_k.png", lambda p: generate_score_by_k(episodes, p, dpi)),
        ("operator_heatmap.png", lambda p: generate_operator_heatmap(episodes, p, dpi)),
        ("uplift_heatmap.png", lambda p: generate_uplift_heatmap(episodes, p, dpi)),
        ("comparison_matrix.png", lambda p: generate_comparison_matrix(episodes, p, dpi)),
        ("uplift_by_novelty.png", lambda p: generate_uplift_by_novelty(episodes, p, dpi)),
        ("k_operator_heatmap.png", lambda p: generate_k_operator_heatmap(episodes, p, dpi)),
        ("baseline_vs_skill.png", lambda p: generate_baseline_vs_skill_bar(summary, p, dpi)),
        ("win_loss.png", lambda p: generate_win_loss_bar(episodes, summary, p, dpi)),
    ]

    for filename, gen_fn in charts:
        path = output_dir / filename
        gen_fn(path)
        if path.exists():
            generated.append(str(path))

    return generated


def main() -> None:
    """CLI entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate SkillMix charts")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with episodes.json and summary.json")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for PNG files")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    args = parser.parse_args()

    generated = generate_all(args.results_dir, args.output_dir, args.dpi)
    for path in generated:
        print(f"  generated: {path}")


if __name__ == "__main__":
    main()
