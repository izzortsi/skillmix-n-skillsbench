"""
b2_benchmarks.skillmix.visualizer

Charts for composed-skill (seq / par / cond / atomic) injection experiments.
Consumes:
    episodes.json  list of {task_uid, model, condition, skill_name, score, passed}
                   where condition is "baseline" or "skill_injected"
    summary.json   {model: {baseline_mean_score, skill_mean_score, ...}}

Six charts:
    score_by_k.png            line: mean score vs k, one line per model (baseline as dashed ref)
    operator_heatmap.png      heatmap: operator x model, mean score (skill_injected only)
    uplift_heatmap.png        heatmap: skill x model, delta = skill_injected_mean − baseline_mean
    k_operator_heatmap.png    heatmap: (k, operator) x model, mean score
    baseline_vs_skill.png     grouped bar: baseline vs skill mean per model (from summary.json)
    win_loss.png              stacked bar: wins / ties / losses per model (per-task comparison)

Skill-name parsing:
    "seq-alpha-then-beta"                 -> ("seq", 2)
    "par-a-and-b-and-c"                   -> ("par", 3)
    "cond-a-then-b-and-c"                 -> ("cond", 3)
    "any-other-name" (no prefix)          -> ("atomic", 1)
    "" / missing                          -> ("baseline", 0)

Ported-and-trimmed from skillsuite/llm-skills.skillmix-evaluation/c2_analytics/visualizer.py.
"""

from __future__ import annotations

import argparse
import json
import re
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
# Loaders + helpers
# ---------------------------------------------------------------------------


def parse_skill_name(skill_name: str) -> Tuple[str, int]:
    """Extract (operator, k) from a composed skill name.

    No-prefix names are atomic (k=1); an empty or missing name is baseline (k=0).
    """
    if not skill_name:
        return ("baseline", 0)
    for prefix in ("seq-", "par-", "cond-"):
        if skill_name.startswith(prefix):
            operator = prefix[:-1]
            body = skill_name[len(prefix):]
            k = len(re.findall(r"-(?:then|and)-", body)) + 1
            return (operator, k)
    return ("atomic", 1)


def _score(ep: Dict[str, Any]) -> float:
    if "score" in ep:
        return float(ep["score"])
    return 1.0 if ep.get("passed") else 0.0


def _load_data(results_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load episodes.json and summary.json from a results directory."""
    with open(results_dir / "episodes.json", "r", encoding="utf-8") as f:
        episodes = json.load(f)
    with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return episodes, summary


def _save(fig, output_path: Path, dpi: int) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def generate_score_by_k(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Line chart: mean score by k, one line per model. Baseline as dashed reference."""
    by_model_k: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    baseline_by_model: Dict[str, List[float]] = defaultdict(list)
    for ep in episodes:
        model = ep.get("model", "")
        cond = ep.get("condition", "")
        if cond == "baseline":
            baseline_by_model[model].append(_score(ep))
        elif cond == "skill_injected":
            _, k = parse_skill_name(ep.get("skill_name", ""))
            by_model_k[model][k].append(_score(ep))

    if not by_model_k:
        return None

    models = sorted(by_model_k.keys())
    all_k = sorted({k for m in models for k in by_model_k[m]})

    fig, ax = plt.subplots(figsize=(max(6, len(all_k) * 1.5 + 2), 5))
    for idx, model in enumerate(models):
        xs: List[int] = []
        ys: List[float] = []
        for k in all_k:
            vals = by_model_k[model].get(k, [])
            if vals:
                xs.append(k)
                ys.append(sum(vals) / len(vals))
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(xs, ys, marker="o", label=model, color=color, linewidth=2)

        bvals = baseline_by_model.get(model, [])
        if bvals:
            ax.axhline(
                y=sum(bvals) / len(bvals),
                color=color, linewidth=0.8, linestyle="--", alpha=0.5,
            )

    ax.set_xlabel("Composition Size (k)", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Score by Composition Size (k)", fontsize=13, pad=12)
    ax.set_xticks(all_k)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, output_path, dpi)


def _operator_scores(episodes: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, List[float]]], set]:
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    models: set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        operator, _ = parse_skill_name(ep.get("skill_name", ""))
        scores[operator][model].append(_score(ep))
        models.add(model)
    return scores, models


def generate_operator_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    scores, models_set = _operator_scores(episodes)
    if not scores:
        return None

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
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=("white" if val > 0.65 else "black"))

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Mean Score", fontsize=10)
    return _save(fig, output_path, dpi)


def generate_uplift_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Heatmap: skill x model, delta = skill_injected_mean − baseline_mean (per model)."""
    baseline_scores: Dict[str, List[float]] = defaultdict(list)
    skill_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    models_set: set = set()

    for ep in episodes:
        model = ep.get("model", "")
        models_set.add(model)
        score = _score(ep)
        if ep.get("condition") == "baseline":
            baseline_scores[model].append(score)
        elif ep.get("condition") == "skill_injected":
            skill_scores[ep.get("skill_name", "")][model].append(score)

    if not skill_scores:
        return None

    baseline_mean = {
        m: (sum(baseline_scores[m]) / len(baseline_scores[m])) if baseline_scores.get(m) else 0.0
        for m in models_set
    }
    skills = sorted(skill_scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(skills), len(models)))
    for i, sk in enumerate(skills):
        for j, m in enumerate(models):
            vals = skill_scores[sk][m]
            sk_mean = sum(vals) / len(vals) if vals else 0.0
            data[i, j] = sk_mean - baseline_mean.get(m, 0.0)

    labels: List[str] = []
    for sk in skills:
        op, k = parse_skill_name(sk)
        if op == "atomic":
            labels.append(sk[:35])
        else:
            body = sk[len(op) + 1:]
            labels.append(f"[{op} k={k}] {body[:25]}")

    vmax = max(abs(data.min()), abs(data.max()), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figsize = (max(6, len(models) * 1.5 + 3), max(4, len(skills) * 0.45 + 2))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdBu", norm=norm, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Uplift Heatmap: Score Delta (Skill − Baseline)", fontsize=13, pad=12)

    for i in range(len(skills)):
        for j in range(len(models)):
            val = data[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7,
                    color=("white" if abs(val) > vmax * 0.6 else "black"))

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Score Delta", fontsize=10)
    return _save(fig, output_path, dpi)


def generate_k_operator_heatmap(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    models_set: set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        operator, k = parse_skill_name(ep.get("skill_name", ""))
        scores[f"k={k} {operator}"][model].append(_score(ep))
        models_set.add(model)

    if not scores:
        return None

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
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color=("white" if val > 0.65 else "black"))

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Mean Score", fontsize=10)
    return _save(fig, output_path, dpi)


def generate_baseline_vs_skill_bar(
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Grouped bar: baseline vs skill mean score per model, from summary.json."""
    models = sorted(m for m in summary.keys()
                    if isinstance(summary.get(m), dict)
                    and "baseline_mean_score" in summary[m])
    if not models:
        return None
    baselines = [summary[m]["baseline_mean_score"] for m in models]
    skills = [summary[m].get("skill_mean_score", 0.0) for m in models]

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
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )
    return _save(fig, output_path, dpi)


def generate_win_loss_bar(
    episodes: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Optional[Path]:
    """Stacked bar: wins / ties / losses per model (per-task baseline vs skill)."""
    by_model_task: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"baseline": [], "skill_injected": []})
    )
    for ep in episodes:
        model = ep.get("model", "")
        task = ep.get("task_uid", ep.get("task_id", ""))
        cond = ep.get("condition", "")
        if cond not in ("baseline", "skill_injected"):
            continue
        by_model_task[model][task][cond].append(_score(ep))

    models = sorted(by_model_task.keys())
    wins: List[int] = []
    ties: List[int] = []
    losses: List[int] = []
    for m in models:
        w = t = l = 0
        for task, conds in by_model_task[m].items():
            b = conds["baseline"]
            s = conds["skill_injected"]
            if not b or not s:
                continue
            b_mean = sum(b) / len(b)
            s_mean = sum(s) / len(s)
            if s_mean > b_mean + 0.001:
                w += 1
            elif s_mean < b_mean - 0.001:
                l += 1
            else:
                t += 1
        wins.append(w)
        ties.append(t)
        losses.append(l)

    if not any(wins) and not any(ties) and not any(losses):
        return None

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
    return _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Batch generator + CLI
# ---------------------------------------------------------------------------


def generate_all(
    results_dir: Path,
    output_dir: Path,
    dpi: int = 150,
) -> List[str]:
    """Generate every chart into `output_dir`; returns list of saved paths."""
    episodes, summary = _load_data(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("score_by_k.png",          lambda p: generate_score_by_k(episodes, p, dpi)),
        ("operator_heatmap.png",    lambda p: generate_operator_heatmap(episodes, p, dpi)),
        ("uplift_heatmap.png",      lambda p: generate_uplift_heatmap(episodes, p, dpi)),
        ("k_operator_heatmap.png",  lambda p: generate_k_operator_heatmap(episodes, p, dpi)),
        ("baseline_vs_skill.png",   lambda p: generate_baseline_vs_skill_bar(summary, p, dpi)),
        ("win_loss.png",            lambda p: generate_win_loss_bar(episodes, p, dpi)),
    ]
    generated: List[str] = []
    for filename, gen in charts:
        path = output_dir / filename
        out = gen(path)
        if out is not None and path.exists():
            generated.append(str(path))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing episodes.json and summary.json")
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    generated = generate_all(args.results_dir, args.output_dir, args.dpi)
    for p in generated:
        print(f"  wrote: {p}")


if __name__ == "__main__":
    main()
