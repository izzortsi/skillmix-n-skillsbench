"""
make_figures.py — Generate the paper's figures from episodes.json files
already in the repo. Outputs PDF (paper) + PNG (README preview) for each.

USAGE
    python paper/figures/make_figures.py

FIGURES
  fig1_progression       BL/CU bar chart across all 11 variants
  fig2_per_skill_v2_0    horizontal Δ per skill at v2.0 (lift/flat/regress)
  fig3_bootstrap         v1.9 vs v2.0 Δ bootstrap distributions
  fig4_attribution       v1.9 lift decomposition (base scaling + SFT)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "data" / "pipeline-runs" / "default"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

# ---------- shared style ---------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# colorblind-safe palette
C_BL  = "#4477AA"   # blue   = baseline
C_CU  = "#EE6677"   # red    = curated
C_BASE = "#999999"  # grey   = base-scaling
C_SFT  = "#228833"  # green  = SFT contribution
C_LIFT = "#228833"
C_FLAT = "#999999"
C_REGR = "#EE6677"


def save(fig, name):
    """Save fig as both PDF (paper) and PNG (preview)."""
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"  -> {pdf.relative_to(ROOT)}")
    print(f"  -> {png.relative_to(ROOT)}")
    plt.close(fig)


# ---------- data loading ---------------------------------------------------

def load_summary(name):
    p = RUNS / name / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_episodes(name):
    p = RUNS / name / "episodes.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------- fig1: progression ---------------------------------------------

def fig1_progression():
    """BL/CU pass-rate bar chart across all variants. Three visual groups
    via background shading: pre-SFT controls, 0.8B SFT iterations,
    larger models."""
    rows = [
        # (label, BL, CU, group)  group: "pre", "0.8b", "scale"
        ("pre 0.8B",     0.625, 0.510, "pre"),
        ("pre 2B",       0.685, 0.710, "pre"),
        ("pre haiku-4-5",0.785, 0.800, "pre"),
        ("v1 (0.8B)",    0.635, 0.585, "0.8b"),
        ("v1.5 (0.8B)",  0.650, 0.425, "0.8b"),
        ("v1.6 (0.8B)",  0.645, 0.565, "0.8b"),
        ("v1.7 (0.8B)",  0.465, 0.615, "0.8b"),
        ("v1.8 (0.8B)",  0.545, 0.570, "0.8b"),
        ("v1.9 (2B)",    0.750, 0.825, "scale"),
        ("v2.0 (4B)",    0.835, 0.880, "scale"),
    ]
    n = len(rows)
    x = np.arange(n)
    w = 0.4

    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    # group shading
    group_bounds = {"pre": (0, 3), "0.8b": (3, 8), "scale": (8, 10)}
    group_colors = {"pre": "#f0f4f8", "0.8b": "#fdf6e3", "scale": "#f0f8f0"}
    for g, (lo, hi) in group_bounds.items():
        ax.axvspan(lo - 0.5, hi - 0.5, color=group_colors[g], zorder=0)

    bls = [r[1] for r in rows]
    cus = [r[2] for r in rows]
    ax.bar(x - w/2, bls, w, label="baseline", color=C_BL,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, cus, w, label="curated", color=C_CU,
           edgecolor="black", linewidth=0.4)

    # haiku reference line
    ax.axhline(0.800, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(0.05, 0.805, "haiku-4-5 CU = 0.800",
            ha="left", va="bottom", fontsize=7.5, alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("pass rate")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.legend(loc="upper left", frameon=False, ncol=2)

    # group labels above
    for g, (lo, hi), title in [
        ("pre", group_bounds["pre"], "pre-SFT controls"),
        ("0.8b", group_bounds["0.8b"], "0.8B SFT iterations"),
        ("scale", group_bounds["scale"], "model-size pivot"),
    ]:
        ax.text((lo + hi) / 2 - 0.5, 1.02, title, ha="center", va="bottom",
                fontsize=8, transform=ax.get_xaxis_transform(),
                fontstyle="italic", alpha=0.7)

    ax.set_title("Baseline vs Curated pass rate across recipe variants and model scales", pad=22)
    save(fig, "fig1_progression")


# ---------- fig2: per-skill v2.0 ------------------------------------------

def fig2_per_skill_v2_0():
    """Horizontal Δ-per-skill bars at v2.0, sorted, colored by cluster."""
    eps = load_episodes("bench-eval-post-sft-v2_0")
    if eps is None:
        print("skip fig2: v2.0 episodes missing")
        return

    task_skill_map = json.loads(
        (RUNS / "synthesis" / "task_skill_map_eval.json").read_text()
    )

    by_skill = defaultdict(lambda: {"bl": [0, 0], "cu": [0, 0]})
    for e in eps:
        sk = task_skill_map.get(e["task_uid"], "unknown")
        cond = e["condition"]
        by_skill[sk]["bl" if cond == "baseline" else "cu"][0] += int(e["passed"])
        by_skill[sk]["bl" if cond == "baseline" else "cu"][1] += 1

    rows = []
    for sk, c in by_skill.items():
        if c["bl"][1] == 0 or c["cu"][1] == 0:
            continue
        bl = c["bl"][0] / c["bl"][1]
        cu = c["cu"][0] / c["cu"][1]
        rows.append((sk, bl, cu, cu - bl))

    rows.sort(key=lambda r: r[3])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.0, max(7.5, n * 0.28)))
    y = np.arange(n)

    deltas = [r[3] for r in rows]
    colors = [C_REGR if d < 0 else (C_FLAT if d == 0 else C_LIFT) for d in deltas]

    ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.3, height=0.75)
    ax.axvline(0, color="black", linewidth=0.6)

    short_names = [r[0] if len(r[0]) <= 36 else r[0][:33] + "..." for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Δ = pass(curated) − pass(baseline)")
    ax.set_xlim(-0.30, 0.50)

    n_lift = sum(1 for d in deltas if d > 0)
    n_flat = sum(1 for d in deltas if d == 0)
    n_regr = sum(1 for d in deltas if d < 0)
    handles = [
        mpatches.Patch(color=C_LIFT, label=f"lift (Δ>0): {n_lift}"),
        mpatches.Patch(color=C_FLAT, label=f"flat (Δ=0): {n_flat}"),
        mpatches.Patch(color=C_REGR, label=f"regression (Δ<0): {n_regr}"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    ax.set_title(f"v2.0 per-skill procedural lift (n=5 per skill, {n} skills)", pad=8)
    save(fig, "fig2_per_skill_v2_0")


# ---------- fig3: bootstrap distributions ---------------------------------

def fig3_bootstrap():
    """Bootstrap distributions of Δ for v1.9 and v2.0, plus the
    difference distribution. Visualizes why the saturation test fails."""
    eps_v19 = load_episodes("bench-eval-post-sft-v1_9")
    eps_v20 = load_episodes("bench-eval-post-sft-v2_0")
    if eps_v19 is None or eps_v20 is None:
        print("skip fig3: episodes missing")
        return

    def paired(eps):
        by_t = defaultdict(dict)
        for e in eps:
            by_t[e["task_uid"]][e["condition"]] = int(e["passed"])
        return [(t["baseline"], t["curated"]) for t in by_t.values()
                if "baseline" in t and "curated" in t]

    p19 = paired(eps_v19)
    p20 = paired(eps_v20)
    n19, n20 = len(p19), len(p20)

    rng = random.Random(42)
    n_boot = 10000
    d19s, d20s, diffs = [], [], []
    for _ in range(n_boot):
        s19 = [p19[rng.randrange(n19)] for _ in range(n19)]
        s20 = [p20[rng.randrange(n20)] for _ in range(n20)]
        d19 = sum(b - a for a, b in s19) / n19
        d20 = sum(b - a for a, b in s20) / n20
        d19s.append(d19); d20s.append(d20); diffs.append(d19 - d20)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # left: side-by-side Δ distributions
    ax = axes[0]
    bins = np.arange(-0.05, 0.20, 0.005)
    ax.hist(d19s, bins=bins, alpha=0.6, color=C_CU,
            edgecolor="black", linewidth=0.3, label="v1.9 (2B)")
    ax.hist(d20s, bins=bins, alpha=0.6, color=C_BL,
            edgecolor="black", linewidth=0.3, label="v2.0 (4B)")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    p19_pt = sum(b - a for a, b in p19) / n19
    p20_pt = sum(b - a for a, b in p20) / n20
    ax.axvline(p19_pt, color=C_CU, linewidth=1.2)
    ax.axvline(p20_pt, color=C_BL, linewidth=1.2)
    ax.text(p19_pt, ax.get_ylim()[1] * 0.92, f"+{p19_pt:.3f}",
            ha="center", color=C_CU, fontsize=8)
    ax.text(p20_pt, ax.get_ylim()[1] * 0.78, f"+{p20_pt:.3f}",
            ha="center", color=C_BL, fontsize=8)

    ax.set_xlabel("Δ = pass(curated) − pass(baseline)")
    ax.set_ylabel("bootstrap frequency")
    ax.set_title(f"Per-model Δ distributions (10k bootstrap)", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # right: difference distribution
    ax = axes[1]
    diff_pt = p19_pt - p20_pt
    diffs_sorted = sorted(diffs)
    ci_lo = diffs_sorted[int(0.025 * n_boot)]
    ci_hi = diffs_sorted[int(0.975 * n_boot)]

    bins = np.arange(-0.10, 0.16, 0.005)
    ax.hist(diffs, bins=bins, color="#999999",
            edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--",
               label="H₀: no shrinkage")
    ax.axvline(diff_pt, color="#CC3311", linewidth=1.2,
               label=f"point: +{diff_pt:.3f}")
    ax.axvspan(ci_lo, ci_hi, alpha=0.2, color="#CC3311",
               label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    p_one_sided = sum(1 for d in diffs if d <= 0) / n_boot
    ax.set_xlabel("Δ_v1.9 − Δ_v2.0")
    ax.set_title(f"Saturation test (one-sided p={p_one_sided:.3f})", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)

    fig.suptitle("Bootstrap analysis of the bench-saturation hypothesis",
                 fontsize=11, y=1.02)
    save(fig, "fig3_bootstrap")


# ---------- fig4: attribution decomposition -------------------------------

def fig4_attribution():
    """Stacked bar showing v1.9's lift over pre-SFT 0.8B decomposed
    into base-scaling and SFT contribution. Two side-by-side stacks
    for ΔBL and ΔCU."""
    # numbers from v1.9 attribution split (matched-path HF, det-mixed scoring)
    # see Table 3 (tab:attribution): base scaling uses pre-SFT 0.8B HF (0.625/0.510)
    # and pre-SFT 2B HF (0.685/0.710); SFT contribution = v1.9 - pre-SFT 2B
    metrics = ["ΔBL", "ΔCU"]
    base_scaling = [0.060, 0.200]
    sft_contrib  = [0.065, 0.115]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    x = np.arange(len(metrics))
    w = 0.5

    p1 = ax.bar(x, base_scaling, w, color=C_BASE,
                edgecolor="black", linewidth=0.4,
                label="base scaling (0.8B → 2B)")
    p2 = ax.bar(x, sft_contrib, w, bottom=base_scaling, color=C_SFT,
                edgecolor="black", linewidth=0.4,
                label="SFT contribution at 2B")

    # annotate each segment
    for i in range(len(metrics)):
        b = base_scaling[i]
        s = sft_contrib[i]
        ax.text(x[i], b / 2, f"+{b:.3f}", ha="center", va="center",
                fontsize=8, color="white")
        ax.text(x[i], b + s / 2, f"+{s:.3f}", ha="center", va="center",
                fontsize=8, color="white")
        ax.text(x[i], b + s + 0.01, f"total +{b+s:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("absolute lift (v1.9 − pre-SFT 0.8B)")
    ax.set_ylim(0, 0.32)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("v1.9 attribution split", pad=8)
    save(fig, "fig4_attribution")


# ---------- main ----------------------------------------------------------

if __name__ == "__main__":
    print("rendering paper figures...")
    fig1_progression()
    fig2_per_skill_v2_0()
    fig3_bootstrap()
    fig4_attribution()
    print("done.")
