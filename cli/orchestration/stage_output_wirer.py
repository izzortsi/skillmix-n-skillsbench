"""
stage_output_wirer.py

Translate PipelineProfile + stage outputs into:
  - the method_id to dispatch (`python -m cli <method_id>`)
  - the argument list that method expects

Every stage runs from the repo root. No cross-repo providers (lmproxy, iosys,
lm-studio, zai) — only anthropic and mock; both are handled by core.providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from config.pipeline_profile import PipelineProfile


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def stage_output_path(run_dir: Path, stage_id: str, filename: str) -> Path:
    """Canonical output path for a stage artifact under the run directory."""
    return run_dir / stage_id / filename


def register_stage_outputs(stage_id: str, run_dir: Path) -> Dict[str, str]:
    """Map stage_id -> {semantic_key: absolute_path_str} for downstream lookup."""
    if stage_id == "catalog":
        return {"skills": str(stage_output_path(run_dir, "catalog", "skills.json"))}
    if stage_id == "compose":
        return {"examples": str(stage_output_path(run_dir, "compose", "examples.json"))}
    if stage_id == "verify":
        return {
            "verified_examples": str(stage_output_path(run_dir, "verify", "verified-examples.json")),
        }
    if stage_id == "skillmix":
        return {"trials": str(stage_output_path(run_dir, "skillmix", "trials.json"))}
    if stage_id == "solve":
        return {
            "solutions": str(stage_output_path(run_dir, "solve", "solutions.jsonl")),
            "traces": str(stage_output_path(run_dir, "solve", "traces.jsonl")),
            "summary": str(stage_output_path(run_dir, "solve", "summary.json")),
        }
    if stage_id == "bench":
        return {
            "episodes": str(stage_output_path(run_dir, "bench", "episodes.json")),
            "summary": str(stage_output_path(run_dir, "bench", "summary.json")),
        }
    if stage_id == "bench_traced":
        return {
            "episodes": str(stage_output_path(run_dir, "bench_traced", "episodes.jsonl")),
            "summary": str(stage_output_path(run_dir, "bench_traced", "summary.json")),
        }
    return {}


# ---------------------------------------------------------------------------
# Command + args builder
# ---------------------------------------------------------------------------


BENCH_MODULE_PATH = "b2_benchmarks.skillsbench.corpus_harness"
BENCH_VIZ_MODULE_PATH = "b2_benchmarks.skillsbench.visualization"
BENCH_CATALOG_MODULE_PATH = "b2_benchmarks.skillsbench.procedural_catalog"


def build_stage_command(
    stage_id: str,
    profile: PipelineProfile,
    run_dir: Path,
    repo_root: Path,
    stage_outputs: Dict[str, Dict[str, str]],
) -> Tuple[str, List[str]]:
    """Return (method_or_module_path, args) for the given stage.

    method_or_module_path:
      - a method id like "s2.m1" -> invoke `python -m cli <method_id> <args>`
      - a module path prefixed with "__module__:" -> invoke `python -m <path>
        <args>` (used for b2_benchmarks.skillsbench which lives outside
        the cli method dispatcher)
    """
    catalog_out = stage_outputs.get("catalog", {}).get(
        "skills", str(stage_output_path(run_dir, "catalog", "skills.json"))
    )
    compose_out = stage_outputs.get("compose", {}).get(
        "examples", str(stage_output_path(run_dir, "compose", "examples.json"))
    )

    if stage_id == "catalog":
        return _build_catalog(profile, run_dir)
    if stage_id == "compose":
        return _build_compose(profile, run_dir, catalog_out)
    if stage_id == "verify":
        return _build_verify(profile, run_dir, catalog_out, compose_out)
    if stage_id == "skillmix":
        return _build_skillmix(profile, run_dir, catalog_out)
    if stage_id == "solve":
        return _build_solve(profile, run_dir)
    if stage_id == "bench_catalog":
        return _build_bench_catalog(profile, run_dir, catalog_out)
    if stage_id == "bench":
        bench_catalog_out = stage_outputs.get("bench_catalog", {}).get(
            "skills", str(stage_output_path(run_dir, "bench_catalog", "skills.json"))
        )
        return _build_bench(profile, run_dir, bench_catalog_out)
    if stage_id == "bench_traced":
        bench_catalog_out = stage_outputs.get("bench_catalog", {}).get(
            "skills", str(stage_output_path(run_dir, "bench_catalog", "skills.json"))
        )
        return _build_bench_traced(profile, run_dir, bench_catalog_out)
    if stage_id == "bench_viz":
        return _build_bench_viz(profile, run_dir, stage_outputs)

    raise ValueError(f"Unknown stage_id: {stage_id!r}")


# ---------------------------------------------------------------------------
# Per-stage builders
# ---------------------------------------------------------------------------


def _build_catalog(profile: PipelineProfile, run_dir: Path) -> Tuple[str, List[str]]:
    out = stage_output_path(run_dir, "catalog", "skills.json")
    method = profile.catalog_method

    if method == "s2.m1":
        return method, [
            "--source", profile.catalog_source,
            "--out", str(out),
        ]
    if method == "s1.m2":
        return method, [
            "--n-roots", str(profile.catalog_n_roots),
            "--max-subskills", str(profile.catalog_max_subskills),
            "--root-batch-size", str(profile.catalog_root_batch_size),
            "--provider", profile.catalog_provider,
            "--out", str(out),
            "-v",
        ]
    if method == "s1.m3":
        return method, [
            "--target-size", str(profile.catalog_target_size),
            "--categories", str(profile.catalog_categories),
            "--batch-size", str(profile.catalog_batch_size),
            "--max-batches-per-category", str(profile.catalog_max_batches_per_category),
            "--concurrency", str(profile.catalog_concurrency),
            "--provider", profile.catalog_provider,
            "--out", str(out),
            "-v",
        ]
    raise ValueError(f"catalog_method={method!r} not recognized")


def _build_compose(
    profile: PipelineProfile, run_dir: Path, catalog_path: str,
) -> Tuple[str, List[str]]:
    out = stage_output_path(run_dir, "compose", "examples.json")
    method = profile.compose_method

    if method == "s3.m1":
        args = [
            "--catalog", catalog_path,
            "--n", str(profile.compose_n),
            "--k", str(profile.compose_k),
            "--seed", str(profile.compose_seed),
            "--provider", profile.compose_provider,
            "--out", str(out),
            "-v",
        ]
        if profile.compose_same_category:
            args.append("--same-category")
        if profile.compose_categories:
            args.extend(["--categories", ",".join(profile.compose_categories)])
        return method, args

    if method == "s3.m2":
        if not profile.compose_domain.strip():
            raise ValueError("compose_method=s3.m2 requires profile.compose_domain")
        return method, [
            "--catalog", catalog_path,
            "--domain", profile.compose_domain,
            "--n", str(profile.compose_n),
            "--k", str(profile.compose_k),
            "--seed", str(profile.compose_seed),
            "--provider", profile.compose_provider,
            "--out", str(out),
            "-v",
        ]

    if method == "s3.m3":
        return method, [
            "--catalog", catalog_path,
            "--topics", profile.compose_topics,
            "--n", str(profile.compose_n),
            "--k", str(profile.compose_k),
            "--sentence-limit", str(profile.compose_sentence_limit),
            "--seed", str(profile.compose_seed),
            "--provider", profile.compose_provider,
            "--out", str(out),
            "-v",
        ]

    raise ValueError(f"compose_method={method!r} not recognized")


def _build_verify(
    profile: PipelineProfile, run_dir: Path, catalog_path: str, examples_path: str,
) -> Tuple[str, List[str]]:
    out_dir = run_dir / "verify"
    args = [
        "--examples", examples_path,
        "--catalog", catalog_path,
        "--out-dir", str(out_dir),
        "--verifier", profile.verify_verifier,
        "--refiner", profile.verify_refiner,
        "--max-rounds", str(profile.verify_max_rounds),
        "-v",
    ]
    if profile.verify_limit > 0:
        args.extend(["--limit", str(profile.verify_limit)])
    return "s3.m4", args


def _build_skillmix(
    profile: PipelineProfile, run_dir: Path, catalog_path: str,
) -> Tuple[str, List[str]]:
    out = stage_output_path(run_dir, "skillmix", "trials.json")
    return "s2.m2", [
        "--skills", catalog_path,
        "--topics", profile.skillmix_topics,
        "--k", str(profile.skillmix_k),
        "--trials", str(profile.skillmix_trials),
        "--sentence-limit", str(profile.skillmix_sentence_limit),
        "--student", profile.skillmix_student,
        "--judge", profile.skillmix_judge,
        "--seed", str(profile.skillmix_seed),
        "--out", str(out),
        "-v",
    ]


def _build_solve(profile: PipelineProfile, run_dir: Path) -> Tuple[str, List[str]]:
    if not profile.solve_tasks.strip():
        raise ValueError("stage 'solve' requires profile.solve_tasks (path to ExtractedTask JSON)")
    out_dir = run_dir / "solve"
    args = [
        "--tasks", profile.solve_tasks,
        "--out-dir", str(out_dir),
        "--model", profile.solve_model,
        "--thinking-budget", str(profile.solve_thinking_budget),
        "--max-tokens", str(profile.solve_max_tokens),
        "-v",
    ]
    if profile.solve_limit > 0:
        args.extend(["--limit", str(profile.solve_limit)])
    return "s4.m4", args


def _build_bench(
    profile: PipelineProfile, run_dir: Path, catalog_path: str,
) -> Tuple[str, List[str]]:
    import json as _json   # local import: wirer is otherwise deps-free

    if not profile.bench_tasks.strip():
        raise ValueError("stage 'bench' requires profile.bench_tasks (path to ExtractedTask JSON)")
    out_dir = run_dir / "bench"
    args = [
        "--tasks", profile.bench_tasks,
        "--catalog", catalog_path,
        "--out-dir", str(out_dir),
        "--judge", profile.bench_judge,
        "--mode", profile.bench_mode,
        "-v",
    ]
    # Multi-student sweep wins; otherwise single student via bench_provider.
    models = [m.strip() for m in (profile.bench_models or []) if m.strip()]
    if models:
        args.extend(["--models", ",".join(models)])
    else:
        args.extend(["--provider", profile.bench_provider])
    if profile.bench_cross_task:
        args.append("--cross-task")
    if profile.bench_limit > 0:
        args.extend(["--limit", str(profile.bench_limit)])

    # Per-task skill injection. cross_task takes precedence in the harness;
    # skip wiring the map when both are set so the user's config is honored
    # predictably.
    if not profile.bench_cross_task:
        map_path: str = ""
        inline_map = profile.bench_task_skill_map or {}
        if inline_map:
            out_dir.mkdir(parents=True, exist_ok=True)
            map_path_obj = out_dir / "task_skill_map.json"
            map_path_obj.write_text(
                _json.dumps(inline_map, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            map_path = str(map_path_obj)
        elif profile.bench_task_skill_map_file.strip():
            map_path = profile.bench_task_skill_map_file.strip()
        if map_path:
            args.extend(["--task-skill-map", map_path])

    return f"__module__:{BENCH_MODULE_PATH}", args


def _build_bench_traced(
    profile: PipelineProfile, run_dir: Path, bench_catalog_path: str,
) -> Tuple[str, List[str]]:
    """Closes Gap A: skill-injected solver with native CoT capture.

    Reuses bench_tasks + bench_task_skill_map (so configs match the bench
    stage), but invokes s4.m4 in --mode inject so each episode carries the
    LinearAgent thinking blocks alongside the injected procedural Skill.
    """
    import json as _json   # local import: wirer is otherwise deps-free

    if not profile.bench_tasks.strip():
        raise ValueError(
            "stage 'bench_traced' requires profile.bench_tasks "
            "(reuses the same task source as the bench stage)"
        )

    out_dir = run_dir / "bench_traced"
    args = [
        "--mode", "inject",
        "--tasks", profile.bench_tasks,
        "--catalog", bench_catalog_path,
        "--out-dir", str(out_dir),
        "--model", profile.bench_traced_model,
        "--thinking-budget", str(profile.bench_traced_thinking_budget),
        "--max-tokens", str(profile.bench_traced_max_tokens),
        "-v",
    ]
    if profile.bench_traced_judge.strip():
        args.extend(["--judge", profile.bench_traced_judge])
    if profile.bench_traced_limit > 0:
        args.extend(["--limit", str(profile.bench_traced_limit)])

    # Reuse the same task->skill map as bench (inline dict wins over file).
    map_path: str = ""
    inline_map = profile.bench_task_skill_map or {}
    if inline_map:
        out_dir.mkdir(parents=True, exist_ok=True)
        map_path_obj = out_dir / "task_skill_map.json"
        map_path_obj.write_text(
            _json.dumps(inline_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        map_path = str(map_path_obj)
    elif profile.bench_task_skill_map_file.strip():
        map_path = profile.bench_task_skill_map_file.strip()
    if not map_path:
        raise ValueError(
            "stage 'bench_traced' requires bench_task_skill_map (inline dict) "
            "or bench_task_skill_map_file (path) — without it, every task "
            "would run baseline-only and the curated half is meaningless."
        )
    args.extend(["--task-skill-map", map_path])

    return "s4.m4", args


def _build_bench_catalog(
    profile: PipelineProfile,
    run_dir: Path,
    catalog_path: str,
) -> Tuple[str, List[str]]:
    """Rewrite declarative catalog → procedural catalog for injection."""
    out = stage_output_path(run_dir, "bench_catalog", "skills.json")
    args = [
        "--input-catalog", catalog_path,
        "--out", str(out),
        "--provider", profile.bench_catalog_provider,
        "-v",
    ]
    if profile.bench_catalog_limit > 0:
        args.extend(["--limit", str(profile.bench_catalog_limit)])
    filter_names = [s.strip() for s in (profile.bench_catalog_skills or []) if s.strip()]
    if filter_names:
        args.extend(["--skills", ",".join(filter_names)])
    return f"__module__:{BENCH_CATALOG_MODULE_PATH}", args


def _build_bench_viz(
    profile: PipelineProfile,
    run_dir: Path,
    stage_outputs: Dict[str, Dict[str, str]],
) -> Tuple[str, List[str]]:
    """Render heatmaps + summary charts from the bench stage's episodes.json."""
    episodes_path = stage_outputs.get("bench", {}).get(
        "episodes", str(stage_output_path(run_dir, "bench", "episodes.json"))
    )
    heatmaps_dir = run_dir / "bench" / "heatmaps"
    args = [
        "--results", episodes_path,
        "--output-dir", str(heatmaps_dir),
        "--type", "all",
        "--dpi", str(profile.bench_viz_dpi),
    ]
    return f"__module__:{BENCH_VIZ_MODULE_PATH}", args
