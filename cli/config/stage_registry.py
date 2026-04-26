"""
stage_registry.py

Registry of pipeline stages aligned with the new 14-method PROJECT_SPECS layout
at `/workspace/llm-skills/`. Each stage delegates to one PROJECT_SPECS method
(chosen via the profile), invoked as `python -m cli <method_id> <args>` from
the repo root.

Stage chain (defaults when `--stages all`):

    catalog  -> compose  -> verify  -> skillmix
                                     |-> solve         (needs external tasks)
                                     |-> bench_catalog -> bench
                                                       |-> bench_traced
                                                       |-> bench_viz

`solve`, `bench`, and `bench_traced` consume ExtractedTasks not produced by
our in-chain methods, so they don't list `compose` as a dependency — drive
them by passing --tasks explicitly in the profile. `bench_traced` reuses
`bench_tasks` and `bench_task_skill_map` and reads procedural skills from
`bench_catalog`.
"""

from config.pipeline_stage import PipelineStage


STAGES = [
    PipelineStage(
        stage_id="catalog",
        name="build-catalog",
        description="Build a Skill catalog (seeded, elicited, or LLM-generated)",
        pipeline_dir="",                             # run from repo root
        commands=[],                                 # method_id resolved from profile.catalog_method
        output_dir="catalog",
        output_files=["skills.json"],
        depends_on=[],
    ),
    PipelineStage(
        stage_id="compose",
        name="compose-examples",
        description="Generate (question, answer) examples exhibiting k-tuples of skills",
        pipeline_dir="",
        commands=[],                                 # profile.compose_method
        output_dir="compose",
        output_files=["examples.json"],
        depends_on=["catalog"],
    ),
    PipelineStage(
        stage_id="verify",
        name="verify-examples",
        description="Agentic verification + refinement of example answers (s3.m4)",
        pipeline_dir="",
        commands=["s3.m4"],
        output_dir="verify",
        output_files=["verified-examples.json"],
        depends_on=["compose", "catalog"],
    ),
    PipelineStage(
        stage_id="skillmix",
        name="skillmix-eval",
        description="Yu et al. Skill-Mix evaluation using the catalog (s2.m2)",
        pipeline_dir="",
        commands=["s2.m2"],
        output_dir="skillmix",
        output_files=["trials.json"],
        depends_on=["catalog"],
    ),
    PipelineStage(
        stage_id="solve",
        name="solve-tasks",
        description="Solve ExtractedTasks with thinking + exact-match verification (s4.m4)",
        pipeline_dir="",
        commands=["s4.m4"],
        output_dir="solve",
        output_files=["solutions.jsonl", "traces.jsonl", "summary.json"],
        depends_on=[],                               # needs external tasks.json via profile.solve_tasks
    ),
    PipelineStage(
        stage_id="bench_catalog",
        name="bench-catalog",
        description="Rewrite the declarative catalog into procedural Skills for injection",
        pipeline_dir="",
        commands=[],                                 # see wirer -> procedural_catalog module
        output_dir="bench_catalog",
        output_files=["skills.json"],
        depends_on=["catalog"],
    ),
    PipelineStage(
        stage_id="bench",
        name="skillsbench-eval",
        description="SkillsBench corpus evaluation (baseline vs curated pass rate)",
        pipeline_dir="",
        commands=[],                                 # invoked directly via module path, see wirer
        output_dir="bench",
        output_files=["episodes.json", "summary.json"],
        depends_on=["bench_catalog"],                # reads procedural skills, not raw catalog
    ),
    PipelineStage(
        stage_id="bench_traced",
        name="bench-traced",
        description="Skill-injected solver with native CoT capture (closes Gap A; emits SFT-ready episodes)",
        pipeline_dir="",
        commands=["s4.m4"],                          # dispatched via cli with --mode inject
        output_dir="bench_traced",
        output_files=["episodes.jsonl", "summary.json"],
        depends_on=["bench_catalog"],                # reads procedural skills + reuses bench_task_skill_map
    ),
    PipelineStage(
        stage_id="bench_viz",
        name="skillsbench-viz",
        description="Generate heatmaps + summary charts from bench episodes",
        pipeline_dir="",
        commands=[],                                 # b2_benchmarks.skillsbench.visualization via wirer
        output_dir="bench/heatmaps",
        output_files=[
            "uplift_heatmap.png",
            "baseline_pass_rate.png",
            "baseline_score_heatmap.png",
            "curated_score_heatmap.png",
            "combined_heatmap.png",
            "combined_score_heatmap.png",
            "score_comparison_heatmap.png",
            "win_loss_bar.png",
            "delta_by_mode_bar.png",
            "baseline_vs_curated.png",
        ],
        depends_on=["bench"],
    ),
]


STAGE_MAP = {stage.stage_id: stage for stage in STAGES}
ALL_STAGE_IDS = [stage.stage_id for stage in STAGES]


def get_stage(stage_id: str) -> PipelineStage:
    """Look up a stage by stage_id. Raises KeyError if not found."""
    if stage_id not in STAGE_MAP:
        raise KeyError(
            f"Unknown stage_id {stage_id!r}. Valid IDs: {', '.join(ALL_STAGE_IDS)}"
        )
    return STAGE_MAP[stage_id]


def parse_stage_range(range_str: str) -> list:
    """Parse a stage range string into an ordered list of stage IDs.

    Accepted formats:
        "all"                     -> ALL_STAGE_IDS
        "extract" | "extraction"  -> ["catalog", "compose", "verify"]
        "eval" | "evaluation"     -> ["skillmix", "solve", "bench_catalog", "bench", "bench_viz"]
        "viz" | "visualization"   -> ["bench_viz"]
        "catalog,compose"         -> comma list
        "catalog"                 -> single
    """
    r = range_str.strip().lower()

    if r == "all":
        return list(ALL_STAGE_IDS)
    if r in ("extract", "extraction"):
        return ["catalog", "compose", "verify"]
    if r in ("eval", "evaluation"):
        return ["skillmix", "solve", "bench_catalog", "bench", "bench_traced", "bench_viz"]
    if r in ("viz", "visualization"):
        return ["bench_viz"]

    if "," in r:
        ids = [s.strip() for s in r.split(",") if s.strip()]
        for sid in ids:
            if sid not in STAGE_MAP:
                raise ValueError(
                    f"Unknown stage id {sid!r} in range {range_str!r}. "
                    f"Valid: {', '.join(ALL_STAGE_IDS)}"
                )
        return ids

    if r in STAGE_MAP:
        return [r]

    raise ValueError(
        f"Cannot parse stage range {range_str!r}. "
        f"Use: all, extract, eval, a comma list, or a single id "
        f"({', '.join(ALL_STAGE_IDS)})."
    )
