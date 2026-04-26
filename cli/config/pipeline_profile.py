"""
pipeline_profile.py

Experiment profile: every value the orchestrator needs to run a pipeline
end-to-end. Serializable to/from YAML. Each stage reads a coherent subset of
fields named `<stage_id>_*`.

Provider spec shape everywhere: "provider:model", e.g. "anthropic:claude-opus-4-7".
Use "mock:" (no model) for dry runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
DEFAULT_HAIKU = "claude-haiku-4-5-20251001"
DEFAULT_SONNET = "claude-sonnet-4-6"


# Known method ids per stage — the orchestrator validates profile values
# against these at runtime so a typo is caught before the subprocess launches.
CATALOG_METHODS = ("s1.m2", "s1.m3", "s2.m1")
COMPOSE_METHODS = ("s3.m1", "s3.m2", "s3.m3")


@dataclass
class PipelineProfile:
    # identity
    profile_name: str = "default"
    run_dir: str = "data/pipeline-runs/default"     # relative to repo root

    # ---- Stage: catalog (build Skill catalog) ------------------------------
    catalog_method: str = "s2.m1"                    # s1.m2 | s1.m3 | s2.m1
    catalog_provider: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"

    # s2.m1 (wikipedia seed)
    catalog_source: str = "data/wikipedia-seed/source.json"

    # s1.m2 (novel elicitation)
    catalog_n_roots: int = 5
    catalog_max_subskills: int = 4
    catalog_root_batch_size: int = 5

    # s1.m3 (generated catalog)
    catalog_target_size: int = 100
    catalog_categories: int = 10
    catalog_batch_size: int = 15
    catalog_max_batches_per_category: int = 5
    catalog_concurrency: int = 1

    # ---- Stage: compose (skill-tuple examples) -----------------------------
    compose_method: str = "s3.m1"                    # s3.m1 | s3.m2 | s3.m3
    compose_provider: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    compose_n: int = 20
    compose_k: int = 2
    compose_seed: int = 42
    compose_same_category: bool = False
    compose_categories: List[str] = field(default_factory=list)     # pool filter; empty = all
    # s3.m2 domain (required when compose_method == s3.m2)
    compose_domain: str = ""
    # s3.m3 fiction (topics file + sentence limit)
    compose_topics: str = "data/topics/topics.json"
    compose_sentence_limit: int = 2

    # ---- Stage: verify (agentic verify+refine) ----------------------------
    verify_verifier: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    verify_refiner: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    verify_max_rounds: int = 2
    verify_limit: int = 0                            # 0 = all

    # ---- Stage: skillmix (Yu et al. Skill-Mix) ----------------------------
    skillmix_student: str = f"anthropic:{DEFAULT_SONNET}"
    skillmix_judge: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    skillmix_k: int = 2
    skillmix_trials: int = 10
    skillmix_sentence_limit: int = 2
    skillmix_topics: str = "data/topics/topics.json"
    skillmix_seed: int = 42

    # ---- Stage: solve (s4.m4 exact-match evaluator) -----------------------
    # Tasks are supplied externally (not produced in the chain).
    solve_tasks: str = ""                            # required to run this stage
    solve_model: str = DEFAULT_ANTHROPIC_MODEL       # LinearAgent model name
    solve_thinking_budget: int = 4096
    solve_max_tokens: int = 8000
    solve_limit: int = 0

    # ---- Stage: bench_catalog (declarative -> procedural Skill rewrite) ---
    # Reads catalog/skills.json, writes bench_catalog/skills.json with
    # procedure / constraints / detection-framed when_to_use / dual example.
    # This is what `bench` actually injects; skipping or mis-configuring this
    # stage silently degrades bench quality (injection becomes few-shot style).
    bench_catalog_provider: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"  # Opus by default
    bench_catalog_limit: int = 0                     # 0 = rewrite every skill in the input
    bench_catalog_skills: List[str] = field(default_factory=list)
    # ^ if non-empty, only these skill names are rewritten; others are skipped

    # ---- Stage: bench (SkillsBench corpus eval) ---------------------------
    bench_tasks: str = ""                            # required to run this stage
    bench_provider: str = f"anthropic:{DEFAULT_HAIKU}"
    # Multi-student sweep: when non-empty, every spec in the list is benchmarked
    # and all episodes land in one file (tagged with .model). bench_provider is
    # then ignored. Example in default.yaml comments.
    bench_models: List[str] = field(default_factory=list)
    bench_judge: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    bench_mode: str = "singlecall"                   # singlecall | guided
    # Per-task skill injection: {task_uid: skill_name}. Each listed task runs
    # baseline AND one curated episode with its mapped skill. Unlisted tasks
    # run baseline only. Ignored when bench_cross_task is true.
    bench_task_skill_map: dict = field(default_factory=dict)
    # Optional: path to an external JSON of the same shape. Used only when
    # bench_task_skill_map (inline dict) is empty.
    bench_task_skill_map_file: str = ""
    bench_cross_task: bool = False
    bench_limit: int = 0

    # ---- Stage: bench_traced (skill-injected solver with native CoT capture) ----
    # Closes Gap A: per task, run baseline + curated episodes through the
    # s4.m4 LinearAgent solver (thinking enabled), inject the procedural Skill
    # from bench_catalog into the curated system prompt. Each episode carries
    # both procedural_steps (the model's own CoT) and the injected skill —
    # exactly the (CoT, injected-skill) pair an SFT pipeline needs.
    #
    # Reuses bench_tasks + bench_task_skill_map (same task universe / mapping
    # as the bench stage) so configs stay consistent across the two evals.
    bench_traced_model: str = DEFAULT_ANTHROPIC_MODEL
    bench_traced_thinking_budget: int = 4096
    bench_traced_max_tokens: int = 8000
    bench_traced_judge: str = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    bench_traced_limit: int = 0                      # 0 = all tasks

    # ---- Stage: bench_viz (heatmaps + summary charts from bench output) --
    bench_viz_dpi: int = 150


MINIMAL_OVERRIDES = {
    "catalog_target_size": 10,
    "catalog_categories": 3,
    "catalog_batch_size": 5,
    "catalog_max_batches_per_category": 1,
    "catalog_concurrency": 1,
    "catalog_n_roots": 2,
    "catalog_max_subskills": 2,
    "compose_n": 3,
    "compose_k": 2,
    "verify_max_rounds": 1,
    "verify_limit": 3,
    "skillmix_trials": 2,
    "solve_limit": 2,
    "bench_limit": 2,
}


def apply_minimal(profile: PipelineProfile) -> PipelineProfile:
    """Minimum-API-calls overrides — useful for smoke runs."""
    for key, value in MINIMAL_OVERRIDES.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    return profile


def validate(profile: PipelineProfile) -> List[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errs: List[str] = []
    if profile.catalog_method not in CATALOG_METHODS:
        errs.append(
            f"catalog_method={profile.catalog_method!r} must be one of {CATALOG_METHODS}"
        )
    if profile.compose_method not in COMPOSE_METHODS:
        errs.append(
            f"compose_method={profile.compose_method!r} must be one of {COMPOSE_METHODS}"
        )
    if profile.compose_method == "s3.m2" and not profile.compose_domain.strip():
        errs.append("compose_method=s3.m2 requires compose_domain (a Skill.category string)")
    if profile.bench_mode not in ("singlecall", "guided"):
        errs.append(f"bench_mode={profile.bench_mode!r} must be 'singlecall' or 'guided'")
    return errs
