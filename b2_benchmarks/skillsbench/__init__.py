"""
b2_benchmarks.skillsbench

SkillsBench-style corpus evaluation: measure whether injecting a skill into
an LLM's system prompt improves its ability to answer an ExtractedTask
correctly. Outside the 14-method PROJECT_SPECS grid, but the only thing in
this repo that measures end-to-end whether extracted skills actually help.

Ported-and-trimmed from skillsuite/llm-skills.skillsbench-evaluation/. Dropped:
  - proof_verifier.py       (508 LOC; task-specific proof checking — out of scope)
  - self-generated condition (requires a separate generator-model config)
  - cross-repo providers    (replaced by core.providers: anthropic / mock only)

Public surface:
    LLMJudgeEvaluator        — LLM-as-judge scorer for FREE_FORM and deterministic queries
    format_skill_as_system   — system-prompt block carrying a single Skill
    run_singlecall_episode   — one-turn baseline or skill-injected episode
    run_guided_episode       — skill.procedure-guided multi-turn episode (fallback to singlecall if procedure empty)
    run_corpus_evaluation    — orchestrator across tasks + optional skills
    compute_pass_rate_delta  — bootstrap CI + permutation p-value between two conditions
    generate_* / generate_all — visualization charts (uplift / pass-rate / combined / win-loss / delta / scatter)
"""

from b2_benchmarks.skillsbench.corpus_harness import (
    CorpusEpisode,
    StepTrace,
    run_corpus_evaluation,
    run_guided_episode,
    run_singlecall_episode,
)
from b2_benchmarks.skillsbench.effectiveness import (
    bootstrap_ci,
    compute_pass_rate_delta,
    mean,
    pass_rate,
    pass_rate_delta_pp,
    permutation_test,
)
from b2_benchmarks.skillsbench.llm_judge import (
    JudgeResult,
    LLMJudgeEvaluator,
)
from b2_benchmarks.skillsbench.skill_injection import (
    DEFAULT_READING_COMPREHENSION_PROMPT,
    format_skill_as_system,
    get_default_system_prompt,
)
from b2_benchmarks.skillsbench.visualization import (
    build_task_model_uplift_matrix,
    generate_all,
    generate_baseline_vs_curated_scatter,
    generate_combined_heatmap,
    generate_combined_score_heatmap,
    generate_delta_by_mode_bar,
    generate_score_comparison_heatmap,
    generate_pass_rate_heatmap,
    generate_score_heatmap,
    generate_uplift_heatmap,
    generate_win_loss_bar,
    load_episodes,
)

__all__ = [
    "CorpusEpisode",
    "DEFAULT_READING_COMPREHENSION_PROMPT",
    "JudgeResult",
    "LLMJudgeEvaluator",
    "StepTrace",
    "bootstrap_ci",
    "build_task_model_uplift_matrix",
    "compute_pass_rate_delta",
    "format_skill_as_system",
    "generate_all",
    "generate_baseline_vs_curated_scatter",
    "generate_combined_heatmap",
    "generate_combined_score_heatmap",
    "generate_delta_by_mode_bar",
    "generate_score_comparison_heatmap",
    "generate_pass_rate_heatmap",
    "generate_score_heatmap",
    "generate_uplift_heatmap",
    "generate_win_loss_bar",
    "get_default_system_prompt",
    "load_episodes",
    "mean",
    "pass_rate",
    "pass_rate_delta_pp",
    "permutation_test",
    "run_corpus_evaluation",
    "run_guided_episode",
    "run_singlecall_episode",
]
