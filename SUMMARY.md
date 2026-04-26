# Session Summary — 2026-04-25

## Topic
Gap-analysis of the four primitive capabilities (Extract / Name / Describe / Show task and CoT trace using skill), then implementation of the missing piece (Gap A).

## Gap Analysis (resolved)

| Capability                          | Status   | Where                                                                                         |
|-------------------------------------|----------|-----------------------------------------------------------------------------------------------|
| Extract skills                      | DONE     | `s1.m1`, `s1.m2`, `s1.m3`, `s2.m1` (Wikipedia-seed default)                                   |
| Name skills                         | DONE     | snake_case (s1.m1) / kebab-case (s1.m2/m3/s2.m1); `Skill.skill_uid` stable across stages      |
| Describe skill                      | DONE     | declarative (`catalog/skills.json`) + procedural (`bench_catalog/skills.json` via Opus rewrite) |
| Show task and CoT trace using skill | **PARTIAL → now DONE** | s4.m4 had CoT, bench had injection, but they didn't cross. Closed in this session. |

## Pipeline status report
Written to `b1.reports/260425.pipeline-status.txt`. 8 stages, dual-catalog architecture, dual evaluation modes (Skill-Mix vs SkillsBench), achieved + remaining work.

## Gap A — implementation

### Design
Extended `s4_extracting_skill_usage_instances/m4_skill_composition_testing.py` rather than adding a new method (PROJECT_SPECS stays at 14 methods). Trace plumbing was already there; only needed skill-injection wiring.

### Changes
- `solve_one(agent, task, skill: Optional[Skill] = None)` — when `skill` is provided, drops the redundant "Required skills:" line from the user prompt and tags the output with `condition="curated"` + `injected_skill_name`/`injected_skill_uid`. Caller is responsible for constructing `agent` with the right system prompt (via `format_skill_as_system`).
- `ReasoningTrace` gained `condition`, `injected_skill_name`, `injected_skill_uid`.
- `solve_with_injection(tasks, skill_map, judge, ...)` — new top-level driver. Per task: baseline episode (always) + curated episode (if task is in `skill_map`). Both judged; both adapted to traces; both emitted as one episode record carrying solution + trace + judge verdict.
- `--mode inject` CLI subcommand on s4.m4: `--catalog`, `--task-skill-map`, `--judge` (defaults to exact-match if absent). Outputs `episodes.jsonl` + `summary.json` + `failures.jsonl`.
- New orchestrator stage `bench_traced`:
  - `cli/config/stage_registry.py` — registered, depends_on `bench_catalog`, output_files `episodes.jsonl + summary.json`. Added to `eval` range alias.
  - `cli/config/pipeline_profile.py` — fields `bench_traced_model`, `_thinking_budget`, `_max_tokens`, `_judge`, `_limit`. Reuses `bench_tasks` + `bench_task_skill_map`.
  - `cli/orchestration/stage_output_wirer.py` — `_build_bench_traced` constructs the `s4.m4 --mode inject` command.
  - `cli/profiles/default.yaml` — populated with defaults.
  - `cli/README.md` — stage table updated.

### Smoke test results
Ran on `t1` (modus-ponens) and `t8` (spatial-reasoning) with limit=1 each, judge=anthropic:claude-opus-4-7.

**Structural plumbing: works.**
- 2 episodes emitted per task (baseline + curated)
- `condition`, `injected_skill_name`, `injected_skill_uid` correctly tagged on both episode and embedded trace
- System prompt sizes confirm injection: baseline 197 chars vs curated 2117 chars
- Judge verdict (passed / score / conclusion_reached / rationale) captured

**Caveat: Opus emits no thinking blocks on these tasks.**
- `thinking: []` and `output_tokens=11-13` per call
- LinearAgent's request *does* enable thinking (`thinking: {type: enabled, budget_tokens: 4096}`), but Opus 4.7 chooses not to use it — likely because:
  1. `SOLVER_SYSTEM_PROMPT` says "give your final answer as **one concise** conclusion" → model reads this as "be terse, skip thinking"
  2. OAuth path prepends `CLAUDE_CODE_IDENTITY` ("You are Claude Code...") which primes CLI-style terseness
- Trace falls back to `_split_into_steps(response_text)` → `procedural_steps = ["ANSWER: yes"]`. Structurally valid, semantically useless for SFT distillation.

## Open question (next decision)

To make `bench_traced` produce SFT-quality traces, two candidate paths:

| Option | Action | Risk |
|--------|--------|------|
| (a)    | Add an injection-mode-specific solver prompt that explicitly requests step-by-step reasoning before the ANSWER line. Loosens the "concise" framing. | Low; targeted change. |
| (b)    | Test with API-key auth (no `CLAUDE_CODE_IDENTITY` prefix) to isolate whether OAuth identity is the cause. | Diagnostic only — doesn't change runtime. |

User has not yet chosen. Alternative: park Gap A as "structurally complete" and move to **Gap B (SFT dataset stage)** scoping.

## Files touched
- `s4_extracting_skill_usage_instances/m4_skill_composition_testing.py` — major (+~280 lines)
- `cli/config/pipeline_profile.py` — added `bench_traced_*` fields
- `cli/config/stage_registry.py` — added `bench_traced` stage
- `cli/orchestration/stage_output_wirer.py` — added `_build_bench_traced` builder + outputs
- `cli/profiles/default.yaml` — populated bench_traced defaults
- `cli/README.md` — stage table
- `b1.reports/260425.pipeline-status.txt` — overview report (created earlier this session)
