# cli

Pipeline orchestrator for the `llm-skills` project. Chains PROJECT_SPECS
methods — `s1.m1..s4.m4` plus `b2_benchmarks.skillsbench` — into one
subprocess-isolated, YAML-profile-driven run with crash recovery.

This sits alongside the simple method dispatcher at `../cli.py`:

    ../cli.py        -> python -m cli <method_id> <args>    (one method, one call)
    ./cli/main.py    -> python -m cli.main run ...          (chained stages, profiles)

## quickstart

```bash
cd /workspace/llm-skills/cli

# Full chain with the default profile
python3 -m cli.main run --stages all --clean

# Minimal run (2 examples, 1 batch) for sanity-checking wiring
python3 -m cli.main run --stages all --minimal --clean

# Extraction-only (catalog -> compose -> verify)
python3 -m cli.main run --stages extract --clean

# Override every LLM role to anthropic:<tier>
python3 -m cli.main run --stages all --anthropic opus

# Single stage
python3 -m cli.main run --stages catalog

# Preflight + interactive provider/model selection
python3 -m cli.main setup                          # updates default.yaml by default
python3 -m cli.main setup --non-interactive        # preflight + discovery only, no prompts
python3 -m cli.main setup --profile my-experiment  # update a named profile

# What completed in the run directory
python3 -m cli.main status --profile default
```

## stages

| stage_id      | method(s)                                              | output                                       |
|---------------|--------------------------------------------------------|----------------------------------------------|
| catalog       | s2.m1 (default) / s1.m2 / s1.m3                        | catalog/skills.json                          |
| compose       | s3.m1 (default) / s3.m2 / s3.m3                        | compose/examples.json                        |
| verify        | s3.m4                                                  | verify/verified-examples.json                |
| skillmix      | s2.m2                                                  | skillmix/trials.json                         |
| solve         | s4.m4  (requires `solve_tasks` in profile)             | solve/solutions.jsonl + traces.jsonl         |
| bench_catalog | b2_benchmarks.skillsbench.procedural_catalog           | bench_catalog/skills.json                    |
| bench         | b2_benchmarks.skillsbench  (requires `bench_tasks`)    | bench/episodes.json + summary.json           |
| bench_traced  | s4.m4 --mode inject  (skill-injected solver + CoT)     | bench_traced/episodes.jsonl + summary.json   |
| bench_viz     | b2_benchmarks.skillsbench.visualization (post-bench)   | bench/heatmaps/*.png                         |

Stage ranges accepted by `--stages`:

    all                       catalog, compose, verify, skillmix, solve, bench_catalog, bench, bench_traced, bench_viz
    extract | extraction      catalog, compose, verify
    eval | evaluation         skillmix, solve, bench_catalog, bench, bench_traced, bench_viz
    viz | visualization       bench_viz
    <id1>,<id2>               comma-separated subset
    <single_id>               just that stage

## providers

Three providers are wired into `core.providers.create_provider`:

| provider  | spec                              | source                                     |
|-----------|-----------------------------------|--------------------------------------------|
| anthropic | `anthropic:claude-opus-4-7`       | OAuth via `~/.claude/.credentials.json` (preferred), or `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_OAUTH_TOKEN` |
| ollama    | `ollama:llama3.1:8b`              | local Ollama daemon at `$OLLAMA_HOST` (default `http://localhost:11434`) |
| mock      | `mock:`                           | always available, returns canned text — for dry-runs |

Use `python -m cli.main setup` to interactively pick frontier and student
providers; it discovers reachable providers, lists installed Ollama models,
and writes the picks back into a profile.

## profiles

YAML files in `profiles/`. Every PipelineProfile field is a flat key;
`config show <name>` pretty-prints them. See `profiles/default.yaml` for the
full schema.

Provider spec shape everywhere: `"provider:model"`, e.g.
`"anthropic:claude-opus-4-7"`. Use `"mock:"` (note the trailing colon — quote
it in YAML) for dry runs against `core.providers.MockProvider`.

```bash
python3 -m cli.main config list                   # list saved profiles
python3 -m cli.main config show default           # pretty-print one
python3 -m cli.main config create my-experiment   # cp default.yaml -> my-experiment.yaml
python3 -m cli.main config delete my-experiment
```

## crash recovery

Each stage checks whether its declared output files exist before running. If
the pipeline fails mid-run, re-invoking the same `run` skips completed stages
and resumes from the first incomplete one. Use `--clean` to force a full
re-run, or `--clean-stages` to wipe only the requested stages' output.

## structure

    cli/
      cli/                    main.py + command_{run,config,status,setup}.py + rich_ui.py
      config/                 pipeline_stage.py, pipeline_profile.py, stage_registry.py
      tools/                  profile_loader.py, stage_runner.py, output_inspector.py, provider_checker.py
      orchestration/          pipeline_executor.py, stage_output_wirer.py
      profiles/               default.yaml + user-created YAMLs
      conftest.py             pytest bootstrap
      requirements.txt        optional extras (rich, InquirerPy, pyyaml)

## dependencies

Required: `anthropic`, `httpx`, `pyyaml`, `anthropic-oauth` (in-tree at
`/workspace/anthropic-oauth/` or pip-installed).

Optional: `rich` for colored terminal output (falls back to plain text if
missing), `InquirerPy` for arrow-key prompts (not wired for the current
non-interactive flow).

## standalone method dispatch

Individual methods can also be run without the orchestrator via the root
dispatcher:

    cd /workspace/llm-skills
    python -m cli s2.m1 --source data/wikipedia-seed/source.json --out skills.json
    python -m cli s3.m1 --catalog skills.json --n 5 --k 2
    python -m cli s3.m4 --examples examples.json --catalog skills.json --out-dir /tmp/verify
