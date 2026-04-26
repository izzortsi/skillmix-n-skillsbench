# llm-skills

Faithful implementation of [`PROJECT_SPECS.md`](PROJECT_SPECS.md). Every method
in the spec has a corresponding module; names are 1:1 so the correspondence is
obvious at a glance. 10 of 14 spec methods are FULL; the 4 remaining are OOS
(out of scope: training infrastructure or mechanistic interpretability).

## layout

```
llm-skills/
├── PROJECT_SPECS.md                           # authoritative spec (4 sections, 14 methods)
├── cli.py                                     # python -m cli <method>
├── core/                                      # single canonical schemas + utilities
│   ├── schemas.py       Skill, Topic, SkillExample, SkillMixTrial, ExtractedTask
│   ├── providers.py     AnthropicProvider (OAuth bridge), MockProvider
│   ├── judge.py         per_skill_rubric_judge, task_acceptance_judge
│   ├── formatters.py    skill/task <-> markdown converters
│   └── operators.py     seq / par / cond / sem composition (k-way)
├── harness/                                   # streaming LinearAgent + transcript capture
│   ├── agent.py         LinearAgent, OAuth wire path
│   ├── transcript.py    Transcript + record() for thinking-block capture
│   ├── events.py        14 typed stream events
│   └── ollama_agent.py  OllamaAgent (no tools, no thinking)
├── s1_extracting_skill_names/                 # §1 — Extracting Skills (Names)
│   ├── m1_task_based_labeling.py              [FULL]
│   ├── m2_direct_elicitation.py               [FULL]
│   └── m3_catalog_generation.py               [FULL]
├── s2_extracting_skills_from_text/            # §2 — Extracting Skills from Text
│   ├── m1_wikipedia_seeder.py                 [FULL]
│   ├── m2_skill_mix_evaluation.py             [FULL]  true Yu et al. replication
│   └── m3_mini_theories.py                    [OOS]   mechanistic interpretability
├── s3_generating_skill_examples/              # §3 — Generating Skill Examples
│   ├── m1_random_pair_composition.py          [FULL]  k=1..5, same-category supported
│   ├── m2_domain_specific_combination.py      [FULL]
│   ├── m3_synthetic_construction.py           [FULL]
│   └── m4_agentic_answer_verification.py      [FULL]  verify + refine
├── s4_extracting_skill_usage_instances/       # §4 — Skill Usage Instances from Text
│   ├── m1_context_enhanced_learning.py        [OOS]   training infra
│   ├── m2_curriculum_internalization.py       [OOS]   training infra
│   ├── m3_mechanistic_analysis.py             [OOS]   interp tooling
│   └── m4_skill_composition_testing.py        [FULL]  eval half (solver + trace adapter)
├── b2_benchmarks/skillsbench/                 # SkillsBench-style corpus evaluation
│   ├── corpus_harness.py     run_singlecall / run_guided episodes
│   ├── llm_judge.py          LLMJudgeEvaluator
│   ├── skill_injection.py    format_skill_as_system
│   └── effectiveness.py      bootstrap CI, permutation test, aggregations
├── data/
│   ├── wikipedia-seed/source.json             # 10 Yu et al. skills (paper Table 5)
│   ├── wikipedia-seed/skills.json             # seeded Skill records
│   └── topics/topics.json                     # 10 Yu et al. topics (paper Table 6)
├── b1.reports/                                # dated engineering reports
├── resources/                                 # papers + meta-analysis
└── .gitignore
```

Status legend:
- **FULL** — implemented end-to-end, runnable.
- **OOS** — out of scope. Needs infrastructure beyond inference (training
  harness, mechanistic interpretability tooling). Left as docstring stubs so
  the PROJECT_SPECS ↔ code mapping stays 1:1.

## single canonical schema

Every method imports dataclasses from `core/schemas.py`. Six types, deduped:

| Type              | Where produced              | Where consumed                          |
|-------------------|-----------------------------|-----------------------------------------|
| `Skill`           | s1.m1/m2/m3, s2.m1          | every generator + eval                  |
| `Topic`           | s2.m1 (from topics.json)    | s2.m2, s3.m3                            |
| `SkillExample`    | s3.m1/m2/m3                 | s3.m4 (verify/refine)                   |
| `SkillMixTrial`   | s2.m2                       | (terminal)                              |
| `ExtractedTask`   | s3 generators, b2 consumers | s4.m4, b2 skillsbench                   |
| *(helpers)*       | `stable_uid`, `to_kebab`, `save_json`, `load_*`, `validate_free_form_single_answer` |

Methods attribute their origin via the `source` field on `Skill` (e.g.
`"s2.m1.wikipedia-seed"`, `"s1.m3.catalog"`).

## providers

`core/providers.py` wraps the [anthropic-oauth](https://github.com/anthropics/anthropic-oauth)
package so API calls reuse Claude Code's local credentials at
`~/.claude/.credentials.json` (no extra setup; falls back to `ANTHROPIC_API_KEY`
and then `CLAUDE_CODE_OAUTH_TOKEN`). The `system` field is converted to the
two-block array the OAuth endpoint requires — without this, Opus/Sonnet
requests return HTTP 429 (see `b1.reports/260421.frontier-extraction-pipeline.txt`
ADDENDUM A for the full story).

`MockProvider` is available for tests that must not hit the network.

For streaming + thinking-block capture, use `harness.LinearAgent` instead —
same OAuth path, but yields typed events (`ThinkingStart/Delta/Stop`,
`ToolUseStart/InputDelta/Stop`, etc.) through `record()` into a `Transcript`.
`s4.m4` is the only method that uses it directly; everything else goes
through `core.providers`.

## quickstart

Run from the repo root (data paths are relative):

```bash
cd /workspace/llm-skills

# List all methods + status
python -m cli

# Seed 10 Yu et al. skills (Wikipedia-style)
python -m cli s2.m1

# Yu Skill-Mix at k=2 with Sonnet as student, Opus as judge
python -m cli s2.m2 --k 2 --trials 5 --verbose

# Dry-run with mock providers (no API calls)
python -m cli s2.m2 --student mock: --judge mock: --k 2 --trials 2 --verbose

# Generate k=3 same-category Q&A pairs from the Wikipedia seed
python -m cli s3.m1 --k 3 --same-category --n 10

# Agentic verify + refine the s3.m1 output
python -m cli s3.m4 --examples data/s3-m1-examples.json \
                    --catalog data/wikipedia-seed/skills.json \
                    --out-dir data/s3-m4-out

# Generate a ~200-skill catalog with 3-way concurrency
python -m cli s1.m3 --target-size 200 --categories 10 --concurrency 3
```

## composition operators

`core/operators.py` provides `compose_seq`, `compose_par`, `compose_cond`, and
`compose_sem` (LLM-fused) — non-PROJECT_SPECS utilities that turn a list of
`Skill` records into a `ComposedSkill`. `generate_all_compositions(skills,
max_k)` enumerates every `itertools.combinations` k-tuple and builds seq/par/
cond versions of each.

## b2_benchmarks/skillsbench

Outside the spec grid but the only end-to-end measurement of "do extracted
skills actually improve a solver model?":

```bash
python -m b2_benchmarks.skillsbench.corpus_harness \
    --tasks     data/holdout-tasks.json \
    --catalog   data/wikipedia-seed/skills.json \
    --out-dir   data/skillsbench-out \
    --provider  anthropic:claude-haiku-4-5-20251001 \
    --judge     anthropic:claude-opus-4-7 \
    --mode      singlecall --cross-task
```

Outputs `episodes.json` and `summary.json` (baseline vs curated pass rate +
bootstrap CI + permutation p-value).

## out of scope (OOS)

The spec lists four methods this repo cannot build:

| id    | name                                    | reason |
|-------|-----------------------------------------|--------|
| s2.m3 | Mini-theories / context analysis        | Needs mechanistic interpretability tooling (white-box activations, circuit probes). |
| s4.m1 | Context-enhanced learning               | Needs SFT training harness + GPU. |
| s4.m2 | Curriculum internalization              | Same as s4.m1. |
| s4.m3 | Mechanistic analysis of skill storage   | Same as s2.m3. |

Each OOS module is a docstring stub explaining what the method would do and
what infrastructure it needs, so the PROJECT_SPECS ↔ code map stays complete.
