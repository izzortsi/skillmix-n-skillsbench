# skillmix-n-skillsbench

Skill extraction, evaluation, and SFT-corpus generation for LLMs. Implements
the 14-method [`PROJECT_SPECS.md`](PROJECT_SPECS.md) plus the downstream
pieces — procedural-skill rewrite, task synthesis, bench harness, SFT row
emission, training scripts — needed to take a base student model from "no
SFT" to "trained on Opus demonstrations and re-evaluated on held-out tasks."

11 of 14 spec methods are FULL; 3 are OOS (training infrastructure or
mechanistic interpretability). One non-spec method (`s3.m5`) was added to
bridge `s3` example output to the bench harness's task input format.

## two skill paradigms

The codebase carries both, deliberately. They are different objects with the
same name and produce different evaluations.

| | Arora's *skill* | SkillsBench's *skill* |
|---|---|---|
| Where it lives | Latent in model weights | External text block in system prompt |
| Access | Reference by name (`modus_ponens`) | Inject procedural text (when_to_use / procedure / constraints / example) |
| Test | Compositional generation (k random skills → coherent text) | Task pass-rate Δ between baseline (no skill in system) and curated (skill block injected) |
| In this repo | `data/pipeline-runs/.../catalog/skills.json` (declarative) | `data/pipeline-runs/.../bench_catalog/skills.json` (procedural) |
| Used by | `s2.m2` Skill-Mix evaluator | `b2_benchmarks/skillsbench/` + `s4.m4 --mode inject` (SFT path) |

The **declarative catalog** is what `s2.m1` (Wikipedia seeder) and `s1.m1-m3`
produce. The **procedural catalog** is the same set of skills rewritten by
`b2_benchmarks/skillsbench/procedural_catalog.py` via Opus, adding numbered
procedures, constraints, and dual positive/negative examples.

For SFT, the procedural form is what gets injected into the student's system
prompt during corpus generation; the declarative form is what `s2.m2` uses
to probe latent skills via composition.

## pipeline (end-to-end)

```
              ┌──────────────────────────────┐
              │  data/wikipedia-seed/        │  curated source list
              │  source.json (40 entries)    │  (or s1.m1/m2/m3 output)
              └──────────────┬───────────────┘
                             │ s2.m1 wikipedia_seeder
                             ▼
              ┌──────────────────────────────┐
              │  catalog/skills.json         │  Skill (declarative form)
              │  40 records                  │  → consumed by s2.m2 Skill-Mix
              └──────────────┬───────────────┘
                             │ b2_benchmarks/skillsbench/procedural_catalog.py
                             │ (Opus rewrites each skill into procedural form)
                             ▼
              ┌──────────────────────────────┐
              │  bench_catalog/skills.json   │  Skill (procedural form)
              │  40 records                  │  when_to_use + procedure + constraints
              └──────────────┬───────────────┘
                             │ s3.m5 task_synthesis
                             │ (Opus generates N tasks per skill)
                             ▼
              ┌──────────────────────────────┐
              │  synthesis/tasks.json        │  ExtractedTask, tagged with
              │  N×40 records                │  skill_uid in acceptance_criteria
              │  + task_skill_map.json       │
              └──────────────┬───────────────┘
                             │ split: 10/skill train, 5/skill eval
                             ▼
                   ┌─────────┴─────────┐
                   ▼                   ▼
              ┌─────────┐         ┌─────────┐
              │  train  │         │  eval   │
              │  set    │         │  set    │
              └────┬────┘         └────┬────┘
                   │                   │
   s4.m4 --mode    │                   │ corpus_harness or
   inject (Opus)   │                   │ training/eval_qwen35_lora.py
                   ▼                   ▼
              ┌─────────┐         ┌──────────────┐
              │ bench_  │         │  bench-eval- │
              │ traced/ │         │  pre-sft/    │  ← multi-student
              │ episodes│         │  episodes    │    characterization
              └────┬────┘         └──────────────┘
                   │ sft_dataset filter
                   ▼
              ┌─────────────────────────────┐
              │ sft_dataset/dataset.jsonl   │  curated-passing chat-format rows
              └────┬────────────────────────┘
                   │ training/train_qwen35_lora.py
                   ▼
              ┌─────────────────────────────┐
              │  qwen35-0.8b-skill-lora/    │  LoRA adapter
              └────┬────────────────────────┘
                   │ training/eval_qwen35_lora.py (post-SFT)
                   ▼
              ┌─────────────────────────────┐
              │  bench-eval-post-sft/       │  same shape as pre-SFT
              │  episodes                   │  → compare_pre_post.py
              └─────────────────────────────┘
```

## layout

```
skillmix-n-skillsbench/
├── PROJECT_SPECS.md                           authoritative 14-method spec
├── cli.py                                     python cli.py <method>
├── core/
│   ├── schemas.py       Skill, ExtractedTask, SkillExample, SkillMixTrial
│   ├── providers.py     AnthropicProvider (OAuth bridge), Ollama, Mock
│   ├── judge.py         per_skill_rubric_judge
│   ├── formatters.py    skill/task <-> markdown converters
│   └── operators.py     k-way composition operators
├── harness/                                   streaming LinearAgent + transcript
│   ├── agent.py         LinearAgent with thinking-block capture
│   ├── transcript.py    Transcript + record()
│   └── ollama_agent.py  OllamaAgent
│
│ ── PROJECT_SPECS methods ───────────────────────────────
├── s1_extracting_skill_names/
│   ├── m1_task_based_labeling.py              FULL
│   ├── m2_direct_elicitation.py               FULL
│   └── m3_catalog_generation.py               FULL
├── s2_extracting_skills_from_text/
│   ├── m1_wikipedia_seeder.py                 FULL
│   ├── m2_skill_mix_evaluation.py             FULL  Yu et al. replication
│   └── m3_mini_theories.py                    OOS   needs interp tooling
├── s3_generating_skill_examples/
│   ├── m1_random_pair_composition.py          FULL  k=1..5
│   ├── m2_domain_specific_combination.py      FULL
│   ├── m3_synthetic_construction.py           FULL
│   ├── m4_agentic_answer_verification.py      FULL  verify + refine
│   └── m5_extracted_task_synthesis.py         NEW  bridges procedural skill → ExtractedTask
├── s4_extracting_skill_usage_instances/
│   ├── m1_context_enhanced_learning.py        OOS   training infra
│   ├── m2_curriculum_internalization.py       OOS   training infra
│   ├── m3_mechanistic_analysis.py             OOS   interp tooling
│   └── m4_skill_composition_testing.py        FULL  --mode solve | --mode inject
│
│ ── beyond the spec ─────────────────────────────────────
├── b2_benchmarks/skillsbench/
│   ├── corpus_harness.py     run_singlecall / run_guided episodes
│   ├── procedural_catalog.py declarative → procedural rewrite (Opus)
│   ├── skill_injection.py    format_skill_as_system
│   ├── llm_judge.py          LLMJudgeEvaluator (deterministic + LLM judge paths)
│   ├── sft_dataset.py        bench_traced episodes → SFT chat-format rows
│   ├── effectiveness.py      bootstrap CI, permutation test
│   ├── visualization.py      heatmaps + summary charts
│   └── rescore.py            post-hoc verdict re-derivation
├── training/
│   ├── train_qwen35_lora.py        LoRA SFT (HF transformers + TRL + peft)
│   ├── eval_qwen35_lora.py         post-SFT eval, emits bench-shape episodes.json
│   ├── patch_qwen35_template.py    chat-template patcher for assistant_only_loss
│   └── compare_pre_post.py         per-skill diff between pre/post-SFT eval
├── cli/                                       orchestrator (older, separate)
│   ├── cli/             command_run, command_setup, etc.
│   ├── config/          PipelineProfile, stage_registry
│   ├── orchestration/   stage_output_wirer, pipeline_executor
│   └── profiles/        default.yaml
├── data/
│   ├── wikipedia-seed/source.json             40 named language skills (curated)
│   ├── topics/topics.json                     10 topics (Yu et al. Table 6)
│   ├── mini-tasks.json                        8 hand-written ExtractedTasks
│   └── pipeline-runs/default/                 catalog/, bench_catalog/, synthesis/,
│                                              bench_traced/, sft_dataset/, bench-eval-*
├── b1.reports/                                dated engineering reports
├── resources/                                 papers + meta-analysis
└── PROJECT_SPECS.md
```

## quickstart

```bash
# List all PROJECT_SPECS methods + status
python cli.py

# Seed 40 Wikipedia-style skills into catalog/skills.json
python cli.py s2.m1 --source data/wikipedia-seed/source.json \
                    --out data/pipeline-runs/default/catalog/skills.json

# Rewrite catalog → procedural form (~$5, ~10 min on 40 skills)
python -m b2_benchmarks.skillsbench.procedural_catalog \
  --input-catalog data/pipeline-runs/default/catalog/skills.json \
  --out           data/pipeline-runs/default/bench_catalog/skills.json \
  --provider anthropic:claude-opus-4-7

# Synthesize ExtractedTasks from procedural skills (15 per skill = 600 total)
python cli.py s3.m5 \
  --catalog data/pipeline-runs/default/bench_catalog/skills.json \
  --out     data/pipeline-runs/default/synthesis/tasks.json \
  --n 15 --provider anthropic:claude-opus-4-7

# Run Skill-Mix evaluation (Yu et al.) — uses declarative catalog
python cli.py s2.m2 --skills data/pipeline-runs/default/catalog/skills.json \
                    --topics data/topics/topics.json \
                    --k 2 --trials 5 \
                    --student anthropic:claude-sonnet-4-6 \
                    --judge   anthropic:claude-opus-4-7

# Dry-run anything with mocks (no API calls)
python cli.py s2.m2 --student mock: --judge mock: --k 2 --trials 2
```

## SFT pipeline (the longer story)

For the full chain `procedural-catalog → tasks → bench_traced → SFT corpus →
LoRA-trained student → post-SFT eval`, see
[`b1.reports/260427.sft-v1-experiment-specification.txt`](b1.reports/260427.sft-v1-experiment-specification.txt)
which formally specifies every stage as `(state, inputs, outputs, metrics,
transitions)` and reports SFT v1 outcomes.

Concrete commands:

```bash
# Generate the SFT corpus (Opus solver + judge over 400 train tasks)
python cli.py s4.m4 --mode inject \
  --tasks    data/pipeline-runs/default/synthesis/tasks_train.json \
  --catalog  data/pipeline-runs/default/bench_catalog/skills.json \
  --task-skill-map data/pipeline-runs/default/synthesis/task_skill_map_train.json \
  --out-dir  data/pipeline-runs/default/bench_traced/ \
  --judge    anthropic:claude-opus-4-7 \
  --model    claude-opus-4-7 --thinking-budget 4096

# Filter passing curated episodes into SFT chat-format rows
python -m b2_benchmarks.skillsbench.sft_dataset \
  --episodes data/pipeline-runs/default/bench_traced/episodes.jsonl \
  --out      data/pipeline-runs/default/sft_dataset/dataset.jsonl \
  --conditions curated --skip-skills <ceiling-skills>

# Patch the chat template so TRL ≥0.18 can mask loss to assistant tokens only
python training/patch_qwen35_template.py \
  --model Qwen/Qwen3.5-0.8B \
  --out   ./qwen35-0.8b-tokenizer-patched

# LoRA fine-tune on host with GPU
python training/train_qwen35_lora.py \
  --data           data/pipeline-runs/default/sft_dataset/dataset.jsonl \
  --model          Qwen/Qwen3.5-0.8B \
  --tokenizer-path ./qwen35-0.8b-tokenizer-patched \
  --output-dir     ./qwen35-0.8b-skill-lora

# Eval on the held-out 200 tasks; produces bench-shape episodes.json
python training/eval_qwen35_lora.py \
  --base           Qwen/Qwen3.5-0.8B \
  --tokenizer-path ./qwen35-0.8b-tokenizer-patched \
  --adapter        ./qwen35-0.8b-skill-lora \
  --tasks          data/pipeline-runs/default/synthesis/tasks_eval.json \
  --catalog        data/pipeline-runs/default/bench_catalog/skills.json \
  --task-skill-map data/pipeline-runs/default/synthesis/task_skill_map_eval.json \
  --out-dir        data/pipeline-runs/default/bench-eval-post-sft \
  --judge anthropic:claude-opus-4-7

# Per-skill diff vs pre-SFT
python training/compare_pre_post.py
```

## canonical schemas

Every method imports dataclasses from `core/schemas.py`. The five user-facing
types:

| Type            | Where produced              | Where consumed                       |
|-----------------|-----------------------------|--------------------------------------|
| `Skill`         | s1.m1/m2/m3, s2.m1, bench_catalog rewrite | every generator + eval |
| `Topic`         | s2.m1 (from topics.json)    | s2.m2, s3.m3                         |
| `SkillExample`  | s3.m1/m2/m3                 | s3.m4 (verify/refine)                |
| `SkillMixTrial` | s2.m2                       | (terminal)                           |
| `ExtractedTask` | s3.m5, hand-written         | s4.m4, b2 skillsbench, training/eval |

`stable_uid(seed)` produces deterministic 16-hex identifiers. Skills carry a
`source` field (e.g. `"s2.m1.wikipedia-seed"`, `"bench_catalog.procedural-v1"`)
for provenance.

## providers

`core/providers.py` wraps three:

| provider     | spec                                | source                                    |
|--------------|-------------------------------------|-------------------------------------------|
| `anthropic`  | `anthropic:claude-opus-4-7`         | OAuth via `~/.claude/.credentials.json`, falls back to `ANTHROPIC_API_KEY` |
| `ollama`     | `ollama:qwen3.5:0.8b`               | local Ollama daemon at `$OLLAMA_HOST`     |
| `mock`       | `mock:`                             | always available, canned responses        |

`AnthropicProvider` includes the system-array OAuth wire fix (without it,
Opus/Sonnet 429s — see
[`b1.reports/260421.frontier-extraction-pipeline.txt`](b1.reports/260421.frontier-extraction-pipeline.txt)
ADDENDUM A). `OllamaProvider` sets `think: false` and `keep_alive: "0"` for
sweep memory eviction. Increase `max_tokens` via `create_provider(...,
max_tokens=N)` for prompts that ask for many records in one call (s3.m5 sets
16384).

For streaming + native thinking-block capture, use `harness.LinearAgent`
directly (`s4.m4` is the only spec method that does).

## status (live)

- **Pipeline**: catalog → procedural rewrite → task synthesis → bench_traced →
  sft_dataset all functional. `data/pipeline-runs/default/` contains a full
  v2 run (40 skills × 15 tasks = 600 tasks).
- **SFT v1** (curated-only training, 353 rows, qwen3.5:0.8b LoRA): trained
  cleanly. Eval result: **baseline +12.5pp, curated flat, Δ flipped sign**.
  Findings in
  [`b1.reports/260427.sft-v1-experiment-specification.txt`](b1.reports/260427.sft-v1-experiment-specification.txt).
- **SFT v1.5** (added baseline rows + dropped 5 ceiling skills + chat-template
  patch): worse than v1 (BL 0.650 / CU 0.425). Contrastive-rows hypothesis
  falsified.
- **SFT v1.6** (curated-only + drop ceiling + chat patch — isolates the patch
  effect): BL 0.645 / CU 0.565. Matches v1's CU; chat patch alone did not
  unlock differential lift.
- **SFT v1.7** (partial FT, top-6 layers + lm_head + final norm,
  adamw_bnb_8bit): BL 0.465 / CU 0.615. The only +Δ run, but BL collapsed —
  mode collapse: the model hallucinated "SKILL block" prefaces when none was
  present at inference. Apparent Δ was a BL-collapse artifact.
- **SFT v1.8** (partial FT + `--strip-skill-block` so the SKILL frame is
  removed from training-row system prompts): BL 0.545 / CU 0.570. As
  predicted, mode collapse fixed (BL recovered) but no new ceiling
  unlocked (CU stayed in the 0.565-0.585 band shared by v1, v1.6, v1.8).

After five recipe variants, three well-formed runs (v1, v1.6, v1.8) cluster
CU within a 2 pp band — strong evidence the ceiling is **not** in the
training recipe at this model size.

- **SFT v1.9** (same v1 recipe — curated-only, 353 rows — on
  **Qwen/Qwen3.5-2B** with LoRA r=16, alpha=32): **WIN.** BL 0.750 /
  CU 0.825 / Δ +0.075. CU exceeds pre-SFT haiku-4-5's 0.800 reference.
  Cleared both decision gates (CU > 0.65; Δ ≥ 5 pp). Confirms the 0.8B
  ceiling was a model-capacity ceiling, not a corpus or objective
  ceiling — the SAME recipe at 2B lifts CU by +24 pp over the best
  0.8B recipe.
- **SFT v2.0** (same v1 recipe on **Qwen/Qwen3.5-4B** with LoRA r=32,
  alpha=64, run on a vast.ai 3090 Ti): **soft WIN.** BL 0.835 / CU 0.880
  / Δ +0.045. Cleared CU > 0.85 absolute gate; Δ is a statistical tie
  with v1.9 at n=200 noise (~5 pp). Both BL and CU lifted v1.9 → v2.0,
  but the SFT-attributable Δ shrank from +0.075 → +0.045 — the bench
  is approaching saturation: at 4B the base model handles many tasks
  without the procedure. v1.7-style mode collapse confirmed cosmetic
  at 4B (35/200 baseline responses reference a phantom skill block;
  33 of those 35 still PASS).

The original SFT hypothesis (procedural-skill demonstrations teach a
student to apply the procedure when shown) is supported across two
model sizes (2B, 4B) on a 353-row corpus. The 200-task / 40-skill
bench saturates at 4B+; further model scaling will lift absolutes
but Δ will keep compressing. The next bottleneck is the bench, not
the model.

**Format-sensitivity artifact (added 2026-05-06):** pre-SFT 4B could
not be benched cleanly under the deterministic judge (which requires
literal `ANSWER:` lines that base 4B doesn't emit reliably). See
[`b1.reports/260506.bench-format-sensitivity-finding.txt`](b1.reports/260506.bench-format-sensitivity-finding.txt).

**LLM-only re-judge across all 7 configurations (updated 2026-05-09):**
re-routing every episode through the LLM judge regardless of task type
bypasses the format-compliance gate. Under matched-path HF + LLM-only
scoring, the pre-SFT base trajectory is **W-shaped**: 0.8B Δ −0.075,
2B Δ +0.060, 4B Δ −0.010, haiku Δ +0.030. The post-SFT trajectory rises
then falls (peak at 2B): v1 Δ −0.005, v1.9 Δ +0.100, v2.0 Δ +0.065. The
**SFT-attributable Δ-lift is roughly capacity-invariant**: +0.070 (0.8B),
+0.040 (2B), +0.075 (4B) — a 4 pp band across an order of magnitude in
base capacity. The 4 pp band is regime-asymmetric: SFT works harder
where the base struggles with the procedure (deep negative pre-SFT Δ
at 0.8B; shallow negative at 4B; +0.07 lift at both) and less where the
base is already in a procedure-friendly regime (positive pre-SFT Δ at
2B; +0.04 lift). v2.0's CU ties haiku at 0.985 (197/200 identical pass
count) — the bench is saturated in absolute pass rate but the SFT
mechanism is not. See
[`b1.reports/260506.llm-only-rejudge-findings.txt`](b1.reports/260506.llm-only-rejudge-findings.txt).

**Two earlier framings are now path-mismatch artifacts.** (a) v1's
"format-only learning at 0.8B" diagnosis was based on comparing v1's Δ
−0.005 against pre-SFT 0.8B Ollama Δ +0.055; under matched HF +
LLM-only scoring (pre-SFT 0.8B Δ −0.075), v1's Δ-lift is +0.070, on par
with 4B. (b) An earlier framing of "SFT contribution at 4B is the
largest in the experiment" is overstated — 4B's +0.075 Δ-lift is on
par with 0.8B's +0.070, not uniquely large. The accurate claim is
mechanism-uniformity rather than capacity-conditional amplification.

**Cross-family judge validation (added 2026-05-08):** GPT-5.4 via
OpenRouter graded the same model responses as Opus 4.7 across 6 of
7 datasets (2400 episodes; pre-SFT 0.8B HF cross-family validation
left for follow-up). Per-episode agreement ≥93.25%, Cohen's κ ≥ 0.754,
headline Δ shifts ≤0.035 pp. v1.9 is judge-invariant (100% agreement,
κ=1.000); v2.0 shifts by 0.005 pp; haiku shifts by 0.010 pp; pre-SFT
4B (the dataset most worth cross-validating) shifts by 0.035 pp,
direction preserved. The judge-overlap concern (paper §7.1) is bounded
quantitatively under a non-Anthropic-family second judge on
non-Anthropic infrastructure. See
[`b1.reports/260508.cross-family-judge-validation.txt`](b1.reports/260508.cross-family-judge-validation.txt).

Findings across v1-v1.8 are consolidated in
[`b1.reports/260426.findings-pre-post-sft-iterations.txt`](b1.reports/260426.findings-pre-post-sft-iterations.txt);
v1.9 result and v1.9 generality probe in
[`b1.reports/260428.sft-v1_9-2b-result.txt`](b1.reports/260428.sft-v1_9-2b-result.txt) and
[`b1.reports/260428.v1_9-generality-probe.txt`](b1.reports/260428.v1_9-generality-probe.txt);
v2.0 result and next-axis options in
[`b1.reports/260428.sft-v2_0-4b-result.txt`](b1.reports/260428.sft-v2_0-4b-result.txt).
The
[`b1.reports/`](b1.reports/) directory is the running engineering log; each
report is a self-contained snapshot at the date in its filename
(`yymmdd.description.txt`).

## relevant papers

- Yu et al. "Skill-Mix: A Flexible and Expandable Family of Evaluations for
  AI Models" (NeurIPS 2024) — `data/topics/topics.json` and the original 12
  skills come from Table 5 / 6.
- Arora & Goyal "A Theory for Emergence of Complex Skills in Language Models"
  (2023) — the latent-skill / compositional framing the catalog uses.
- Li et al. "SkillsBench: Benchmarking the Effectiveness of Skill Injection
  on LLMs" (2026) — the procedural-injection framing `bench_traced` uses.

Copies and a meta-analysis are in [`resources/`](resources/).

## out of scope

The three OOS methods need infrastructure beyond inference:

| id    | name                               | reason                                    |
|-------|------------------------------------|-------------------------------------------|
| s2.m3 | Mini-theories / context analysis   | Needs mechanistic interpretability        |
| s4.m1 | Context-enhanced learning          | Needs SFT training harness (now exists for `s4.m4 --mode inject`; not retrofitted to s4.m1's pre-training framing) |
| s4.m2 | Curriculum internalization         | Same as s4.m1                             |
| s4.m3 | Mechanistic analysis of storage    | Same as s2.m3                             |

Each OOS module is a docstring stub explaining what the method would do, so
the PROJECT_SPECS ↔ code map stays complete.
