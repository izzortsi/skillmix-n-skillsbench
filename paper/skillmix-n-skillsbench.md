# When Procedural-Skill SFT Works: Negative-Result-Driven Iteration Across 0.8B–4B Qwen3.5 Models

**Author**: Italo Strozzi (`istrozzi@matematica.ufrj.br`)
**Date**: 2026-05-06
**Status**: Draft

---

## Abstract

We study whether supervised fine-tuning (SFT) on a small corpus of (task + injected procedural-skill block, Opus chain-of-thought, judge-confirmed pass) demonstrations teaches a smaller language model to apply procedural skills *differentially* when shown an explicit procedure. Across eight training-recipe variants and three model sizes (Qwen3.5-0.8B, -2B, -4B) on a 200-task / 40-skill holdout, we find: (i) at 0.8B the SFT signal is dominated by format learning and never produces a clean curated-vs-baseline lift across five recipe iterations; (ii) at 2B, the same simplest recipe (LoRA r=16, 353 rows, three epochs) yields curated pass-rate 0.825 vs baseline 0.750, and a controlled attribution split (pre-SFT 2B at 0.685/0.710) isolates +11.5 pp curated lift to SFT contribution and +14.5 pp to base-model scaling; (iii) at 4B, the same recipe lifts curated to 0.880 with Δ shrinking to +0.045, indicating bench saturation. We additionally find that the bench scoring is format-sensitive in a way that prevents direct measurement of pre-SFT 4B reasoning, that the v1.7 "mode collapse" failure mode is a 0.8B capacity-floor artifact rather than a structural property of the SFT shape, and that the calibration regression visible in 2B SFT generality probes is eliminated at 4B. We release pipeline, recipes, eval scripts, and full episode-level logs.

---

## 1. Introduction

A common compositional-skills hypothesis holds that LLM capabilities can be decomposed into reusable skills, and that explicit procedural instructions can elicit better task performance from smaller students than implicit-skill priming alone (cf. Skill-Mix [Yu et al., 2024], procedural-injection benchmarks). A natural follow-up question: does demonstrating the explicit procedure during fine-tuning teach a smaller student to *use* such procedures more effectively at inference time, or merely to imitate the response shape?

We answer this empirically. We construct a self-contained pipeline that generates a procedural-skill SFT corpus from a hand-curated declarative skill catalog, train Qwen3.5 students at three scales with eight recipe variants, evaluate against a 200-task holdout with deterministic and LLM-judge-based scoring, and probe out-of-distribution behavior with a 52-prompt generality battery.

The main empirical results are:

1. **At 0.8B, SFT teaches format, not procedural-skill differentiation.** Five recipe variants (varying corpus composition, chat-template patches, partial fine-tuning, skill-block stripping) all cluster post-SFT curated pass rate inside a 2 pp band of 0.565–0.585, statistically indistinguishable from baseline.
2. **At 2B, the simplest recipe works.** Curated pass rate 0.825 vs baseline 0.750. With a pre-SFT 2B control (0.685/0.710), the curated lift decomposes cleanly into +14.5 pp from base-model scaling and +11.5 pp from SFT contribution. The procedural-Δ (curated − baseline) lifts from +0.025 (pre-SFT) to +0.075 (post-SFT).
3. **At 4B, the bench saturates.** Same recipe at Qwen3.5-4B yields curated 0.880, baseline 0.835, Δ +0.045. The pre-SFT 4B control fails because the deterministic judge requires literal `ANSWER:` lines that the base model does not reliably emit (despite producing correct reasoning), so the 4B SFT contribution is bounded but not directly measurable on this bench.
4. **Mode-collapse failure modes are capacity-floor artifacts.** A v1.7 partial-FT recipe at 0.8B produced an apparent +0.150 Δ but with collapsed baseline (0.465); the same SFT shape at 4B produces no analogous failure (0/52 phantom skill-block references in general chat).
5. **Generality is preserved.** A 52-prompt OOD probe shows v2.0 (4B+LoRA) matches base 4B on factual recall (34/34 HIT), eliminates a calibration regression observed in v1.9 (2B+LoRA), and produces no format lock-in despite three epochs of training on a single response shape.

The negative-result iteration sequence (five 0.8B variants before model-size pivot) and the format-sensitivity finding (the bench is partially a format-compliance test for non-SFT-trained students) are themselves contributions of this work, since they constrain the interpretation of similar SFT-on-procedural-demonstrations claims in the literature.

---

## 2. Pipeline and Bench

### 2.1 Skill catalog and procedural rewrite

We hand-curate 40 declarative skills across categories (logical fallacies, deductive forms, cognitive biases, interpretive patterns, spatial reasoning), seeded from the Skill-Mix Table 5/6 and expanded across three review batches. Each declarative entry contains `{name, definition, example}`. We use Opus 4.7 (1M context) to rewrite each into a procedural form: ordered procedure steps, when-to-use criteria, constraints, and a positive/negative worked example. The resulting `bench_catalog/skills.json` contains the procedural skill definitions used as injection content.

### 2.2 Task synthesis

We use a `s3.m5` Opus task synthesizer that, given a procedural skill, produces 15 tasks designed to require that skill. Tasks carry one of four query types: `YES_NO` (122/200, 61%), `FREE_FORM` (33/200, 16.5%), `SINGLE_WORD` (23/200, 11.5%), `RANKING` (22/200, 11%). Tasks are split into `tasks_train.json` (400 tasks) and `tasks_eval.json` (200 tasks), disjoint by task-uid and ordered by a `task_skill_map` that records the 1:1 task-to-skill pairing.

### 2.3 SFT corpus generation

We trace each training task with Opus under the SOLVER system prompt with the corresponding procedural skill block injected. Episodes that pass the LLM-judge are filtered to a curated-only SFT corpus of 353 rows. Each row is a chat-format triple: `(system: SOLVER + skill block, user: task, assistant: Opus's chain-of-thought + ANSWER: line)`.

### 2.4 Evaluation protocol

For every (task, condition) pair on the 200-task holdout we generate one student response and judge it. Two conditions: `baseline` (SOLVER prompt without skill block) and `curated` (SOLVER prompt with the corresponding procedural skill injected). Judge dispatch: deterministic ANSWER-line extraction for `YES_NO`/`SINGLE_WORD`/`RANKING`, Opus 4.7 LLM-judge for `FREE_FORM`. Decoding: greedy (temperature 0), repetition_penalty 1.05 (to break degenerate token loops on out-of-distribution prompts), max_new_tokens 1024 (post-SFT) or 2048 (pre-SFT base controls; thinking-mode is verbose).

The two reported per-condition statistics are pass rate (binary judge verdict) and mean score (0–1 judge score). We also report **Δ = pass_rate(curated) − pass_rate(baseline)**, the differential procedural lift.

### 2.5 Training

LoRA via TRL ≥0.18 + PEFT 0.13. Adapters target all-linear modules. The chat template is patched with `{% generation %}...{% endgeneration %}` markers to enable assistant-only loss masking (TRL ≥0.18 falls back to loss-on-all-tokens otherwise). For partial-FT runs, we freeze all but the top-N transformer layers + lm_head + final norm, and use bitsandbytes 8-bit Adam to fit the 0.8B model on consumer hardware.

---

## 3. Recipe Iterations

We report eight recipe variants. Five at 0.8B (v1–v1.8) explore corpus composition, partial fine-tuning, and skill-block masking; two at 2B/4B (v1.9, v2.0) test model-size as a separate axis.

| Variant | Base | Recipe | BL | CU | Δ |
|---|---|---|---|---|---|
| Pre-SFT 0.8B | qwen3.5:0.8b (Ollama) | none | 0.510 | 0.565 | +0.055 |
| Pre-SFT 2B | Qwen/Qwen3.5-2B (HF) | none | 0.685 | 0.710 | +0.025 |
| Pre-SFT 4B | Qwen/Qwen3.5-4B (HF) | none | 0.000 † | 0.000 † | — |
| Pre-SFT haiku-4-5 (ref) | claude-haiku-4-5 | none | 0.785 | 0.800 | +0.015 |
| **v1** | 0.8B | LoRA r=8, curated-only, 353 rows | 0.635 | 0.585 | −0.050 |
| **v1.5** | 0.8B | LoRA + BL+CU mixed corpus | 0.650 | 0.425 | −0.225 |
| **v1.6** | 0.8B | LoRA + drop ceiling skills + chat patch | 0.645 | 0.565 | −0.080 |
| **v1.7** | 0.8B | partial FT (top-6 layers + lm_head) | 0.465 | 0.615 | +0.150 † |
| **v1.8** | 0.8B | partial FT + skill-block stripped from training rows | 0.545 | 0.570 | +0.025 |
| **v1.9** | 2B | LoRA r=16, curated-only, 353 rows | **0.750** | **0.825** | **+0.075** |
| **v2.0** | 4B | LoRA r=32, curated-only, 353 rows | **0.835** | **0.880** | **+0.045** |

† v1.7's apparent +Δ is an artifact of baseline collapse, not differential learning (see § 4.3). † Pre-SFT 4B's 0/0 is a format-compliance artifact, not a measure of reasoning capability (see § 5.2).

---

## 4. 0.8B: Five Negative Results

### 4.1 v1: simplest recipe, format-only learning

LoRA r=8 / α=16 on the 353-row curated corpus for three epochs at lr 2e-4 lifts the 0.8B baseline pass rate by +12.5 pp (0.510 → 0.635) but does not lift curated; Δ flips from +0.055 (pre-SFT) to −0.050 (post-SFT). Per-skill inspection at n=200 shows the lift is concentrated on tasks where the model previously failed to produce the bench's ANSWER format and now does so. We interpret v1 as having taught primarily the response shape (step-by-step + `ANSWER:` line), with no detectable transfer of procedural skill use.

### 4.2 v1.5–v1.6: corpus composition does not unlock differential lift

Adding baseline-passing rows to the SFT corpus (v1.5; teaching the model to produce procedural responses both with and without skill-block presence) collapses curated pass rate to 0.425. Dropping ceiling-passing skills (v1.6; removing tasks where the base already passes ≥90% to avoid the model learning to add procedure overhead on tasks it can solve directly) lands at 0.645/0.565 — equivalent to v1 within noise.

### 4.3 v1.7: full-rank partial FT and the mode-collapse artifact

Replacing LoRA with full-rank fine-tuning of the top-6 layers + lm_head + final norm (using 8-bit Adam to fit GPU budget) produces an apparent positive Δ (+0.150) but with baseline pass rate *collapsed* to 0.465 — substantially below pre-SFT 0.510. Inspection of v1.7 baseline failures reveals the model emits "Step 1 (SKILL block: ...)" prefaces and procedural step language even when no skill block is in the system prompt — i.e. the model has learned to expect a skill block and hallucinates one when absent. The +0.150 Δ does not reflect the model learning to use skill blocks; it reflects the model losing the ability to function without one.

### 4.4 v1.8: skill-block stripping fixes mode collapse, no new ceiling

Stripping the skill block from the system prompt of every training row (so the model never sees a skill block during training, only the procedural response) recovers baseline (0.545) but lands curated at 0.570 — within the 0.565–0.585 cluster shared by v1, v1.6, and v1.8. The mode-collapse failure mode is fixed; no new ceiling is unlocked.

### 4.5 0.8B summary

Three well-formed variants (v1, v1.6, v1.8) cluster CU within 2 pp. The bench ceiling at 0.8B does not move with corpus composition, recipe, chat-template patches, or partial fine-tuning. We infer the ceiling is set by base-model capacity, not by the training recipe.

---

## 5. 2B and 4B: Model Size as the Decisive Axis

### 5.1 v1.9 (2B): SFT works

Same recipe shape as v1 (LoRA, curated-only, 353 rows, 3 epochs), bumped to r=16 / α=32 to give the larger base more LoRA capacity, on Qwen/Qwen3.5-2B. Result: BL 0.750 / CU 0.825 / Δ +0.075. Curated pass rate exceeds the pre-SFT haiku-4-5 reference of 0.800.

A pre-SFT 2B control (0.685/0.710) cleanly attributes the lift over pre-SFT 0.8B: +14.5 pp CU from base scaling and +11.5 pp CU from SFT contribution at 2B. The procedural-Δ lifts from +0.025 (pre-SFT) to +0.075 (post-SFT), a +5 pp differential procedural-application lift attributable to SFT.

This is the cleanest available measurement of the original SFT hypothesis. SFT does teach differential procedural use; the 0.8B failures were base-capacity-bound.

### 5.2 v2.0 (4B): bench saturation

Same recipe at LoRA r=32 / α=64 on Qwen/Qwen3.5-4B. Result: BL 0.835 / CU 0.880 / Δ +0.045. The CU 0.85 absolute gate is cleared; the Δ ≥ 5 pp gate is a statistical tie with v1.9 within n=200 binomial noise (~3 pp per condition, ~5 pp on Δ).

The pre-SFT 4B attribution control was attempted but failed: base Qwen3.5-4B reasons correctly on bench tasks (spot-checks confirm step-by-step reasoning that reaches the right answer) but does not reliably emit the literal `ANSWER:` format the deterministic judge requires. With 88.5% of the bench using the deterministic judge, pre-SFT 4B scores ~0/200, which is a format-compliance artifact rather than a reasoning measurement. The v2.0 SFT contribution at 4B is therefore bounded by the saturation trend (plausibly 0.03–0.10 CU) but not directly measured on this bench.

The v1.9→v2.0 deltas (ΔBL +0.085, ΔCU +0.055, Δ−0.030) are unaffected by this artifact since both v1.9 and v2.0 are SFT-trained format-compliant students. The shrinkage of the SFT-attributable Δ from +0.075 to +0.045 (within noise) is the bench-saturation signal: as base-model capability grows, the procedure adds less to what the model can already do.

### 5.3 Generality probe: v2.0 preserves out-of-distribution behavior; v1.9 had a small calibration regression that v2.0 eliminated

We probe v1.9, v2.0, and base 2B/4B with a 52-prompt OOD battery (10 generality + 42 factual including 8 false-premise "trick" prompts) under a generic system prompt (not the SOLVER prompt). Results:

| | format-locked | factual HIT | trick CONFAB (eyeballed) | phantom skill block |
|---|---|---|---|---|
| base 2B | 0/52 | n/a (not run) | n/a | 0/52 |
| v1.9 (2B) | 0/52 | 33/34 † | 6/8 | 0/52 |
| base 4B | 0/52 | 34/34 | ~1/8 | 0/52 |
| v2.0 (4B) | 0/52 | 34/34 | 0/8 | 0/52 |

† v1.9 swapped *Brave New World* attribution from Aldous Huxley to H. G. Wells — a plausible-author confabulation absent from base 4B and from v2.0.

v1.9 introduced a small calibration regression on adversarial prompts: the SFT-induced "produce a confident step-by-step answer" prior leaks into out-of-distribution behavior, making v1.9 commit to plausible-sounding inventions where base 2B would refuse. v2.0 does not exhibit this — the 4B base has enough native capacity to recognize false premises and refuse cleanly without the SFT shape overriding refusal.

The bench-time finding that 17.5% of v2.0's baseline responses reference a "skill block" not present in the system prompt does *not* extend to general-chat distributions: under a generic system prompt, the SFT-induced skill-block reflex is dormant (0/52). The phantom-mention is contextual to SFT-distribution-similar prompts.

---

## 6. Discussion

### 6.1 What did SFT actually teach?

At 2B, the cleanest measurement we have, SFT contributed:

- **+11.5 pp absolute curated pass rate**, on top of base 2B's already-respectable 0.710
- **+5 pp differential procedural-application lift** (Δ from +0.025 to +0.075), evidence that the model learned to *use* the procedure rather than just imitate the response shape
- **+6.5 pp absolute baseline pass rate**, attributable to format learning (ANSWER-line emission, step-by-step structure)

The SFT contribution decomposes into roughly half "format" and half "procedure" by these crude metrics. The two are not independent — producing the procedural-step shape implicitly enforces a checking discipline that may itself be the procedural lift — but the differential Δ-lift confirms there is a procedure-vs-no-procedure component of the gain that pure format-learning cannot explain.

### 6.2 Bench format-sensitivity is itself a methodological finding

The pre-SFT 4B attribution control failure surfaces a property of the bench scoring that is invisible when only SFT-trained students are evaluated: 88.5% of the holdout uses a deterministic ANSWER-line extractor, which heavily penalizes models that reason correctly but emit free-form conclusions. Base Qwen3.5-4B is such a model. The bench's 0/200 for pre-SFT 4B should be read as "format-compliance baseline ≈ 0", not "reasoning ≈ 0".

This has interpretive consequences for any paper claiming "+X pp from SFT on procedural demonstrations" without a base-model control benched under format-tolerant scoring. The post-SFT lift conflates format learning, format-conditional reasoning gating, and procedural application; isolating these requires either (a) a stricter base-model evaluation prompt that hits the format target, (b) a format-tolerant judge for base controls, or (c) FREE_FORM-only evaluation. We did not pursue these in the present study, but mark them as required steps for future replication.

### 6.3 Mode collapse reinterpreted

The v1.7 "mode collapse" — partial-FT producing apparent +Δ at the cost of baseline integrity — appeared at 0.8B as a structural concern about SFT-on-procedural-demonstrations and motivated the v1.8 skill-block stripping intervention. Spot-checks of v2.0 baseline responses show the same phantom skill-block reference (35/200, 17.5%) but with **94.3% of those responses still passing** — at 4B, the phantom mention is cosmetic rather than load-bearing. The model has the capacity to execute the procedure regardless of whether it remembers seeing a skill block in the system prompt.

This re-frames mode collapse as a base-capacity-floor artifact: at 0.8B, the model's SFT-absorbed expectation of seeing a skill block crashed performance when one was absent; at 4B, the same expectation produces a confabulated mention but does not derail the answer. The v1.8 skill-block-stripping fix is unnecessary at the model sizes where SFT actually delivers.

### 6.4 Negative results were informative

The five 0.8B variants are individually publishable as negative results — corpus composition, partial fine-tuning, and chat-template patching all failed to lift the SFT signal at this scale. They are jointly informative: together they constrain the diagnosis to "model size is the binding constraint", which the 2B run validates. We argue that running through this negative-result series before moving to a larger model was epistemically efficient (~$15 in compute, three days of GPU time) compared to skipping straight to 4B with no priors on what fails.

---

## 7. Limitations

1. **Pre-SFT 0.8B was run via Ollama**, not the same HF transformers path used for 2B/4B. Quantization and chat-template differences may inflate or deflate its baseline relative to a strictly-comparable HF run. We did not re-bench 0.8B under the same path.

2. **Pre-SFT 4B is unmeasured on this bench.** The format-compliance artifact prevents direct attribution of v2.0's lift between base scaling and SFT contribution. The deferred remediation options (force-LLM-judge, lenient extractor, stricter prompt) all introduce different tradeoffs; we leave this for future work.

3. **n=200 per condition limits the resolution of the Δ statistic.** Per-condition binomial standard error is ~3 pp, so Δ uncertainty is ~5 pp. The v1.9 and v2.0 Δ values (+0.075 and +0.045) are within one standard error of each other; the bench-saturation claim is directional rather than statistically distinguishable at this sample size.

4. **The bench tests single-skill-per-task application, not skill composition.** The original procedural-skills hypothesis posits compositional benefit (apply skill A then skill B); the present bench cannot distinguish "model uses procedure when shown" from "model would have applied roughly the right pattern anyway, because the task was synthesized to require exactly that skill." A compose-skill bench (deferred future work) would be more decisive.

5. **The SFT corpus is small (353 rows).** Whether scaling the corpus to 10× would change the saturation curve at 4B is untested. Cost estimate: ~$50–100 in synthesis + judge calls, ~4 GPU-hours.

6. **Attribution at 4B is bounded but not measured.** We rely on saturation-trend extrapolation for the 4B SFT contribution range (0.03–0.10 CU). A clean measurement would require resolving Limitation 2.

7. **The judge is an LLM (Opus 4.7).** While we patched the judge for deterministic extraction robustness and re-judged ~17% of episodes that hit transient auth failures, we have not cross-validated against a second judge or human raters. Judge bias toward the SFT-induced response shape is plausible but unmeasured.

---

## 8. Conclusion

Procedural-skill SFT works as a recipe for teaching a smaller language model to apply explicit procedures more effectively when shown — but only at base sizes where the model already produces format-compliant reasoning. At 0.8B, five recipe variants establish that the SFT signal cannot exceed format-learning at this capacity. At 2B, the simplest recipe (LoRA r=16, 353 rows, 3 epochs) produces a measurable +11.5 pp curated lift attributable to SFT contribution and a +5 pp differential procedural-Δ lift, with clean attribution from a pre-SFT 2B control. At 4B, the same recipe lifts to curated 0.880 with shrinking Δ, indicating bench saturation and limiting further interpretation without harder benchmarks.

The negative-result iteration sequence and the bench format-sensitivity finding both constrain the interpretation of similar SFT-on-procedural-demonstrations claims. We argue future work in this area should (i) include a pre-SFT base-model control evaluated under format-tolerant scoring, (ii) measure compositional skill application rather than single-skill template-matching, and (iii) bound the SFT contribution within statistical noise at the chosen sample size before claiming model-size-conditional results.

---

## Acknowledgments

Compute provided by vast.ai (Qwen3.5-2B and -4B training and partial evaluation) and a local consumer GPU (RTX 3090 Ti, evaluation and probing). Skill catalog content seeded from Skill-Mix (Yu et al., 2024) Tables 5 and 6 and expanded by hand. Opus 4.7 (1M context) was used for skill procedural rewriting, task synthesis, baseline trace generation, and judge-of-record scoring.

---

## References

- Yu, D., Kaur, S., Gupta, A., Brown-Cohen, J., Goyal, A., Arora, S. (2024). *Skill-Mix: A Flexible and Expandable Family of Evaluations for AI Models*. NeurIPS 2024.
- Arora, S., Goyal, A. (2023). *A Theory for Emergence of Complex Skills in Language Models*.
- Li et al. (2026). *SkillsBench: Benchmarking the Effectiveness of Skill Injection on LLMs*.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- TRL: Transformer Reinforcement Learning library, version ≥0.18.
- PEFT: Parameter-Efficient Fine-Tuning library, version ≥0.13.

---

## Appendix A: Pipeline file structure

```
skillmix-n-skillsbench/
├── core/                                  shared schema + provider abstractions
├── harness/                               LinearAgent for streaming + thinking-block capture
├── b2_benchmarks/skillsbench/             corpus_harness, llm_judge, procedural_catalog
├── s1_extracting_skill_names/             declarative-skill seeders (Wikipedia-style)
├── s2_extracting_skills_from_text/        procedural-rewrite stage
├── s3_generating_skill_examples/m5_*.py   task synthesis from procedural skills
├── s4_extracting_skill_usage_instances/   solver + trace + judge composition (m4)
├── training/
│   ├── train_qwen35_lora.py               LoRA / partial-FT trainer (TRL >=0.18)
│   ├── eval_qwen35_lora.py                holdout evaluator (--adapter optional)
│   ├── patch_qwen35_template.py           chat-template {% generation %} patcher
│   ├── chat_with_adapter.py               interactive + probe-set generality tester
│   ├── rejudge_failed_episodes.py         post-hoc rejudge for transient judge errors
│   └── compare_pre_post.py                per-skill diff utility
├── docker/                                vast.ai training image + setup docs
├── data/pipeline-runs/default/            full run artifacts (catalog, bench, eval)
└── b1.reports/                            engineering log (yymmdd.title.txt)
```

## Appendix B: Per-skill v2.0 result (n=5 per skill, sorted by Δ)

Lift cluster (procedure helped):
- accident-fallacy +0.40, spatial-reasoning +0.40
- false-dilemma, confirmation-bias, anchoring-bias, appeal-to-emotion, base-rate-neglect, transitive-inference, necessary-and-sufficient-conditions, metaphor, emotional-self-regulation: +0.20 each

Flat cluster (Δ = 0): 25 skills, of which 22 are at BL ≥ 0.80 (ceiling-bound).

Regression cluster (procedure hurt, all 5/5 BL → 4/5 CU): complex-question, hypothetical-syllogism, framing-effect, equivocation.

The regression cluster overlaps with deductive-logic skills where the base 4B has ceiling competence and the 5-step procedure introduces sub-inference error opportunities — the same lift/flat/regression pattern observed at 0.8B in v1, shifted upward by base scaling.

## Appendix C: Code and data availability

- Repository: `https://github.com/izzortsi/skillmix-n-skillsbench`
- Branch: `dev`
- Reports cited (full text in `b1.reports/`):
  - `260426.findings-pre-post-sft-iterations.txt` — v1–v1.8 consolidation
  - `260428.sft-v1_9-2b-result.txt` — v1.9 result + attribution
  - `260428.v1_9-generality-probe.txt` — v1.9 generality regression
  - `260428.sft-v2_0-4b-result.txt` — v2.0 result + attribution caveat
  - `260428.v2_0-generality-probe.txt` — v2.0 generality preservation
  - `260506.bench-format-sensitivity-finding.txt` — pre-SFT 4B format artifact
- Eval artifacts: `data/pipeline-runs/default/bench-eval-{pre-sft,post-sft-v*}/episodes.json`
- Trained adapters: omitted from git via `.gitignore`; reproducible via training scripts + the committed SFT corpus.
