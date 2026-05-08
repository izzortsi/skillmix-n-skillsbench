"""
eval_qwen35_lora.py — Run a LoRA-fine-tuned Qwen3.5-0.8B on the holdout
eval task corpus and emit episodes.json in the same shape as the bench
harness output.

Used to compare PRE-SFT numbers (from b2_benchmarks.skillsbench.corpus_harness
multi-student bench) against POST-SFT numbers from the trained model.

PREREQUISITES
    pip install "transformers>=4.46" "peft>=0.13" "datasets>=3.0" \
                "accelerate>=1.0" torch anthropic

USAGE (from training/ directory)

    python eval_qwen35_lora.py \\
        --base Qwen/Qwen3.5-0.8B \\
        --adapter ./qwen35-0.8b-skill-lora \\
        --tasks ../data/pipeline-runs/default/synthesis/tasks_eval.json \\
        --catalog ../data/pipeline-runs/default/bench_catalog/skills.json \\
        --task-skill-map ../data/pipeline-runs/default/synthesis/task_skill_map_eval.json \\
        --out-dir ../data/pipeline-runs/default/bench-eval-post-sft \\
        --judge anthropic:claude-opus-4-7

OUTPUT
    out_dir/episodes.json    same shape as bench/episodes.json (corpus_harness)
                              keyed fields: task_uid, model, condition,
                              skill_name, response, passed, score,
                              judge_rationale
    out_dir/summary.json     per-condition pass rate

The output is directly comparable to the pre-SFT bench output via:
    python compare_pre_post.py \\
        --pre  ../data/pipeline-runs/default/bench-eval-pre-sft/episodes.json \\
        --post ../data/pipeline-runs/default/bench-eval-post-sft/episodes.json
(separate compare script, drafted after first run)

NOTES
- Uses the same default reading-comprehension prompt that corpus_harness uses
  for baseline (see core.providers / b2_benchmarks.skillsbench.skill_injection)
  so the comparison is apples-to-apples with the pre-SFT bench numbers.
- Model name in episodes is set to "{base}+lora:{adapter_dir}" so downstream
  visualization can distinguish pre/post SFT runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reach into the project's core/b2_benchmarks for shared helpers
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from b2_benchmarks.skillsbench.skill_injection import (  # noqa: E402
    format_skill_as_system,
)
from b2_benchmarks.skillsbench.llm_judge import LLMJudgeEvaluator  # noqa: E402
# Use the same SOLVER_SYSTEM_PROMPT the SFT corpus was generated under, so
# the post-SFT model sees the same prompt distribution at inference as it
# did during training. Pre-SFT bench should also be re-run with this prompt
# (corpus_harness --system-prompt solver) for an apples-to-apples comparison.
from s4_extracting_skill_usage_instances.m4_skill_composition_testing import (  # noqa: E402
    SOLVER_SYSTEM_PROMPT,
)
from core.providers import create_provider  # noqa: E402
from core.schemas import load_extracted_tasks, load_skills  # noqa: E402


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_user_prompt(task) -> str:
    """Mirror the user-prompt shape used by corpus_harness for fair comparison."""
    return f"Passage: {task.passage}\n\nChallenge: {task.challenge}"


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Single chat completion via HF transformers."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # repetition_penalty=1.05 prevents greedy-decoding token loops that the
    # raw base model is prone to on OOD-feeling prompts (e.g. SOLVER prompt
    # against pre-SFT base, where the model is not in its instruction-tuned
    # distribution). 1.05 is mild enough not to distort SFT'd-model output.
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Slice off the prompt
    response_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-task episode pair (baseline + curated)
# ---------------------------------------------------------------------------


def run_task_pair(
    task,
    skill,
    model,
    tokenizer,
    judge: LLMJudgeEvaluator,
    model_label: str,
    verbose: bool = False,
    force_llm_judge: bool = False,
) -> List[Dict[str, Any]]:
    """Run baseline + curated for one task; return 2 episode dicts.

    force_llm_judge: when True, route every task through the LLM judge
    by overriding query_type to FREE_FORM. Useful for pre-SFT base
    evals where the model reasons correctly but doesn't reliably emit
    the literal ANSWER: format the deterministic judge requires —
    bypassing the format-compliance gate lets the LLM judge measure
    substantive correctness instead. The LLM judge already accepts
    free-form conclusions; the routing change just bypasses the
    deterministic-extractor short-circuit for non-FREE_FORM types.
    """
    base_system = SOLVER_SYSTEM_PROMPT  # match SFT corpus prompt distribution
    user = build_user_prompt(task)

    eps: List[Dict[str, Any]] = []
    for cond, sys_prompt, sk_name in [
        ("baseline", base_system, ""),
        ("curated", format_skill_as_system(base_system, skill), skill.name),
    ]:
        t0 = time.time()
        try:
            response = generate_response(model, tokenizer, sys_prompt, user)
        except Exception as e:
            response = f"[GENERATION ERROR] {e}"
        elapsed = round(time.time() - t0, 2)

        # Judge — route to LLM judge for everything when force_llm_judge.
        effective_query_type = (
            "FREE_FORM" if force_llm_judge
            else getattr(task, "query_type", "FREE_FORM")
        )
        verdict = judge.evaluate(
            response=response,
            passage=task.passage,
            challenge=task.challenge,
            acceptance_criteria=task.acceptance_criteria,
            query_type=effective_query_type,
        )

        ep = {
            "task_uid": task.task_uid,
            "model": model_label,
            "condition": cond,
            "skill_name": sk_name,
            "mode": "singlecall",
            "response": response,
            "passed": bool(verdict.passed),
            "score": float(verdict.score),
            "tokens": 0,
            "elapsed_s": elapsed,
            "judge_rationale": verdict.rationale,
        }
        eps.append(ep)
        if verbose:
            tag = "[PASS]" if verdict.passed else "[FAIL]"
            short = response[:80].replace("\n", " ")
            print(f"  {cond:<8} {tag} score={verdict.score:.2f} {elapsed}s :: {short!r}")
    return eps


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", default="Qwen/Qwen3.5-0.8B",
                        help="HF model id or local path for the BASE WEIGHTS "
                             "(default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Tokenizer source override. Use this if the "
                             "patched tokenizer dir does not contain model "
                             "weights. Default: tries the LoRA adapter dir, "
                             "then falls back to --base.")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Path to LoRA adapter directory (or full-FT "
                             "model directory). Omit to evaluate the BASE "
                             "model directly — useful for pre-SFT controls.")
    parser.add_argument("--tasks", type=Path, required=True,
                        help="Path to tasks_eval.json")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="Path to bench_catalog/skills.json (procedural)")
    parser.add_argument("--task-skill-map", type=Path, required=True,
                        help="Path to task_skill_map_eval.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judge", default="anthropic:claude-opus-4-7")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate the first N tasks (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--force-llm-judge", action="store_true",
                        help="Route every task through the LLM judge "
                             "regardless of query_type. Useful for pre-SFT "
                             "base evals where the model reasons correctly "
                             "but doesn't reliably emit literal ANSWER: "
                             "lines (default deterministic judge would "
                             "score those 0). LLM judge cost: ~$0.05 per "
                             "episode, ~$20 for a full 400-episode run.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ---- Tokenizer ----
    # Preference order: explicit --tokenizer-path > the adapter dir (if
    # provided and contains tokenizer files saved by train_qwen35_lora.py)
    # > the base. Lets the user point at a chat-template-patched tokenizer
    # dir while keeping --base pointed at the weights source.
    if args.tokenizer_path:
        tokenizer_src = args.tokenizer_path
    elif args.adapter and (args.adapter / "tokenizer_config.json").exists():
        tokenizer_src = str(args.adapter)
    else:
        tokenizer_src = args.base
    print(f"Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    # Three modes:
    #   adapter omitted  -> evaluate the base model directly (pre-SFT control)
    #   adapter is LoRA  -> base + PeftModel adapter
    #   adapter is full  -> dir IS the model (full / partial fine-tune output)
    if args.adapter is None:
        print(f"Loading base model (no adapter) from: {args.base}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model_label = f"base:{args.base}"
    else:
        adapter_config = args.adapter / "adapter_config.json"
        is_lora = adapter_config.exists()
        if is_lora:
            print(f"Loading base model from: {args.base}")
            base = AutoModelForCausalLM.from_pretrained(
                args.base,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Applying LoRA adapter: {args.adapter}")
            model = PeftModel.from_pretrained(base, str(args.adapter))
            model_label = f"{args.base}+lora:{args.adapter.name}"
        else:
            print(f"Loading full fine-tuned model from: {args.adapter}")
            model = AutoModelForCausalLM.from_pretrained(
                str(args.adapter),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model_label = f"full:{args.adapter.name}"
    model.eval()

    print(f"Model label for episodes: {model_label}")

    # ---- Tasks ----
    tasks = load_extracted_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Tasks: {len(tasks)}")

    catalog = load_skills(args.catalog)
    catalog_by_name = {s.name: s for s in catalog}

    with args.task_skill_map.open() as f:
        task_skill_map = json.load(f)
    print(f"Catalog: {len(catalog)} skills, map: {len(task_skill_map)} entries")

    # ---- Judge ----
    j_name, _, j_model = args.judge.partition(":")
    judge_provider = create_provider(j_name, j_model)
    judge = LLMJudgeEvaluator(judge_provider)
    print(f"Judge: {args.judge}")

    # ---- Run ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_eps: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        sk_name = task_skill_map.get(task.task_uid)
        skill = catalog_by_name.get(sk_name) if sk_name else None
        if skill is None:
            print(f"[{i}/{len(tasks)}] {task.task_uid}: no skill mapped, skipping curated")
            continue
        print(f"[{i}/{len(tasks)}] {task.task_uid} + {sk_name}")
        eps = run_task_pair(
            task, skill, model, tokenizer, judge, model_label,
            verbose=args.verbose, force_llm_judge=args.force_llm_judge,
        )
        all_eps.extend(eps)

    # ---- Save ----
    (args.out_dir / "episodes.json").write_text(
        json.dumps(all_eps, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Summary stats
    by_cond: Dict[str, Dict[str, float]] = {}
    for ep in all_eps:
        c = ep["condition"]
        b = by_cond.setdefault(c, {"n": 0, "passed": 0, "score_sum": 0.0})
        b["n"] += 1
        b["passed"] += int(bool(ep["passed"]))
        b["score_sum"] += float(ep["score"])

    summary = {
        "total_episodes": len(all_eps),
        "model": model_label,
        "base": args.base,
        "adapter": str(args.adapter),
        "tasks": str(args.tasks),
        "per_condition": {
            c: {
                "n": int(b["n"]),
                "passed": int(b["passed"]),
                "pass_rate": round(b["passed"] / b["n"], 4) if b["n"] else 0.0,
                "mean_score": round(b["score_sum"] / b["n"], 4) if b["n"] else 0.0,
            }
            for c, b in by_cond.items()
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\nWrote {len(all_eps)} episodes -> {args.out_dir}")
    for c, s in summary["per_condition"].items():
        print(f"  {c}: pass_rate={s['pass_rate']} mean_score={s['mean_score']} (n={s['n']})")


if __name__ == "__main__":
    main()
