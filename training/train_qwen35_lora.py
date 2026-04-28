"""
train_qwen35_lora.py — LoRA fine-tune Qwen3.5-0.8B on the SFT corpus.

Trains a LoRA adapter on the (system + user, assistant CoT) demonstrations
emitted by sft_dataset/dataset.jsonl. Runs on a single GPU host.

PREREQUISITES
    pip install "transformers>=4.46" "trl>=0.14" "peft>=0.13" \
                "datasets>=3.0" "accelerate>=1.0" torch

HARDWARE
    Single GPU. 0.8B params + LoRA r=8 in bf16 fits in ~3-4 GB VRAM.
    Tested target: any modern consumer GPU (RTX 3060 12GB and up).

USAGE
    # Default args — train 3 epochs on the v2 SFT corpus.
    python train_qwen35_lora.py \\
        --data ../data/pipeline-runs/default/sft_dataset/dataset.jsonl

    # Quick smoke test — 1 epoch, lower lr, save fewer checkpoints.
    python train_qwen35_lora.py \\
        --data ../data/pipeline-runs/default/sft_dataset/dataset.jsonl \\
        --epochs 1 --lr 1e-4 --save-steps 999999

OUTPUT
    output_dir/
        adapter_model.safetensors      LoRA weights
        adapter_config.json            LoRA config
        tokenizer.json + special_tokens_map.json
        training_args.bin              full training args
        checkpoint-*/                  intermediate (kept up to save_total_limit)

LOADING THE TRAINED MODEL (inference)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype="bfloat16", device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, "./qwen35-0.8b-skill-lora")
    tokenizer = AutoTokenizer.from_pretrained("./qwen35-0.8b-skill-lora")

    msgs = [
        {"role": "system", "content": "<base prompt + injected SKILL block>"},
        {"role": "user",   "content": "<task passage + question>"},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = model.generate(**tokenizer(text, return_tensors="pt").to(model.device),
                         max_new_tokens=1024)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def freeze_for_partial_ft(
    model,
    n_trainable_layers: int,
    train_embeddings: bool = False,
) -> tuple[int, int]:
    """Freeze base, keep top-N transformer layers + final norm + lm_head trainable.

    Layer access path is the standard HF Llama/Qwen layout: model.model.layers
    (a ModuleList of N transformer blocks), model.model.norm (final RMSNorm),
    model.lm_head (output projection), model.model.embed_tokens (input).

    Returns (n_trainable_params, n_total_params) for logging.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Locate the layers list (Qwen3 / Llama-style)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise RuntimeError(
            f"Could not find model.model.layers on a {type(model).__name__}. "
            f"This freeze helper assumes Qwen/Llama-style architecture; "
            f"adjust for your model class."
        )
    n_total_layers = len(layers)
    if n_trainable_layers > n_total_layers:
        n_trainable_layers = n_total_layers

    # 3. Unfreeze top N transformer layers
    for layer in layers[n_total_layers - n_trainable_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    # 4. Always train final norm (cheap, important for output distribution)
    if hasattr(model.model, "norm"):
        for p in model.model.norm.parameters():
            p.requires_grad = True

    # 5. Always train LM head (output projection)
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True

    # 6. Optionally train input embeddings
    if train_embeddings and hasattr(model.model, "embed_tokens"):
        for p in model.model.embed_tokens.parameters():
            p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total


def load_sft_jsonl(path: Path) -> Dataset:
    """Load the bench_traced -> sft_dataset JSONL.

    Each row in the file:
      {"messages": [
         {"role": "system",    "content": "<base + skill>"},
         {"role": "user",      "content": "<task>"},
         {"role": "assistant", "content": "<CoT + ANSWER:>"}
       ],
       "metadata": {...}}

    We keep only `messages` for training; `metadata` is for inspection.
    """
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs:
                continue
            rows.append({"messages": msgs})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to sft_dataset/dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="HF model id or local path for the BASE WEIGHTS "
                             "(default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Override the tokenizer source. Use this when you "
                             "have a chat-template-patched tokenizer dir "
                             "(weights + config still come from --model). "
                             "Default: same as --model.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: './qwen35-0.8b-skill-{mode}' where {mode} is "
                             "the --mode value.")

    # Training mode — LoRA (parameter-efficient) vs full fine-tune vs partial
    parser.add_argument("--mode", choices=["lora", "full", "partial"], default="lora",
                        help="lora (default): LoRA r=8 on all-linear, ~6 GB VRAM, "
                             "lr=2e-4, 3 epochs. full: full fine-tune, ~14-19 GB "
                             "VRAM for 0.8B, lr=5e-6, 2 epochs. partial: layerwise — "
                             "freeze base, train top N transformer layers + lm_head + "
                             "final norm. Set N via --n-trainable-layers (default 6). "
                             "Memory between LoRA and full; better behavioral capacity "
                             "than LoRA, fits where full doesn't.")
    parser.add_argument("--n-trainable-layers", type=int, default=6,
                        help="[partial only] number of top transformer layers to keep "
                             "trainable; lower base layers are frozen. Default 6.")
    parser.add_argument("--train-embeddings", action="store_true",
                        help="[partial only] also keep input embeddings trainable. "
                             "Default frozen.")
    parser.add_argument("--optim", default=None,
                        help="HF TrainingArguments optim string. Default depends on "
                             "mode. For tight VRAM, set 'adamw_bnb_8bit' (requires "
                             "bitsandbytes; cuts optimizer memory ~4x).")

    # Training hyperparameters — defaults differ by --mode; sentinel None
    # means "use mode-appropriate default" (resolved after parse_args).
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: lora=3, full=2")
    parser.add_argument("--lr", type=float, default=None,
                        help="default: lora=2e-4, full=5e-6")
    parser.add_argument("--lr-scheduler", default=None,
                        choices=["cosine", "linear", "constant", "constant_with_warmup"],
                        help="default: lora=cosine, full=linear")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="default: lora=0.03, full=0.05")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="default: lora=0.0, full=0.01")
    parser.add_argument("--batch", type=int, default=1,
                        help="per-device batch size (default 1; combine with --grad-accum)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="gradient accumulation steps (effective batch = batch * grad_accum)")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="curated system prompts run ~2k-3k chars; 4096 is comfortable")

    # LoRA hyperparameters (ignored when --mode full)
    parser.add_argument("--r", type=int, default=8, help="[lora only] LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="[lora only] LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="[lora only] LoRA dropout")
    parser.add_argument("--target-modules", default="all-linear",
                        help='[lora only] LoRA target modules. "all-linear" (default) '
                             'is safest for a new architecture like qwen35; alternatively '
                             'pass a comma list like "q_proj,k_proj,v_proj,o_proj"')

    # Logging / checkpointing
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Loss masking
    parser.add_argument("--no-assistant-only-loss", action="store_true",
                        help="By default, loss is computed only on assistant tokens (correct "
                             "SFT behavior). Pass this flag to compute loss on every token.")

    args = parser.parse_args()

    # Resolve mode-dependent defaults
    _MODE_DEFAULTS = {
        "lora":    {"epochs": 3, "lr": 2e-4, "lr_scheduler": "cosine",
                    "warmup_ratio": 0.03, "weight_decay": 0.0},
        "full":    {"epochs": 2, "lr": 5e-6, "lr_scheduler": "linear",
                    "warmup_ratio": 0.05, "weight_decay": 0.01},
        # partial: between lora and full. Higher lr than full because most
        # layers are frozen; lower than lora because we still update full
        # parameters in the unfrozen layers (no low-rank regularization).
        "partial": {"epochs": 2, "lr": 2e-5, "lr_scheduler": "linear",
                    "warmup_ratio": 0.05, "weight_decay": 0.01},
    }
    md = _MODE_DEFAULTS[args.mode]
    if args.epochs is None:        args.epochs = md["epochs"]
    if args.lr is None:            args.lr = md["lr"]
    if args.lr_scheduler is None:  args.lr_scheduler = md["lr_scheduler"]
    if args.warmup_ratio is None:  args.warmup_ratio = md["warmup_ratio"]
    if args.weight_decay is None:  args.weight_decay = md["weight_decay"]
    if args.output_dir is None:    args.output_dir = Path(f"./qwen35-0.8b-skill-{args.mode}")

    print(f"Training mode: {args.mode}")
    print(f"  epochs={args.epochs}  lr={args.lr}  lr_scheduler={args.lr_scheduler}")
    print(f"  warmup_ratio={args.warmup_ratio}  weight_decay={args.weight_decay}")
    print(f"  output_dir={args.output_dir}")

    # ---------- model + tokenizer ----------
    tokenizer_src = args.tokenizer_path or args.model
    print(f"Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen models often ship without a pad_token; using EOS is conventional.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # gradient checkpointing trades a small amount of compute for ~30-40% memory savings
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Apply layerwise freeze if --mode partial
    if args.mode == "partial":
        n_trainable, n_total = freeze_for_partial_ft(
            model,
            n_trainable_layers=args.n_trainable_layers,
            train_embeddings=args.train_embeddings,
        )
        pct = 100.0 * n_trainable / n_total
        print(f"Partial FT: top {args.n_trainable_layers} transformer layers + lm_head"
              f"{' + embeddings' if args.train_embeddings else ''} trainable.")
        print(f"  trainable: {n_trainable:,} / {n_total:,} params ({pct:.1f}%)")

    # ---------- data ----------
    print(f"Loading SFT data: {args.data}")
    dataset = load_sft_jsonl(args.data)
    print(f"  rows: {len(dataset)}")
    if len(dataset) == 0:
        raise SystemExit("No rows loaded — check --data path and JSONL format.")

    # Quick sanity: confirm chat template works on row 0
    sample = tokenizer.apply_chat_template(
        dataset[0]["messages"], tokenize=False, add_generation_prompt=False,
    )
    print(f"  sample row (first 200 chars after chat template):\n    {sample[:200]!r}...")

    # ---------- LoRA config (only built when --mode lora) ----------
    lora_config = None
    if args.mode == "lora":
        if args.target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        print(f"LoRA: r={args.r} alpha={args.alpha} dropout={args.dropout} "
              f"target_modules={target_modules}")

        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules=target_modules,
            lora_dropout=args.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        print("Full fine-tune mode: all parameters trainable, no PEFT adapter.")

    # ---------- SFTConfig (kwargs vary across TRL versions; introspect) ----------
    import inspect
    sft_sig = inspect.signature(SFTConfig.__init__)
    sft_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        packing=False,
        seed=args.seed,
        report_to="none",  # disable wandb/tensorboard by default
    )
    # Optimizer override (e.g. adamw_bnb_8bit) — accepted by HF TrainingArguments
    if args.optim:
        sft_kwargs["optim"] = args.optim
        print(f"Optimizer: {args.optim}")
    # Sequence-length kwarg renamed across TRL versions:
    #   <0.15: max_seq_length    >=0.15: max_length
    if "max_length" in sft_sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    # assistant_only_loss only exists in TRL >=0.14, AND in TRL >=0.18 it
    # additionally requires the chat template to contain {% generation %}
    # markers so TRL can locate the assistant turn for loss masking. Qwen3
    # family templates don't ship these markers by default.
    want_assistant_only = not args.no_assistant_only_loss
    if "assistant_only_loss" in sft_sig.parameters:
        template = tokenizer.chat_template or ""
        if want_assistant_only and "{% generation %}" not in template:
            print("WARNING: tokenizer chat template lacks `{% generation %}` markers "
                  "required by TRL for assistant_only_loss. Falling back to "
                  "loss-on-all-tokens. To enable proper assistant-only masking, "
                  "patch the chat template (see TRL docs: "
                  "https://huggingface.co/docs/trl/en/sft_trainer#train-on-assistant-messages-only) "
                  "or pass --no-assistant-only-loss to suppress this warning.")
            sft_kwargs["assistant_only_loss"] = False
        else:
            sft_kwargs["assistant_only_loss"] = want_assistant_only
    elif want_assistant_only:
        print("Note: TRL is too old for `assistant_only_loss` — loss will be on all tokens. "
              "Upgrade TRL >=0.14 for proper SFT behavior.")
    sft_config = SFTConfig(**sft_kwargs)

    print(f"Effective batch size: {args.batch * args.grad_accum}")
    print(f"Total optimizer steps (~): {len(dataset) * args.epochs // (args.batch * args.grad_accum)}")

    # ---------- trainer (tokenizer kwarg renamed processing_class in newer TRL) -
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )
    # peft_config only when training a LoRA adapter; in full mode the trainer
    # updates all model parameters directly.
    if lora_config is not None:
        trainer_kwargs["peft_config"] = lora_config
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # ---------- train ----------
    print("\nStarting training...")
    trainer.train()

    # ---------- save ----------
    artifact_label = {
        "lora":    "LoRA adapter",
        "full":    "full fine-tuned model",
        "partial": f"partial fine-tuned model (top {args.n_trainable_layers} layers)",
    }[args.mode]
    print(f"\nSaving {artifact_label} + tokenizer to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Persist a small manifest so we know what produced these weights
    manifest = {
        "mode": args.mode,
        "base_model": args.model,
        "data": str(args.data),
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora": ({
            "r": args.r, "alpha": args.alpha, "dropout": args.dropout,
            "target_modules": target_modules,
        } if args.mode == "lora" else None),
        "partial": ({
            "n_trainable_layers": args.n_trainable_layers,
            "train_embeddings": args.train_embeddings,
        } if args.mode == "partial" else None),
        "optim": args.optim,
        "effective_batch": args.batch * args.grad_accum,
        "n_train_rows": len(dataset),
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    print("\nDone. To run inference, see the LOADING section in this script's docstring.")


if __name__ == "__main__":
    main()
