"""
patch_qwen35_template.py — Inject {% generation %}/{% endgeneration %} markers
into a Qwen3.5 (or generally Qwen-style) chat template so TRL >=0.18 can
compute assistant_only_loss correctly during SFT.

WHY
    TRL >=0.18 requires the tokenizer's chat template to demarcate the
    assistant turn with Jinja {% generation %}...{% endgeneration %} markers.
    Without them, TRL falls back to loss-on-all-tokens, which dilutes the
    gradient across system + user prompt boilerplate and amplifies format-
    learning over response-content learning. (Symptom in v1: BL pass rose
    +12.5pp from format adoption while CU pass barely moved.)

    Stock Qwen3 / Qwen3.5 templates do not include the markers. This script
    inserts them around the assistant content rendering, validates the
    patched template still tokenizes cleanly, optionally runs TRL's
    `get_training_chat_template` to confirm assistant_only_loss is now usable,
    and saves the patched tokenizer to a local directory.

USAGE

    # Patch a HF model id, save to local dir
    python training/patch_qwen35_template.py \\
        --model Qwen/Qwen3.5-0.8B \\
        --out ./qwen35-0.8b-tokenizer-patched

    # Patch an existing local tokenizer (e.g. one already downloaded), in-place
    python training/patch_qwen35_template.py \\
        --model ./some/local/tokenizer/dir \\
        --out   ./some/local/tokenizer/dir-patched

    # Pass --print to dump templates before/after instead of saving
    python training/patch_qwen35_template.py --model Qwen/Qwen3.5-0.8B --print

THEN, in the train command, point at the patched dir:

    python training/train_qwen35_lora.py \\
        --data sft_dataset_v1_5/dataset.jsonl \\
        --model ./qwen35-0.8b-tokenizer-patched \\
        --output-dir ./qwen35-0.8b-skill-lora-v1.5

    (--model accepts a local path; the patched dir contains the tokenizer
    config + the original config.json copied across so the trainer can
    still load the model class.)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoConfig, AutoTokenizer


# ---------------------------------------------------------------------------
# Pattern-driven patcher
# ---------------------------------------------------------------------------


# String-level substitutions for known template families. The first one that
# matches (substring contains) wins. Pattern-level regex fallbacks follow.
_LITERAL_SUBS = [
    # ----- Qwen3.5 / Qwen3 ChatML with reasoning_content + <think> blocks ----
    # The template renders the assistant message in two if-branches based on
    # whether the message is the latest assistant turn (with <think>...</think>
    # reasoning) or an earlier turn (plain content). We wrap each branch's
    # content tokens with {% generation %}...{% endgeneration %}, leaving
    # the <|im_start|>assistant\n prefix outside the loss span.
    (
        # branch A: post-last-query (with <think>)
        "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}",
        "{{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}{% endgeneration %}",
    ),
    (
        # branch B: pre-last-query (plain content)
        "{{- '<|im_start|>' + message.role + '\\n' + content }}",
        "{{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{- content }}{% endgeneration %}",
    ),
]

# Pattern-level fallbacks for templates we haven't seen the exact form of.
_PATTERNS = [
    # Generic "role == 'assistant'" branch with '.content' attribute access
    (
        r"(\{%\s*(?:elif|if)\s+[^%]*assistant[^%]*%\}.*?)"
        r"(\{\{\s*message\.content\s*\}\})",
        r"\1{% generation %}\2{% endgeneration %}",
    ),
]


def patch_template(template: str) -> tuple[Optional[str], str]:
    """Try literal substitutions, then regex patterns. Return
    (patched_template, strategy_used) or (None, 'no-match')."""
    if "{% generation %}" in template:
        return template, "already-patched"

    # Pass 1: literal substring substitutions (most precise; targets known
    # template families by exact source string).
    new = template
    matched_subs = []
    for i, (find, rep) in enumerate(_LITERAL_SUBS):
        if find in new:
            new = new.replace(find, rep, 1)
            matched_subs.append(f"literal-{i}")
    if matched_subs:
        return new, ",".join(matched_subs)

    # Pass 2: regex pattern fallbacks
    for i, (pat, rep) in enumerate(_PATTERNS):
        new = re.sub(pat, rep, template, count=1, flags=re.DOTALL)
        if new != template:
            return new, f"regex-{i}"

    return None, "no-match"


def verify_with_trl(tokenizer) -> tuple[bool, str]:
    """Run TRL's get_training_chat_template to confirm the patched template
    is acceptable. Returns (ok, message)."""
    try:
        from trl.chat_template_utils import get_training_chat_template
    except ImportError:
        return True, "TRL not installed; skipped verification"
    try:
        get_training_chat_template(tokenizer)
        return True, "TRL accepts the patched template"
    except Exception as e:
        return False, f"TRL rejected: {e}"


# ---------------------------------------------------------------------------
# Smoke-render: produce a sample chat to eyeball the markers landed correctly
# ---------------------------------------------------------------------------


def smoke_render(tokenizer) -> str:
    msgs = [
        {"role": "system",    "content": "[SYSTEM PROMPT]"},
        {"role": "user",      "content": "[USER PROMPT]"},
        {"role": "assistant", "content": "[ASSISTANT RESPONSE]"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-0.8B",
        help="HF model id or local tokenizer dir (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--out", type=Path,
        help="Output directory for the patched tokenizer. "
             "Required unless --print is set.",
    )
    parser.add_argument(
        "--print", action="store_true", dest="just_print",
        help="Print template before and after; do not save.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.just_print and not args.out:
        parser.error("--out is required unless --print is set")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print("ERROR: tokenizer has no chat_template — nothing to patch.", file=sys.stderr)
        sys.exit(1)

    if args.verbose or args.just_print:
        print("\n----- ORIGINAL TEMPLATE -----")
        print(tokenizer.chat_template)
        print("----- END ORIGINAL -----\n")

    patched, reason = patch_template(tokenizer.chat_template)

    if patched is None:
        print(
            "ERROR: could not auto-detect the assistant render branch in this "
            "template. Patch manually:\n"
            "  1. Find the line that renders message.content for an assistant\n"
            "     role (e.g. `{{ message.content }}` inside an `assistant` branch).\n"
            "  2. Wrap that line with `{% generation %}...{% endgeneration %}`.\n"
            "  3. Re-run with --print to verify.\n"
            "  4. Or set tokenizer.chat_template = <patched_string> and save.",
            file=sys.stderr,
        )
        sys.exit(2)

    if reason == "already-patched":
        print("NOTE: template already contains {% generation %} markers; nothing to do.")
    else:
        print(f"Patched via {reason}")

    tokenizer.chat_template = patched

    if args.verbose or args.just_print:
        print("\n----- PATCHED TEMPLATE -----")
        print(tokenizer.chat_template)
        print("----- END PATCHED -----\n")

    # Smoke render
    sample = smoke_render(tokenizer)
    print("\n----- SMOKE RENDER -----")
    print(sample)
    print("----- END SMOKE -----\n")

    # Verify with TRL if available
    ok, msg = verify_with_trl(tokenizer)
    print(f"TRL verification: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        sys.exit(3)

    if args.just_print:
        return

    # Save: copy the source dir if it's local, otherwise create a fresh dir
    args.out.mkdir(parents=True, exist_ok=True)
    src = Path(args.model)
    if src.is_dir():
        # Copy non-tokenizer files so the dir is a usable drop-in replacement
        for item in src.iterdir():
            if item.is_file() and not item.name.startswith("tokenizer"):
                shutil.copy2(item, args.out / item.name)

    tokenizer.save_pretrained(str(args.out))

    # Critical: also save the model config so AutoTokenizer.from_pretrained
    # on the patched dir can resolve the tokenizer class. Without this,
    # transformers can't find config.json in the dir and falls through to
    # treating the path as an HF repo id (and the './' prefix trips its
    # validator). This step is cheap — config.json is a few KB.
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.save_pretrained(str(args.out))

    print(f"\nSaved patched tokenizer + config -> {args.out}")
    print(f"\nNext: pass `--model {args.out}` to train_qwen35_lora.py")


if __name__ == "__main__":
    main()
