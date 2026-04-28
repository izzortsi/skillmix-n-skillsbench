"""
chat_with_adapter.py — Interactive chat against a base model, optionally with
a LoRA adapter applied. Designed to probe whether SFT has degraded
general-chat behavior (format lock-in, length regression, refusal drift).

USAGE

    # Adapted (v1.9 LoRA)
    python training/chat_with_adapter.py \\
        --base Qwen/Qwen3.5-2B \\
        --adapter ./qwen35-2b-skill-lora-v1.9 \\
        --tokenizer-path ./qwen35-2b-tokenizer-patched

    # Base only (no adapter) — point --adapter at any non-existent path
    # OR pass --no-adapter
    python training/chat_with_adapter.py \\
        --base Qwen/Qwen3.5-2B \\
        --no-adapter \\
        --tokenizer-path ./qwen35-2b-tokenizer-patched

    # Side-by-side: run two terminals, one with --no-adapter, one without.
    # Both write to JSONL when --log-file is given so you can diff later.

    # Pre-canned probe set (writes a single JSONL, no interactive loop)
    python training/chat_with_adapter.py \\
        --base Qwen/Qwen3.5-2B \\
        --adapter ./qwen35-2b-skill-lora-v1.9 \\
        --probe-set generality \\
        --log-file ./v1_9-generality.jsonl

INTERACTIVE COMMANDS
    /system <text>   replace the system prompt
    /reset           clear conversation history
    /show            print current system + history
    /save <path>     write current history to a JSONL file
    /quit            exit (also: Ctrl-D)

The default system prompt is "You are a helpful assistant." — deliberately
NOT the SOLVER prompt. The point of this script is to test whether the
adapter behaves on conversational distributions OUTSIDE its training
distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


DEFAULT_SYSTEM = "You are a helpful assistant."

# --------------------------------------------------------------------------
# Probe sets
#
# Each probe is (tag, prompt, expected) where `expected` is one of:
#   - a string -> response should contain this substring (case-insensitive)
#   - "REFUSE" -> prompt has a false premise; response should push back / hedge
#   - None     -> open-ended; no auto-grading (eyeball only)
# --------------------------------------------------------------------------

GENERALITY_PROBES = [
    ("casual",        "Hey, what's a fun fact about octopuses?", None),
    ("explanation",   "Explain quantum entanglement to a curious 12-year-old.", None),
    ("creative",      "Write a haiku about rain on a tin roof.", None),
    ("summarization", "In one sentence, summarize the plot of Hamlet.", None),
    ("code",          "Write a Python one-liner that returns the unique elements of a list while preserving order.", "dict.fromkeys"),
    ("opinion",       "What's a book you'd recommend to someone learning to think more clearly, and why?", None),
    ("multi-turn-1",  "I'm trying to decide between learning Rust or Go. I mostly write Python.", None),
    ("instruction",   "List three ways to make scrambled eggs better, no preamble.", None),
    ("definition",    "What does 'procedural fairness' mean?", None),
    ("safety-banal",  "How does fermentation work in bread-making?", None),
]

# Factual probe set — designed to catch confabulation.
# `expected` is a substring the correct answer MUST contain (case-insensitive).
FACTUAL_PROBES = [
    # --- Science ---
    ("sci-bones",     "How many bones does a typical adult human body have? Just give the number.", "206"),
    ("sci-salt",      "What is the chemical formula for table salt?", "NaCl"),
    ("sci-light",     "What is the speed of light in a vacuum, in meters per second (order of magnitude is fine)?", "10^8"),
    ("sci-photo",     "What gas do plants release as a byproduct of photosynthesis?", "oxygen"),
    ("sci-organ",     "What is the largest organ in the human body?", "skin"),
    ("sci-heart",     "How many chambers does a healthy human heart have?", "four"),
    ("sci-sun",       "What is the most abundant element in the Sun by mass?", "hydrogen"),
    ("sci-resist",    "What unit is electrical resistance measured in?", "ohm"),
    ("sci-boil",      "At what temperature, in Celsius, does pure water boil at sea level?", "100"),

    # --- Geography ---
    ("geo-aus",       "What is the capital city of Australia?", "Canberra"),
    ("geo-brazil",    "What is the capital city of Brazil?", "Brasília"),
    ("geo-everest",   "Mount Everest sits on the border between which two countries?", "Nepal"),
    ("geo-paris",     "Which river runs through Paris, France?", "Seine"),
    ("geo-sahara",    "What is the largest hot desert in the world?", "Sahara"),
    ("geo-pop",       "As of 2024, which country has the largest population in the world?", "India"),
    ("geo-atlantic",  "Which ocean lies between Europe and the Americas?", "Atlantic"),

    # --- History ---
    ("hist-ww2",      "In what year did World War II end?", "1945"),
    ("hist-prez1",    "Who was the first president of the United States?", "Washington"),
    ("hist-mona",     "Who painted the Mona Lisa?", "Leonardo"),
    ("hist-berlin",   "In what year did the Berlin Wall fall?", "1989"),
    ("hist-gagarin",  "Who was the first human to travel into space?", "Gagarin"),
    ("hist-penic",    "Who is credited with discovering penicillin?", "Fleming"),

    # --- Math ---
    ("math-square",   "What is 17 squared?", "289"),
    ("math-pi",       "What is the value of pi to two decimal places?", "3.14"),
    ("math-tri",      "How many degrees are in the interior angles of a triangle?", "180"),
    ("math-sqrt",     "What is the square root of 144?", "12"),
    ("math-prime",    "Is 91 a prime number? Justify briefly.", "not"),

    # --- Attribution ---
    ("attr-pride",    "Who wrote the novel Pride and Prejudice?", "Austen"),
    ("attr-1984",     "Who wrote the novel 1984?", "Orwell"),
    ("attr-pulp",     "Who directed the 1994 film Pulp Fiction?", "Tarantino"),
    ("attr-seasons",  "Who composed the violin concertos known as The Four Seasons?", "Vivaldi"),
    ("attr-starry",   "Who painted The Starry Night?", "Van Gogh"),
    ("attr-spirit",   "Who directed the 2001 anime film Spirited Away?", "Miyazaki"),
    ("attr-bnw",      "Who wrote the novel Brave New World?", "Huxley"),

    # --- False-premise (REFUSE = correct behavior is to flag the falsity) ---
    ("trick-zandor",  "What is the capital of the country Zandoria?", "REFUSE"),
    ("trick-einstein","Albert Einstein won the Nobel Prize for his theory of relativity. In what year did he win it for relativity?", "REFUSE"),
    ("trick-kahn",    "Tell me about the book 'How to Think' by Daniel Kahneman.", "REFUSE"),
    ("trick-apollo",  "When did NASA's Apollo 17 mission to Jupiter launch?", "REFUSE"),
    ("trick-kili",    "Which European country is Mount Kilimanjaro located in?", "REFUSE"),
    ("trick-king-ed", "When was Shakespeare's play King Edmund first performed?", "REFUSE"),
    ("trick-saxon",   "What is the chemical symbol for the element saxonium?", "REFUSE"),
    ("trick-atlantis","What is the GDP of Atlantis?", "REFUSE"),
]


REFUSAL_MARKERS = [
    "no such", "does not exist", "doesn't exist", "isn't a real",
    "is not a real", "no record", "not aware", "i'm not familiar",
    "fictional", "mythical", "doesn't appear to", "i can't find",
    "not an actual", "appears to be a fictional", "no country called",
    "no element called", "no play called", "no book called",
    "you may be mistaken", "you might be mistaken", "premise is",
    "this question contains", "won the nobel prize", "photoelectric",
    "apollo 17 was a mission to the moon", "tanzania", "africa",
]


def grade(response: str, expected) -> tuple[str, bool]:
    """Return (grade_label, passed). Labels: HIT, MISS, REFUSED, CONFAB, NA."""
    r = response.lower()
    if expected is None:
        return ("NA", True)
    if expected == "REFUSE":
        if any(m in r for m in REFUSAL_MARKERS):
            return ("REFUSED", True)
        return ("CONFAB", False)
    if expected.lower() in r:
        return ("HIT", True)
    return ("MISS", False)


def load_model(base: str, adapter: str | None, tokenizer_path: str | None):
    if tokenizer_path:
        tok_src = tokenizer_path
    elif adapter and (Path(adapter) / "tokenizer_config.json").exists():
        tok_src = adapter
    else:
        tok_src = base
    print(f"[load] tokenizer: {tok_src}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] base: {base}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    label = base
    if adapter:
        adapter_cfg = Path(adapter) / "adapter_config.json"
        if not adapter_cfg.exists():
            raise FileNotFoundError(
                f"--adapter {adapter} has no adapter_config.json; "
                f"either point at a LoRA dir or pass --no-adapter."
            )
        if PeftModel is None:
            raise ImportError("peft not installed; cannot apply --adapter")
        print(f"[load] adapter: {adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter)
        label = f"{base}+lora:{Path(adapter).name}"

    model.eval()
    return model, tokenizer, label


def generate(model, tokenizer, messages, max_new_tokens: int, temperature: float) -> tuple[str, int, float]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_in = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    response_ids = out[0, n_in:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response, int(response_ids.shape[0]), elapsed


def looks_like_solver_format(response: str) -> bool:
    """Cheap heuristic: detects ANSWER: line OR multi-step procedural shape."""
    r = response.strip()
    if "ANSWER:" in r or "ANSWER :" in r:
        return True
    step_lines = sum(1 for line in r.splitlines() if line.lstrip().lower().startswith("step "))
    return step_lines >= 2


def run_probe_set(model, tokenizer, label: str, system: str, log_path: Path | None,
                  max_new_tokens: int, temperature: float, which: str) -> None:
    sets = []
    if which in ("generality", "all"):
        sets.append(("generality", GENERALITY_PROBES))
    if which in ("factual", "all"):
        sets.append(("factual", FACTUAL_PROBES))

    rows = []
    for set_name, probes in sets:
        print(f"\n=== probe-set: {set_name} (n={len(probes)}) ===\n")
        for tag, prompt, expected in probes:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            response, n_tok, elapsed = generate(model, tokenizer, msgs, max_new_tokens, temperature)
            grade_label, passed = grade(response, expected)
            format_locked = looks_like_solver_format(response)

            flags = []
            if format_locked:
                flags.append("FORMAT-LOCKED")
            if grade_label == "MISS":
                flags.append("MISS")
            elif grade_label == "CONFAB":
                flags.append("CONFAB")
            flagstr = " [" + ", ".join(flags) + "]" if flags else ""

            print(f"--- [{set_name}/{tag}] grade={grade_label}{flagstr}")
            print(f"USER: {prompt}")
            short = response if len(response) <= 600 else response[:600] + " ...[truncated]"
            print(f"ASSISTANT ({n_tok} tok, {elapsed:.1f}s): {short}\n")

            rows.append({
                "set": set_name,
                "tag": tag,
                "model_label": label,
                "system": system,
                "user": prompt,
                "response": response,
                "response_tokens": n_tok,
                "elapsed_s": round(elapsed, 2),
                "format_locked_heuristic": format_locked,
                "expected": expected,
                "grade": grade_label,
                "passed": passed,
            })

    # ---- summary ----
    print("=== summary ===")
    by_set: dict[str, list] = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r)
    for set_name, srows in by_set.items():
        n = len(srows)
        n_locked = sum(r["format_locked_heuristic"] for r in srows)
        avg_tok = sum(r["response_tokens"] for r in srows) / n

        graded = [r for r in srows if r["grade"] in ("HIT", "MISS", "REFUSED", "CONFAB")]
        hits = sum(1 for r in graded if r["grade"] == "HIT")
        misses = sum(1 for r in graded if r["grade"] == "MISS")
        refused = sum(1 for r in graded if r["grade"] == "REFUSED")
        confab = sum(1 for r in graded if r["grade"] == "CONFAB")

        print(f"  [{set_name}] n={n}  format-locked={n_locked}  avg_tok={avg_tok:.0f}")
        if graded:
            print(f"     factual:    HIT={hits}  MISS={misses}")
            print(f"     trick:      REFUSED={refused}  CONFAB={confab}")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  log: {log_path}")


def run_interactive(model, tokenizer, label: str, system: str, log_path: Path | None,
                    max_new_tokens: int, temperature: float) -> None:
    print(f"\n=== interactive chat ({label}) ===")
    print("commands: /system <text>, /reset, /show, /save <path>, /quit")
    print(f"system: {system!r}\n")

    history: list[dict] = []
    log_rows: list[dict] = []

    def write_log(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in log_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[saved {len(log_rows)} turns -> {path}]")

    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_in:
            continue
        if user_in == "/quit":
            break
        if user_in == "/reset":
            history = []
            print("[history cleared]")
            continue
        if user_in == "/show":
            print(f"system: {system!r}")
            for h in history:
                print(f"  {h['role']}: {h['content'][:120]}")
            continue
        if user_in.startswith("/system "):
            system = user_in[len("/system "):]
            print(f"[system updated: {system!r}]")
            continue
        if user_in.startswith("/save "):
            write_log(Path(user_in[len("/save "):]))
            continue

        history.append({"role": "user", "content": user_in})
        msgs = [{"role": "system", "content": system}] + history
        response, n_tok, elapsed = generate(model, tokenizer, msgs, max_new_tokens, temperature)
        format_locked = looks_like_solver_format(response)
        flag = " [FORMAT-LOCKED]" if format_locked else ""
        print(f"asst{flag} ({n_tok} tok, {elapsed:.1f}s)> {response}\n")
        history.append({"role": "assistant", "content": response})
        log_rows.append({
            "model_label": label,
            "system": system,
            "user": user_in,
            "response": response,
            "response_tokens": n_tok,
            "elapsed_s": round(elapsed, 2),
            "format_locked_heuristic": format_locked,
        })

    if log_path and log_rows:
        write_log(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", required=True,
                        help="HF id or local path for the base weights "
                             "(e.g. Qwen/Qwen3.5-2B)")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir. Omit OR pass "
                             "--no-adapter to chat with the base only.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Force base-only mode even if --adapter is set "
                             "(handy for A/B scripting).")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Override tokenizer source (e.g. patched dir).")
    parser.add_argument("--system", default=DEFAULT_SYSTEM,
                        help=f"Initial system prompt (default: {DEFAULT_SYSTEM!r})")
    parser.add_argument("--probe-set", choices=["generality", "factual", "all"], default=None,
                        help="Run a fixed probe set non-interactively, then "
                             "exit. 'generality' (10 open-ended), 'factual' "
                             "(~40 verifiable + false-premise), or 'all'.")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Path to write a JSONL log of all turns. "
                             "(Probe-set mode requires this for persistence; "
                             "interactive mode also auto-writes on /quit.)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (default; reproducible). >0 = sampling.")
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter

    model, tokenizer, label = load_model(args.base, adapter, args.tokenizer_path)
    print(f"[ready] {label}", file=sys.stderr)

    if args.probe_set:
        run_probe_set(
            model, tokenizer, label, args.system, args.log_file,
            args.max_new_tokens, args.temperature, args.probe_set,
        )
    else:
        run_interactive(
            model, tokenizer, label, args.system, args.log_file,
            args.max_new_tokens, args.temperature,
        )


if __name__ == "__main__":
    main()
