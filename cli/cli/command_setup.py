"""
command_setup.py

The `setup` command:
  1. Pre-flight checks (anthropic + credentials + pyyaml + optional rich)
  2. Discover reachable providers (anthropic tiers, ollama models, mock)
  3. Interactively pick a FRONTIER provider+model (used for catalog / compose /
     verify / skillmix_judge / bench_judge / solve_model)
  4. Interactively pick a STUDENT provider+model (used for skillmix_student /
     bench_provider — typically a cheaper/weaker model)
  5. Write the selection back to the chosen profile (default: the loaded
     profile, or `default.yaml` if none was loaded)

Non-interactive: pass `--non-interactive` (or `-q`) to skip steps 3-5 and
only run preflight + discovery.

Usage:
    python -m cli.main setup [--profile NAME] [--non-interactive]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from config.pipeline_profile import PipelineProfile
from tools.profile_loader import load_profile, save_profile, PROFILES_DIR
from tools.provider_checker import run_preflight_checks
from tools.provider_discovery import (
    ANTHROPIC_MODELS,
    ProviderInfo,
    discover_providers,
)

try:
    from cli.rich_ui import console, HAS_RICH
except ImportError:  # pragma: no cover
    HAS_RICH = False
    class _Fallback:
        def print(self, *a, **k):
            print(" ".join(str(x) for x in a))
    console = _Fallback()


# ---------------------------------------------------------------------------
# Small UI helpers (plain input() so no InquirerPy dependency is required)
# ---------------------------------------------------------------------------


def _print(msg: str) -> None:
    if HAS_RICH:
        console.print(msg)
    else:
        print(msg)


def _pick(prompt: str, options: List[str], default_index: int = 0) -> int:
    """Numbered-list picker. Empty input picks `default_index`. Returns an index."""
    _print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = "[default]" if i - 1 == default_index else ""
        _print(f"  {i}. {opt} {marker}")
    while True:
        raw = input("select [1-{}]: ".format(len(options))).strip()
        if not raw:
            return default_index
        try:
            n = int(raw)
            if 1 <= n <= len(options):
                return n - 1
        except ValueError:
            pass
        _print(f"(enter 1..{len(options)} or blank for default)")


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------


def _select_provider_model(
    label: str,
    providers: dict,
    prefer_local: bool = False,
) -> Tuple[str, str]:
    """Prompt for provider then model; return (provider_name, model_name)."""
    names = list(providers.keys())
    # Reorder so unreachable providers sort last without dropping them.
    ordered = sorted(names, key=lambda n: (0 if providers[n].reachable else 1, n))

    labeled = [
        f"{n:<10} {'[reachable]' if providers[n].reachable else '[unreachable]':<14} {providers[n].detail}"
        for n in ordered
    ]
    # Default to ollama when prefer_local=True and reachable; else first reachable.
    default_idx = 0
    if prefer_local and providers.get("ollama") and providers["ollama"].reachable:
        default_idx = ordered.index("ollama")
    else:
        for i, n in enumerate(ordered):
            if providers[n].reachable:
                default_idx = i
                break

    chosen_provider = ordered[_pick(f"{label} provider:", labeled, default_idx)]
    info = providers[chosen_provider]

    if not info.reachable:
        _print(f"warning: {chosen_provider} is not reachable ({info.detail}); continuing anyway")

    if chosen_provider == "mock":
        return chosen_provider, "mock"

    if chosen_provider == "anthropic":
        tiers = [f"{label} ({model})" for label, model in ANTHROPIC_MODELS] + ["custom..."]
        default_tier = len(ANTHROPIC_MODELS) - 1  # opus
        idx = _pick(f"{label} anthropic tier:", tiers, default_tier)
        if idx == len(ANTHROPIC_MODELS):
            model = _ask("custom model name", default=ANTHROPIC_MODELS[default_tier][1])
        else:
            model = ANTHROPIC_MODELS[idx][1]
        return chosen_provider, model

    if chosen_provider == "ollama":
        models = list(info.models)
        if not models:
            _print(
                "warning: no ollama models installed; pull one first with "
                "`ollama pull llama3.1:8b`"
            )
            model = _ask(f"{label} ollama model", default="llama3.1:8b")
        else:
            idx = _pick(f"{label} ollama model:", models, 0)
            model = models[idx]
        return chosen_provider, model

    return chosen_provider, ""


def _apply_to_profile(profile: PipelineProfile, frontier: Tuple[str, str], student: Tuple[str, str]) -> None:
    """Write the selected specs into frontier/student slots on the profile."""
    fp = f"{frontier[0]}:{frontier[1]}"
    sp = f"{student[0]}:{student[1]}"
    profile.catalog_provider = fp
    profile.compose_provider = fp
    profile.verify_verifier = fp
    profile.verify_refiner = fp
    profile.skillmix_judge = fp
    profile.bench_judge = fp
    profile.solve_model = frontier[1]     # s4.m4 takes a raw model name
    profile.skillmix_student = sp
    profile.bench_provider = sp


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills setup",
        description="Preflight + interactive provider/model selection",
    )
    parser.add_argument(
        "--profile", type=str, default="",
        help="Profile to update (default: loads 'default' if no name given)",
    )
    parser.add_argument(
        "--non-interactive", "-q", action="store_true",
        help="Only run preflight + discovery; skip provider selection",
    )
    args = parser.parse_args()

    if HAS_RICH:
        console.print("[bold]llm-skills setup[/bold]\n")
    else:
        _print("llm-skills setup\n")

    # --- Step 1: preflight -------------------------------------------------
    results = run_preflight_checks()
    passed = failed = 0
    for r in results:
        if r.passed:
            passed += 1
            _print(f"  PASS  {r.name}: {r.message}")
        else:
            failed += 1
            _print(f"  FAIL  {r.name}: {r.message}")
    _print(f"\n{passed} passed, {failed} failed out of {len(results)} checks\n")

    # --- Step 2: discovery -------------------------------------------------
    providers = discover_providers()
    _print("provider discovery:")
    for name, info in providers.items():
        tag = "[reachable]" if info.reachable else "[unreachable]"
        _print(f"  {tag:<14} {name:<10} {info.detail}")

    if args.non_interactive:
        return

    # --- Step 3: load (or create) profile ----------------------------------
    profile_name = args.profile or "default"
    try:
        profile = load_profile(profile_name)
        _print(f"\nloaded profile: {profile_name} (from {PROFILES_DIR / f'{profile_name}.yaml'})")
    except FileNotFoundError:
        profile = PipelineProfile(profile_name=profile_name)
        _print(f"\nprofile {profile_name!r} not found — starting from defaults")

    # --- Step 4: frontier + student selection ------------------------------
    frontier = _select_provider_model("frontier", providers, prefer_local=False)
    student = _select_provider_model("student", providers, prefer_local=True)

    _print(f"\nfrontier: {frontier[0]}:{frontier[1]}")
    _print(f"student:  {student[0]}:{student[1]}")

    # --- Step 5: persist ---------------------------------------------------
    _apply_to_profile(profile, frontier, student)
    save_choice = _pick(
        "save to...",
        [
            f"update {profile.profile_name}.yaml",
            "write to new profile (enter name)",
            "discard (print but do not save)",
        ],
        default_index=0,
    )
    if save_choice == 2:
        _print("\nchanges discarded. Previewed profile:")
        _print(f"  catalog_provider  = {profile.catalog_provider}")
        _print(f"  compose_provider  = {profile.compose_provider}")
        _print(f"  skillmix_student  = {profile.skillmix_student}")
        _print(f"  bench_provider    = {profile.bench_provider}")
        _print(f"  solve_model       = {profile.solve_model}")
        return

    if save_choice == 1:
        new_name = _ask("new profile name", default=f"{profile.profile_name}-custom")
        profile.profile_name = new_name

    path = save_profile(profile)
    _print(f"\nwrote profile: {path}")
    _print(
        f"\ntry it:\n  python -m cli.main run --profile {profile.profile_name} --stages extract --minimal --clean"
    )


if __name__ == "__main__":
    main()
