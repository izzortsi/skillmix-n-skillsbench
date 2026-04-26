"""
command_run.py

The `run` command: execute pipeline stages with profile-based configuration.

Usage:
    python -m cli.main run [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.pipeline_profile import PipelineProfile, apply_minimal
from tools.profile_loader import load_profile, save_profile, PROFILES_DIR
from orchestration.pipeline_executor import execute_pipeline
from cli.rich_ui import print_header, print_summary


_PROVIDER_TIERS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


def _apply_provider_shortcut(profile: PipelineProfile, tier: str) -> None:
    """Set every LLM role on the profile to anthropic:<tier>."""
    model = _PROVIDER_TIERS[tier]
    spec = f"anthropic:{model}"
    profile.catalog_provider = spec
    profile.compose_provider = spec
    profile.verify_verifier = spec
    profile.verify_refiner = spec
    profile.skillmix_student = spec
    profile.skillmix_judge = spec
    profile.solve_model = model           # solver takes a raw model name
    profile.bench_provider = spec
    profile.bench_judge = spec


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills run",
        description="Run pipeline stages (catalog -> compose -> verify / skillmix / solve / bench)",
    )
    parser.add_argument("--profile", type=str, default="",
                        help="Named profile from profiles/ directory")
    parser.add_argument("--stages", type=str, default="all",
                        help="Range: all | extract | eval | <id> | <id1>,<id2>,...")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe previous output and re-run all stages")
    parser.add_argument("--clean-stages", action="store_true",
                        help="Wipe output of only the requested stages before running")
    parser.add_argument("--minimal", action="store_true",
                        help="Override with minimal settings (fewest API calls)")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Override run directory path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stage subprocess output")
    parser.add_argument("--anthropic", type=str, nargs="?", const="haiku",
                        choices=list(_PROVIDER_TIERS.keys()),
                        help="Override every LLM role to anthropic:<tier>.")
    parser.add_argument("--bench-models", type=str, default="",
                        help="Comma-separated 'provider:model' specs for the bench sweep. "
                             "Overrides profile.bench_models (and ignores profile.bench_provider).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    # Auto-load 'default' when no --profile is given so edits to default.yaml
    # take effect without an explicit flag. Fall back to dataclass defaults
    # only if the YAML is missing entirely.
    profile_name = args.profile or "default"
    try:
        profile = load_profile(profile_name)
    except FileNotFoundError:
        if args.profile:
            print(f"Profile {args.profile!r} not found in {PROFILES_DIR}")
            sys.exit(1)
        profile = PipelineProfile()

    if args.anthropic:
        _apply_provider_shortcut(profile, args.anthropic)

    if args.bench_models.strip():
        profile.bench_models = [
            s.strip() for s in args.bench_models.split(",") if s.strip()
        ]

    if args.minimal:
        apply_minimal(profile)

    if args.run_dir:
        profile.run_dir = args.run_dir

    verbose = not args.quiet

    print_header(
        profile_name=profile.profile_name,
        stages=args.stages,
        run_dir=profile.run_dir,
        is_minimal=args.minimal,
        is_clean=args.clean,
    )

    results = execute_pipeline(
        profile=profile,
        stage_range=args.stages,
        repo_root=repo_root,
        clean=args.clean,
        clean_stages=args.clean_stages,
        verbose=verbose,
    )

    print_summary(results)
