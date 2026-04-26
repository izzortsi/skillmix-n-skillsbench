"""
command_status.py

The "status" command: display pipeline run directory state.

Usage:
    python -m cli.main status [--run-dir PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.pipeline_profile import PipelineProfile
from tools.output_inspector import inspect_run_dir
from cli.rich_ui import console, print_status_table


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills status",
        description="Show pipeline run status",
    )
    parser.add_argument("--run-dir", type=str, default="",
                        help="Path to run directory")
    parser.add_argument("--profile", type=str, default="",
                        help="Load run-dir from named profile")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Auto-load 'default' so status picks up edits to default.yaml.
        from tools.profile_loader import load_profile
        try:
            profile = load_profile(args.profile or "default")
            run_dir = Path(profile.run_dir)
        except FileNotFoundError:
            if args.profile:
                print(f"Profile {args.profile!r} not found")
                return
            run_dir = Path(PipelineProfile().run_dir)

    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir

    if not run_dir.exists():
        console.print(f"Run directory does not exist: {run_dir}")
        console.print(f"Run the pipeline first: [bold]llm-skills run --stages all[/bold]")
        return

    statuses = inspect_run_dir(run_dir)
    print_status_table(str(run_dir), statuses)
