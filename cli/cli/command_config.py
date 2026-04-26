"""
command_config.py

The "config" command: manage experiment profiles.

Usage:
    python -m cli.main config <subcommand> [args]

Subcommands:
    create NAME     Create a new profile with default values
    list            List all saved profiles
    show NAME       Display profile contents
    delete NAME     Delete a profile
"""

from __future__ import annotations

import argparse
import sys

from config.pipeline_profile import PipelineProfile
from tools.profile_loader import (
    load_profile,
    save_profile,
    list_profiles,
    delete_profile,
    PROFILES_DIR,
)
from cli.rich_ui import console, print_profile, print_profiles_list


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills config",
        description="Manage experiment profiles",
    )
    parser.add_argument("subcommand", choices=["create", "list", "show", "delete"],
                        help="Config subcommand")
    parser.add_argument("name", nargs="?", default="",
                        help="Profile name (for create, show, delete)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Create profile interactively (for 'create' subcommand)")

    args = parser.parse_args()

    if args.subcommand == "list":
        print_profiles_list(list_profiles())

    elif args.subcommand == "create":
        if args.interactive:
            console.print(
                "[yellow]Interactive profile creation is not wired for the new "
                "stage set yet. Create a YAML manually by copying "
                "profiles/default.yaml.[/yellow]"
            )
            sys.exit(2)
        if not args.name:
            console.print("Usage: llm-skills config create <name>")
            sys.exit(1)
        profile = PipelineProfile(profile_name=args.name)
        path = save_profile(profile)
        console.print(f"Created profile '[cyan]{args.name}[/cyan]' at {path}")
        console.print(
            f"Edit the YAML, then run: "
            f"[bold]python3 -m cli.main run --profile {args.name}[/bold] (from cli/)"
        )

    elif args.subcommand == "show":
        if not args.name:
            console.print("Usage: llm-skills config show <name>")
            sys.exit(1)
        try:
            profile = load_profile(args.name)
        except FileNotFoundError:
            console.print(f"Profile '{args.name}' not found in {PROFILES_DIR}")
            sys.exit(1)
        print_profile(profile)

    elif args.subcommand == "delete":
        if not args.name:
            console.print("Usage: llm-skills config delete <name>")
            sys.exit(1)
        if delete_profile(args.name):
            console.print(f"Deleted profile '[cyan]{args.name}[/cyan]'")
        else:
            console.print(f"Profile '{args.name}' not found")
            sys.exit(1)
