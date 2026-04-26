"""
main.py

Unified CLI entry point for the llm-skills pipeline suite.

Usage:
    cd cli
    python -m cli.main <command> [options]

Commands:
    run       Run pipeline stages (full or partial)
    config    Manage experiment profiles
    status    Show pipeline run status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_repo = _project.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills",
        description="Unified CLI for the llm-skills pipeline suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run       Run pipeline stages (full or partial)
  config    Manage experiment profiles (create, list, show, delete)
  status    Show pipeline run directory status
  setup     Pre-flight checks and environment validation

Examples:
  python3 -m cli.main run --stages all --minimal --clean
  python3 -m cli.main run --stages 1-4 --profile my-experiment
  python3 -m cli.main run --stages evaluation
  python3 -m cli.main run --interactive
  python3 -m cli.main config list
  python3 -m cli.main config create --interactive
  python3 -m cli.main status
  python3 -m cli.main setup
""",
    )
    parser.add_argument("command", help="Command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Command arguments")

    args = parser.parse_args()

    if args.command == "run":
        sys.argv = ["llm-skills run"] + args.args
        from cli.command_run import main as cmd_main
        cmd_main()

    elif args.command == "config":
        sys.argv = ["llm-skills config"] + args.args
        from cli.command_config import main as cmd_main
        cmd_main()

    elif args.command == "status":
        sys.argv = ["llm-skills status"] + args.args
        from cli.command_status import main as cmd_main
        cmd_main()

    elif args.command == "setup":
        sys.argv = ["llm-skills setup"] + args.args
        from cli.command_setup import main as cmd_main
        cmd_main()

    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
