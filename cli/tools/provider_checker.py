"""
provider_checker.py

Pre-flight checks for pipeline execution: confirm the Anthropic SDK and
credentials are present (OAuth via ~/.claude/.credentials.json, or API key),
plus a handful of optional Python packages. The new layout only supports
anthropic + mock providers; the old cross-provider checks (lmproxy, ollama,
iosys, z.ai, lm-studio) are gone.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def check_package(module_name: str, required: bool = True) -> CheckResult:
    """Verify a Python package is importable."""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        return CheckResult(module_name, True, "installed")
    severity = "required" if required else "optional"
    return CheckResult(module_name, not required, f"not installed ({severity})")


def check_anthropic_credentials() -> CheckResult:
    """ANTHROPIC_API_KEY env var, ~/.claude/.credentials.json, or CLAUDE_CODE_OAUTH_TOKEN."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult("Anthropic credentials", True, "ANTHROPIC_API_KEY set")
    cred_file = Path.home() / ".claude" / ".credentials.json"
    if cred_file.exists():
        return CheckResult(
            "Anthropic credentials",
            True,
            f"Claude Code OAuth file at {cred_file}",
        )
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return CheckResult(
            "Anthropic credentials",
            True,
            "CLAUDE_CODE_OAUTH_TOKEN set",
        )
    return CheckResult(
        "Anthropic credentials",
        False,
        "no credentials found (set ANTHROPIC_API_KEY, run `claude login`, or set CLAUDE_CODE_OAUTH_TOKEN)",
    )


def check_anthropic_oauth_package() -> CheckResult:
    """The anthropic-oauth package (either in-tree at /workspace or pip-installed)."""
    path = Path("/workspace/anthropic-oauth")
    if path.is_dir():
        return CheckResult("anthropic-oauth", True, f"in-tree at {path}")
    spec = importlib.util.find_spec("anthropic_oauth")
    if spec is not None:
        return CheckResult("anthropic-oauth", True, "pip-installed")
    return CheckResult(
        "anthropic-oauth",
        False,
        "missing (needed for OAuth path; pip install anthropic-oauth or clone to /workspace/anthropic-oauth)",
    )


def run_preflight_checks(profile=None) -> List[CheckResult]:
    """Run all pre-flight checks. `profile` is unused for now — kept as a
    parameter for forward compatibility with profile-specific checks."""
    return [
        check_package("anthropic", required=True),
        check_package("httpx", required=True),
        check_package("yaml", required=True),
        check_anthropic_oauth_package(),
        check_anthropic_credentials(),
        check_package("rich", required=False),
        check_package("InquirerPy", required=False),
    ]
