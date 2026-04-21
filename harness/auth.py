"""
Credential resolution for the harness.

Priority:
  1. ANTHROPIC_API_KEY  -> api_key auth
  2. ~/.claude/.credentials.json  -> oauth (Claude Code)
  3. CLAUDE_CODE_OAUTH_TOKEN env var  -> oauth
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple


def _claude_credentials_path() -> Path:
    return Path.home() / ".claude" / ".credentials.json"


def _load_claude_oauth_token() -> Optional[str]:
    path = _claude_credentials_path()
    if not path.exists():
        return None
    try:
        creds = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return creds.get("claudeAiOauth", {}).get("accessToken")


def resolve_credentials() -> Tuple[str, str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return "api_key", key
    token = _load_claude_oauth_token()
    if token:
        return "oauth", token
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return "oauth", token
    raise ValueError(
        "No Anthropic credentials. Set ANTHROPIC_API_KEY, run `claude login`, "
        "or set CLAUDE_CODE_OAUTH_TOKEN."
    )
