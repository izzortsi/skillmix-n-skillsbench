"""
provider_discovery.py

Probe the three supported providers so `llm-skills setup` can present an
interactive menu grounded in what's actually reachable:

    anthropic  -- OAuth via Claude Code credentials or API key
    ollama     -- local Ollama daemon at $OLLAMA_HOST (or localhost:11434)
    mock       -- always available, always returns canned text

For anthropic we hard-code the known Claude tiers (the inferred-from-env
path is too chatty for a setup menu). For ollama we call `/api/tags` to list
installed models; on failure we return reachable=False.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


# Known Claude tiers, ordered by capability. Used as the menu shown when
# the user picks "anthropic" at setup time.
ANTHROPIC_MODELS = [
    ("haiku",  "claude-haiku-4-5-20251001"),
    ("sonnet", "claude-sonnet-4-6"),
    ("opus",   "claude-opus-4-7"),
]


@dataclass
class ProviderInfo:
    name: str
    reachable: bool
    detail: str = ""
    models: List[str] = field(default_factory=list)


def _discover_anthropic() -> ProviderInfo:
    """Reachable iff ANTHROPIC_API_KEY is set, Claude Code credentials exist,
    or CLAUDE_CODE_OAUTH_TOKEN is set. Model list is the Claude tier catalog."""
    models = [m for (_, m) in ANTHROPIC_MODELS]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderInfo("anthropic", True, "ANTHROPIC_API_KEY set", models)
    cred = Path.home() / ".claude" / ".credentials.json"
    if cred.exists():
        return ProviderInfo("anthropic", True, f"OAuth via {cred}", models)
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return ProviderInfo("anthropic", True, "CLAUDE_CODE_OAUTH_TOKEN set", models)
    return ProviderInfo("anthropic", False, "no credentials found", models)


def _discover_ollama(host: str = "") -> ProviderInfo:
    """GET {host}/api/tags and collect model names. Returns reachable=False on error."""
    resolved = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    try:
        req = urllib.request.Request(f"{resolved}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return ProviderInfo("ollama", False, f"{resolved}: {e}", [])

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return ProviderInfo("ollama", False, f"malformed /api/tags from {resolved}", [])

    models = sorted(str(m.get("name", "")) for m in payload.get("models", []) if m.get("name"))
    detail = f"{resolved} ({len(models)} model{'s' if len(models) != 1 else ''})"
    return ProviderInfo("ollama", True, detail, models)


def _discover_mock() -> ProviderInfo:
    return ProviderInfo("mock", True, "always available", ["mock"])


def discover_providers(ollama_host: str = "") -> Dict[str, ProviderInfo]:
    """Probe every supported provider. Returns {name: ProviderInfo}."""
    return {
        "anthropic": _discover_anthropic(),
        "ollama":    _discover_ollama(ollama_host),
        "mock":      _discover_mock(),
    }
