"""
core/providers.py

LLM provider abstraction. Non-streaming — use `harness.LinearAgent` for
streaming + thinking-block capture.

Supported providers:
  - "anthropic"    Anthropic API. OAuth via ~/.claude/.credentials.json when
                   present (same code path as harness/agent.py), falls back
                   to ANTHROPIC_API_KEY.
  - "mock"         Deterministic echo for tests.

Usage:
    from core.providers import create_provider
    provider = create_provider("anthropic", "claude-opus-4-7")
    result = provider.chat([{"role": "user", "content": "hi"}])
    text = result.text
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Credential resolution (OAuth bridge from Claude Code)
# ---------------------------------------------------------------------------


def _claude_credentials_path() -> Path:
    return Path.home() / ".claude" / ".credentials.json"


def _load_claude_oauth_token() -> Optional[str]:
    """Extract OAuth access token from Claude Code's ~/.claude/.credentials.json."""
    path = _claude_credentials_path()
    if not path.exists():
        return None
    try:
        creds = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    oauth = creds.get("claudeAiOauth", {})
    return oauth.get("accessToken")


def resolve_anthropic_credentials() -> tuple[str, str]:
    """Return (auth_mode, token). auth_mode is 'api_key' or 'oauth'.

    Priority:
      1. ANTHROPIC_API_KEY env var -> api_key
      2. Claude Code OAuth token -> oauth
      3. CLAUDE_CODE_OAUTH_TOKEN env var -> oauth
    """
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


# ---------------------------------------------------------------------------
# OAuth client bridge
#
# Hand-rolled OAuth (just oauth-2025-04-20 beta + auth_token on
# anthropic.Anthropic) 429s on claude-opus-4-7 and claude-sonnet-4-6 because
# the OAuth endpoint expects the `system` field as a list of text content
# blocks with CLAUDE_CODE_IDENTITY as the first block, not a concatenated
# string. The anthropic-oauth package's RequestTransformer handles that
# plus the header / URL fingerprint overrides.
#
# Mirrors harness/agent.py:_build_oauth_client — keep in sync.
# ---------------------------------------------------------------------------


CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _build_oauth_client():
    """Construct an anthropic.Anthropic bound to the Claude Code OAuth session."""
    import anthropic
    import httpx

    # Prefer the in-tree checkout when present; fall back to pip-installed.
    _anthropic_oauth_path = "/workspace/anthropic-oauth"
    if os.path.isdir(_anthropic_oauth_path) and _anthropic_oauth_path not in sys.path:
        sys.path.insert(0, _anthropic_oauth_path)

    from anthropic_oauth import (  # noqa: E402
        OAuthManager,
        OAuthTokens,
        OAuthTransport,
        RequestTransformer,
    )

    class ClaudeCodeOAuthManager(OAuthManager):
        """Read/write ~/.claude/.credentials.json in Claude Code's shape so the
        desktop session's tokens are reused (and refreshed in place)."""

        def __init__(self):
            super().__init__(token_path="~/.claude/.credentials.json")

        def load_tokens(self):
            try:
                raw = self._token_path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                return None
            cco = (data or {}).get("claudeAiOauth") or {}
            access = cco.get("accessToken")
            refresh = cco.get("refreshToken")
            expires_at_ms = cco.get("expiresAt")
            if not (
                isinstance(access, str)
                and isinstance(refresh, str)
                and isinstance(expires_at_ms, (int, float))
            ):
                return None
            self._tokens = OAuthTokens(
                access=access,
                refresh=refresh,
                expires=float(expires_at_ms) / 1000.0,
            )
            return self._tokens

        def save_tokens(self, tokens):
            try:
                raw = self._token_path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                data = {}
            data.setdefault("claudeAiOauth", {})
            data["claudeAiOauth"]["accessToken"] = tokens.access
            data["claudeAiOauth"]["refreshToken"] = tokens.refresh
            data["claudeAiOauth"]["expiresAt"] = int(tokens.expires * 1000)
            self._token_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._tokens = tokens

    manager = ClaudeCodeOAuthManager()
    transport_kwargs: Dict[str, Any] = {"transformer": RequestTransformer()}
    # Older anthropic-oauth builds do not expose auto_auth; guard for it.
    if "auto_auth" in inspect.signature(OAuthTransport.__init__).parameters:
        transport_kwargs["auto_auth"] = False
    transport = OAuthTransport(manager, **transport_kwargs)
    return anthropic.Anthropic(
        api_key="placeholder",
        http_client=httpx.Client(transport=transport),
    )


# ---------------------------------------------------------------------------
# Common result type
# ---------------------------------------------------------------------------


@dataclass
class ChatResult:
    text: str
    usage: Dict[str, int]
    raw: Any = None


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"


class AnthropicProvider:
    """Anthropic chat provider with OAuth-or-API-key auth."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 4096):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        self.model = model
        self.max_tokens = max_tokens

        auth_mode, token = resolve_anthropic_credentials()
        self._auth_mode = auth_mode
        if auth_mode == "oauth":
            self._client = _build_oauth_client()
        else:
            import anthropic
            self._client = anthropic.Anthropic(api_key=token)

    @property
    def model_name(self) -> str:
        return self.model

    def _effective_system(self, user_system: str) -> str:
        """Prepend CLAUDE_CODE_IDENTITY under OAuth so RequestTransformer emits
        the required two-block system array (identity then user system)."""
        if self._auth_mode != "oauth":
            return user_system
        if not user_system:
            return CLAUDE_CODE_IDENTITY
        return CLAUDE_CODE_IDENTITY + "\n\n" + user_system

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 1.0) -> ChatResult:
        user_system = ""
        convo: List[Dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                user_system = msg["content"]
            else:
                convo.append(msg)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": convo,
            "max_tokens": self.max_tokens,
        }
        # Newer Claude models (opus-4-7+, sonnet-4-6+) rejected requests that
        # include `temperature` at all with
        #     BadRequestError: `temperature` is deprecated for this model.
        # so pass it only when the caller explicitly overrides the API default,
        # and transparently retry without it if the server still rejects.
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        system_text = self._effective_system(user_system)
        if system_text:
            kwargs["system"] = system_text

        import anthropic
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if "temperature" in kwargs and "temperature" in msg and "deprecated" in msg:
                kwargs.pop("temperature", None)
                response = self._client.messages.create(**kwargs)
            else:
                raise

        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        text = "".join(parts)

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        return ChatResult(text=text, usage=usage, raw=response)


# ---------------------------------------------------------------------------
# OllamaProvider (local models over HTTP)
# ---------------------------------------------------------------------------


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaProvider:
    """Chat with a local Ollama daemon. No thinking blocks, no tools.

    `host` falls through: explicit arg -> OLLAMA_HOST env var -> localhost:11434.
    Model names are passed through verbatim (e.g. "llama3.1:8b", "qwen2.5:3b").
    """

    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        think: bool = False,
        keep_alive: str = "0",
    ):
        if not model:
            raise ValueError("OllamaProvider requires a model name (e.g. 'llama3.1:8b')")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Qwen3 / DeepSeek-R1 default to reasoning mode, which spends every
        # token inside a hidden <think>...</think> block and leaves
        # message.content empty. We disable it by default so small reasoning
        # models actually produce answers the judge can score.
        self.think = think
        # Tell Ollama to unload the model immediately after each call.
        # Multi-student sweeps otherwise leave the previous (possibly heavy)
        # model resident, and the next load fails with HTTP 500 "model failed
        # to load ... resource limitations". "0" = unload now; "5m" keeps it
        # warm for reuse.
        self.keep_alive = keep_alive
        resolved = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        self.host = resolved.rstrip("/")

    @property
    def model_name(self) -> str:
        return self.model

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 1.0) -> ChatResult:
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
            "stream": False,
            "think": self.think,                   # Qwen3 / DeepSeek-R1 flag
            "keep_alive": self.keep_alive,          # "0" = unload after each call
            "options": {
                "num_predict": self.max_tokens,
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama request to {self.host} failed: {e.reason}. "
                f"Is `ollama serve` running?"
            ) from e

        response = json.loads(body)
        message = response.get("message") or {}
        text = message.get("content", "") or ""
        # If the model still ran in thinking mode (e.g. Ollama daemon too old
        # to honor `think: false`), content may be empty and all output sits
        # under `message.thinking`. Surface that so the judge has something.
        if not text:
            text = message.get("thinking", "") or ""
        in_tok = int(response.get("prompt_eval_count", 0) or 0)
        out_tok = int(response.get("eval_count", 0) or 0)
        return ChatResult(
            text=text,
            usage={
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
            raw=response,
        )


# ---------------------------------------------------------------------------
# MockProvider (deterministic, for smoke tests)
# ---------------------------------------------------------------------------


class MockProvider:
    """Returns a canned response keyed on the last user message (for tests)."""

    def __init__(self, model: str = "mock", responses: Optional[Dict[str, str]] = None):
        self.model = model
        self._responses = responses or {}

    @property
    def model_name(self) -> str:
        return self.model

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 1.0) -> ChatResult:
        last_user = ""
        for msg in messages:
            if msg["role"] == "user":
                last_user = msg["content"]
        text = self._responses.get(last_user, "MOCK: " + last_user[:80])
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return ChatResult(text=text, usage=usage, raw=None)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(name: str = "anthropic", model: str = "", **kwargs) -> Any:
    """Create a provider by name. Nice defaults for common cases."""
    if name == "anthropic":
        return AnthropicProvider(model=model or DEFAULT_ANTHROPIC_MODEL)
    if name == "ollama":
        # Strip an optional "ollama:" prefix in case the spec was stored with it.
        resolved = model[len("ollama:"):] if model.startswith("ollama:") else model
        return OllamaProvider(model=resolved, host=kwargs.get("host"))
    if name == "mock":
        return MockProvider(model=model or "mock")
    raise ValueError(f"Unknown provider: {name!r}. Supported: anthropic, ollama, mock.")
