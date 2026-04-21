"""
LinearAgent: single-thread, streaming, tool-dispatching chat loop.

Usage:
    from harness import LinearAgent, Tool, ThinkingDelta, TextDelta

    def get_time(args): return "2026-04-21T12:00:00Z"
    tools = [Tool("get_time", "Current UTC time", {"type": "object", "properties": {}}, get_time)]

    agent = LinearAgent(model="claude-opus-4-7", tools=tools, thinking_budget=4096)
    for ev in agent.run("What time is it?"):
        if isinstance(ev, ThinkingDelta):
            print("[think]", ev.text, end="")
        elif isinstance(ev, TextDelta):
            print(ev.text, end="")

The generator yields typed events (see events.py) as the SSE stream
arrives. Thinking blocks are preserved with their signature across turns
so tool-use loops remain valid for extended-thinking models.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterator, List, Optional

from .auth import resolve_credentials
from .events import (
    Event,
    MessageStart,
    MessageStop,
    TextDelta,
    TextStart,
    TextStop,
    ThinkingDelta,
    ThinkingStart,
    ThinkingStop,
    ToolResult,
    ToolUseInputDelta,
    ToolUseStart,
    ToolUseStop,
    TurnEnd,
    TurnStart,
)
from .tools import Tool


DEFAULT_MODEL = "claude-opus-4-7"

# OAuth identity that Claude Code injects as the FIRST system content
# block. The anthropic-oauth package handles this automatically; we
# keep the constant here for backward-compatible _effective_system().
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


# Claude Code stores its tokens at ~/.claude/.credentials.json in a
# different shape than the anthropic-oauth package's default
# ~/.config/anthropic-oauth/tokens.json. Subclass OAuthManager to read
# Claude Code's format so we share the same active session tokens.
def _build_oauth_client(token_ignored: str):
    import inspect
    import os

    import anthropic
    import httpx

    # Prefer the in-tree anthropic-oauth checkout when present (dev
    # container), otherwise fall back to the pip-installed package on
    # the host.
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
            # Write back to Claude Code's file in its own shape so the
            # desktop session picks up the new tokens too.
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
    if "auto_auth" in inspect.signature(OAuthTransport.__init__).parameters:
        transport_kwargs["auto_auth"] = False
    transport = OAuthTransport(manager, **transport_kwargs)
    return anthropic.Anthropic(
        api_key="placeholder",
        http_client=httpx.Client(transport=transport),
    )


class LinearAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system: str = "",
        tools: Optional[List[Tool]] = None,
        thinking_budget: int = 4096,
        max_tokens: int = 16000,
        max_turns: int = 10,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        self.model = model
        self.system = system
        self.tools = tools or []
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.max_turns = max_turns

        auth_mode, token = resolve_credentials()
        self._auth_mode = auth_mode
        if auth_mode == "oauth":
            self._client = _build_oauth_client(token)
        else:
            self._client = anthropic.Anthropic(api_key=token)

        self._tool_map: Dict[str, Tool] = {t.name: t for t in self.tools}

    def _effective_system(self) -> str:
        if self._auth_mode != "oauth":
            return self.system
        if not self.system:
            return CLAUDE_CODE_IDENTITY
        return CLAUDE_CODE_IDENTITY + "\n\n" + self.system

    def run(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Event]:
        messages: List[Dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_message})

        for turn in range(1, self.max_turns + 1):
            yield TurnStart(turn=turn)
            assistant_blocks, stop_reason = yield from self._stream_turn(messages)
            messages.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use":
                yield TurnEnd(turn=turn)
                return

            tool_result_blocks: List[Dict[str, Any]] = []
            for block in assistant_blocks:
                if block.get("type") != "tool_use":
                    continue
                content, is_error = self._dispatch(block["name"], block.get("input", {}))
                yield ToolResult(
                    tool_use_id=block["id"],
                    name=block["name"],
                    content=content,
                    is_error=is_error,
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": content,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_result_blocks})
            yield TurnEnd(turn=turn)

    def _dispatch(self, name: str, args: Dict[str, Any]):
        tool = self._tool_map.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}", True
        try:
            return tool.handler(args), False
        except Exception as e:
            return f"Error: {e}", True

    def _stream_turn(self, messages: List[Dict[str, Any]]):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget},
        }
        system = self._effective_system()
        if system:
            kwargs["system"] = system
        if self.tools:
            kwargs["tools"] = [t.to_api() for t in self.tools]

        blocks: Dict[int, Dict[str, Any]] = {}
        stop_reason = "end_turn"

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                et = event.type

                if et == "message_start":
                    yield MessageStart(model=event.message.model)

                elif et == "content_block_start":
                    idx = event.index
                    cb = event.content_block
                    if cb.type == "thinking":
                        blocks[idx] = {"type": "thinking", "thinking": "", "signature": ""}
                        yield ThinkingStart(index=idx)
                    elif cb.type == "text":
                        blocks[idx] = {"type": "text", "text": ""}
                        yield TextStart(index=idx)
                    elif cb.type == "tool_use":
                        blocks[idx] = {
                            "type": "tool_use",
                            "id": cb.id,
                            "name": cb.name,
                            "input_json": "",
                        }
                        yield ToolUseStart(index=idx, id=cb.id, name=cb.name)

                elif et == "content_block_delta":
                    idx = event.index
                    d = event.delta
                    dt = d.type
                    blk = blocks.get(idx)
                    if blk is None:
                        continue
                    if dt == "thinking_delta":
                        blk["thinking"] += d.thinking
                        yield ThinkingDelta(index=idx, text=d.thinking)
                    elif dt == "signature_delta":
                        blk["signature"] += d.signature
                    elif dt == "text_delta":
                        blk["text"] += d.text
                        yield TextDelta(index=idx, text=d.text)
                    elif dt == "input_json_delta":
                        blk["input_json"] += d.partial_json
                        yield ToolUseInputDelta(index=idx, partial_json=d.partial_json)

                elif et == "content_block_stop":
                    idx = event.index
                    blk = blocks.get(idx)
                    if blk is None:
                        continue
                    if blk["type"] == "thinking":
                        yield ThinkingStop(
                            index=idx,
                            text=blk["thinking"],
                            signature=blk["signature"],
                        )
                    elif blk["type"] == "text":
                        yield TextStop(index=idx, text=blk["text"])
                    elif blk["type"] == "tool_use":
                        raw = blk["input_json"]
                        try:
                            parsed = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        blk["input"] = parsed
                        yield ToolUseStop(
                            index=idx,
                            id=blk["id"],
                            name=blk["name"],
                            input=parsed,
                        )

                elif et == "message_delta":
                    if getattr(event.delta, "stop_reason", None):
                        stop_reason = event.delta.stop_reason

            final = stream.get_final_message()
            stop_reason = final.stop_reason or stop_reason
            yield MessageStop(
                stop_reason=stop_reason,
                usage={
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                },
            )

        assistant_blocks: List[Dict[str, Any]] = []
        for idx in sorted(blocks.keys()):
            b = blocks[idx]
            if b["type"] == "thinking":
                assistant_blocks.append({
                    "type": "thinking",
                    "thinking": b["thinking"],
                    "signature": b["signature"],
                })
            elif b["type"] == "text":
                if b["text"]:
                    assistant_blocks.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": b["id"],
                    "name": b["name"],
                    "input": b.get("input", {}),
                })

        return assistant_blocks, stop_reason
