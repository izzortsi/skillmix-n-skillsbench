"""
OllamaAgent: chat-completion adapter for Ollama that mimics LinearAgent's
public surface (`run()` generator yielding harness events, compatible
with `record()`).

No tool-use, no thinking blocks, no streaming — Ollama is typically run
against small open-weight models where those features are absent or
unreliable. The generator yields a single text block per turn to keep
Transcript materialization identical to LinearAgent.

Model naming: pass `ollama:<model_tag>` (e.g. `ollama:llama3.1:8b`) or
just the raw Ollama model tag when constructing. Host defaults to
`http://localhost:11434` and can be overridden with the `OLLAMA_HOST`
environment variable or the `host` constructor argument.

Usage:
    from harness import OllamaAgent, record

    agent = OllamaAgent(model="ollama:llama3.1", system="You are concise.")
    transcript = record(agent.run("Say hi in 3 words."))
    print(transcript.final_text)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from .events import (
    Event,
    MessageStart,
    MessageStop,
    TextDelta,
    TextStart,
    TextStop,
    TurnEnd,
    TurnStart,
)
from .tools import Tool


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MODEL_PREFIX = "ollama:"


def strip_prefix(model: str) -> str:
    return model[len(MODEL_PREFIX):] if model.startswith(MODEL_PREFIX) else model


class OllamaAgent:
    def __init__(
        self,
        model: str,
        system: str = "",
        tools: Optional[List[Tool]] = None,
        thinking_budget: int = 0,
        max_tokens: int = 2000,
        max_turns: int = 1,
        host: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        if tools:
            raise ValueError("OllamaAgent does not support tools")
        self.model = strip_prefix(model)
        self.system = system
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.temperature = temperature
        self.timeout = timeout
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        # Kept for API parity with LinearAgent; unused.
        self.thinking_budget = thinking_budget
        self.tools: List[Tool] = []

    def run(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Event]:
        messages: List[Dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for m in (history or []):
            messages.append(_coerce_message(m))
        messages.append({"role": "user", "content": user_message})

        yield TurnStart(turn=1)
        yield MessageStart(model=self.model)

        response = self._chat(messages)
        text = (response.get("message") or {}).get("content", "") or ""

        yield TextStart(index=0)
        if text:
            yield TextDelta(index=0, text=text)
        yield TextStop(index=0, text=text)

        usage = {
            "input_tokens": int(response.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(response.get("eval_count", 0) or 0),
        }
        yield MessageStop(
            stop_reason=response.get("done_reason") or "end_turn",
            usage=usage,
        )
        yield TurnEnd(turn=1)

    def _chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
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
        return json.loads(body)


def _coerce_message(m: Dict[str, Any]) -> Dict[str, Any]:
    role = m.get("role", "user")
    content = m.get("content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        content = "".join(parts)
    return {"role": role, "content": content}
