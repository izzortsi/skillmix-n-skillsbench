"""
Structured transcript capture for LinearAgent runs.

`record()` consumes the event iterator from LinearAgent.run() and
materializes a Transcript: per-turn thinking blocks (with signatures),
text blocks, tool_use calls, tool_results, stop_reason, and usage.

Usage:
    from harness import LinearAgent, record

    agent = LinearAgent(model="claude-opus-4-7", tools=tools)
    transcript = record(agent.run("hello"))

    for turn in transcript.turns:
        for t in turn.thinking: print(t.text, t.signature[:16])
        for u in turn.tool_uses: print(u.name, u.input)
        for r in turn.tool_results: print(r.name, r.content)
    print(transcript.final_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .events import (
    Event,
    MessageStart,
    MessageStop,
    TextStop,
    ThinkingStop,
    ToolResult,
    ToolUseStop,
    TurnStart,
)


@dataclass
class ThinkingBlock:
    text: str
    signature: str


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    name: str
    content: str
    is_error: bool


@dataclass
class Turn:
    index: int
    model: Optional[str] = None
    thinking: List[ThinkingBlock] = field(default_factory=list)
    text: List[TextBlock] = field(default_factory=list)
    tool_uses: List[ToolUseBlock] = field(default_factory=list)
    tool_results: List[ToolResultBlock] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class Transcript:
    turns: List[Turn] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        if not self.turns:
            return ""
        return "".join(b.text for b in self.turns[-1].text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turns": [
                {
                    "index": t.index,
                    "model": t.model,
                    "thinking": [{"text": b.text, "signature": b.signature} for b in t.thinking],
                    "text": [{"text": b.text} for b in t.text],
                    "tool_uses": [
                        {"id": b.id, "name": b.name, "input": b.input} for b in t.tool_uses
                    ],
                    "tool_results": [
                        {
                            "tool_use_id": b.tool_use_id,
                            "name": b.name,
                            "content": b.content,
                            "is_error": b.is_error,
                        }
                        for b in t.tool_results
                    ],
                    "stop_reason": t.stop_reason,
                    "usage": t.usage,
                }
                for t in self.turns
            ]
        }


def record(events: Iterator[Event]) -> Transcript:
    transcript = Transcript()
    current: Optional[Turn] = None

    for ev in events:
        if isinstance(ev, TurnStart):
            current = Turn(index=ev.turn)
            transcript.turns.append(current)
        elif current is None:
            continue
        elif isinstance(ev, MessageStart):
            current.model = ev.model
        elif isinstance(ev, ThinkingStop):
            current.thinking.append(ThinkingBlock(text=ev.text, signature=ev.signature))
        elif isinstance(ev, TextStop):
            if ev.text:
                current.text.append(TextBlock(text=ev.text))
        elif isinstance(ev, ToolUseStop):
            current.tool_uses.append(
                ToolUseBlock(id=ev.id, name=ev.name, input=ev.input)
            )
        elif isinstance(ev, ToolResult):
            current.tool_results.append(
                ToolResultBlock(
                    tool_use_id=ev.tool_use_id,
                    name=ev.name,
                    content=ev.content,
                    is_error=ev.is_error,
                )
            )
        elif isinstance(ev, MessageStop):
            current.stop_reason = ev.stop_reason
            current.usage = dict(ev.usage)

    return transcript
