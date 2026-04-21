"""
Streaming events emitted by LinearAgent.run().

One event class per semantic boundary the caller might want to intercept.
Thinking blocks are first-class: ThinkingStart/Delta/Stop fire alongside
text and tool_use events, preserving the signature needed to replay the
block on subsequent turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Union


@dataclass
class MessageStart:
    model: str


@dataclass
class MessageStop:
    stop_reason: str
    usage: Dict[str, int]


@dataclass
class TurnStart:
    turn: int


@dataclass
class TurnEnd:
    turn: int


@dataclass
class ThinkingStart:
    index: int


@dataclass
class ThinkingDelta:
    index: int
    text: str


@dataclass
class ThinkingStop:
    index: int
    text: str
    signature: str


@dataclass
class TextStart:
    index: int


@dataclass
class TextDelta:
    index: int
    text: str


@dataclass
class TextStop:
    index: int
    text: str


@dataclass
class ToolUseStart:
    index: int
    id: str
    name: str


@dataclass
class ToolUseInputDelta:
    index: int
    partial_json: str


@dataclass
class ToolUseStop:
    index: int
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    tool_use_id: str
    name: str
    content: str
    is_error: bool = False


Event = Union[
    MessageStart,
    MessageStop,
    TurnStart,
    TurnEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingStop,
    TextStart,
    TextDelta,
    TextStop,
    ToolUseStart,
    ToolUseInputDelta,
    ToolUseStop,
    ToolResult,
]
