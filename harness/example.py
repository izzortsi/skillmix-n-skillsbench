"""
Minimal demo of LinearAgent streaming thinking + text + a tool call.

Run:
    cd /workspace/llm-skills
    python -m harness.example
"""

from __future__ import annotations

import sys

from harness import (
    LinearAgent,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ThinkingStop,
    Tool,
    ToolResult,
    ToolUseStart,
    TurnStart,
)


def _get_time(args):
    return "2026-04-21T12:00:00Z"


def main():
    tools = [
        Tool(
            name="get_time",
            description="Return the current UTC time as an ISO-8601 string.",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=_get_time,
        ),
    ]

    agent = LinearAgent(
        model="claude-opus-4-7",
        system="You are a concise assistant. Think before answering.",
        tools=tools,
        thinking_budget=2048,
        max_tokens=4096,
    )

    user = sys.argv[1] if len(sys.argv) > 1 else "What time is it right now?"
    print(f"\n=== user: {user}\n")

    thinking_shown = False
    for ev in agent.run(user):
        if isinstance(ev, TurnStart):
            print(f"\n--- turn {ev.turn} ---")
        elif isinstance(ev, ThinkingDelta):
            if not thinking_shown:
                sys.stdout.write("[think] ")
                thinking_shown = True
            sys.stdout.write(ev.text)
            sys.stdout.flush()
        elif isinstance(ev, ThinkingStop):
            sys.stdout.write("\n")
            thinking_shown = False
        elif isinstance(ev, TextDelta):
            sys.stdout.write(ev.text)
            sys.stdout.flush()
        elif isinstance(ev, ToolUseStart):
            print(f"\n[tool_use] {ev.name} (id={ev.id})")
        elif isinstance(ev, ToolResult):
            print(f"[tool_result] {ev.name} -> {ev.content!r}")
        elif isinstance(ev, MessageStop):
            print(f"\n[stop={ev.stop_reason}, usage={ev.usage}]")

    print()


if __name__ == "__main__":
    main()
