"""
Live end-to-end test: real Anthropic stream into structured transcript.

Forces a tool call so the transcript exercises thinking + tool_use +
tool_result + final text across multiple turns.

Run:
    cd /workspace/llm-skills
    python3 -m harness.test_live
"""

from __future__ import annotations

import json
import sys

from harness import LinearAgent, Tool, record


CALLS: list[dict] = []


def _get_time(args):
    CALLS.append({"tool": "get_time", "args": args})
    return "2026-04-21T12:34:56Z"


def main() -> int:
    tools = [
        Tool(
            name="get_time",
            description="Return the current UTC time as an ISO-8601 string.",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=_get_time,
        ),
    ]

    agent = LinearAgent(
        model="claude-haiku-4-5-20251001",
        system="Think briefly, then call get_time and report the result.",
        tools=tools,
        thinking_budget=1024,
        max_tokens=1500,
        max_turns=4,
    )

    transcript = record(agent.run("What time is it? Use the tool."))

    print("=" * 60)
    print(json.dumps(transcript.to_dict(), indent=2))
    print("=" * 60)
    print(f"turns:        {len(transcript.turns)}")
    print(f"tool calls:   {CALLS}")
    print(f"final_text:   {transcript.final_text!r}")

    assert len(transcript.turns) >= 2, "expected at least one tool-use turn + one final turn"

    tool_turn = next(
        (t for t in transcript.turns if t.tool_uses),
        None,
    )
    assert tool_turn is not None, "expected a turn containing a tool_use block"
    assert tool_turn.stop_reason == "tool_use"
    assert any(u.name == "get_time" for u in tool_turn.tool_uses), "expected get_time tool_use"
    assert any(r.name == "get_time" for r in tool_turn.tool_results), "expected get_time tool_result"

    thinking_turn = next(
        (t for t in transcript.turns if t.thinking),
        None,
    )
    assert thinking_turn is not None, "expected at least one thinking block"
    first_thinking = thinking_turn.thinking[0]
    assert first_thinking.signature, "thinking block must carry a signature"

    final_turn = transcript.turns[-1]
    assert final_turn.stop_reason == "end_turn"
    assert transcript.final_text, "expected non-empty final text"

    assert CALLS and CALLS[0]["tool"] == "get_time", "tool handler must have been invoked"

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
