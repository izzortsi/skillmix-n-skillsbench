"""
Minimal tool registry for LinearAgent.

A Tool bundles the API schema with a Python handler. The agent resolves
tool_use blocks against the registry and dispatches to the handler; the
return value is sent back as a tool_result in the next turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], str]

    def to_api(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
