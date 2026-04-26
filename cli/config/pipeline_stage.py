"""
pipeline_stage.py

Immutable definition of a single pipeline stage. Each PipelineStage instance
describes one stage's identity, which pipeline directory contains it, the CLI
command to invoke, expected output files, and dependency relationships.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineStage:
    stage_id: str              # "1a", "1b", "2", "3", "4", "5", "6", "7"
    name: str                  # "extract-passages", "capture-traces", etc.
    description: str           # human-readable one-liner
    pipeline_dir: str          # "text-extraction-pipeline"
    commands: List[str] = field(default_factory=list)  # ["extract-passages"] or ["traceability-report", "export-csv"]
    output_dir: str = ""       # "stage1-task-extraction" (relative to run_dir)
    output_files: List[str] = field(default_factory=list)  # ["passages.json"]
    depends_on: List[str] = field(default_factory=list)    # ["1a"] for stage "1b"
