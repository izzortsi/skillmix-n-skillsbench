"""
stage_runner.py

Execute a single pipeline stage as a subprocess rooted at the repository.
Every stage runs `python -m <entry> <args>` with `cwd` = repo_root, so the
method modules (`s1_extracting_skill_names.m1_...`, etc.) resolve off the
ambient project layout.

Two entry shapes:
  - method_id: "s2.m1" -> invoked as `python -m cli s2.m1 <args>`
  - module path: "__module__:b2_benchmarks.skillsbench.corpus_harness"
    -> invoked as `python -m b2_benchmarks.skillsbench.corpus_harness <args>`
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


MODULE_PREFIX = "__module__:"


@dataclass
class StageResult:
    stage_id: str
    command: str
    exit_code: int
    duration_seconds: float
    log_path: str


def _resolve_cmd(entry: str, args: List[str]) -> List[str]:
    """Turn (entry, args) into the full subprocess argv."""
    if entry.startswith(MODULE_PREFIX):
        module_path = entry[len(MODULE_PREFIX):]
        return [sys.executable, "-u", "-m", module_path] + args
    # default: method_id dispatched through cli.py
    return [sys.executable, "-u", "-m", "cli", entry] + args


def run_stage_command(
    repo_root: Path,
    entry: str,
    args: List[str],
    log_path: Path,
    verbose: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> StageResult:
    """Execute the stage subprocess and capture its log.

    Args:
        repo_root: cwd for the subprocess (the llm-skills repository root)
        entry:     method_id ("s2.m1") or "__module__:pkg.path"
        args:      CLI arg list passed after the entry
        log_path:  where to write combined stdout/stderr
        verbose:   stream output to console in addition to logging it
        extra_env: env vars merged into os.environ for the subprocess
    """
    full_cmd = _resolve_cmd(entry, args)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # PYTHONUNBUFFERED=1 forces line-buffered output so verbose streaming
    # shows progress in real time instead of waiting on pipe buffers.
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            full_cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )
        for line in process.stdout:
            log_file.write(line)
            if verbose:
                sys.stdout.write(line)
                sys.stdout.flush()
        process.wait()

    return StageResult(
        stage_id="",
        command=" ".join(full_cmd),
        exit_code=process.returncode,
        duration_seconds=time.time() - start_time,
        log_path=str(log_path),
    )
