"""
pipeline_executor.py

Run a sequence of pipeline stages with:
  - dependency checking (does the upstream stage's output exist?)
  - crash recovery   (skip stages whose declared output already exists)
  - clean / clean-stages modes
  - optional rich UI (falls back to plain prints)

Every stage is a subprocess rooted at the repo root. Commands and args come
from `stage_output_wirer.build_stage_command`; we don't special-case any
stage here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from config.pipeline_profile import PipelineProfile, validate as validate_profile
from config.stage_registry import STAGES, get_stage, parse_stage_range
from tools.output_inspector import check_dependencies_met, inspect_run_dir
from tools.stage_runner import StageResult, run_stage_command
from orchestration.stage_output_wirer import (
    build_stage_command,
    register_stage_outputs,
)

try:
    from cli.rich_ui import (
        print_stage_start,
        print_stage_skip,
        print_stage_complete,
        print_stage_fail,
        print_stage_info,
        print_dependency_error,
    )
    _HAS_UI = True
except ImportError:
    _HAS_UI = False


# ---------------------------------------------------------------------------
# UI wrappers — plain print fallback when rich isn't installed
# ---------------------------------------------------------------------------


def _p(msg):
    print(msg)


def _ui_start(sid, desc):
    if _HAS_UI:
        print_stage_start(sid, desc)
    else:
        _p(f"\n=== Stage {sid}: {desc} ===")


def _ui_skip(sid):
    if _HAS_UI:
        print_stage_skip(sid)
    else:
        _p("  [skip] output already exists")


def _ui_complete(sid, duration):
    if _HAS_UI:
        print_stage_complete(sid, duration)
    else:
        _p(f"  completed in {duration:.1f}s")


def _ui_fail(sid, code, log):
    if _HAS_UI:
        print_stage_fail(sid, code, log)
    else:
        _p(f"  FAILED (exit {code}). See {log}")


def _ui_info(msg):
    if _HAS_UI:
        print_stage_info(msg)
    else:
        _p(f"  {msg}")


def _ui_dep_error(sid, missing):
    if _HAS_UI:
        print_dependency_error(sid, missing)
    else:
        _p(f"  ERROR: missing deps: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Crash-recovery check — does this stage's output already exist?
# ---------------------------------------------------------------------------


def _stage_output_exists(stage, run_dir: Path) -> bool:
    if not stage.output_files:
        return False
    for output_file in stage.output_files:
        path = run_dir / stage.output_dir / output_file if stage.output_dir else run_dir / output_file
        if not path.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def execute_pipeline(
    profile: PipelineProfile,
    stage_range: str,
    repo_root: Path,
    clean: bool = False,
    clean_stages: bool = False,
    verbose: bool = True,
    print_fn=None,
) -> List[StageResult]:
    """Run `stage_range` under `profile`, rooted at `repo_root`.

    Returns a StageResult per stage attempted. Stops at the first non-zero
    exit so callers can see which stage failed.
    """
    if print_fn is None:
        print_fn = print

    errors = validate_profile(profile)
    if errors:
        for e in errors:
            _ui_info(f"profile error: {e}")
        raise ValueError(
            "profile validation failed:\n  - " + "\n  - ".join(errors)
        )

    run_dir = Path(profile.run_dir)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir

    if clean and run_dir.exists():
        _ui_info(f"CLEAN: removing {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    stage_ids = parse_stage_range(stage_range)

    if clean_stages and not clean:
        for sid in stage_ids:
            stage = get_stage(sid)
            stage_dir = run_dir / stage.output_dir if stage.output_dir else run_dir
            for output_file in stage.output_files:
                fpath = stage_dir / output_file
                if fpath.exists():
                    _ui_info(f"CLEAN-STAGE: removing {fpath}")
                    fpath.unlink()

    # Pre-register outputs from prior runs so dependent stages can find them.
    stage_outputs: Dict[str, Dict[str, str]] = {
        stage.stage_id: register_stage_outputs(stage.stage_id, run_dir)
        for stage in STAGES
    }

    results: List[StageResult] = []

    for stage_id in stage_ids:
        stage = get_stage(stage_id)
        _ui_start(stage_id, stage.description)

        missing_deps = check_dependencies_met(stage, run_dir)
        if missing_deps:
            _ui_dep_error(stage_id, missing_deps)
            break

        if _stage_output_exists(stage, run_dir):
            _ui_skip(stage_id)
            results.append(StageResult(
                stage_id=stage_id,
                command="(skipped)",
                exit_code=0,
                duration_seconds=0.0,
                log_path="",
            ))
            continue

        (run_dir / stage.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            entry, args = build_stage_command(
                stage_id, profile, run_dir, repo_root, stage_outputs,
            )
        except ValueError as e:
            _ui_info(f"skipping {stage_id}: {e}")
            results.append(StageResult(
                stage_id=stage_id,
                command="(not configured)",
                exit_code=2,
                duration_seconds=0.0,
                log_path="",
            ))
            break

        log_path = logs_dir / f"{stage_id}.log"
        result = run_stage_command(
            repo_root=repo_root,
            entry=entry,
            args=args,
            log_path=log_path,
            verbose=verbose,
        )
        result.stage_id = stage_id
        results.append(result)

        if result.exit_code != 0:
            _ui_fail(stage_id, result.exit_code, result.log_path)
            break

        _ui_complete(stage_id, result.duration_seconds)
        stage_outputs[stage_id] = register_stage_outputs(stage_id, run_dir)

    return results
