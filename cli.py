"""
cli.py

Unified entry: `python -m cli <section>.<method> [args]`.

Examples:
    python -m cli s2.m1 --source data/wikipedia-seed/source.json \\
                        --out data/wikipedia-seed/skills.json
    python -m cli s2.m2 --k 2 --trials 10 \\
                        --student anthropic:claude-sonnet-4-6 \\
                        --judge anthropic:claude-opus-4-7

Methods marked [OOS] are out of scope for this repo (training / mechanistic
interpretability). Everything else is FULL.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Map PROJECT_SPECS method ids -> python modules + status
METHODS: dict[str, dict] = {
    # Section 1 — Extracting Skills (Names)
    "s1.m1": {"module": "s1_extracting_skill_names.m1_task_based_labeling",   "status": "FULL",
              "desc": "Task-based skill labeling + clustering"},
    "s1.m2": {"module": "s1_extracting_skill_names.m2_direct_elicitation",     "status": "FULL",
              "desc": "Direct elicitation of novel skills"},
    "s1.m3": {"module": "s1_extracting_skill_names.m3_catalog_generation",     "status": "FULL",
              "desc": "Comprehensive ~1000-skill catalog generation"},

    # Section 2 — Extracting Skills from Text
    "s2.m1": {"module": "s2_extracting_skills_from_text.m1_wikipedia_seeder",  "status": "FULL",
              "desc": "Wikipedia-style seed from a hand-curated list"},
    "s2.m2": {"module": "s2_extracting_skills_from_text.m2_skill_mix_evaluation", "status": "FULL",
              "desc": "Yu et al. Skill-Mix: k-tuple + topic + per-skill rubric"},
    "s2.m3": {"module": "s2_extracting_skills_from_text.m3_mini_theories",     "status": "OOS",
              "desc": "Mini-theories / mechanistic (out of scope)"},

    # Section 3 — Generating Skill Examples
    "s3.m1": {"module": "s3_generating_skill_examples.m1_random_pair_composition", "status": "FULL",
              "desc": "Random k-tuple composition (k=1..5, Q&A)"},
    "s3.m2": {"module": "s3_generating_skill_examples.m2_domain_specific_combination", "status": "FULL",
              "desc": "Domain-specific skill combinations"},
    "s3.m3": {"module": "s3_generating_skill_examples.m3_synthetic_construction",  "status": "FULL",
              "desc": "Fiction-context synthetic construction"},
    "s3.m4": {"module": "s3_generating_skill_examples.m4_agentic_answer_verification", "status": "FULL",
              "desc": "Agentic verify + refine loop"},
    "s3.m5": {"module": "s3_generating_skill_examples.m5_extracted_task_synthesis", "status": "FULL",
              "desc": "Synthesize ExtractedTasks from procedural Skills (bridges to bench_traced)"},

    # Section 4 — Extracting Skill Usage Instances
    "s4.m1": {"module": "s4_extracting_skill_usage_instances.m1_context_enhanced_learning", "status": "OOS",
              "desc": "Context-enhanced learning (needs training infra)"},
    "s4.m2": {"module": "s4_extracting_skill_usage_instances.m2_curriculum_internalization", "status": "OOS",
              "desc": "Curriculum-based internalization (needs training infra)"},
    "s4.m3": {"module": "s4_extracting_skill_usage_instances.m3_mechanistic_analysis", "status": "OOS",
              "desc": "Mechanistic analysis (needs interp tooling)"},
    "s4.m4": {"module": "s4_extracting_skill_usage_instances.m4_skill_composition_testing", "status": "FULL",
              "desc": "Skill composition testing (eval half only)"},
}


def _print_usage() -> None:
    print("Usage: python -m cli <method> [args]")
    print()
    print(f"  {'method':<8} {'status':<6}  description")
    print(f"  {'------':<8} {'------':<6}  -----------")
    for mid, info in METHODS.items():
        print(f"  {mid:<8} {info['status']:<6}  {info['desc']}")
    print()
    print("Status: FULL = implemented; OOS = out of scope (training / mech interp).")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    method_id = sys.argv[1]
    if method_id not in METHODS:
        print(f"Unknown method: {method_id!r}")
        _print_usage()
        sys.exit(1)

    # Make the repo root importable so "from core.schemas import ..." resolves
    # whether this module is invoked as `python cli.py ...` or `python -m cli ...`.
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import importlib
    mod = importlib.import_module(METHODS[method_id]["module"])

    if not hasattr(mod, "main"):
        status = METHODS[method_id]["status"]
        if status == "OOS":
            print(f"{method_id}: out of scope for this repo. "
                  f"See {METHODS[method_id]['module']} docstring for the reason.")
            sys.exit(2)
        print(f"{method_id}: [{status}] no main() entry point; "
              f"see {METHODS[method_id]['module']} for importable functions.")
        sys.exit(2)

    # Strip method_id from argv so the module's argparse sees only its args
    sys.argv = [method_id] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
