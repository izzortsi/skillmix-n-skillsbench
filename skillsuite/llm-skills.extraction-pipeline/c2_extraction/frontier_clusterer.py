"""
frontier_clusterer.py

PROJECT_SPECS section 1 Method 1 step 2 ("semantic clustering of skill
labels"): feed the flat skill-frequency map produced by frontier_labeler
back to a frontier model and ask for a 2-level hierarchy.

Inputs the `skill-frequency.json` emitted by frontier_labeler (flat
{label: count}). Emits a navigable taxonomy and a flat lookup table.

Output:
    skill-clusters.json        - 2-level hierarchy (cluster -> subcluster -> labels)
    label-to-cluster.json      - flat {label: {cluster, subcluster}} lookup
    clustering-transcript.json - full Transcript from the single API call
    coverage-report.json       - coverage audit: unassigned, duplicated, invented

Usage:
    cd /workspace/llm-skills/skillsuite/llm-skills.extraction-pipeline
    python -m c2_extraction.frontier_clusterer \\
        --frequency data/pipeline-runs/frontier-wikipedia/stage0-labeling/skill-frequency.json \\
        --output-dir data/pipeline-runs/frontier-wikipedia/stage0b-clustering
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness import LinearAgent, Transcript, record  # noqa: E402

from c0_utils.text_utils import strip_markdown_fences  # noqa: E402


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_THINKING_BUDGET = 6144
DEFAULT_MAX_TOKENS = 8000
CLUSTERING_METHOD = "frontier-clusterer-v1"


SYSTEM_PROMPT = (
    "You are a taxonomist. You turn flat lists of procedural skill "
    "labels into navigable 2-level hierarchies. You never invent new "
    "labels, never drop labels, and never duplicate labels across "
    "clusters."
)


CLUSTER_PROMPT = """Cluster the skill labels below into a 2-level hierarchy.

Labels (frequency in parentheses; higher = more central):
{labels_block}

Rules:
  - Produce 3 to 12 top-level clusters. Aim for ~sqrt(N) clusters.
  - Each cluster has a snake_case name describing the shared cognitive
    move, plus a one-sentence description.
  - A cluster with >= 4 labels SHOULD be split into 2-4 subclusters.
    A cluster with < 4 labels has no subclusters.
  - Every input label must appear exactly once, verbatim, in some
    cluster or subcluster.
  - Cluster/subcluster names MUST NOT duplicate any input label.
  - No new labels. No renamed labels. Copy each one character-for-character.

Return ONLY valid JSON:

{{
  "clusters": [
    {{
      "name": "<snake_case cluster name>",
      "description": "<one sentence>",
      "subclusters": [
        {{"name": "<snake_case subcluster name>", "labels": ["<label>", ...]}}
      ],
      "labels": ["<labels not under any subcluster>"]
    }}
  ]
}}
"""


def _build_prompt(frequency: Dict[str, int]) -> str:
    sorted_items = sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"  {label} ({count})" for label, count in sorted_items]
    return CLUSTER_PROMPT.format(labels_block="\n".join(lines))


def _parse_json(text: str):
    raw = strip_markdown_fences(text).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{"); end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None


def _final_text(transcript: Transcript) -> str:
    return "".join(b.text for turn in transcript.turns for b in turn.text)


def _flatten_hierarchy(clusters: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    duplicates: List[str] = []
    for cluster in clusters:
        cname = str(cluster.get("name", "")).strip()
        for label in cluster.get("labels", []) or []:
            key = str(label).strip()
            if not key:
                continue
            if key in lookup:
                duplicates.append(key)
            lookup[key] = {"cluster": cname, "subcluster": ""}
        for sub in cluster.get("subclusters", []) or []:
            sname = str(sub.get("name", "")).strip()
            for label in sub.get("labels", []) or []:
                key = str(label).strip()
                if not key:
                    continue
                if key in lookup:
                    duplicates.append(key)
                lookup[key] = {"cluster": cname, "subcluster": sname}
    if duplicates:
        lookup["__duplicates__"] = {"labels": ",".join(sorted(set(duplicates)))}
    return lookup


def _coverage_audit(
    input_labels: List[str], lookup: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    duplicates_entry = lookup.pop("__duplicates__", None)
    clustered = set(lookup.keys())
    input_set = set(input_labels)

    missing = sorted(input_set - clustered)
    invented = sorted(clustered - input_set)
    duplicates = []
    if duplicates_entry and duplicates_entry.get("labels"):
        duplicates = duplicates_entry["labels"].split(",")

    return {
        "input_label_count": len(input_labels),
        "clustered_label_count": len(clustered),
        "missing_from_output": missing,
        "invented_in_output": invented,
        "duplicated_in_output": duplicates,
        "coverage_ok": not missing and not invented and not duplicates,
    }


def cluster_labels(
    frequency_path: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> Dict[str, Any]:
    frequency = json.loads(frequency_path.read_text(encoding="utf-8"))
    if not isinstance(frequency, dict) or not frequency:
        raise ValueError(f"Empty or malformed frequency map at {frequency_path}")

    input_labels = sorted(frequency.keys())
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Clustering {len(input_labels)} labels with {model}...")

    agent = LinearAgent(
        model=model, system=SYSTEM_PROMPT, tools=[],
        thinking_budget=thinking_budget, max_tokens=max_tokens, max_turns=1,
    )

    transcript = record(agent.run(_build_prompt(frequency)))
    text = _final_text(transcript)
    parsed = _parse_json(text)
    if parsed is None:
        (output_dir / "clustering-transcript.json").write_text(
            json.dumps({"final_text": text, "transcript": transcript.to_dict()}, indent=2),
            encoding="utf-8",
        )
        raise ValueError(f"Failed to parse JSON from model response. See {output_dir / 'clustering-transcript.json'}")

    clusters = parsed.get("clusters", []) or []

    hierarchy_path = output_dir / "skill-clusters.json"
    hierarchy_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    lookup = _flatten_hierarchy(clusters)
    audit = _coverage_audit(input_labels, lookup)
    (output_dir / "label-to-cluster.json").write_text(
        json.dumps(lookup, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output_dir / "coverage-report.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output_dir / "clustering-transcript.json").write_text(
        json.dumps({
            "final_text": text,
            "transcript": transcript.to_dict(),
            "model": model,
            "clustering_method": CLUSTERING_METHOD,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "input_labels": len(input_labels),
        "top_level_clusters": len(clusters),
        "coverage_ok": audit["coverage_ok"],
        "missing": len(audit["missing_from_output"]),
        "invented": len(audit["invented_in_output"]),
        "duplicated": len(audit["duplicated_in_output"]),
        "model": model,
        "clustering_method": CLUSTERING_METHOD,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    if verbose:
        print(f"  clusters: {len(clusters)}")
        print(f"  coverage_ok={audit['coverage_ok']}  "
              f"missing={len(audit['missing_from_output'])}  "
              f"invented={len(audit['invented_in_output'])}  "
              f"duplicated={len(audit['duplicated_in_output'])}")
        for c in clusters:
            n_direct = len(c.get("labels", []) or [])
            n_sub = sum(
                len(s.get("labels", []) or []) for s in (c.get("subclusters", []) or [])
            )
            print(f"    {c.get('name', '?'):40s} direct={n_direct} sub={n_sub}")
        print(f"Wrote {hierarchy_path}")

    return {"hierarchy": parsed, "lookup": lookup, "audit": audit, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description="Hierarchically cluster skill labels via a frontier model (PROJECT_SPECS 1.M1 step 2).")
    ap.add_argument("--frequency", required=True, type=Path, help="skill-frequency.json from frontier_labeler.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cluster_labels(
        frequency_path=args.frequency, output_dir=args.output_dir,
        model=args.model, thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens, verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
