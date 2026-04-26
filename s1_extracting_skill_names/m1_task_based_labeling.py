"""
s1.m1 — Task-Based Skill Labeling (PROJECT_SPECS §1 Method 1).

Two stages, one file:

  Stage A — labeling:  task -> LLM -> list of 2-5 snake_case skill labels
  Stage B — clustering: flat label list -> LLM -> 2-level hierarchy
                         (cluster -> subcluster -> labels) with coverage audit

The spec's constrained label format is "strings of four words connected with
underscore" (snake_case of 2-4 words). §1.m1 is the only scaffold method that
uses snake_case skill names; other methods use kebab-case. Skill records
emitted here keep snake_case verbatim in `name` so the spec mapping is exact.

Input: any JSON file `core.schemas.load_extracted_tasks` can read (pipeline
output from s3.m1/m3/m4, frontier_extractor, etc.).

Output (written to --out-dir):
    labeled-tasks.jsonl       per-task labels + rationales + usage
    skill-frequency.json      {label: count} across all tasks
    skill-clusters.json       raw {"clusters": [...]} from the clusterer
    coverage-report.json      {missing, invented, duplicated, coverage_ok}
    skills.json               Skill records (cluster/subcluster/label hierarchy)

Usage:
    python -m cli s1.m1 --tasks path/to/tasks.json --out-dir out/s1m1 \\
                        --provider anthropic:claude-opus-4-7
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schemas import (
    Skill,
    ExtractedTask,
    load_extracted_tasks,
    save_json,
    stable_uid,
)
from core.providers import create_provider


SOURCE_ID = "s1.m1.task-labeling"

MIN_LABELS = 2
MAX_LABELS = 5


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


LABEL_SYSTEM_PROMPT = (
    "You label evaluation tasks with the specific procedural skills a solver "
    "must exercise to reach the correct answer. Labels are snake_case of 2-4 "
    "words. You prefer specific procedural moves over vague topic words."
)


LABEL_USER_PROMPT = """Label this task with the {min_labels}-{max_labels} discrete skills a solver must exercise.

Task title:  {title}
Difficulty:  {difficulty}
Question:    {question}
Context/input:
{input}

Expected answer:
{output}

Rules for labels:
  - Format: snake_case, 2-4 words (e.g. self_serving_bias_identification)
  - Prefer specific procedural moves: "causal_attribution_analysis",
    "loaded_question_detection", "loop_invariant_derivation".
  - Reject vague labels: "reasoning", "thinking", "math", "writing_well".
  - A skill names a procedure or cognitive move, not a topic.
  - Distinct labels; no synonyms of each other.

Return ONLY valid JSON:

{{
  "skills": [
    {{"label": "<snake_case_label>", "rationale": "<one sentence on why the task requires this skill>"}}
  ]
}}
"""


CLUSTER_SYSTEM_PROMPT = (
    "You are a taxonomist. You turn flat lists of procedural skill labels "
    "into navigable 2-level hierarchies. You never invent new labels, never "
    "drop labels, and never duplicate labels across clusters."
)


CLUSTER_USER_PROMPT = """Cluster the skill labels below into a 2-level hierarchy.

Labels (frequency in parentheses; higher = more central):
{labels_block}

Rules:
  - Produce 3 to 12 top-level clusters. Aim for ~sqrt(N) clusters.
  - Each cluster has a snake_case name describing the shared cognitive move,
    plus a one-sentence description.
  - A cluster with >= 4 labels SHOULD be split into 2-4 subclusters. A cluster
    with < 4 labels has no subclusters.
  - Every input label must appear exactly once, verbatim, in some cluster or
    subcluster.
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return text.strip()


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_markdown_fences(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None


def _sanitize_snake_case(raw_label: str) -> str:
    """Normalize model-emitted labels to snake_case (alnum + underscore only)."""
    label = raw_label.strip().lower()
    cleaned = []
    for ch in label:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        elif ch in (" ", "-", "/"):
            cleaned.append("_")
    collapsed = "".join(cleaned).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed


# ---------------------------------------------------------------------------
# Stage A — labeling
# ---------------------------------------------------------------------------


def label_one_task(task: ExtractedTask, provider) -> Dict[str, Any]:
    """Elicit labels for a single task. Returns {task_uid, labels, error, usage}."""
    prompt = LABEL_USER_PROMPT.format(
        min_labels=MIN_LABELS,
        max_labels=MAX_LABELS,
        title=task.title,
        difficulty=task.difficulty or "unspecified",
        question=task.question,
        input=task.input,
        output=task.output or "(not given)",
    )
    result = provider.chat(
        [
            {"role": "system", "content": LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = _parse_json_object(result.text)

    skill_labels: List[Dict[str, str]] = []
    parse_error: Optional[str] = None

    if parsed is None:
        parse_error = "json_parse_failed"
    else:
        for item in parsed.get("skills", []) or []:
            if not isinstance(item, dict):
                continue
            label = _sanitize_snake_case(str(item.get("label", "")))
            if not label:
                continue
            skill_labels.append({
                "label": label,
                "rationale": str(item.get("rationale", "")).strip(),
            })
        if not skill_labels:
            parse_error = "no_valid_skill_labels"

    return {
        "task_uid": task.task_uid,
        "title": task.title,
        "difficulty": task.difficulty,
        "domain": task.domain,
        "skill_labels": skill_labels,
        "final_text": result.text,
        "parse_error": parse_error,
        "usage": dict(result.usage),
        "labeling_method": SOURCE_ID,
    }


def label_tasks(tasks: List[ExtractedTask], provider, verbose: bool = False) -> List[Dict[str, Any]]:
    """Label every task. Returns per-task records (including failures)."""
    records: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"[label {i}/{len(tasks)}] {task.title!r}")
        try:
            rec = label_one_task(task, provider)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            rec = {
                "task_uid": task.task_uid,
                "title": task.title,
                "skill_labels": [],
                "parse_error": f"api_error:{e}",
                "usage": {},
                "labeling_method": SOURCE_ID,
            }
        records.append(rec)
        if verbose and not rec.get("parse_error"):
            print(f"  labels={[s['label'] for s in rec['skill_labels']]}")
    return records


def build_frequency(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate {label: count} across label records, skipping failures."""
    counter: Counter = Counter()
    for rec in records:
        if rec.get("parse_error"):
            continue
        for s in rec.get("skill_labels", []):
            counter[s["label"]] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Stage B — clustering
# ---------------------------------------------------------------------------


def _flatten_hierarchy(clusters: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Build {label: {cluster, subcluster}}. Captures duplicates as a side entry."""
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


def _coverage_audit(input_labels: List[str], lookup: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Did the model cover every input label exactly once, with no inventions?"""
    dup_entry = lookup.pop("__duplicates__", None)
    clustered = set(lookup.keys())
    input_set = set(input_labels)
    missing = sorted(input_set - clustered)
    invented = sorted(clustered - input_set)
    duplicated: List[str] = []
    if dup_entry and dup_entry.get("labels"):
        duplicated = dup_entry["labels"].split(",")
    return {
        "input_label_count": len(input_labels),
        "clustered_label_count": len(clustered),
        "missing_from_output": missing,
        "invented_in_output": invented,
        "duplicated_in_output": duplicated,
        "coverage_ok": not missing and not invented and not duplicated,
    }


def cluster_labels(frequency: Dict[str, int], provider, verbose: bool = False) -> Dict[str, Any]:
    """Ask the model to organize frequency map into a 2-level hierarchy.

    Returns {"hierarchy", "lookup", "audit", "summary"}. Raises ValueError if
    the response is not parseable JSON.
    """
    if not frequency:
        raise ValueError("cluster_labels: empty frequency map")

    input_labels = sorted(frequency.keys())
    sorted_items = sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))
    labels_block = "\n".join(f"  {label} ({count})" for label, count in sorted_items)
    prompt = CLUSTER_USER_PROMPT.format(labels_block=labels_block)

    if verbose:
        print(f"Clustering {len(input_labels)} labels...")

    result = provider.chat(
        [
            {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = _parse_json_object(result.text)
    if parsed is None:
        raise ValueError(
            "cluster_labels: failed to parse JSON from model response.\n"
            f"Final text (first 500 chars):\n{result.text[:500]}"
        )

    clusters = parsed.get("clusters", []) or []
    lookup = _flatten_hierarchy(clusters)
    audit = _coverage_audit(input_labels, lookup)

    summary = {
        "input_labels": len(input_labels),
        "top_level_clusters": len(clusters),
        "coverage_ok": audit["coverage_ok"],
        "missing": len(audit["missing_from_output"]),
        "invented": len(audit["invented_in_output"]),
        "duplicated": len(audit["duplicated_in_output"]),
        "clustering_method": SOURCE_ID,
        "usage": dict(result.usage),
    }

    if verbose:
        print(
            f"  clusters={len(clusters)} coverage_ok={audit['coverage_ok']} "
            f"missing={len(audit['missing_from_output'])} "
            f"invented={len(audit['invented_in_output'])} "
            f"duplicated={len(audit['duplicated_in_output'])}"
        )

    return {"hierarchy": parsed, "lookup": lookup, "audit": audit, "summary": summary}


# ---------------------------------------------------------------------------
# Stage C — assemble Skill records from labels + clusters
# ---------------------------------------------------------------------------


def skills_from_labels_and_clusters(
    records: List[Dict[str, Any]],
    cluster_result: Dict[str, Any],
) -> List[Skill]:
    """Build a Skill list (clusters as roots, subclusters as children, labels as leaves).

    Skill.name is the snake_case label verbatim per spec §1.m1 (other methods
    use kebab-case). Skill.description on leaves is the first rationale we saw
    for that label across the labeled tasks.
    """
    clusters = cluster_result.get("hierarchy", {}).get("clusters", [])

    # Map label -> first rationale encountered.
    first_rationale: Dict[str, str] = {}
    for rec in records:
        if rec.get("parse_error"):
            continue
        for s in rec.get("skill_labels", []):
            label = s["label"]
            if label not in first_rationale and s.get("rationale"):
                first_rationale[label] = s["rationale"]

    out: List[Skill] = []
    for cluster in clusters:
        cname = str(cluster.get("name", "")).strip()
        if not cname:
            continue
        cluster_uid = stable_uid(f"{SOURCE_ID}|cluster|{cname}")
        out.append(Skill(
            skill_uid=cluster_uid,
            name=cname,
            description=str(cluster.get("description", "")).strip(),
            category="cluster",
            example="",
            procedure=[],
            when_to_use="",
            constraints=[],
            source=SOURCE_ID,
            parent_skill_uid="",
        ))

        # labels directly under the cluster (no subcluster)
        for label in cluster.get("labels", []) or []:
            label_name = str(label).strip()
            if not label_name:
                continue
            out.append(Skill(
                skill_uid=stable_uid(f"{SOURCE_ID}|label|{label_name}"),
                name=label_name,
                description=first_rationale.get(label_name, ""),
                category="label",
                example="",
                procedure=[],
                when_to_use="",
                constraints=[],
                source=SOURCE_ID,
                parent_skill_uid=cluster_uid,
            ))

        # subclusters + labels under each subcluster
        for sub in cluster.get("subclusters", []) or []:
            sname = str(sub.get("name", "")).strip()
            if not sname:
                continue
            sub_uid = stable_uid(f"{SOURCE_ID}|subcluster|{cname}|{sname}")
            out.append(Skill(
                skill_uid=sub_uid,
                name=sname,
                description="",
                category="subcluster",
                example="",
                procedure=[],
                when_to_use="",
                constraints=[],
                source=SOURCE_ID,
                parent_skill_uid=cluster_uid,
            ))
            for label in sub.get("labels", []) or []:
                label_name = str(label).strip()
                if not label_name:
                    continue
                out.append(Skill(
                    skill_uid=stable_uid(f"{SOURCE_ID}|label|{label_name}"),
                    name=label_name,
                    description=first_rationale.get(label_name, ""),
                    category="label",
                    example="",
                    procedure=[],
                    when_to_use="",
                    constraints=[],
                    source=SOURCE_ID,
                    parent_skill_uid=sub_uid,
                ))

    return out


# ---------------------------------------------------------------------------
# CLI (end-to-end: label -> cluster -> skills)
# ---------------------------------------------------------------------------


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        name, model = spec.split(":", 1)
        return name.strip(), model.strip()
    return spec.strip(), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--tasks", type=Path, required=True,
                        help="JSON file of ExtractedTask records")
    parser.add_argument("--out-dir", type=Path, default=Path("data/s1-m1-out"),
                        help="Directory for labeled-tasks.jsonl, skill-frequency.json, "
                             "skill-clusters.json, coverage-report.json, skills.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only label the first N tasks (0 = all)")
    parser.add_argument("--provider", default="anthropic:claude-opus-4-7",
                        help="Provider spec for both labeling and clustering")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tasks = load_extracted_tasks(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    provider_name, provider_model = _parse_provider_spec(args.provider)
    provider = create_provider(provider_name, provider_model)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: labeling
    print("\n=== Stage A: labeling ===")
    records = label_tasks(tasks, provider, verbose=args.verbose)
    labeled_path = args.out_dir / "labeled-tasks.jsonl"
    with labeled_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    parse_errors = sum(1 for r in records if r.get("parse_error"))
    print(f"  labeled {len(records) - parse_errors}/{len(records)} -> {labeled_path}")

    frequency = build_frequency(records)
    freq_path = args.out_dir / "skill-frequency.json"
    freq_path.write_text(json.dumps(frequency, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  unique labels: {len(frequency)} -> {freq_path}")

    if not frequency:
        print("No labels extracted; aborting before clustering.")
        return

    # Stage B: clustering
    print("\n=== Stage B: clustering ===")
    cluster_result = cluster_labels(frequency, provider, verbose=args.verbose)
    (args.out_dir / "skill-clusters.json").write_text(
        json.dumps(cluster_result["hierarchy"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "coverage-report.json").write_text(
        json.dumps(cluster_result["audit"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  coverage_ok={cluster_result['audit']['coverage_ok']}  "
          f"top-level clusters={len(cluster_result['hierarchy'].get('clusters', []))}")

    # Stage C: Skills
    print("\n=== Stage C: Skills ===")
    skills = skills_from_labels_and_clusters(records, cluster_result)
    skills_path = args.out_dir / "skills.json"
    save_json(skills, skills_path)
    n_clusters = sum(1 for s in skills if s.category == "cluster")
    n_subs = sum(1 for s in skills if s.category == "subcluster")
    n_labels = sum(1 for s in skills if s.category == "label")
    print(f"  wrote {len(skills)} Skills -> {skills_path}  "
          f"({n_clusters} clusters, {n_subs} subclusters, {n_labels} labels)")


if __name__ == "__main__":
    main()
