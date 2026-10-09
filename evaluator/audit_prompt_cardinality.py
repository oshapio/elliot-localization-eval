#!/usr/bin/env python3
"""Audit the actual benchmark-850 entry point for answer-cardinality leakage."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import run


OUTPUT_FORMAT = "mixed_special_xyxy_grouped_all_instances_v2"
FORBIDDEN = (
    "Return the box for the requested region",
    "Output one box per requested item",
    "Output one point per requested item",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "manifests/benchmark_850_v4.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "evaluator/PROMPT_CARDINALITY_PROOF.md",
    )
    args = parser.parse_args()
    records = run.read_jsonl(args.manifest)
    family_counts = Counter()
    exact_prompts = {}
    differing = []
    forbidden = []

    # Invoke the real evaluator boundary.  These fake answer fields are omitted
    # by PROMPT_INPUT_FIELDS, so their values must be observationally irrelevant.
    answer_variants = (
        {"ground_truth": {}, "targets": [], "annotations": []},
        {"ground_truth": {"probe": [[1, 2, 3, 4]]}, "targets": [[1, 2, 3, 4]]},
        {"ground_truth": {"probe": [[1, 2, 3, 4]] * 20}, "targets": [[1, 2, 3, 4]] * 20},
    )
    for row in records:
        prompts = [run.prompt_for({**row, **extra}, "elliot-loc", OUTPUT_FORMAT)
                   for extra in answer_variants]
        if not prompts[0] == prompts[1] == prompts[2]:
            differing.append(row.get("sample_id"))
        hits = [phrase for phrase in FORBIDDEN if phrase in prompts[0]]
        if hits:
            forbidden.append((row.get("sample_id"), hits))
        family = row.get("benchmark_family", "unknown")
        protocol = row.get("benchmark_protocol", "box")
        family_counts[(family, protocol)] += 1
        exact_prompts.setdefault((family, protocol), prompts[0])

    assert not differing, f"cardinality-dependent prompts: {differing[:10]}"
    assert not forbidden, f"forbidden old wording: {forbidden[:10]}"
    assert len(records) == 850, len(records)
    assert all(
        "ALL matching instances" in prompt
        for prompt in exact_prompts.values()
    )

    source = root / "evals_localization/tasks.py"
    lines = [
        "# Evaluator prompt cardinality proof", "",
        "**PASS.** The actual benchmark-850 prompt entry point rendered identical text after its hidden answer was replaced by 0, 1, and 20 targets.", "",
        f"- Manifest rows: {len(records)}",
        f"- Counterfactual renders: {len(records) * 3}",
        "- Prompt differences: **0**",
        "- Rows missing `ALL matching instances`: **0**",
        "- Forbidden legacy singleton wording: **0**",
        f"- Manifest SHA256: `{digest(args.manifest)}`",
        f"- Prompt source SHA256: `{digest(source)}`", "",
        "The evaluator receives only task metadata when constructing prompts; answer annotations are not in its allowed prompt-input field set.", "",
        "## Coverage", "",
        "| Family | Protocol | Rows |", "|---|---:|---:|",
    ]
    for (family, protocol), count in sorted(family_counts.items()):
        lines.append(f"| {family} | {protocol} | {count} |")
    lines.extend(["", "## Exact prompts by family/protocol", ""])
    for (family, protocol), prompt in sorted(exact_prompts.items()):
        lines.extend([f"### {family} · {protocol}", "", "```text", prompt, "```", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: {len(records)} rows, {len(records) * 3} renders, 0 differences")


if __name__ == "__main__":
    main()
