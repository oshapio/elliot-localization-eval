#!/usr/bin/env python3
"""Score merged fixed-suite predictions with the existing Rex-Omni scorer."""

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    """Parse merged predictions and score output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path):
    """Read non-empty JSONL records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    """Write scorer-compatible records."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_scorer():
    """Import the established localization scoring implementation."""
    eval_root = Path(__file__).resolve().parents[2] / "evals_localization"
    import_roots = [
        eval_root / "runtime" / "leonardo",
        eval_root / "vendor" / "fastevaluate",
        eval_root,
        eval_root / "vendor" / "rex_omni",
        eval_root / "vendor" / "rex_omni" / "metrics",
        eval_root / "vendor" / "rex_omni" / "utils",
        eval_root / "vendor" / "embodied",
    ]
    for import_root in reversed(import_roots):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    import eval as scorer
    return scorer


def dataset_diagnostics(rows):
    """Summarize parse coverage and generation failures."""
    return {
        "samples": len(rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "nonempty_predictions": sum(bool(row["predictions"]) for row in rows),
        "parse_rate": sum(bool(row["predictions"]) for row in rows) / len(rows),
        "token_cap_hits": sum(bool(row.get("hit_token_cap")) for row in rows),
        "token_cap_retries": sum(bool(row.get("retried_after_token_cap")) for row in rows),
        "predicted_targets": sum(len(targets) for row in rows for targets in row["predictions"].values()),
        "ground_truth_targets": sum(len(targets) for row in rows for targets in row["gt"].values()),
    }


def main():
    """Score each dataset separately and write macro/weighted summaries."""
    args = parse_args()
    rows = read_jsonl(args.predictions)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset_name"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_scorer()
    results = {}
    for dataset, dataset_rows in sorted(grouped.items()):
        answer_path = args.output_dir / f"{dataset}.answer.jsonl"
        write_jsonl(answer_path, dataset_rows)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metrics = scorer.score(answer_path)
        results[dataset] = {
            "metrics": metrics,
            "diagnostics": dataset_diagnostics(dataset_rows),
        }
    primary = {dataset: result["metrics"]["F1@mean"] for dataset, result in results.items()}
    counts = {dataset: result["diagnostics"]["samples"] for dataset, result in results.items()}
    summary = {
        "samples": len(rows),
        "macro_f1_mean": sum(primary.values()) / len(primary),
        "weighted_f1_mean": sum(primary[name] * counts[name] for name in primary) / sum(counts.values()),
        "datasets": results,
    }
    output = args.output_dir / "scores.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
