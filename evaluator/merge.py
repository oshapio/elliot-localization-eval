#!/usr/bin/env python3
"""Merge evaluator shards in canonical manifest order."""

import argparse
import json
from pathlib import Path


def parse_args():
    """Parse the manifest, shards, and merged output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path):
    """Read non-empty JSONL records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    """Validate complete coverage and write records in manifest order."""
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    manifest = manifest[:args.limit] if args.limit else manifest
    expected = [row["sample_id"] for row in manifest]
    by_id = {}
    for shard in args.shard:
        for row in read_jsonl(shard):
            sample_id = row["sample_id"]
            if sample_id in by_id:
                raise ValueError(f"duplicate sample ID across shards: {sample_id}")
            by_id[sample_id] = row
    missing = [sample_id for sample_id in expected if sample_id not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"coverage mismatch: {len(missing)} missing, {len(extra)} extra")
    adapters = {row["adapter"] for row in by_id.values()}
    if len(adapters) != 1:
        raise ValueError(f"shards use different adapters: {sorted(adapters)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample_id in expected:
            handle.write(json.dumps(by_id[sample_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    errors = sum(row["status"] == "error" for row in by_id.values())
    print(json.dumps({"samples": len(expected), "errors": errors, "adapter": next(iter(adapters))}, indent=2))


if __name__ == "__main__":
    main()
