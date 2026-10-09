#!/usr/bin/env python3
"""Fail before model load if a prepared evaluation contract has drifted."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_shape(path: Path) -> tuple[int, int]:
    count = 0
    groups = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if row.get("benchmark_dataset") is not None and row.get("benchmark_protocol") is not None:
                groups.add((str(row["benchmark_dataset"]), str(row["benchmark_protocol"])))
    return count, len(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("eval_repo", type=Path)
    parser.add_argument("prompt_family")
    args = parser.parse_args()

    contract = json.loads((args.run_dir / "contract.json").read_text())
    manifest = args.run_dir / "inputs/manifest.jsonl"
    config_path = args.run_dir / "inputs/config.json"
    checkpoint = Path(contract["checkpoint"])
    checks = {
        "manifest_sha256": sha256(manifest),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint),
    }
    for key, actual in checks.items():
        if actual != contract[key]:
            raise SystemExit(f"{key} mismatch: expected {contract[key]}, got {actual}")
    samples, groups = manifest_shape(manifest)
    if samples != contract["manifest_samples"]:
        raise SystemExit(f"manifest sample mismatch: {samples}")
    if groups != contract["manifest_groups"]:
        raise SystemExit(f"manifest group mismatch: {groups}")
    if args.prompt_family != contract["prompt_family"]:
        raise SystemExit("PROMPT_FAMILY does not match prepared contract")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.eval_repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if commit != contract["eval_commit"]:
        raise SystemExit(f"evaluator commit mismatch: expected {contract['eval_commit']}, got {commit}")
    config = json.loads(config_path.read_text())
    if config["model_path"] != str(args.run_dir.resolve() / "hf_export"):
        raise SystemExit("config model_path does not point at this run's hf_export")
    print(json.dumps({"verified": True, "samples": samples, "groups": groups, "commit": commit}))


if __name__ == "__main__":
    main()
