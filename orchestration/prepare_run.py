#!/usr/bin/env python3
"""Freeze and optionally submit one reproducible Elliot vLLM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


SUITES = {
    "routine850": {
        "manifest": "manifests/benchmark_850_v4.jsonl",
        "samples": 850,
        "groups": None,
        "launcher": "orchestration/vllm_checkpoint.sbatch",
        "tasks": 4,
    },
    "full228133": {
        "manifest": None,
        "samples": 228133,
        "groups": 44,
        "launcher": "orchestration/vllm_full_32.sbatch",
        "tasks": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hf-template", type=Path, required=True)
    parser.add_argument("--minimal-infer-dir", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-repo", type=Path, default=Path(
        "/leonardo_work/EUHPC_E07_125/nas/elliot-vl-train"
    ))
    parser.add_argument("--prompt-family", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_source(repo: Path) -> str:
    commit = run(["git", "rev-parse", "HEAD"], repo)
    dirty = run(["git", "status", "--porcelain"], repo)
    if dirty:
        raise SystemExit(
            f"Evaluator checkout is dirty: {repo}. Use a clean committed checkout; "
            "evaluation source exceptions are not reproducible."
        )
    return commit


def load_config(path: Path, name: str, model_path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = {
        "max_model_len", "max_num_seqs", "tensor_parallel_size",
        "max_pixels", "min_pixels", "coordinate_max", "output_format",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise SystemExit(f"Config is missing required keys: {missing}")
    payload["name"] = name
    payload["model_path"] = str(model_path)
    return payload


def validate_manifest(path: Path, expected_samples: int, expected_groups: int | None) -> tuple[int, int]:
    count = 0
    groups = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Malformed manifest line {line_number}: {error}") from error
            count += 1
            if row.get("benchmark_dataset") is not None and row.get("benchmark_protocol") is not None:
                groups.add((str(row["benchmark_dataset"]), str(row["benchmark_protocol"])))
            else:
                group = row.get("group") or row.get("group_name") or row.get("benchmark_group")
                if group is not None:
                    groups.add(str(group))
    if count != expected_samples:
        raise SystemExit(f"Manifest count {count} != expected {expected_samples}: {path}")
    if expected_groups is not None and len(groups) != expected_groups:
        raise SystemExit(f"Manifest groups {len(groups)} != expected {expected_groups}: {path}")
    return count, len(groups)


def main() -> None:
    args = parse_args()
    suite = SUITES[args.suite]
    repo = args.eval_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint sidecar does not exist: {checkpoint}")
    if not args.hf_template.joinpath("config.json").is_file():
        raise SystemExit(f"HF template has no config.json: {args.hf_template}")
    if not args.eval_python.is_file():
        raise SystemExit(f"Evaluation Python does not exist: {args.eval_python}")
    if not args.minimal_infer_dir.is_dir():
        raise SystemExit(f"Minimal inference directory does not exist: {args.minimal_infer_dir}")

    manifest = args.manifest
    if manifest is None:
        relative = suite["manifest"]
        if relative is None:
            raise SystemExit(f"--manifest is required for {args.suite}")
        manifest = repo / relative
    manifest = manifest.resolve()
    config_source = args.config.resolve()
    count, group_count = validate_manifest(manifest, suite["samples"], suite["groups"])
    manifest_digest = sha256(manifest)
    if args.suite == "full228133":
        identity = json.loads((repo / "manifests/full_228133_v1.json").read_text())
        if manifest_digest != identity["sha256"] or manifest.stat().st_size != identity["uncompressed_bytes"]:
            raise SystemExit(
                "Full manifest does not match full_228133_v1.json identity "
                f"(sha256={manifest_digest}, bytes={manifest.stat().st_size})"
            )
    commit = validate_source(repo)

    run_dir = args.run_dir.resolve()
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frozen_manifest = inputs / "manifest.jsonl"
    frozen_config = inputs / "config.json"
    if frozen_manifest.exists() or frozen_config.exists():
        raise SystemExit(f"Refusing to overwrite prepared inputs in {inputs}")
    frozen_manifest.write_bytes(manifest.read_bytes())
    config = load_config(config_source, args.name, run_dir / "hf_export")
    frozen_config.write_text(json.dumps(config, indent=2) + "\n")

    contract = {
        "version": 1,
        "suite": args.suite,
        "name": args.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "eval_repo": str(repo),
        "eval_commit": commit,
        "eval_source_dirty": False,
        "manifest_source": str(manifest),
        "manifest_sha256": manifest_digest,
        "manifest_samples": count,
        "manifest_groups": group_count,
        "config_source": str(config_source),
        "config_sha256": sha256(frozen_config),
        "prompt_family": args.prompt_family,
        "output_format": config["output_format"],
        "max_new_tokens": 4096,
        "retry_max_new_tokens": 0,
        "shards": suite["tasks"],
        "hf_template": str(args.hf_template.resolve()),
        "hf_template_config_sha256": sha256(args.hf_template / "config.json"),
        "minimal_infer_dir": str(args.minimal_infer_dir.resolve()),
        "train_repo": str(args.train_repo.resolve()),
        "launcher": suite["launcher"],
    }
    atomic_json(run_dir / "contract.json", contract)

    exports = {
        "RUN_DIR": str(run_dir),
        "EVAL_REPO": str(repo),
        "MCORE_CHECKPOINT": str(checkpoint),
        "HF_TEMPLATE_PATH": str(args.hf_template.resolve()),
        "MINIMAL_INFER_DIR": str(args.minimal_infer_dir.resolve()),
        "TRAIN_REPO": str(args.train_repo.resolve()),
        "EVAL_PYTHON": str(args.eval_python.resolve()),
        "EVAL_DATA": str(args.eval_data.resolve()),
        "PROMPT_FAMILY": args.prompt_family,
        "EXPECTED_MANIFEST_SHA256": contract["manifest_sha256"],
    }
    if suite["groups"] is not None:
        exports["EXPECTED_GROUP_COUNT"] = str(suite["groups"])
    launcher = repo / suite["launcher"]
    command = [
        "sbatch", "--parsable", f"--job-name={args.name[:80]}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
        str(launcher),
    ]
    preview = {"command": command, "contract_sha256": sha256(run_dir / "contract.json")}
    atomic_json(run_dir / "submission-preview.json", preview)
    print(json.dumps(preview, indent=2))

    if not args.submit:
        return
    receipt = run_dir / "submission.json"
    if receipt.exists():
        raise SystemExit(f"Refusing duplicate submission; receipt exists: {receipt}")
    job_id = run(command, repo).split(";")[0]
    atomic_json(receipt, {**preview, "job_id": job_id})
    print(f"submitted job={job_id}")


if __name__ == "__main__":
    main()
