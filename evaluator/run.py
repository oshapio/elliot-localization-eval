#!/usr/bin/env python3
"""Run one adapter over a deterministic shard of the fixed benchmark."""

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path


SCHEMA_VERSION = 1

ROLLOUT_PROVENANCE_FIELDS = (
    "split", "prepared_target", "source_view", "source_row", "target_count",
    "format", "image_reference", "media_root",
)


def _load_task_registry():
    """Load the prompt implementation together with its protocol identity."""
    eval_root = Path(__file__).resolve().parents[1] / "evals_localization"
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    import tasks
    return tasks


TASK_REGISTRY = _load_task_registry()
PROMPT_PROTOCOL = TASK_REGISTRY.LOCALIZATION_PROTOCOL_ID

PROMPT_INPUT_FIELDS = {
    "dataset_name", "task_name", "categories", "benchmark_dataset",
    "benchmark_family", "benchmark_protocol", "evaluation", "keypoint_names",
    "visual_prompt",
}

ADAPTER_MODULES = {
    "baseline_vllm": "adapters.baselines.baseline_vllm",
    "locateanything_native": "adapters.baselines.locateanything_native",
    "empty": "adapters.testing.empty",
    "oracle": "adapters.testing.oracle",
}


def parse_args():
    """Parse the benchmark, adapter, and shard assignment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--prompt-family", default="elliot-loc")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retry-max-new-tokens", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-group-count", type=int)
    return parser.parse_args()


def read_jsonl(path):
    """Read non-empty JSONL records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_config(path):
    """Read an optional JSON adapter configuration."""
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_sha256(value):
    """Hash JSON content independently of whitespace and key order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_contract(args, records):
    """Fail before model loading when a launcher points at the wrong suite."""
    actual_hash = file_sha256(args.manifest)
    if args.expected_manifest_sha256 and actual_hash != args.expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA256 mismatch: expected {args.expected_manifest_sha256}, "
            f"got {actual_hash}"
        )
    groups = {
        (row.get("benchmark_dataset", row.get("dataset_name")),
         row.get("benchmark_protocol"))
        for row in records
    }
    if args.expected_group_count is not None and len(groups) != args.expected_group_count:
        raise ValueError(
            f"benchmark group-count mismatch: expected {args.expected_group_count}, "
            f"got {len(groups)}"
        )


def load_adapter(name, config):
    """Load an explicitly named adapter module."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid adapter name: {name}")
    module = importlib.import_module(ADAPTER_MODULES.get(name, f"adapters.{name}"))
    adapter = module.create_adapter(config)
    required = {"name", "predict_batch"}
    missing = required - set(adapter)
    if missing:
        raise ValueError(f"Adapter {name} is missing keys: {sorted(missing)}")
    adapter["config_sha256"] = json_sha256(config)
    return adapter


def run_metadata(args, adapter):
    """Describe every setting that affects an output shard."""
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "adapter": adapter["name"],
        "adapter_config_sha256": adapter["config_sha256"],
        "prompt_family": args.prompt_family,
        "prompt_protocol": prompt_protocol_for(args.prompt_family),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "limit": args.limit,
    }


def validate_metadata(output, metadata):
    """Refuse to resume an output created with different settings."""
    path = Path(str(output) + ".meta.json")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(f"resume metadata mismatch: {path}")
    else:
        if output.exists() and output.stat().st_size:
            raise ValueError(f"cannot resume output without provenance metadata: {output}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prompt_for(record, family, output_format=None):
    """Build one complete prompt from annotation-free task metadata."""
    if family == "manifest-exact":
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("manifest-exact needs a non-empty prompt field")
        return prompt
    if family == "elliot-v022":
        raise ValueError(
            "elliot-v022 is retired: its prompt text depended on GT cardinality"
        )
    safe_entry = {key: record.get(key) for key in PROMPT_INPUT_FIELDS if key in record}
    return TASK_REGISTRY.build_prompt_for_entry(safe_entry, family, output_format)


def prompt_protocol_for(family):
    if family == "manifest-exact":
        return "manifest-exact-v1"
    if family == "elliot-v022":
        raise ValueError(
            "elliot-v022 is retired: its prompt text depended on GT cardinality"
        )
    return PROMPT_PROTOCOL


def shard_records(records, shard_index, num_shards, limit):
    """Select a stable strided shard from manifest order."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    selected = records[:limit] if limit else records
    return selected[shard_index::num_shards]


def completed_ids(path):
    """Read completed IDs so interrupted runs resume safely."""
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in existing output: {path}")
    return set(ids)


def validate_predictions(predictions):
    """Validate the normalized label-to-spatial-target schema."""
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be an object mapping labels to targets")
    normalized = {}
    for label, targets in predictions.items():
        if not isinstance(targets, list):
            raise ValueError(f"targets for {label!r} must be a list")
        clean_targets = []
        for target in targets:
            if not isinstance(target, (list, tuple)) or len(target) not in (2, 4):
                raise ValueError(f"target for {label!r} must have 2 or 4 coordinates")
            values = [float(value) for value in target]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"target for {label!r} contains a non-finite coordinate")
            clean_targets.append(values)
        normalized[str(label)] = clean_targets
    return normalized


def prediction_record(record, prompt, adapter, result, max_new_tokens,
                      prompt_protocol=PROMPT_PROTOCOL):
    """Create one versioned normalized prediction record."""
    predictions = validate_predictions(result.get("predictions", {}))
    output = {
        "schema_version": SCHEMA_VERSION,
        "prompt_protocol": prompt_protocol,
        "sample_id": record["sample_id"],
        "dataset_name": record["dataset_name"],
        "task_name": record.get("task_name"),
        "benchmark_dataset": record.get("benchmark_dataset"),
        "benchmark_family": record.get("benchmark_family"),
        "benchmark_protocol": record.get("benchmark_protocol"),
        "data_type": record.get("data_type"),
        "data_source": record.get("data_source"),
        "categories": record.get("categories", []),
        "evaluation": record.get("evaluation"),
        "image_path": record["image_path"],
        "prompt": prompt,
        "predictions": predictions,
        "extracted_predictions": predictions,
        "gt": record["gt"],
        "raw_response": result.get("raw_response", ""),
        "generation_events": result.get("generation_events"),
        "generated_tokens": result.get("generated_tokens"),
        "processed_image_size": result.get("processed_image_size"),
        "image_tokens": result.get("image_tokens"),
        "token_ids": result.get("token_ids"),
        "token_logprobs": result.get("token_logprobs"),
        "cumulative_logprob": result.get("cumulative_logprob"),
        "mean_logprob": result.get("mean_logprob"),
        "spatial_targets": result.get("spatial_targets"),
        "hit_token_cap": bool(result.get("hit_token_cap", False)),
        "retried_after_token_cap": False,
        "status": "ok",
        "error": None,
        "adapter": adapter["name"],
        "adapter_config_sha256": adapter["config_sha256"],
        "max_new_tokens": max_new_tokens,
    }
    output.update({key: record.get(key) for key in ROLLOUT_PROVENANCE_FIELDS
                   if key in record})
    return output


def error_record(record, prompt, adapter, error, max_new_tokens,
                 prompt_protocol=PROMPT_PROTOCOL):
    """Preserve failed samples without silently dropping them."""
    output = {
        "schema_version": SCHEMA_VERSION,
        "prompt_protocol": prompt_protocol,
        "sample_id": record["sample_id"],
        "dataset_name": record["dataset_name"],
        "task_name": record.get("task_name"),
        "benchmark_dataset": record.get("benchmark_dataset"),
        "benchmark_family": record.get("benchmark_family"),
        "benchmark_protocol": record.get("benchmark_protocol"),
        "data_type": record.get("data_type"),
        "data_source": record.get("data_source"),
        "categories": record.get("categories", []),
        "evaluation": record.get("evaluation"),
        "image_path": record["image_path"],
        "prompt": prompt,
        "predictions": {},
        "extracted_predictions": {},
        "gt": record["gt"],
        "raw_response": "",
        "generation_events": None,
        "generated_tokens": None,
        "token_logprobs": None,
        "cumulative_logprob": None,
        "mean_logprob": None,
        "spatial_targets": None,
        "hit_token_cap": False,
        "retried_after_token_cap": False,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "adapter": adapter["name"],
        "adapter_config_sha256": adapter["config_sha256"],
        "max_new_tokens": max_new_tokens,
    }
    output.update({key: record.get(key) for key in ROLLOUT_PROVENANCE_FIELDS
                   if key in record})
    return output


def run_batch(adapter, records, prompts, max_new_tokens, fail_fast,
              prompt_protocol=PROMPT_PROTOCOL):
    """Run a batch, falling back to individual samples after batch failure."""
    try:
        results = adapter["predict_batch"](records, prompts, max_new_tokens)
        if len(results) != len(records):
            raise ValueError(f"adapter returned {len(results)} results for {len(records)} records")
        return [prediction_record(record, prompt, adapter, result, max_new_tokens, prompt_protocol)
                for record, prompt, result in zip(records, prompts, results)]
    except Exception as batch_error:
        if fail_fast:
            raise
        outputs = []
        for record, prompt in zip(records, prompts):
            try:
                result = adapter["predict_batch"]([record], [prompt], max_new_tokens)
                if len(result) != 1:
                    raise ValueError(f"adapter returned {len(result)} results for one record")
                outputs.append(prediction_record(
                    record, prompt, adapter, result[0], max_new_tokens, prompt_protocol))
            except Exception as sample_error:
                outputs.append(error_record(
                    record, prompt, adapter, sample_error, max_new_tokens, prompt_protocol))
        print(f"batch failed and was isolated: {type(batch_error).__name__}: {batch_error}", file=sys.stderr)
        return outputs


def retry_capped(adapter, records, prompts, outputs, retry_max_new_tokens, fail_fast,
                 prompt_protocol=PROMPT_PROTOCOL):
    """Retry only generations that exhausted their initial token budget."""
    indices = [
        index for index, output in enumerate(outputs)
        if output["hit_token_cap"] and retry_max_new_tokens > output["max_new_tokens"]
    ]
    if not indices:
        return outputs
    retry_records = [records[index] for index in indices]
    retry_prompts = [prompts[index] for index in indices]
    try:
        results = adapter["predict_batch"](retry_records, retry_prompts, retry_max_new_tokens)
        if len(results) != len(indices):
            raise ValueError(f"adapter returned {len(results)} retries for {len(indices)} samples")
        for index, record, prompt, result in zip(indices, retry_records, retry_prompts, results):
            replacement = prediction_record(
                record, prompt, adapter, result, retry_max_new_tokens, prompt_protocol)
            replacement["retried_after_token_cap"] = True
            outputs[index] = replacement
    except Exception as error:
        if fail_fast:
            raise
        print(f"token-cap retry failed: {type(error).__name__}: {error}", file=sys.stderr)
    return outputs


def run(args):
    """Evaluate the assigned records with resumable JSONL output."""
    records = read_jsonl(args.manifest)
    validate_input_contract(args, records)
    records = shard_records(records, args.shard_index, args.num_shards, args.limit)
    adapter = load_adapter(args.adapter, read_config(args.adapter_config))
    validate_metadata(args.output, run_metadata(args, adapter))
    done = completed_ids(args.output)
    assigned_ids = {record["sample_id"] for record in records}
    unexpected = done - assigned_ids
    if unexpected:
        raise ValueError(f"existing output contains {len(unexpected)} IDs outside this shard")
    pending = [record for record in records if record["sample_id"] not in done]
    prompt_protocol = prompt_protocol_for(args.prompt_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            output_format = adapter.get("output_format")
            prompts = [prompt_for(record, args.prompt_family, output_format)
                       for record in batch]
            outputs = run_batch(
                adapter, batch, prompts, args.max_new_tokens, args.fail_fast,
                prompt_protocol,
            )
            outputs = retry_capped(
                adapter, batch, prompts, outputs, args.retry_max_new_tokens,
                args.fail_fast, prompt_protocol,
            )
            for output in outputs:
                handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                errors += output["status"] == "error"
            handle.flush()
            print(f"{len(done) + written}/{len(records)} complete, {errors} errors", flush=True)
    return {"assigned": len(records), "resumed": len(done), "written": written, "errors": errors}


def main():
    """Run the command-line evaluator."""
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
