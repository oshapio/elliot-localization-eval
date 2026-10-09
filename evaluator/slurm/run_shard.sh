#!/bin/bash
# Run one fixed-suite shard on the GPU assigned by Slurm.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_shard.sh EVALUATOR_DIR MANIFEST.jsonl CONFIG.json OUTPUT_DIR LIMIT

Required environment:
  ADAPTER             adapter module name (for example: elliot_vllm)
  SLURM_PROCID        global shard index, injected by srun
  SLURM_NTASKS        total shard count, injected by srun
  SLURM_LOCALID       node-local task index, injected by srun

Optional environment:
  PROMPT_FAMILY       prompt family (default: elliot-loc)
  MAX_NEW_TOKENS      generation budget (default: 512)
  BATCH_SIZE          per-engine request batch (default: 1)
  GPUS_PER_MODEL      GPUs assigned to one vLLM replica (default: 1)
  EXPECTED_MANIFEST_SHA256  optional exact input identity
  EXPECTED_GROUP_COUNT      optional dataset/protocol group count
EOF
}

die() {
  printf 'run_shard.sh: ERROR: %s\n' "$*" >&2
  exit 64
}

[[ $# -eq 5 ]] || { usage; die "expected 5 positional arguments, received $#"; }

# These variables are injected independently into every process started by
# srun. PROCID is the global shard index; NTASKS is the total shard count;
# LOCALID is the process index on the current node and is used for GPU binding.
[[ -n "${SLURM_PROCID:-}" ]] || die "SLURM_PROCID is missing; launch through srun"
[[ -n "${SLURM_NTASKS:-}" ]] || die "SLURM_NTASKS is missing; launch through srun"
[[ -n "${SLURM_LOCALID:-}" ]] || die "SLURM_LOCALID is missing; launch through srun"

evaluator=$1
manifest=$2
config=$3
output=$4
limit=$5
[[ -n "${ADAPTER:-}" ]] || die "ADAPTER is missing; select the adapter explicitly"
adapter=$ADAPTER
max_new_tokens=${MAX_NEW_TOKENS:-512}
batch_size=${BATCH_SIZE:-1}

[[ -f "$evaluator/run.py" ]] || die "evaluator entry point not found: $evaluator/run.py"
[[ -r "$manifest" ]] || die "manifest is not readable: $manifest"
[[ -r "$config" ]] || die "adapter config is not readable: $config"
[[ "$adapter" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid ADAPTER name: $adapter"
[[ "$SLURM_PROCID" =~ ^[0-9]+$ ]] || die "SLURM_PROCID must be a non-negative integer"
[[ "$SLURM_NTASKS" =~ ^[1-9][0-9]*$ ]] || die "SLURM_NTASKS must be a positive integer"
[[ "$SLURM_LOCALID" =~ ^[0-9]+$ ]] || die "SLURM_LOCALID must be a non-negative integer"
(( SLURM_PROCID < SLURM_NTASKS )) || die "SLURM_PROCID must be smaller than SLURM_NTASKS"
[[ "$limit" =~ ^[0-9]+$ ]] || die "LIMIT must be a non-negative integer"
[[ "$max_new_tokens" =~ ^[1-9][0-9]*$ ]] || die "MAX_NEW_TOKENS must be a positive integer"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer"
mkdir -p "$output" || die "cannot create output directory: $output"
[[ -d "$output" && -w "$output" ]] || die "output directory is not writable: $output"

# Validate syntax before importing an adapter or allocating GPU memory. Field
# requirements remain adapter-owned because baseline adapters use other schemas.
python3 - "$config" <<'PY' || die "adapter config must be a readable JSON object: $config"
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise TypeError("top-level JSON value must be an object")
PY

# Leonardo can expose the entire node's GPUs to every Slurm task. vLLM
# workers use a local device index, so give each shard its own GPU set.
if [[ "$adapter" == *vllm || "$adapter" == locateanything_native ]]; then
  gpus_per_model=${GPUS_PER_MODEL:-1}
  [[ "$gpus_per_model" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_MODEL must be a positive integer"
  gpu_assignment=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}
  [[ -n "$gpu_assignment" ]] || die "no GPU assignment; CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS are empty"
  IFS=, read -r -a visible_gpus <<< "$gpu_assignment"
  if (( ${#visible_gpus[@]} > gpus_per_model )); then
    first=$(( ${SLURM_LOCALID:?Missing local rank} * gpus_per_model ))
    visible_gpus=("${visible_gpus[@]:first:gpus_per_model}")
  fi
  (( ${#visible_gpus[@]} == gpus_per_model )) || { echo "Incorrect GPU assignment" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible_gpus[*]}")
fi

contract_args=()
[[ -n "${EXPECTED_MANIFEST_SHA256:-}" ]] && \
  contract_args+=(--expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256")
[[ -n "${EXPECTED_GROUP_COUNT:-}" ]] && \
  contract_args+=(--expected-group-count "$EXPECTED_GROUP_COUNT")

# run.py selects records by manifest[SLURM_PROCID::SLURM_NTASKS]. Therefore
# ranks never share a sample, and their union is the complete manifest.
python3 "$evaluator/run.py" \
  --manifest "$manifest" \
  --adapter "$adapter" \
  --prompt-family "${PROMPT_FAMILY:-elliot-loc}" \
  --adapter-config "$config" \
  --output "$output/shard_${SLURM_PROCID}.jsonl" \
  --shard-index "$SLURM_PROCID" \
  --num-shards "$SLURM_NTASKS" \
  --limit "$limit" \
  --batch-size "$batch_size" \
  --max-new-tokens "$max_new_tokens" \
  "${contract_args[@]}"
