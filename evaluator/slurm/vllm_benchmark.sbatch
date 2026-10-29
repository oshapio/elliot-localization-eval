#!/bin/bash
# Run the existing evaluator/scorer with the Elliot vLLM adapter.
#SBATCH --job-name=elliot-vllm-eval
#SBATCH --account=EUHPC_E07_125
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=qos_llm_prod
#SBATCH --nodes=1
# The next directive makes Slurm export SLURM_NTASKS=4 to the batch shell.
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
set -euo pipefail
repo=${EVAL_REPO:-$SLURM_SUBMIT_DIR}
evaluator="$repo/evaluator"
manifest=${MANIFEST:-$repo/manifests/benchmark_850_v4.jsonl}
python=${EVAL_PYTHON:?Set EVAL_PYTHON to the isolated vLLM Python executable}
config=${CONFIG:?Set CONFIG to an Elliot vLLM adapter config}
output=${OUTPUT_DIR:?Set OUTPUT_DIR to a new durable result directory}
eval_data=${EVAL_DATA:-/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData}
export PATH="$(dirname "$python"):$PATH"
# The shared environment may have an editable plugin installed from another
# checkout. Resolve Elliot from the evaluator checkout selected for this run.
export PYTHONPATH="$repo/integrations/elliot_vllm${PYTHONPATH:+:$PYTHONPATH}"
export ADAPTER=elliot_vllm
export BATCH_SIZE=${BATCH_SIZE:-64}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS=elliot
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$output"

# Fail before starting four GPU engines if the CPU scoring environment is not
# complete. The vendored FasterEvaluate extension is compiled locally and is
# intentionally not committed as a platform-specific binary.
"$python" - "$evaluator" "$repo/integrations/elliot_vllm/elliot_vllm" <<'PY'
import sys
from pathlib import Path

import elliot_vllm

plugin = Path(elliot_vllm.__file__).resolve()
if plugin.parent != Path(sys.argv[2]).resolve():
    raise RuntimeError(f"Elliot plugin resolved outside the selected checkout: {plugin}")
print(f"Elliot plugin: {plugin}", flush=True)

sys.path.insert(0, sys.argv[1])
from scoring.rex import load_scorer

load_scorer()
PY

# Launch exactly SLURM_NTASKS=4 copies of run_shard.sh. Slurm
# gives them global ranks SLURM_PROCID=0..3; each copy writes shard_<rank>.jsonl.
# srun is blocking, so the batch shell continues only after all four exit.
srun --ntasks="$SLURM_NTASKS" --gpu-bind=closest "$evaluator/slurm/run_shard.sh" \
    "$evaluator" "$manifest" "$config" "$output" 0

# SLURM_NTASKS comes from #SBATCH --ntasks above. Enumerate the exact files the
# four ranks must have produced; merge.py rejects missing, duplicate, or extra
# sample coverage and restores the original manifest order.
shards=()
for ((rank=0; rank<SLURM_NTASKS; rank++)); do
    shards+=(--shard "$output/shard_${rank}.jsonl")
done
"$python" "$evaluator/merge.py" --manifest "$manifest" \
    "${shards[@]}" --output "$output/predictions.jsonl"

# Scoring is intentionally single-process after the deterministic merge.
"$python" "$evaluator/score.py" --predictions "$output/predictions.jsonl" \
    --eval-data "$eval_data" --output-dir "$output/scores"
