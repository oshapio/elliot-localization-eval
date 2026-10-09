#!/bin/bash
# Sixteen persistent TP=1 replicas, each processing a shard of all benchmark queries.
#SBATCH --job-name=elliot-full-tp1
#SBATCH --account=EUHPC_E07_125
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=qos_llm_prod
#SBATCH --nodes=4
# The next directive makes Slurm export SLURM_NTASKS=16 to the batch shell.
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
set -euo pipefail
run=${RUN_DIR:?Set RUN_DIR to the prepared reproducible run folder}
repo="$run/source"
evaluator="$repo/evaluator"
python=${EVAL_PYTHON:?Set EVAL_PYTHON to the isolated vLLM Python executable}
eval_data=${EVAL_DATA:?Set EVAL_DATA to the benchmark image/annotation root}
export PATH="$(dirname "$python"):$PATH"
export PYTHONPATH="$repo/integrations/elliot_vllm${PYTHONPATH:+:$PYTHONPATH}"
export ADAPTER=${ADAPTER:-elliot_vllm} GPUS_PER_MODEL=${GPUS_PER_MODEL:-1}
export BATCH_SIZE=${BATCH_SIZE:-64} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS=elliot OMP_NUM_THREADS=8
mkdir -p "$run/predictions" "$run/logs"
printf '%s\n' "$SLURM_JOB_ID" > "$run/job_id.txt"
date -u +'%FT%TZ' > "$run/started_at.txt"
trap 'rc=$?; printf "{\"exit_code\":%d,\"job_id\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" > "$run/exit_status.json"; date -u +"%FT%TZ" > "$run/finished_at.txt"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$run/slurm_job.txt"

# Prove that the CPU scoring environment is complete before starting any GPU
# engine. In particular, FasterEvaluate is a platform-local compiled extension.
"$python" - "$evaluator" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from scoring.rex import load_scorer

load_scorer()
PY

# Launch exactly SLURM_NTASKS=16 copies of run_shard.sh. Each
# process receives a unique global SLURM_PROCID=0..15 and writes its matching
# shard_<rank>.jsonl. This command blocks until every inference rank exits.
srun --ntasks="$SLURM_NTASKS" --label --gpu-bind=none --output="$run/logs/rank_%t-$SLURM_JOB_ID.out" --error="$run/logs/rank_%t-$SLURM_JOB_ID.err" \
    "$evaluator/slurm/run_shard.sh" "$evaluator" "$run/inputs/manifest.jsonl" \
    "$run/inputs/config.json" "$run/predictions" 0
date -u +'%FT%TZ' > "$run/inference_finished_at.txt"

# SLURM_NTASKS is the value from #SBATCH --ntasks above. Construct the complete
# expected shard list; merge.py validates exact manifest coverage and restores
# manifest order rather than concatenating files blindly.
shards=()
for ((rank=0; rank<SLURM_NTASKS; rank++)); do
    shards+=(--shard "$run/predictions/shard_${rank}.jsonl")
done
"$python" "$evaluator/merge.py" --manifest "$run/inputs/manifest.jsonl" \
    "${shards[@]}" --output "$run/predictions/predictions.jsonl"

# Score only the validated, deterministic merged file.
"$python" "$evaluator/score.py" --predictions "$run/predictions/predictions.jsonl" \
    --eval-data "$eval_data" --output-dir "$run/scores"
