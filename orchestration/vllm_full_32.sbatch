#!/bin/bash
# Export one MCore checkpoint and evaluate the full suite with 32 TP=1 ranks.
# Each shard is stable and resumable by sample_id through evaluator/run.py.
#SBATCH --job-name=elliot-vllm-full
#SBATCH --account=EUHPC_E07_125
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=qos_llm_prod
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
set -euo pipefail

repo=${EVAL_REPO:?Set EVAL_REPO to a clean elliot-localization-eval checkout}
run=${RUN_DIR:?Set RUN_DIR to the prepared run folder}
checkpoint=${MCORE_CHECKPOINT:?Set MCORE_CHECKPOINT to model_optim_rng.pt}
python=${EVAL_PYTHON:?Set EVAL_PYTHON to the isolated vLLM Python executable}
eval_data=${EVAL_DATA:?Set EVAL_DATA to the benchmark root}
evaluator="$repo/evaluator"
orchestration="$repo/orchestration"

"$python" "$orchestration/verify_run.py" "$run" "$repo" "${PROMPT_FAMILY:?Set the reviewed prompt family explicitly}"
bash "$orchestration/export_mcore_to_hf.sh" "$run" "$checkpoint"

export PATH="$(dirname "$python"):$PATH"
export PYTHONPATH="$repo/integrations/elliot_vllm${PYTHONPATH:+:$PYTHONPATH}"
export ADAPTER=elliot_vllm GPUS_PER_MODEL=1
export BATCH_SIZE=${BATCH_SIZE:-64} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
export RETRY_MAX_NEW_TOKENS=0
export PROMPT_FAMILY
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS=elliot OMP_NUM_THREADS=8
mkdir -p "$run/predictions" "$run/logs"
printf '%s\n' "$SLURM_JOB_ID" > "$run/job_id.txt"
scontrol show job "$SLURM_JOB_ID" > "$run/slurm_job.txt"

srun --label --gpu-bind=none \
  --output="$run/logs/rank_%t-$SLURM_JOB_ID.out" \
  --error="$run/logs/rank_%t-$SLURM_JOB_ID.err" \
  "$repo/orchestration/run_shard.sh" "$evaluator" "$run/inputs/manifest.jsonl" \
  "$run/inputs/config.json" "$run/predictions" 0

shards=()
for ((rank=0; rank<SLURM_NTASKS; rank++)); do
  shards+=(--shard "$run/predictions/shard_${rank}.jsonl")
done
"$python" "$evaluator/merge.py" --manifest "$run/inputs/manifest.jsonl" \
  "${shards[@]}" --output "$run/predictions/predictions.jsonl"
"$python" "$evaluator/score.py" --predictions "$run/predictions/predictions.jsonl" \
  --eval-data "$eval_data" --output-dir "$run/scores"
date -u +'%FT%TZ' > "$run/completed_at.txt"
