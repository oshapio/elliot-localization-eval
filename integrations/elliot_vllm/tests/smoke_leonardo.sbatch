#!/bin/bash
#SBATCH --job-name=elliot-vllm-smoke
#SBATCH --account=EUHPC_E07_125
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=qos_llm_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=outputs/vllm-smoke-%j.out
#SBATCH --error=outputs/vllm-smoke-%j.err
set -euo pipefail
repo=${EVAL_REPO:?Set EVAL_REPO to the elliot-localization-eval checkout}
python=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
model=${MODEL_PATH:-/leonardo_work/EUHPC_E07_125/nas/checkpoints/locateanything/step36000_4mp_hf_grpo}
image=${IMAGE_PATH:-/leonardo_scratch/large/userexternal/auselis0/elliot-localization/post_training/002_rex_omni_scale_grpo/manifest_r1/images/07410b34d3265ffc9b30e1e1.png}
output="$repo/outputs/vllm_smoke/$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS=elliot
export TRITON_CACHE_DIR="/leonardo_scratch/large/userexternal/auselis0/.cache/triton/elliot-vllm-$SLURM_JOB_ID"
mkdir -p "$output" "$TRITON_CACHE_DIR"
cd "$repo"
"$python" integrations/elliot_vllm/tests/smoke.py --backend hf --model "$model" --image "$image" --output "$output/hf.json"
"$python" integrations/elliot_vllm/tests/smoke.py --backend vllm --model "$model" --image "$image" --output "$output/vllm.json"
