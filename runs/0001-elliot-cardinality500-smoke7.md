# 0001 — Elliot cardinality-repair local-500 smoke test

Status: prepared; not submitted. Authorized by the user for Leonardo debug QoS.
This checks the evaluation path, not benchmark performance.

## Model, code and inputs

- Evaluator and Elliot vLLM plugin: repository commit
  `656a60e` (merged PR #2). The launcher selects this checkout's plugin.
- Remote checkout: `/leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval`.
- Model, including its saved tokenizer/processor and custom HF code:
  `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/cache/eval-hf-exports/58552791`.
  File identities will be captured in the external `asset-identities.json` receipt.
- Config: [`evaluator/configs/current/elliot_vllm.json`](../evaluator/configs/current/elliot_vllm.json).
- Manifest: [`smoke_manifest_7_v1.jsonl`](../manifests/smoke_manifest_7_v1.jsonl).
  SHA-256: `0ecb79959cf9e38e869781bcab9a1e3f2d879e4a35d9ba362b2a9c8acebb09ca`.
  Seven queries across seven dataset/protocol groups and seven localization families.
- Prompt templates: `elliot-loc` in `evals_localization/tasks.py`; mixed-special
  grouped output instructions and decoder selected by the config's `output_format`.
- Coordinates: integer tokens on a 0–1000 scale, decoded into original-image
  pixel boxes or points. GUI scoring uses the predicted target center.
- Generation: greedy, 4096 maximum output tokens, repetition penalty 1.05,
  one query per batch, no token-cap retry. One TP=1 engine, 16 active sequences,
  16,384 context tokens, 65,536–4,194,304 image pixels, GPU memory utilization 0.8,
  CUDA graphs enabled. Special tokens are retained for decoding.
- Runtime Python: `/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python`;
  installed vLLM reports `0.24.0+cu129`. Other versions go in the external receipt.
- Scorer: tracked vendored source plus the existing platform-specific
  `fastevaluate.cpython-311-x86_64-linux-gnu.so`, linked from
  `/leonardo_work/EUHPC_E07_125/nas/elliot-vl-eval-vllm/evals_localization/vendor/fastevaluate/`.
  The binary hash goes in the receipt; it is excluded from Git.

## Attempt 1 — prepared command

Leonardo exposes debug service as QoS `boost_qos_dbg` on partition
`boost_usr_prod`, with a 30-minute QoS limit. Request one GPU for 20 minutes.
There was no existing `elliot-smoke7` job or run directory during preparation.
Job ID: pending.

```bash
cd /leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval
git checkout --detach 656a60e
export EVAL_REPO="$PWD"
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export CONFIG="$EVAL_REPO/evaluator/configs/current/elliot_vllm.json"
export MANIFEST="$EVAL_REPO/manifests/smoke_manifest_7_v1.jsonl"
export OUTPUT_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0001-elliot-cardinality500-smoke7
# Named prompt templates; not a filesystem path.
export PROMPT_FAMILY=elliot-loc
export BATCH_SIZE=1 MAX_NEW_TOKENS=4096
# Reviewed manifest identity and number of distinct dataset/protocol groups.
export EXPECTED_MANIFEST_SHA256=0ecb79959cf9e38e869781bcab9a1e3f2d879e4a35d9ba362b2a9c8acebb09ca
export EXPECTED_GROUP_COUNT=7
unset PYTHONPATH
mkdir -p "$OUTPUT_DIR/logs"
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=boost_qos_dbg --job-name=elliot-smoke7 --nodes=1 \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 \
  --time=00:20:00 --output="$OUTPUT_DIR/logs/%x-%j.out" \
  --error="$OUTPUT_DIR/logs/%x-%j.err" orchestration/vllm_benchmark.sbatch
```

## Artifacts and verification

All artifacts use `OUTPUT_DIR` above:

- Raw answers, token IDs and decoded predictions: `shard_0.jsonl` and `predictions.jsonl`.
- Scores and diagnostics: `scores/scores.json`.
- Batch stdout/stderr: `logs/elliot-smoke7-<job-id>.out` and `.err`.
- Model/config/binary hashes and package versions: `asset-identities.json`.
- Submission command/receipt: `launch.sh` and `submission.json`.

Before submission: all seven image paths and the model config exist; nine
adapter tests and six orchestration tests passed locally. GPU output and scoring
are pending. After completion, inspect unique ID coverage (7/7), error count,
raw/decoded box and point responses, coordinate bounds, token-cap hits and all
seven scored groups. Do not interpret this small subset as a performance comparison.
