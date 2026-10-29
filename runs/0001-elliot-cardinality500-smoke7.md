# 0001 — Elliot cardinality-repair local-500 smoke test

Status: **completed; smoke checks passed**. Slurm job **62797056**, submitted
2026-10-29 19:49:50 UTC, completed with exit `0:0` in **6m 38s**. Authorized by the user for Leonardo debug QoS.
This checks the evaluation path, not benchmark performance.

## Model, code and inputs

- Evaluator and Elliot vLLM plugin: repository commit
  `656a60e` (merged PR #2). The launcher selects this checkout's plugin.
- Remote checkout: `/leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval`.
- Model, including its saved tokenizer/processor and custom HF code:
  `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/cache/eval-hf-exports/58552791`.
  All 18 export files (including weights, tokenizer/processor and custom code)
  have SHA-256 identities in the external `asset-identities.json` receipt.
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
  vLLM `0.24.0+cu129`, torch `2.11.0+cu129`, Transformers `5.12.1`,
  NumPy `2.3.5`, pycocotools `2.0.11`, Shapely `2.1.2`, qwen-vl-utils `0.0.14`,
  Pillow `12.3.0`. Versions are also saved in the external receipt.
- Scorer: tracked vendored source plus the existing platform-specific
  `fastevaluate.cpython-311-x86_64-linux-gnu.so`, linked from
  `/leonardo_work/EUHPC_E07_125/nas/elliot-vl-eval-vllm/evals_localization/vendor/fastevaluate/`.
  The binary hash is saved in the receipt; the binary is excluded from Git.

## Attempt 1 — executed command

Leonardo exposes debug service as QoS `boost_qos_dbg` on partition
`boost_usr_prod`, with a 30-minute QoS limit. Request one GPU for 20 minutes.
There was no existing `elliot-smoke7` job or run directory during preparation.
Job ID: **62797056**, node `lrdn2647`. One attempt; no retry.

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

Before submission, all seven image paths and the model config existed; nine
adapter tests and six orchestration tests passed locally. The CPU scorer preflight
passed. The ignored compiled extension was made available in the clean checkout:

```bash
ln -s /leonardo_work/EUHPC_E07_125/nas/elliot-vl-eval-vllm/evals_localization/vendor/fastevaluate/fastevaluate.cpython-311-x86_64-linux-gnu.so \
  /leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval/evals_localization/vendor/fastevaluate/fastevaluate.cpython-311-x86_64-linux-gnu.so
```

Final evidence is saved in `verification.json` alongside the other artifacts:

- Seven unique predictions in exact manifest order; seven nonempty responses.
- Zero error rows and zero token-cap hits; all seven groups scored.
- All decoded targets have the expected dimension (boxes or points), ordered box
  corners and coordinates within their original image bounds.
- Raw integer tokens survive generation and decode to original-image pixels.
  For example, banana `[<674>,<429>,<994>,<989>]` on a 500×364 image becomes
  `[337.0,156.156,497.0,359.996]`. Point responses remain two-dimensional.
- Slurm reports the job and inference step completed with exit `0:0`.

| Group | Primary score (0–1) |
| --- | ---: |
| COCO / box | 0.000000 |
| VisDrone / box | 0.650000 |
| DocLayNet / box | 0.633333 |
| screenspot_v2 / GUI | 1.000000 |
| IC15 / box | 0.000000 |
| COCO / point | 0.285714 |
| HumanRef / box | 1.000000 |

These are seven individual-query results, not benchmark estimates. Output is
structurally valid but not always correct: the banana box is much smaller than
its ground truth, the OCR box is at the wrong location, and pointing returns six
donuts against one annotated target. Passing this smoke confirms the execution
and decoding/scoring path; it does not establish model accuracy or parity with
historical evaluations.
