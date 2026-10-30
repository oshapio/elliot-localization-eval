# 0006 — language-rl300: canonical compatibility smoke

Status: **failed before generation**. Job **62806537**, exit `1:0`, elapsed **4m 21s**. User authorized additional version smoke tests.
Seven queries, one GPU, debug QoS, 20-minute limit. This is not a historical
score reproduction or a full benchmark evaluation.

## Explicit model and protocol

Lineage: SFT worker-safe local 7145 → language-only GRPO step 300; vision/projector frozen.

The pre-RL worker-safe 7145 anchor is [run 0004](0004-worker-safe-7145-mixed-smoke7.md).
These RL exports historically used the external native-processing plugin. This
smoke explicitly tests the canonical plugin and does not reproduce those scores.

- HF export: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/post_training/localization_grpo_native7145_20261012/runs/native7145-fast-kernels-production-v003/evaluation_automation/step_300/hf_export`. Its own tokenizer, processor and custom HF model code are loaded.
- Config: [`0006-language-rl300.json`](inputs/0006-language-rl300.json); all model settings are explicit there.
- Source: committed and pushed campaign revision `62c579e`, in isolated Git
  worktree `/leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-rl-smokes`. Runtime matches merged PR #2; only records/configs differ.
- Manifest: [`smoke_manifest_7_v1.jsonl`](../manifests/smoke_manifest_7_v1.jsonl),
  seven queries/groups with the SHA-256 pinned in the command below.
- Prompt family: `elliot-loc`, built in `evals_localization/tasks.py` from query
  categories/descriptions, without GT coordinates or target counts. The adapter
  adds Elliot's system/user/assistant non-thinking chat envelope.
- Output format/decoder: **`mixed_special_xyxy_grouped_all_instances_v2`**. REF/BOX parses `<ref>label</ref><box><N>…</box>`;
  mixed output parses label-to-target-list JSON with `<N>` coordinate tokens.
  Mixed output also appends grouped all-instances response instructions to the prompt.
- Coordinates: x then y, normalized 0–1000, converted to original-image pixels.
  Boxes use xyxy and points use xy; GUI uses box centers for scoring.
- Greedy generation, 4096 output-token cap, no retry, repetition penalty 1.05,
  TP=1, batch=1, 16K context, 4MP cap. Platform environment/scorer setup is the
  same as [run 0001](0001-elliot-cardinality500-smoke7.md).

Historical differences: this campaign deliberately uses the canonical current
`elliot-loc` prompt protocol. Older v022 evaluations used different prompts;
recovery-10000 used an external native-processing plugin with extra validation.
A pass here does not establish equivalence to those historical implementations.
Legacy checkpoints may not support all seven requested tasks or point outputs;
empty/invalid predictions must remain visible rather than changing the decoder
until something parses.

## Attempt 1

Job ID: **62806537**. One submission; no retry.

The checkout must be at the committed campaign revision before executing:

```bash
cd /leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-rl-smokes
export EVAL_REPO="$PWD"
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export CONFIG="$EVAL_REPO/runs/inputs/0006-language-rl300.json"
export MANIFEST="$EVAL_REPO/manifests/smoke_manifest_7_v1.jsonl"
export OUTPUT_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0006-language-rl300-smoke7
export PROMPT_FAMILY=elliot-loc BATCH_SIZE=1 MAX_NEW_TOKENS=4096
export EXPECTED_MANIFEST_SHA256=0ecb79959cf9e38e869781bcab9a1e3f2d879e4a35d9ba362b2a9c8acebb09ca
export EXPECTED_GROUP_COUNT=7
unset PYTHONPATH
mkdir -p "$OUTPUT_DIR/logs"
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=boost_qos_dbg --job-name=elliot-smoke0006 --nodes=1 \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 \
  --time=00:20:00 --output="$OUTPUT_DIR/logs/%x-%j.out" \
  --error="$OUTPUT_DIR/logs/%x-%j.err" orchestration/vllm_benchmark.sbatch
```

## Evidence and outcome

Artifacts: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0006-language-rl300-smoke7`.
`asset-identities.json` holds export/config/binary hashes, package versions,
source commit and tokenizer-coordinate checks. `prompts.jsonl` preserves
all seven rendered prompts. `launch.sh` and `submission.json` preserve submission.
`predictions.jsonl` retains raw responses, token IDs and decoded predictions;
`scores/scores.json` holds group diagnostics and scores. Batch logs are under
`logs/`; `verification.json` will hold final coverage and geometry checks.

Asset preflight passed: all model shards and seven images exist; 1001 distinct
coordinate tokens map to IDs 248077–249077. Model/config/scorer SHA-256 values,
package versions and all seven rendered prompts are saved externally. The ignored
compiled scorer binary is linked into this worktree as in run 0001.
No predictions or scores were produced. The exported `processing_elliot.py`
imports `native_processor_call` from `elliot_grpo.native_resolution`. Transformers
rejects processor loading because this dependency is absent. The canonical
plugin loaded correctly; this is an exported-processor dependency failure.
Do not rerun until this dependency is tracked and verified with CPU processor
loading. No checkpoint or exported processor has been modified. Passing execution, parse coverage and model accuracy are
separate findings; failures will be recorded without substituting another model.

## Attempt 2 — restore the historical processor dependency

User authorized a new evaluation of this exact GRPO v003 step-300 checkpoint.
No export, tokenizer, weights, prompt or decoder is modified. The missing runtime
module is supplied from the same GRPO source directory used by its successful
historical full evaluation. This is an external runtime dependency, not a second
evaluator. The canonical repository still owns inference, decoding and scoring.

Prepared command (execute only after CPU processor preflight succeeds):

```bash
cd /leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-grpo300
export EVAL_REPO="$PWD"
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export CONFIG="$PWD/runs/inputs/0006-language-rl300.json"
export MANIFEST="$PWD/manifests/smoke_manifest_7_v1.jsonl"
export OUTPUT_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0006-language-rl300-smoke7/attempt2
export PYTHONPATH=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/post_training/localization_grpo_native7145_20261012/runs/native7145-fast-kernels-production-v003/source/localization-posttraining/code/07_grpo
export PROMPT_FAMILY=elliot-loc BATCH_SIZE=1 MAX_NEW_TOKENS=4096 RETRY_MAX_NEW_TOKENS=0
export EXPECTED_MANIFEST_SHA256=0ecb79959cf9e38e869781bcab9a1e3f2d879e4a35d9ba362b2a9c8acebb09ca EXPECTED_GROUP_COUNT=7
mkdir -p "$OUTPUT_DIR/logs"
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
 --qos=boost_qos_dbg --job-name=elliot-smoke0006b --nodes=1 --ntasks=1 \
 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 --time=00:20:00 \
 --output="$OUTPUT_DIR/logs/%x-%j.out" --error="$OUTPUT_DIR/logs/%x-%j.err" \
 orchestration/vllm_benchmark.sbatch
```

Status: CPU dependency preflight pending; no second job submitted yet. Full-suite
submission remains gated on inspecting the smoke's raw responses, coordinates,
coverage and scoring. Historical full result is a separate artifact; it is not
relabelled as this fresh execution.

Dependency identity for attempt 2: `elliot_grpo/native_resolution.py` SHA-256
`1f15a6f803511fcb0168841ffcbb627246c5ab721c40ec3fb4aa2a685f502065`.
The runtime's ignored compiled FasterEvaluate extension is symlinked from the
existing score-repair checkout; evaluator Python code remains this PR's checkout.
