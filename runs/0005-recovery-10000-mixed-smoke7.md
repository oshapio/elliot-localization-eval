# 0005 — recovery-10000-mixed: canonical compatibility smoke

Status: job **62805250** completed with exit `0:0` in **6m 21s**.
Predictions and scores exist; detailed raw-response/geometry review pending. User authorized additional version smoke tests.
Seven queries, one GPU, debug QoS, 20-minute limit. This is not a historical
score reproduction or a full benchmark evaluation.

## Explicit model and protocol

- HF export: `/leonardo_scratch/large/userexternal/auselis0/elliot-sft-recovery/experiments/original25k-general80-local20-n32-gbs256-lr1e5-v1/evaluation_automation/general/exports/recovery10000`. Its own tokenizer, processor and custom HF model code are loaded.
- Config: [`0005-recovery-10000-mixed.json`](inputs/0005-recovery-10000-mixed.json); all model settings are explicit there.
- Source: committed and pushed campaign revision `3d437eb`. Runtime matches merged PR #2; only records/configs differ.
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

Job ID: **62805250**. One attempt; no retry.

The checkout must be at the committed campaign revision before executing:

```bash
cd /leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval
export EVAL_REPO="$PWD"
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export CONFIG="$EVAL_REPO/runs/inputs/0005-recovery-10000-mixed.json"
export MANIFEST="$EVAL_REPO/manifests/smoke_manifest_7_v1.jsonl"
export OUTPUT_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0005-recovery-10000-mixed-smoke7
export PROMPT_FAMILY=elliot-loc BATCH_SIZE=1 MAX_NEW_TOKENS=4096
export EXPECTED_MANIFEST_SHA256=0ecb79959cf9e38e869781bcab9a1e3f2d879e4a35d9ba362b2a9c8acebb09ca
export EXPECTED_GROUP_COUNT=7
unset PYTHONPATH
mkdir -p "$OUTPUT_DIR/logs"
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=boost_qos_dbg --job-name=elliot-smoke0005 --nodes=1 \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 \
  --time=00:20:00 --output="$OUTPUT_DIR/logs/%x-%j.out" \
  --error="$OUTPUT_DIR/logs/%x-%j.err" orchestration/vllm_benchmark.sbatch
```

## Evidence and outcome

Artifacts: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0005-recovery-10000-mixed-smoke7`.
`asset-identities.json` holds export/config/binary hashes, package versions,
source commit and tokenizer-coordinate checks. `prompts.jsonl` preserves
all seven rendered prompts. `launch.sh` and `submission.json` preserve submission.
`predictions.jsonl` retains raw responses, token IDs and decoded predictions;
`scores/scores.json` holds group diagnostics and scores. Batch logs are under
`logs/`; `verification.json` will hold final coverage and geometry checks.

Preflight passed: all referenced weight shards and images exist; each tokenizer
contains 1001 distinct coordinate tokens, IDs 248077–249077. CPU scorer imports
succeeded. Generation and scoring completed: seven rows and seven scored groups, zero
error rows, seven nonempty predictions and zero token-cap hits. Detailed
raw-response/geometry review and historical parity remain unverified. Passing execution, parse coverage and model accuracy are
separate findings; failures will be recorded without substituting another model.
