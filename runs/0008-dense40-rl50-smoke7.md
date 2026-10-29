# 0008 — dense40-rl50: canonical compatibility smoke

Status: prepared; not submitted. User authorized additional version smoke tests.
Seven queries, one GPU, debug QoS, 20-minute limit. This is not a historical
score reproduction or a full benchmark evaluation.

## Explicit model and protocol

Lineage: Separate dense40 native-linear GRPO branch, step 50. Its parent identity must be read from the export/parent receipts; it is not assumed to descend from worker-safe 7145.

The pre-RL worker-safe 7145 anchor is [run 0004](0004-worker-safe-7145-mixed-smoke7.md).
These RL exports historically used the external native-processing plugin. This
smoke explicitly tests the canonical plugin and does not reproduce those scores.

- HF export: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/post_training/localization_grpo_native7145_20261012/runs/rex-dense40-native-linear-50-v001/evaluation_automation/step_50/hf_export`. Its own tokenizer, processor and custom HF model code are loaded.
- Config: [`0008-dense40-rl50.json`](inputs/0008-dense40-rl50.json); all model settings are explicit there.
- Source: campaign revision committed and pushed before launch; exact revision
  will be recorded in asset-identities.json and below after preparation. Runtime matches merged PR #2; only records/configs differ.
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

Job ID: pending.

The checkout must be at the committed campaign revision before executing:

```bash
cd /leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-rl-smokes
export EVAL_REPO="$PWD"
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export CONFIG="$EVAL_REPO/runs/inputs/0008-dense40-rl50.json"
export MANIFEST="$EVAL_REPO/manifests/smoke_manifest_7_v1.jsonl"
export OUTPUT_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0008-dense40-rl50-smoke7
export PROMPT_FAMILY=elliot-loc BATCH_SIZE=1 MAX_NEW_TOKENS=4096
export EXPECTED_MANIFEST_SHA256=0ecb79959cf9e38e869781bcab9a1e3f2d879e4a35d9ba362b2a9c8acebb09ca
export EXPECTED_GROUP_COUNT=7
unset PYTHONPATH
mkdir -p "$OUTPUT_DIR/logs"
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=boost_qos_dbg --job-name=elliot-smoke0008 --nodes=1 \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 \
  --time=00:20:00 --output="$OUTPUT_DIR/logs/%x-%j.out" \
  --error="$OUTPUT_DIR/logs/%x-%j.err" orchestration/vllm_benchmark.sbatch
```

## Evidence and outcome

Artifacts: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0008-dense40-rl50-smoke7`.
`asset-identities.json` holds export/config/binary hashes, package versions,
source commit and tokenizer-coordinate checks. `prompts.jsonl` preserves
all seven rendered prompts. `launch.sh` and `submission.json` preserve submission.
`predictions.jsonl` retains raw responses, token IDs and decoded predictions;
`scores/scores.json` holds group diagnostics and scores. Batch logs are under
`logs/`; `verification.json` will hold final coverage and geometry checks.

Asset/tokenizer preflight, generation and scoring are pending. Passing execution, parse coverage and model accuracy are
separate findings; failures will be recorded without substituting another model.
