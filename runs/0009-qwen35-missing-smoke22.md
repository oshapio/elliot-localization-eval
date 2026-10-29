# 0009 — qwen35-missing-smoke22

Status: completed; 22-query missing-group execution/geometry/scoring checks passed.
Job **62810773**. Twenty-two scored groups, no error rows; all outputs nonempty. User authorized full baseline coverage work;
this small smoke checks the previously missing task groups before the backfill.

- Config: [`0009-qwen35-missing-smoke22.json`](inputs/0009-qwen35-missing-smoke22.json); model path and runtime settings explicit.
- Runtime: committed canonical Git worktree at `f4cb946`, linked at `RUN_DIR/source`; no copied evaluator.
- Manifest: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/canonical-full-campaign-preflight-humanref-fixed-20261029/missing_baseline_smoke22.jsonl`, one query per missing group.
  SHA-256: `25d56861abcdf7c42f8bfe6cf4d7499109be8bb0204ce796edca7ea5047030fb`.
  Derived by selecting full-suite IDs absent from the historical 179,959-query
  manifest, then the first row per dataset/protocol. Full source SHA-256 is
  `08a7f24ddb75d87285813366e54deb6872f84b4250cce61ce588f73b72d2f960`.
- Prompt family: `qwen` in `evals_localization/tasks.py`. Model-specific
  chat template and output parsing use `adapters/baselines/baseline_vllm.py`.
- Rex uses its vendor system prompt, patched tokenizer and 0–999 special
  coordinates. Qwen uses its own non-thinking template, native box/point JSON,
  normalized 0–1000 coordinates. Both decode to original-image pixels.
- Greedy, 4096 output tokens, no cap retry, TP=1, batch=1. Processor image/context
  limits and repetition penalty come from the config, preserving baseline settings.
- Environment and compiled scorer: as run 0001; package/model/config identities
  are captured in `asset-identities.json`. Rex restore verification is at
  `/leonardo_work/EUHPC_E07_125/nas/checkpoints/Rex-Omni-0e5693d/RESTORE_VERIFIED.json`;
  all 11 historical asset hashes match, including both weight shards.

## Attempt 1

Job ID: **62810773**, first attempt. CPU processor, assets and image checks passed. Prepare `inputs/manifest.jsonl`, `inputs/config.json`, logs and
source symlink before executing. The external receipt and launch.sh pin the exact manifest SHA-256 shown above.

```bash
export EXPECTED_MANIFEST_SHA256=25d56861abcdf7c42f8bfe6cf4d7499109be8bb0204ce796edca7ea5047030fb
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export RUN_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0009-qwen35-missing-smoke22
export ADAPTER=baseline_vllm PROMPT_FAMILY=qwen
export BATCH_SIZE=1 MAX_NEW_TOKENS=4096 GPUS_PER_MODEL=1
export EXPECTED_GROUP_COUNT=22
unset PYTHONPATH
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=boost_qos_dbg --job-name=baseline-smoke0009 --nodes=1 \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 \
  --time=00:20:00 --output="$RUN_DIR/logs/batch-%j.out" \
  --error="$RUN_DIR/logs/batch-%j.err" "$RUN_DIR/source/orchestration/vllm_full.sbatch"
```

## Evidence

Root: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0009-qwen35-missing-smoke22`. Predictions: `predictions/predictions.jsonl`; scores:
`scores/scores.json`; raw/decoded responses and tokens remain in prediction rows.
Receipts: `asset-identities.json`, `submission.json`, `launch.sh`, `slurm_job.txt`.
Logs: `logs/`. Check 22 unique IDs, zero errors, point/box dimensions and scorer
coverage for all 22 groups. No benchmark accuracy claim from these 22 queries.

Final verification: external `verification.json` records exact manifest order,
all box/point dimensions and bounds, raw examples and coverage. One query hit the fixed 4096-token cap; no cap retry was performed.
