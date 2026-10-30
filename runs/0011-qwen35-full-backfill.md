# 0011 — qwen35-full-backfill

Status: **completed**, Slurm job **62812166**, exit `0:0`, elapsed **28m 27s**.
The corresponding 22-query smoke passed execution, geometry and scoring checks. Using the stated default: backfill missing/error rows and retain verified
compatible old rows, with original provenance preserved.

## Coverage and provenance

- Complete target: **228,133 queries / 44 groups**, full manifest SHA-256
  `08a7f24ddb75d87285813366e54deb6872f84b4250cce61ce588f73b72d2f960`.
- New inference: **48,174 queries / 22 groups**; the 48,174 previously
  missing IDs.
- Reuse **179,959 successful rows** from
  `/leonardo_work/EUHPC_E07_125/nas/elliot-vl-eval-vllm/outputs/full_qwen35_vllm_20260905/predictions/predictions.jsonl`.
  Retain their original metadata; these are historical generations, not new ones.
- Every reused rendered prompt matches the current family template exactly;
  output cap is 4096 with no token-cap retry. Historical model/tokenizer asset
  hashes match the current assets. Different generation environments remain
  separately attributable; this is a combined-coverage result, not one fresh run.
- Model config: [`0009-qwen35-missing-smoke22.json`](inputs/0009-qwen35-missing-smoke22.json). Same model-specific processor,
  prompts, coordinate conventions and parser as the corresponding baseline smoke.
- Canonical source worktree at commit `f4cb946`, linked at `RUN_DIR/source`.
  This includes the HumanRef prompt preservation fix; no copied evaluator source.
- Preparation recipe: [`0011-0012-prepare.py`](inputs/0011-0012-prepare.py).
  It validates the pinned full manifest, unique IDs, existing prompts and budgets;
  excludes old error rows and selects their replacements plus missing IDs.
  It refuses to overwrite an existing run directory.
- Prepared inference manifest SHA-256: `824d0bf38d3a5ec9ccc08f8254d5961abfe192fa05042a3da79edbd632561790`.
- Resource request: four nodes / 16 GPUs, TP=1 replicas, two hours. All model,
  image/context and sampling settings remain in the config and command.

## Attempt 1 — prepared command

Job ID: **62812166**, first attempt. Before submitting, check the smoke, current jobs and receipts;
never launch concurrent writers to this run's shards.

```bash
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export RUN_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0011-qwen35-full-backfill
export ADAPTER=baseline_vllm PROMPT_FAMILY=qwen
export BATCH_SIZE=32 MAX_NEW_TOKENS=4096 GPUS_PER_MODEL=1
export EXPECTED_MANIFEST_SHA256=824d0bf38d3a5ec9ccc08f8254d5961abfe192fa05042a3da79edbd632561790
export EXPECTED_GROUP_COUNT=22
unset PYTHONPATH
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=qos_llm_prod --job-name=baseline-backfill0011 --nodes=4 \
  --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=8 \
  --time=02:00:00 --output="$RUN_DIR/logs/batch-%j.out" \
  --error="$RUN_DIR/logs/batch-%j.err" --wrap='set -eu
bash "$RUN_DIR/source/orchestration/vllm_full.sbatch"
mkdir -p "$RUN_DIR/combined"
"$EVAL_PYTHON" "$RUN_DIR/source/evaluator/merge.py" \
  --manifest /leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/full/full_v022_matched_step77938_228k_32gpu_debug_20260915/inputs/manifest.jsonl \
  --shard "$RUN_DIR/inputs/reused_predictions.jsonl" \
  --shard "$RUN_DIR/predictions/predictions.jsonl" \
  --output "$RUN_DIR/combined/predictions.jsonl"
"$EVAL_PYTHON" "$RUN_DIR/source/evaluator/score.py" \
  --predictions "$RUN_DIR/combined/predictions.jsonl" \
  --eval-data "$EVAL_DATA" --output-dir "$RUN_DIR/combined/scores"'
```

## Evidence and acceptance

Root: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0011-qwen35-full-backfill`.
`preparation.json` records selection counts and hashes of the old predictions and
full/backfill manifests. Inputs are under `inputs/`, new inference and its metadata
under `predictions/`, subset scores under `scores/`, and job/rank logs under `logs/`.
Copy the submitted command to `launch.sh` and save the job receipt before handoff.

**The full comparison result is `combined/scores/scores.json`**, from the exact
full-manifest merge of historical compatible rows and new rows. Require exactly
228,133 unique IDs, 44 scored groups, report all error/invalid rows and retain raw
responses and per-row origin metadata. Never report subset scores as full scores.

Backfill submitted with 16 GPUs / four nodes and a two-hour limit. Both image
and annotation preflights passed. Existing rows are retained separately; complete
combined scores and job completion are still pending.

## Completed result

`combined/scores/scores.json` reports **228,133 samples / 44 groups / zero error
rows**. The new inference subset has 48,174 rows / 22 groups / zero errors.
Slurm inference and batch steps completed successfully. These are combined
historical/new generations, with origin metadata retained as described above.

## Newly covered benchmark scores

Qwen3.5-2B, read directly from `combined/scores/scores.json`:

| Benchmark | Metric | Score (×100) |
| --- | --- | ---: |
| ODinW13 | mean dataset mAP | 23.62 |
| RefSpatialBench (277 queries) | accuracy | 24.19 |
| RefCOCO val | Acc@IoU0.5 | 86.18 |
| RefCOCO testA | Acc@IoU0.5 | 87.86 |
| RefCOCO testB | Acc@IoU0.5 | 82.32 |
| RefCOCO+ val | Acc@IoU0.5 | 76.50 |
| RefCOCO+ testA | Acc@IoU0.5 | 80.88 |
| RefCOCO+ testB | Acc@IoU0.5 | 71.04 |

RefCOCOg box accuracy was already present in the reused baseline coverage:
val 84.93%, test 84.80%. Do not confuse box accuracy with the separate pointing
F1 results. Full combined coverage is 228,133 queries / 44 groups / zero error rows.
