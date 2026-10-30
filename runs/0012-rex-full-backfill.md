# 0012 — rex-full-backfill

Status: inference complete; initial scoring failed. Job **62812168**, exit `1:0`,
elapsed **12m 02s**; inference step completed with exit `0:0` in **11m 42s**.
The corresponding 22-query smoke passed execution, geometry and scoring checks. Using the stated default: backfill missing/error rows and retain verified
compatible old rows, with original provenance preserved.

## Coverage and provenance

- Complete target: **228,133 queries / 44 groups**, full manifest SHA-256
  `08a7f24ddb75d87285813366e54deb6872f84b4250cce61ce588f73b72d2f960`.
- New inference: **48,196 queries / 28 groups**; the 48,174 previously
  missing IDs plus Rex’s 22 historical error rows.
- Reuse **179,937 successful rows** from
  `/leonardo_work/EUHPC_E07_125/nas/elliot-vl-eval-vllm/outputs/full_rex_vllm_20260905/predictions/predictions.jsonl`.
  Retain their original metadata; these are historical generations, not new ones.
- Every reused rendered prompt matches the current family template exactly;
  output cap is 4096 with no token-cap retry. Historical model/tokenizer asset
  hashes match the current assets. Different generation environments remain
  separately attributable; this is a combined-coverage result, not one fresh run.
- Model config: [`0010-rex-missing-smoke22.json`](inputs/0010-rex-missing-smoke22.json). Same model-specific processor,
  prompts, coordinate conventions and parser as the corresponding baseline smoke.
- Canonical source worktree at commit `f4cb946`, linked at `RUN_DIR/source`.
  This includes the HumanRef prompt preservation fix; no copied evaluator source.
- Preparation recipe: [`0011-0012-prepare.py`](inputs/0011-0012-prepare.py).
  It validates the pinned full manifest, unique IDs, existing prompts and budgets;
  excludes old error rows and selects their replacements plus missing IDs.
  It refuses to overwrite an existing run directory.
- Prepared inference manifest SHA-256: `05fa505255f559f4faec1fc4c272a627a1f70e2561670b423d2f561efea762b5`.
- Resource request: four nodes / 16 GPUs, TP=1 replicas, two hours. All model,
  image/context and sampling settings remain in the config and command.

## Attempt 1 — prepared command

Job ID: **62812168**, first attempt. Before submitting, check the smoke, current jobs and receipts;
never launch concurrent writers to this run's shards.

```bash
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export RUN_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0012-rex-full-backfill
export ADAPTER=baseline_vllm PROMPT_FAMILY=rex-omni
export BATCH_SIZE=32 MAX_NEW_TOKENS=4096 GPUS_PER_MODEL=1
export EXPECTED_MANIFEST_SHA256=05fa505255f559f4faec1fc4c272a627a1f70e2561670b423d2f561efea762b5
export EXPECTED_GROUP_COUNT=28
unset PYTHONPATH
sbatch --parsable --account=EUHPC_E07_125 --partition=boost_usr_prod \
  --qos=qos_llm_prod --job-name=baseline-backfill0012 --nodes=4 \
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

Root: `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0012-rex-full-backfill`.
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

## Scoring recovery — no inference rerun

New inference has all 48,196 rows, with 19 error rows. Five additional box-query
responses contain two-coordinate points. The referring scorer crashed indexing
those points as boxes. PR #7 counts an invalid first prediction as a miss, keeps
the query in the denominator, and reports invalid dimensions separately. Existing
raw/decoded predictions and partial scores remain unchanged.

Scoring source: commit `3165c72`, separate Git worktree below. Inference source
remains `f4cb946`. Four scorer regression tests and nine adapter tests passed.
The retry writes separate score/combined directories and preserves the failed job.

Prepared CPU-only command (scorer worktree and compiled extension staged first):

```bash
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export SCORE_REPO=/leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-score-repair
export RUN_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0012-rex-full-backfill
export OMP_NUM_THREADS=8
sbatch --parsable --account=EUHPC_E07_125 --partition=lrd_all_serial --qos=normal \
  --job-name=rex0012-score-repair --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=24G --time=01:00:00 --output="$RUN_DIR/logs/score-repair-%j.out" \
  --error="$RUN_DIR/logs/score-repair-%j.err" --wrap='set -eu
"$EVAL_PYTHON" "$SCORE_REPO/evaluator/score.py" \
  --predictions "$RUN_DIR/predictions/predictions.jsonl" \
  --eval-data "$EVAL_DATA" --output-dir "$RUN_DIR/scores-repair-v1"
mkdir -p "$RUN_DIR/combined-repair-v1"
"$EVAL_PYTHON" "$SCORE_REPO/evaluator/merge.py" \
  --manifest /leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/full/full_v022_matched_step77938_228k_32gpu_debug_20260915/inputs/manifest.jsonl \
  --shard "$RUN_DIR/inputs/reused_predictions.jsonl" \
  --shard "$RUN_DIR/predictions/predictions.jsonl" \
  --output "$RUN_DIR/combined-repair-v1/predictions.jsonl"
"$EVAL_PYTHON" "$SCORE_REPO/evaluator/score.py" \
  --predictions "$RUN_DIR/combined-repair-v1/predictions.jsonl" \
  --eval-data "$EVAL_DATA" --output-dir "$RUN_DIR/combined-repair-v1/scores"'
```

Scoring-retry job ID: pending. The intended final full result is
`combined-repair-v1/scores/scores.json`. Report the 19 error rows and five
wrong-dimension predictions explicitly; do not label this error-free.

The user subsequently requested running scoring on the login node. The exact
score/merge/score body above is running directly on login02 with the same
scoring revision and `OMP_NUM_THREADS=8`; no Slurm scoring job was submitted.
Executed script: `score-repair-login.sh`; logs: `logs/score-repair-login.out`
and `.err`. No inference was rerun.
