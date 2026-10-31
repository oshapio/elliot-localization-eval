# 0013 — GRPO v003 step 300, fresh canonical full evaluation

Status: **COMPLETED 0:0**, job 62986106, elapsed **44m05s**.
User requested this exact model after identifying its historical detection scores.

Model and tokenizer/processor: same HF export as [0006](0006-language-rl300-smoke7.md),
`native7145-fast-kernels-production-v003/evaluation_automation/step_300/hf_export`.
Historical contract checkpoint identity:
`6526afc627c9120948f113eea391a3fe7874dbee45c8ec5f7884e946a04ba29e`.
Use unchanged export and its custom tokenizer; mixed-special grouped xyxy / xy,
0–1000 coordinates converted to original pixels, `elliot-loc` prompt family.
Dependency path and SHA-256 are pinned in run 0006. No FindIt decoder is involved.

Full suite: 228133 queries, 44 groups; manifest SHA-256
`08a7f24ddb75d87285813366e54deb6872f84b4250cce61ce588f73b72d2f960`.
Config is [0006-language-rl300.json](inputs/0006-language-rl300.json), copied byte-for-byte
to the run inputs. Greedy, 4096 output tokens, no token-cap retry, 4MP / 16K,
TP1, compiled vLLM, repetition penalty 1.05, batch 32, 16 replicas on 4 nodes.
This is a fresh canonical-repository execution, distinct from historical job 60569506.

Prepared launch, after validating inputs and checking queue/output duplication:

```bash
repo=/leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-grpo300-full
export RUN_DIR=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/0013-grpo300-full
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export EVAL_DATA=/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData
export PYTHONPATH=/leonardo_scratch/large/userexternal/auselis0/elliot-localization/post_training/localization_grpo_native7145_20261012/runs/native7145-fast-kernels-production-v003/source/localization-posttraining/code/07_grpo
export ADAPTER=elliot_vllm PROMPT_FAMILY=elliot-loc BATCH_SIZE=32 MAX_NEW_TOKENS=4096 RETRY_MAX_NEW_TOKENS=0 GPUS_PER_MODEL=1
export EXPECTED_MANIFEST_SHA256=08a7f24ddb75d87285813366e54deb6872f84b4250cce61ce588f73b72d2f960 EXPECTED_GROUP_COUNT=44
sbatch --parsable --job-name=grpo300-full0013 \
 --output="$RUN_DIR/logs/%x-%j.out" --error="$RUN_DIR/logs/%x-%j.err" \
 "$repo/orchestration/vllm_full.sbatch"
```

Run `source` is a symlink to the committed Git worktree above, not copied evaluator
code. Predictions: `predictions/shard_*.jsonl`, merged `predictions/predictions.jsonl`;
scores: `scores/scores.json`; logs: `logs/`. Submission evidence records source,
config, manifest and export identities. No score is claimed before completion.

Submission: **62986106**, source revision `0926feeb4a52c1f328a34aa65281df62458ed71b`.
Queue audit immediately before submission returned no jobs. Input checks confirmed
228133 unique IDs / 44 groups. Config SHA-256
`6bc927522a6ff3611080440babba52533e0a9b3a58bc224abd79f28463bacbce`.
`preflight.json` records export asset hashes and resolved paths;
`submitted_job_id.txt` and `submission-slurm.txt` preserve scheduler evidence.
Smoke 62985196 completed 0:0; seven independent decode/geometry checks passed.
No full-run score exists yet.

## Completed score readback

Final `scores/scores.json`: **228133 samples, 44 groups, zero diagnostic error rows**.
COCO box F1@0.5 71.135269%; Dense200 54.692667%; LVIS 64.221105%;
VisDrone 58.905412%; HumanRef 83.208210%; ODinW13 mean dataset mAP 36.628944%.
These are fresh-run values, not identical to the historical column. This readback
verifies final score coverage and scheduler completion; full per-row provenance
audit and comparison-table import remain pending. GPU allocation is released.
