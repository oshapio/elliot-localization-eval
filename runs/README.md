# Evaluation run history

Run numbers are sequential identifiers, not checkpoint steps. Each record holds
the command, model and input identities, job attempts, artifact paths and outcome.
Retries stay in the same record. Historical evaluations predate this index;
number 0001 is the first run recorded through this workflow.

| Run | Model | Manifest | Status |
| --- | --- | --- | --- |
| [0001](0001-elliot-cardinality500-smoke7.md) | Elliot cardinality-repair local-500 | smoke_manifest_7_v1 (7 queries) | Completed; smoke passed (62797056) |
| [0002](0002-v022-77938-refbox-smoke7.md) | v022-77938-refbox | smoke_manifest_7_v1 (7 queries) | Completed; raw output and geometry checked (62805245) |
| [0003](0003-family-balanced-119000-mixed-smoke7.md) | family-balanced-119000-mixed | smoke_manifest_7_v1 (7 queries) | Completed; raw output and geometry checked (62805247) |
| [0004](0004-worker-safe-7145-mixed-smoke7.md) | worker-safe-7145-mixed | smoke_manifest_7_v1 (7 queries) | Completed; raw output and geometry checked (62805249) |
| [0005](0005-recovery-10000-mixed-smoke7.md) | recovery-10000-mixed | smoke_manifest_7_v1 (7 queries) | Completed; raw output and geometry checked (62805250) |
| [0006](0006-language-rl300-smoke7.md) | language-rl300 | smoke_manifest_7_v1 (7 queries) | Failed: missing elliot_grpo (62806537) |
| [0007](0007-allmodule-rl600-smoke7.md) | allmodule-rl600 | smoke_manifest_7_v1 (7 queries) | Failed: missing elliot_grpo (62806539) |
| [0008](0008-dense40-rl50-smoke7.md) | dense40-rl50 | smoke_manifest_7_v1 (7 queries) | Submission rejected: debug submit limit |
| [0009](0009-qwen35-missing-smoke22.md) | qwen3.5-2b-vllm | missing-groups smoke (22 queries) | Completed; 1 token-cap hit (62810773) |
| [0010](0010-rex-missing-smoke22.md) | rex-omni-3b-vllm | missing-groups smoke (22 queries) | Completed; checks passed (62810776) |

## Historical full-suite evidence

Worker-safe SFT local 7145 already has a completed full result at Leonardo:
`/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/full/worker-safe-v006-59984237-v5-final-step7145`.
Its contract pins the 228,133-query / 44-group manifest and checkpoint hash;
its scores report 228,133 samples, 44 groups and zero error rows. Existing
`validation.json` and prediction metadata remain the evidence to audit before
combining comparisons; no duplicate full inference has been submitted.
| [0011](0011-qwen35-full-backfill.md) | qwen | complete 44-group coverage via backfill | Submitted: 62812166 |
| [0012](0012-rex-full-backfill.md) | rex-omni | complete 44-group coverage via backfill | Submitted: 62812168 |
