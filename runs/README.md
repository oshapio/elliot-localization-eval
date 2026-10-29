# Evaluation run history

Run numbers are sequential identifiers, not checkpoint steps. Each record holds
the command, model and input identities, job attempts, artifact paths and outcome.
Retries stay in the same record. Historical evaluations predate this index;
number 0001 is the first run recorded through this workflow.

| Run | Model | Manifest | Status |
| --- | --- | --- | --- |
| [0001](0001-elliot-cardinality500-smoke7.md) | Elliot cardinality-repair local-500 | smoke_manifest_7_v1 (7 queries) | Completed; smoke passed (62797056) |
| [0002](0002-v022-77938-refbox-smoke7.md) | v022-77938-refbox | smoke_manifest_7_v1 (7 queries) | Prepared; not submitted |
| [0003](0003-family-balanced-119000-mixed-smoke7.md) | family-balanced-119000-mixed | smoke_manifest_7_v1 (7 queries) | Prepared; not submitted |
| [0004](0004-worker-safe-7145-mixed-smoke7.md) | worker-safe-7145-mixed | smoke_manifest_7_v1 (7 queries) | Prepared; not submitted |
| [0005](0005-recovery-10000-mixed-smoke7.md) | recovery-10000-mixed | smoke_manifest_7_v1 (7 queries) | Prepared; not submitted |
