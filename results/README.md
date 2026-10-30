# Localization results and checkpoint comparison

[Open the standalone comparison](comparison.html) · [Download the full CSV](comparison.csv)

The HTML has model toggles, a benchmark filter and source details. Download/open it
locally to use the interactive controls; GitHub displays HTML as source. The CSV
has all imported checkpoint columns side by side, including individual ODinW datasets.

## Current comparison

All values are raw fractions ×100. Each row names its metric. HumanRef F1@0.5
and mean F1 over IoU 0.5–0.95 are different rows. RefSpatial overall is weighted
across 277 queries (100 Location, 100 Placement, 77 Unseen).


## Checkpoint identities and locations

| Table label | Checkpoint / export location | Step | Recorded checkpoint hash / identity |
| --- | --- | --- | --- |
| v035 step 131000 | `Not imported` | 131000 | `Not imported` |
| bbox2d-100k local 1000 | `Not imported` | 1000 | `Not imported` |
| synthetic20 local 1000 | `Not imported` | 1000 | `Not imported` |
| v037 step 134500 | `Not imported` | 134500 | `Not imported` |
| v037 step 139500 | `Not imported` | 139500 | `Not imported` |
| v037 step 137000 | `Not imported` | 137000 | `Not imported` |
| format repair local 500 | `Not imported` | 500 | `Not imported` |
| dense/GUI local 5000 | `Not imported` | 5000 | `Not imported` |
| cardinality-neutral local 2000 | `Not imported` | 2000 | `b8d01917095e5b9f60b89208962117aa6139c9723ca580022e0b55901bd6dd11` |
| worker-safe local 1750 | `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/exports/worker-safe-v006-59984237-step1750-sidecar/converted/torch/iter_0001750/mp_rank_00/model_optim_rng.pt` | 1750 | `27bda5b8a847cd81e76b665c03b145203be39e87a142643bc1ca722f2ff3447e` |
| worker-safe local 7145 (pre-RL parent) | `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations/exports/worker-safe-v006-59984237-step7145-sidecar/converted/torch/iter_0007145/mp_rank_00/model_optim_rng.pt` | 7145 | `5b63c8197ef2b3812b814d7ceed50455cffbfc4f05287db1d6e14b2e2c280eab` |
| packing-corrected local 500 | `Not imported` | 500 | `67275b6a81d1e09c960717b19a8fb6744c8f4eea5e84396069a3d47ff889f312` |
| synthetic40 local 5000 | `Not imported` | 5000 | `Not imported` |
| Qwen3.5-2B | `Not imported` | Not imported | `Not imported` |
| Rex-Omni | `Not imported` | Not imported | `Not imported` |
| Elliot GRPO v003 / local 300 | `/leonardo_scratch/large/userexternal/auselis0/elliot-localization/post_training/localization_grpo_native7145_20261012/runs/native7145-fast-kernels-production-v003/evaluation_automation/step_300/hf_export` | 300 | `6526afc627c9120948f113eea391a3fe7874dbee45c8ec5f7884e946a04ba29e` |

## Detection: SFT parent versus the recalled GRPO checkpoint

| Benchmark | Metric | SFT worker-safe 7145 | GRPO v003 step 300 |
| --- | --- | ---: | ---: |
| COCO / boxes | F1@0.5 | 70.90 | 71.14 |
| Dense200 / boxes | F1@0.5 | 51.70 | 54.57 |
| LVIS / boxes | F1@0.5 | 63.76 | 64.28 |
| VisDrone / boxes | F1@0.5 | 56.65 | 58.81 |
| ODinW13 overall | mean_dataset_mAP | 35.48 | 36.47 |

The five recalled scores match **GRPO v003 step 300**, not its SFT 7145 parent. GRPO values come from archived score content plus its contract/validation; remote files have not been rechecked this session. `checkpoint_identity` is the contract identity, not a newly computed weights checksum.

## Referring comparison (SFT parent and baselines)

| Benchmark | Metric | Elliot SFT7145 | Qwen3.5-2B | Rex-Omni |
| --- | --- | ---: | ---: | ---: |
| ODinW13 overall | mean_dataset_mAP | 35.48 | 23.62 | 42.73 |
| HumanRef / boxes | F1@0.5 | 82.37 | 72.50 | 84.42 |
| HumanRef / boxes (IoU sweep) | F1@mean | 76.84 | 66.42 | 78.94 |
| RefCOCO+_testA | Acc@IoU0.5 | 91.29 | 80.88 | 85.12 |
| RefCOCO+_testB | Acc@IoU0.5 | 83.90 | 71.04 | 73.29 |
| RefCOCO+_val | Acc@IoU0.5 | 87.80 | 76.50 | 80.53 |
| RefCOCO_testA | Acc@IoU0.5 | 94.20 | 87.86 | 89.48 |
| RefCOCO_testB | Acc@IoU0.5 | 91.34 | 82.32 | 83.73 |
| RefCOCO_val | Acc@IoU0.5 | 92.37 | 86.18 | 87.87 |
| RefCOCOg_test | Acc@IoU0.5 | 90.04 | 84.80 | 87.20 |
| RefCOCOg_val | Acc@IoU0.5 | 89.45 | 84.93 | 87.11 |
| RefSpatialBench / Location | accuracy | 57.00 | 42.00 | 52.00 |
| RefSpatialBench / Placement | accuracy | 39.00 | 12.00 | 51.50 |
| RefSpatialBench / Unseen | accuracy | 35.06 | 16.88 | 40.26 |
| RefSpatialBench overall | query_weighted_accuracy | 44.40 | 24.19 | 48.56 |

Elliot here is the **worker-safe local-7145 pre-RL parent**, not the older local-500
smoke target. Qwen is a completed combined 228,133-query / 44-group result.
Rex's selected rows were verified from completed scoring groups; its merged
prediction artifact has 19 error rows and five wrong-dimension box answers.
Rex’s login-node full rescore was killed with exit 137; its full scoring is
incomplete. The selected rows shown above were completed before that failure.
Neither baseline is represented as a fresh generation of every historical row.

## Archived full checkpoint evaluations

These 13 local score files each contain 228,133 samples and 44 groups. Their
SHA-256 values match archived publication receipts. This verifies the stored
numbers, not every historical inference command or per-query protocol. Labels
identify distinct artifacts; row order is not a claim of linear ancestry. The
full CSV/HTML includes all their detailed scores, not only this overview.

| Checkpoint artifact | HumanRef F1@0.5 | RefCOCO val Acc@0.5 | RefSpatial overall | ODinW13 mean mAP | Prompt protocol in imported receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| v035 step 131000 | 39.16 | 92.81 | 44.40 | 27.56 | Not imported |
| bbox2d-100k local 1000 | 32.64 | 30.88 | 23.35 | 8.98 | Not imported |
| synthetic20 local 1000 | 81.65 | 92.13 | 39.04 | 34.60 | Not imported |
| v037 step 134500 | 7.97 | 92.71 | 51.99 | 8.86 | Not imported |
| v037 step 139500 | 74.60 | 92.72 | 31.77 | 31.85 | Not imported |
| v037 step 137000 | 63.15 | 92.16 | 44.04 | 29.15 | Not imported |
| format repair local 500 | 80.05 | 91.80 | 33.09 | 34.93 | Not imported |
| dense/GUI local 5000 | 81.53 | 92.39 | 37.30 | 34.75 | Not imported |
| cardinality-neutral local 2000 | 80.68 | 91.52 | 42.78 | 36.63 | localization-v5-cardinality-neutral-trained-contract |
| worker-safe local 1750 | 82.42 | 92.23 | 39.71 | 37.28 | localization-v5-cardinality-neutral-trained-contract |
| worker-safe local 7145 (pre-RL parent) | 82.37 | 92.37 | 44.40 | 35.48 | localization-v5-cardinality-neutral-trained-contract |
| packing-corrected local 500 | 80.26 | 91.77 | 41.70 | 36.25 | localization-v5-cardinality-neutral-trained-contract |
| synthetic40 local 5000 | 81.30 | 92.16 | 39.35 | 35.29 | Not imported |

## What ran, in plain terms

1. A numbered record under [`runs/`](../runs/README.md) identifies the model,
   committed code, config, manifest, command, environment, job and output paths.
2. `orchestration/vllm_full.sbatch` allocates inference shards through
   `orchestration/run_shard.sh`. The wrapper calls `evaluator/run.py`; it does not
   implement a second evaluator. `ADAPTER` and `PROMPT_FAMILY` select model execution
   and the task templates. Ground-truth coordinates/counts are not used in prompts.
3. Each adapter loads that model's tokenizer/processor and chat template.
   Elliot uses its custom vLLM integration and selected REF/BOX or mixed decoder.
   Rex uses its vendor protocol and 0–999 coordinates; Qwen uses its native
   box/point response format and 0–1000 coordinates. Decoders scale to original
   image pixels, not resized input pixels. Raw text and token IDs remain saved.
4. For the baseline coverage repair, 48,174 IDs were missing from each old
   179,959-query result. We checked all old prompts and model assets for compatibility.
   Qwen reuses 179,959 rows and generates 48,174. Rex reuses 179,937 successful rows
   and generates 48,196, including replacements for 22 old error rows. Historical
   row metadata remains intact, rather than being relabelled as new inference.
5. `evaluator/merge.py` checks exact IDs/uniqueness against the full manifest.
   `evaluator/score.py` scores each group. The complete baseline result is under
   `combined/...` (or Rex's separate `combined-repair-v1/...`), not the subset scores.
6. Rex emitted points for five box questions. The scoring repair counts each as
   a miss without removing the query, changing its answer or selecting a later box.
   Nineteen separate error rows remain disclosed. Recovery used saved predictions
   on the login node; no new inference was needed.

## Commands, evidence and remaining gaps

- [Qwen full coverage and command](../runs/0011-qwen35-full-backfill.md)
- [Rex full coverage, command and scoring recovery](../runs/0012-rex-full-backfill.md)
- [All numbered runs, including smoke and failed attempts](../runs/README.md)
- [Machine-readable values and provenance](data.json)

This snapshot covers **13 archived full checkpoint artifacts plus selected Qwen/Rex
readbacks**. It is not every saved training checkpoint. The seven-query v022-77938,
family-balanced-119000, cardinality-repair-500 and recovery-10000 smoke results are
not inserted as full accuracy columns. Their other historical full results have
not been imported here. RL dependencies/results remain separately unresolved.
Historical commands absent from the imported receipt are not invented.

Remote SSH access expired during this import. Baseline raw full-score files and
additional remote-only checkpoint results need the renewed connection; missing
cells stay blank. Per-model source paths, available checkpoint/protocol hashes,
audit limits and archive checksums are in data.json and the HTML source details.

Regenerate the CSV, HTML and this page from the checked-in snapshot:

```bash
python3 results/build.py
```

On the original workspace, `python3 results/build.py --verify-archives` additionally
checks all 13 external archived score files against their recorded checksums.
