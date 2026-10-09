# NaViT 4 MP validation and full run (2026-09-05)

The earlier measurements below used a fixed 512×512 processor. They do not
establish the intended NaViT evaluation performance. The corrected checkpoint's
training run records max_pixels=4194304, min_pixels=65536, image_scale=1.0 and
seq_len=16384. The training encoder resizes at 32-pixel alignment and disables
the saved processor resize.

- Mixed box/point/OCR/GUI smoke **56138542**: 7/7 nonempty predictions, zero
  runtime errors and zero output caps, at a 4096-token output maximum.
- HF/vLLM comparison **56138543**: the stress image becomes 2496×1664 with
  4056 image tokens. Both prompts and generated token sequences match exactly
  for both queries; single-request and batched vLLM outputs also match.
- SDPA is checked against the checkpoint attention in float32 and bf16; it
  keeps the projections and 2-D rotary embeddings and avoids a full attention
  matrix. Floating-point implementations are not generally bitwise equivalent.
- All 44,289 images pass resize validation. Maximum image-token count: 4096.
  The conservative full-suite prompt-plus-output bound is 8762, below 16384.
- 31 evaluator tests pass, including rejection of a silent image-token count mismatch.

Full job **56140410**: 64 TP=1 replicas, 16 nodes, request batch 32,
max_num_seqs=16, context 16384, output cap 4096, no token-cap retries.
Artifacts on Leonardo: `outputs/full_elliot_navit_4mp_64gpu_20260905/`.
That folder contains frozen source, checkpoint hashes, validation outputs,
training preprocessing references and `launch.sh`. Predictions record resized
image dimensions and image-token counts checked against actual vLLM input IDs.
The existing 512-resolution full run is retained; the 4 MP run has fresh outputs.

These checks validate the corrected inference path; full benchmark scores are
produced separately by the full run.

---

# GPU validation

Latest default-budget check: job `56108068` completed all 850 examples and scoring
with a single 4096-token limit, zero errors and zero retries. Four TP=1 engines,
request batch 64 and 32 active sequences per GPU produced 155,080 tokens; 29
responses exceeded 512 tokens and 22 hit the 4096-token ceiling. Total job time
was 8m34s. Engines reported ready at 19:04:53 and the last engines began shutdown
at 19:06:50: approximately 1m57s including generation and first-use compilation.
Artifacts: `outputs/vllm_corrected_sft_36000_plus100_850_v4_graph_b64_4096_no_retry/`
in the isolated checkout. Earlier 512-token measurements below are historical.

Validated runtime: Python 3.11.7, PyTorch 2.11.0+cu129, Transformers 5.12.1,
vLLM 0.24.0+cu129, four A100 GPUs on Leonardo (one independent engine per GPU).

## Checkpoints and smoke tests

Both checkpoints under `/leonardo_work/EUHPC_E07_125/nas/checkpoints/locateanything/`
load through the same plugin:

- `step36000_4mp_hf_grpo`: both batched smoke outputs match HF exactly.
- `corrected_sft_36000_plus100_hf_eval`: one of two batched smoke outputs matches
  HF exactly. The other differs in coordinates and box count.

Both expanded smoke prompts match HF token-for-token for both checkpoints.
Single versus batched output is not always identical. These are numerical
agreement checks on two prompts, not an accuracy estimate. The installed vLLM
rejects batch-invariant mode for its GDN attention backend.

The corrected checkpoint was exported using the existing training repository's
`tools_local/export_elliot_9k_to_hf.py` from corrected-SFT run
`20260831_163449_corrected_sft_step36000_r1/iter_0000100/mp_rank_00/model_optim_rng.pt`.
All 761 mapped tensors loaded, with no missing mappings or shape mismatches.
The export's `eval_export_provenance.json` identifies its source and base model.

## Initial 850-example comparison

Job `56069091`, corrected-SFT 36k +100, `benchmark_850_v4.jsonl`, four independent
GPU shards, batch 8, context 8192, greedy decoding, repetition penalty 1.05.
The original runner used 512 output tokens followed by a 4096-token regeneration
for capped examples. This affected 31 vLLM examples (35 in the saved HF baseline).
**That behavior is now disabled. These historical results include those retries.**

All 850 predictions completed with zero errors. The job's final scoring step
initially failed because the separate checkout lacked the compiled scorer;
scoring completed after restoring the existing scorer runtime. The failed Slurm
status does not mean generation failed.

| Dataset | Metric | N | Saved HF | vLLM |
|---|---|---:|---:|---:|
| RefCOCOg val | First-box accuracy, IoU ≥ 0.5 | 50 | 80.00 | 80.00 |
| RefCOCOg test | First-box accuracy, IoU ≥ 0.5 | 50 | 90.00 | 92.00 |
| COCO boxes | F1@0.5 | 50 | 75.71 | 75.59 |
| Dense200 boxes | F1@0.5 | 50 | 51.23 | 49.71 |
| HumanRef boxes | F1@0.5 | 50 | 72.19 | 72.41 |
| VisDrone boxes | F1@0.5 | 50 | 31.12 | 32.74 |
| ScreenSpot v2 | Accuracy | 50 | 54.00 | 54.00 |
| ScreenSpot Pro | Accuracy | 50 | 0.00 | 0.00 |

All values are percentages. The largest point-task change was Dense200
55.71 → 63.47 F1@0.5 on only 15 examples. Inspection found two saved HF outputs
(`point:Dense200:000041` and `000184`) that repeated until the 4096-token cap,
producing 1023 parsed points each; vLLM stopped with 44 and 39 points respectively
(135 and 42 ground-truth points). Such generation differences materially affect
precision and limit any claim of exact reproduction. Only 288/850 parsed predictions were
identical. Similar aggregate scores do not establish exact backend equivalence.
This is a development subset across 22 dataset/protocol groups, not full official
benchmark coverage.

Durable artifacts in the isolated checkout:
`outputs/vllm_corrected_sft_36000_plus100_850_v4_b8/` contains predictions,
per-shard settings, and `scores/scores.json`.

## Runtime diagnosis

The initial four-GPU run took 21m13s for the inference Slurm step. The logs show
about 29s loading model weights and 27s for cache profiling/warm-up. Roughly seven
minutes elapsed before useful inference, including Python/engine startup and
first-use kernel compilation; it was not seven minutes reading the weights.

The run used eager execution and admitted only eight requests per GPU at a time,
waiting for the entire group before submitting more. It produced 149,202 final
tokens plus discarded attempts, with uneven output lengths across shards.
These settings and timings are not a tuned vLLM throughput result.

`tests/throughput.py` compares the same 88 examples with request batches of 8 and
64, and eager execution versus CUDA graphs. Every scored example is generated
once with a 512-token cap. Import, engine startup, warm-up, and generation times
are recorded separately; generated tokens and cap counts accompany throughput.

Job `56084052` completed all three modes on separate GPUs on the same node:

| Mode | Active sequences | Request batch | Generation time | Output tokens/s | Capped |
|---|---:|---:|---:|---:|---:|
| Eager | 8 | 8 | 89.88s | 85.0 | 3 |
| Eager | 32 | 64 | 32.91s | 223.8 | 4 |
| CUDA graphs | 32 | 64 | 12.90s | 580.0 | 4 |

CUDA graphs and a larger queue improved measured generation throughput 6.8×
over the initial eager/batch-8 setting. Output token counts varied slightly
(7643, 7367, 7484); these are not identical token streams. Each mode warmed up
separately before measurement, but later shape-specific kernel compilation is
included in generation time. The selected defaults are request batch 64,
32 active sequences, CUDA graphs enabled, and no token-cap retries.

Startup remains substantial: importing vLLM/PyTorch took 118s, engine setup
292–299s, and explicit warm-up 21–27s in this three-process test. The 12.90s
figure excludes those phases. Model loading itself took about 10s. This profile
does not establish an end-to-end speedup over HF.

## Final run without retries

Job `56087727` completed successfully, including merge and scoring. It used the
same corrected checkpoint and 850-example manifest, request batch 64, 32 active
sequences per GPU, CUDA graphs, and a single 512-token output limit.

- 850/850 predictions, zero errors, zero retries, 31 capped outputs.
- 74,034 generated tokens.
- Total Slurm job time: 7m38s (inference step: 7m25s).
- Engines reported ready at 18:02:32; all four began shutdown at 18:03:26,
  about 54s later, including first-use kernel compilation and generation.
- Startup still dominates: the job began at 17:56:24. These timings do not imply
  that model loading alone takes six minutes.

| Dataset | Metric | N | Saved HF | Final vLLM |
|---|---|---:|---:|---:|
| RefCOCOg val | First-box accuracy, IoU ≥ 0.5 | 50 | 80.00 | 80.00 |
| RefCOCOg test | First-box accuracy, IoU ≥ 0.5 | 50 | 90.00 | 92.00 |
| COCO boxes | F1@0.5 | 50 | 75.71 | 75.64 |
| Dense200 boxes | F1@0.5 | 50 | 51.23 | 51.28 |
| HumanRef boxes | F1@0.5 | 50 | 72.19 | 72.41 |
| VisDrone boxes | F1@0.5 | 50 | 31.12 | 32.68 |
| ScreenSpot v2 | Accuracy | 50 | 54.00 | 54.00 |
| ScreenSpot Pro | Accuracy | 50 | 0.00 | 0.00 |

The saved HF run allowed a second generation up to 4096 tokens, so this final
comparison changes output-budget policy as well as backend. Use the earlier
comparison to assess matching-budget reproduction and the controlled 88-example
test to assess the batching/graph speedup. The total-time reduction from the
initial 21m16s also includes removing retries and producing fewer tokens.

Final durable predictions, metadata and scores:
`outputs/vllm_corrected_sft_36000_plus100_850_v4_graph_b64_no_retry/` in the isolated
checkout. All test GPU allocations have finished. CPU adapter/runner checks:
23 tests and two subtests passed.

## Startup investigation

Job `56101314` measured imports with Python's `-X importtime`, wall-clock and
process CPU timestamps on Leonardo node `lrdn0535`. After loading the engine and
generating a response, a fresh Python process repeated the same imports:

| Import phase | First process | Fresh process, cached files |
|---|---:|---:|
| PyTorch | 79.37s | 1.79s |
| vLLM package | 31.45s | 2.86s |
| vLLM LLM dependencies | 96.91s | 9.02s |
| Elliot plugin registration module | 0.006s | 0.001s |
| Total, including PIL | 208.50s | 13.71s |

The first import sequence consumed only 11.63s of process CPU time; the repeat
consumed 10.58s. Starting Python itself took approximately four seconds on the
first pass and less than a second on the repeat. Both processes loaded the same
runtime under `ElliotVL/envs/dataengine_vllm`, on `/leonardo_work` Lustre storage.
Native-library imports (`torch`, `torch._C`, `triton._C.libtriton`) and Transformers
were the largest individually recorded delays. The separate engine startup then
took 145.95s in the first process.

This isolates the large pre-engine delay to cold runtime-file access/cache state;
it is not checkpoint reading or costly Elliot registration. A preparatory import
alone moves the cold-start cost rather than eliminating it. Node-local runtime
staging or retaining workers between evaluations are candidate remedies; neither
has been benchmarked here. No runtime package or launcher was changed by this
investigation. Detailed logs are in `outputs/startup/56101314/` in the isolated
checkout.
