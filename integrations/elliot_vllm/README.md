# Elliot vLLM integration

Use `evaluator/configs/current/elliot_vllm.json` for the selected
cardinality-repair 70/30 local-500 release: dynamic images capped at 4,194,304
pixels, minimum 65,536, 32-pixel alignment and 16,384 context tokens. The
launcher independently supplies the 4,096 output-token budget. The config's
model path is the verified HF export used for the published evaluation and
Dropbox release.
The evaluator saves each query's resized dimensions and image-token count.

Earlier validation and `outputs/full_tp1_20260905` used a fixed 512×512 exported
processor. Those results describe that lower-resolution setup; they do not validate
training-resolution NaViT performance. The existing run is retained.

This package registers `ElliotVLForCausalLM` through vLLM's normal
[model plugin mechanism](https://docs.vllm.ai/en/latest/design/plugin_system/).
It targets the installed vLLM 0.24 series and Qwen3.5 text decoder. The checkpoint
supplies its existing PyTorch vision encoder and projector; vLLM supplies the
language decoder, cache, and generation scheduler.

The evaluator adapter follows the request format in
`evals_localization/vendor/rex_omni/inference_text_prompt.py`: initialize one
`LLM`, submit image/prompt requests to `generate`, then decode its responses.
It submits the evaluator's batch together and reuses the existing coordinate
parser, prompts, manifests, resume checks, and scorers.

## Compatibility details

- The saved `SiglipImageProcessor` has a fixed resize. Training resizes images
  first with `qwen_vl_utils.smart_resize` and disables that processor resize.
  The dynamic path reproduces these steps and expands the image placeholder
  from the actual patch grid. The 512 configuration was missing those steps.
- NaViT retains its checkpoint projections, positional embeddings and 2-D RoPE.
  PyTorch SDPA avoids materializing the full attention matrix at high resolution.
- Elliot uses ordinary sequential decoder positions across text and image tokens.
  `configure_hf_config` removes Qwen's multimodal RoPE markers in memory; it does
  not modify the checkpoint on disk.
- HF's `inputs_embeds` generation applies repetition penalty only to generated
  tokens. The plugin reuses Transformers' penalty through vLLM's supported logits
  processor adapter to preserve that behavior. The evaluator baseline uses 1.05.
- Coordinate tokens must survive decoding (`skip_special_tokens=False`).
- Vision is evaluated independently for each image; the decoder handles the batch.
- No training environment or installed vLLM source is patched.

## Files

- `elliot_vllm/model.py`: multimodal model, weight loading, Qwen recurrent-state hooks.
- `elliot_vllm/processing.py`: checkpoint processor, image-token counts and expansion.
- `elliot_vllm/sampling.py`: HF-compatible output-only repetition penalty.
- `evaluator/adapters/elliot_vllm.py`: evaluator adapter.
- `tests/smoke.py`: GPU HF/vLLM prompt/output comparison and single/batched comparison.

Installation uses an isolated environment with the existing compatible vLLM
runtime and `pip install --no-deps --no-build-isolation -e integrations/elliot_vllm`.
Launch commands are below; reproduction results are recorded separately.

## Leonardo commands

Set up a clean checkout of this repository on Leonardo.
Its environment is `/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval`;
it reuses the existing compatible runtime packages without installing into the
training environment. Only the `elliot` vLLM plugin is enabled by the launchers.

The checkout also reuses `evals_localization/runtime/leonardo` and the Python 3.11
FastEvaluate extension from the existing evaluation checkout. These compiled
scorer assets are ignored by Git and are required for scoring, independently of
the vLLM environment.

```bash
cd /path/to/elliot-localization-eval
mkdir -p outputs
export EVAL_PYTHON=/leonardo_work/EUHPC_E07_125/nas/envs/elliot-vllm-eval/bin/python
export CONFIG="$PWD/evaluator/configs/current/elliot_vllm.json"
export OUTPUT_DIR="$PWD/outputs/my_checkpoint_850"
sbatch evaluator/slurm/vllm_benchmark.sbatch
```

The config selects an already exported HF checkpoint. Copy it and change
`model_path` and `name` for another ordinary discrete-token checkpoint. The
launcher defaults to four independent GPU shards, request batches of 64, and 4096 new tokens,
with no token-cap retries. Set `BATCH_SIZE` or `MAX_NEW_TOKENS` to change the
batch size or output limit. The config allows 32 active sequences per GPU and
enables CUDA graphs; vLLM schedules requests from each submitted batch as slots
become available. `max_model_len` must cover the
expanded image/text prompt plus the longest allowed output. `MANIFEST` selects
another compatible evaluator manifest. This does not turn the 850-example subset
into a full benchmark evaluation.

The earlier 512-token run capped 31/850 responses. The current default allows
4096 tokens in one generation and stops early at EOS; it does not enable retries.
Compare scores using the same output budget.

Predictions appear in `$OUTPUT_DIR/shard_*.jsonl`, then `predictions.jsonl` and
`scores/scores.json`. Re-submit with the same output directory to resume compatible
outputs. Changing the batch size or config requires a new output directory.

The two-backend GPU smoke is:

```bash
MODEL_PATH=/path/to/exported/hf/checkpoint \
  sbatch integrations/elliot_vllm/tests/smoke_leonardo.sbatch
```

It compares expanded prompt tokens, records exact HF/vLLM output agreement and
single-versus-batched agreement, and requires valid decoded boxes. Token differences
are reported without failing the default smoke; add `--require-exact` when invoking
`tests/smoke.py` directly to require bitwise agreement too. Benchmark score
reproduction is a separate check and is not implied by a passing loading smoke.

## Full supplied benchmark suite

The prepared run folder on Leonardo is
`outputs/full_tp1_20260905/` inside the isolated checkout. Its `README.md` describes
the artifacts and rerun procedure; `launch.sh` submits sixteen TP=1 replicas on
16 GPUs with a 4096-token limit and no retries. Each replica remains loaded across
all its benchmark groups. This uses the existing offline evaluator; it is not a
long-lived HTTP server for future relabeling requests.

The frozen manifest contains 179,959 queries across 22 dataset/protocol groups
and 44,289 existing images. All supplied annotation rows are considered, OCR
queries cover every transcription in chunks of at most ten, and pointing expands
every available mask category. The inventory records 113 unusable/empty-target
exclusions. Coverage refers to these annotation files, whose counts can differ
from the original benchmark releases.

`monitoring/v1/tools/manifests/builders/build_full_localization.py` builds this manifest and
inventory using the same dataset-root arguments as the V4 split builder. The run
folder preserves its source archive, code commit, checkpoint hashes, package
versions, configuration, annotation hashes, logs, predictions and scores. Dataset
images and checkpoint files remain at their recorded external paths.
