"""Run normal image requests through one persistent vLLM engine (Rex pattern)."""

import io
import math
import os
from functools import partial

from PIL import Image

from decoders.discrete_ntp import decode


def render_prompt(question):
    """Match evals_localization.models.make_elliot's non-thinking chat exactly."""
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _open_image(record):
    """Open a normal image path or an Energon byte-addressed tar member."""
    if record.get("image_file"):
        return Image.open(record["image_file"])
    reference = record.get("image_reference")
    if not isinstance(reference, dict) or reference.get("k") != "tar":
        raise ValueError("record needs image_file or a tar image_reference")
    required = {"p", "o", "n"}
    if required - set(reference):
        raise ValueError(f"tar image_reference is missing {sorted(required - set(reference))}")
    root = record.get("media_root")
    if not root:
        raise ValueError("tar image_reference needs media_root")
    archive = os.path.join(root, reference["p"])
    offset, size = int(reference["o"]), int(reference["n"])
    if offset < 0 or size <= 0:
        raise ValueError("tar image_reference offset/size must be positive")
    with open(archive, "rb", buffering=0) as handle:
        payload = os.pread(handle.fileno(), size, offset)
    if len(payload) != size:
        raise ValueError(f"short image read from {archive}: {len(payload)} != {size}")
    return Image.open(io.BytesIO(payload))


def _chosen_token_logprobs(output):
    """Extract the sampled token's logprob at every generated position."""
    token_ids = list(output.token_ids)
    positions = getattr(output, "logprobs", None)
    if positions is None:
        raise ValueError("vLLM returned no logprobs although return_logprobs was requested")
    if len(positions) != len(token_ids):
        raise ValueError(f"logprob/token length mismatch: {len(positions)} != {len(token_ids)}")
    values = []
    for index, (token_id, candidates) in enumerate(zip(token_ids, positions)):
        candidate = candidates.get(token_id)
        if candidate is None:
            candidate = candidates.get(str(token_id))
        if candidate is None:
            raise ValueError(f"chosen token {token_id} missing from logprobs at position {index}")
        value = getattr(candidate, "logprob", candidate)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite chosen-token logprob at position {index}")
        values.append(value)
    return values


def create_adapter(config):
    output_format = config.get("output_format", "ref_box")
    if output_format not in {"ref_box", "mixed_special_xyxy_grouped_all_instances_v2"}:
        raise ValueError(f"Unknown Elliot output_format: {output_format}")
    decoder = decode
    if output_format == "mixed_special_xyxy_grouped_all_instances_v2":
        from decoders.mixed_special import decode as decoder

    from vllm import LLM, SamplingParams
    from elliot_vllm import configure_hf_config

    engine = LLM(
        model=config["model_path"], trust_remote_code=True,
        hf_overrides=partial(configure_hf_config, max_pixels=config.get("max_pixels"),
                             min_pixels=config.get("min_pixels", 65536)), dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        max_model_len=int(config["max_model_len"]),
        tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.8)),
        max_num_seqs=int(config.get("max_num_seqs", 32)),
        enforce_eager=bool(config.get("enforce_eager", False)),
        logits_processors=["elliot_vllm.sampling:ElliotRepetitionPenalty"],
        enable_prefix_caching=False,
        async_scheduling=False,
    )

    image_token_id = (engine.get_tokenizer().convert_tokens_to_ids("<|image_pad|>")
                      if config.get("max_pixels") else None)

    def predict_batch(records, prompts, max_new_tokens):
        images = []
        try:
            for record in records:
                with _open_image(record) as source:
                    images.append(source.convert("RGB"))
            requests = [{"prompt": render_prompt(prompt), "multi_modal_data": {"image": image}}
                        for prompt, image in zip(prompts, images)]
            sampling = {
                "temperature": 0,
                "max_tokens": max_new_tokens,
                "extra_args": {
                    "elliot_repetition_penalty": float(config.get("repetition_penalty", 1.05))
                },
                "skip_special_tokens": False,
            }
            if config.get("return_logprobs", False):
                sampling["logprobs"] = 1
            generated = engine.generate(
                requests, SamplingParams(**sampling), use_tqdm=False)
            results = []
            for record, image, result in zip(records, images, generated):
                output = result.outputs[0]
                confidence = {}
                if config.get("return_logprobs", False):
                    token_logprobs = _chosen_token_logprobs(output)
                    cumulative = sum(token_logprobs)
                    confidence = {
                        "token_logprobs": token_logprobs,
                        "cumulative_logprob": cumulative,
                        "mean_logprob": cumulative / len(token_logprobs)
                        if token_logprobs else None,
                    }
                geometry = {}
                if config.get("max_pixels"):
                    from qwen_vl_utils.vision_process import smart_resize
                    height, width = smart_resize(image.height, image.width, factor=32,
                        min_pixels=config.get("min_pixels", 65536), max_pixels=config["max_pixels"])
                    image_tokens = list(result.prompt_token_ids).count(image_token_id)
                    expected = (width // 32) * (height // 32)
                    if image_tokens != expected:
                        raise ValueError(f"Image token count {image_tokens} != resized grid {expected}")
                    geometry = {"processed_image_size": [width, height],
                                "image_tokens": image_tokens}
                results.append({
                    **geometry,
                    **confidence,
                    "raw_response": output.text,
                    "predictions": decoder(
                        output.text, image.width, image.height,
                        int(config.get("coordinate_max", 1000)),
                        2 if record.get("benchmark_protocol") == "point" else 4,
                    ),
                    "hit_token_cap": output.finish_reason == "length",
                    "token_ids": list(output.token_ids),
                    "generated_tokens": len(output.token_ids),
                    "prompt_token_ids": list(result.prompt_token_ids),
                })
            return results
        finally:
            for image in images:
                image.close()

    return {
        "name": config.get("name", "elliot-vllm"),
        "predict_batch": predict_batch,
        "output_format": output_format,
        "fail_fast": output_format != "ref_box",
    }
