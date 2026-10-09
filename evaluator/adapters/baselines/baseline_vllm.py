"""Stock vLLM Qwen3.5 and Rex-Omni, using the existing benchmark parsers."""

import sys
from pathlib import Path

from PIL import Image


def create_adapter(config):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    root = Path(__file__).resolve().parents[3] / "evals_localization"
    sys.path.insert(0, str(root))
    from localization_parsing import _parse

    family = config["family"]
    if family not in ("qwen", "rex-omni"):
        raise ValueError(f"Unsupported stock baseline: {family}")
    processor_path = config.get("tokenizer_path", config["model_path"])
    processor_kwargs = config.get("mm_processor_kwargs", {})
    processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)
    if family == "rex-omni":
        sys.path.insert(0, str(root / "vendor/rex_omni"))
        from inference_text_prompt import DEFAULT_PROMPT, parse_prediction
        # Fail before allocating GPUs if Transformers has shifted Rex's IDs.
        for token, expected in (("<0>", 150643), ("<999>", 151642), ("<|im_end|>", 151645)):
            actual = processor.tokenizer.convert_tokens_to_ids(token)
            if actual != expected:
                raise ValueError(f"Rex tokenizer: {token}={actual}, expected {expected}")

    engine = LLM(
        model=config["model_path"], tokenizer=processor_path,
        dtype="bfloat16", trust_remote_code=True,
        max_model_len=int(config["max_model_len"]),
        tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.85)),
        max_num_seqs=int(config.get("max_num_seqs", 16)),
        enforce_eager=bool(config.get("enforce_eager", False)),
        limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs=processor_kwargs,
        enable_prefix_caching=False, async_scheduling=False,
    )

    def predict_batch(records, prompts, max_new_tokens):
        images = []
        try:
            requests = []
            for record, prompt in zip(records, prompts):
                with Image.open(record["image_file"]) as source:
                    image = source.convert("RGB")
                images.append(image)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt},
                ]}]
                if family == "rex-omni":
                    messages.insert(0, {"role": "system", "content": DEFAULT_PROMPT})
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                requests.append({"prompt": text, "multi_modal_data": {"image": image}})
            generated = engine.generate(requests, SamplingParams(
                temperature=0, max_tokens=max_new_tokens,
                repetition_penalty=float(config.get("repetition_penalty", 1.05)),
                skip_special_tokens=False,
            ), use_tqdm=False)
            results = []
            for record, image, result in zip(records, images, generated):
                output = result.outputs[0]
                raw = output.text
                if family == "rex-omni":
                    predictions = parse_prediction(raw, image.width, image.height,
                                                   image.height, image.width, "xyxy",
                                                   record.get("task_name"))
                else:
                    predictions = _parse(raw.split("</think>")[-1], record, family,
                                         image.width, image.height)
                results.append({"raw_response": raw, "predictions": predictions,
                                "hit_token_cap": output.finish_reason == "length",
                                "token_ids": list(output.token_ids),
                                "generated_tokens": len(output.token_ids),
                                "prompt_token_ids": list(result.prompt_token_ids)})
            return results
        finally:
            for image in images:
                image.close()

    return {"name": config["name"], "predict_batch": predict_batch}
