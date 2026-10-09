"""Adapter for the official NVIDIA LocateAnything checkpoint."""

import os

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from decoders.discrete_ntp import decode


def assigned_device():
    """Select the GPU assigned to this Slurm rank."""
    local_rank = 0 if torch.cuda.device_count() == 1 else int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def load_model(config, device):
    """Load one persistent official model with the supported SDPA backend."""
    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_config._attn_implementation = "sdpa"
    model_config.text_config._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    return tokenizer, processor, model


def prepare_inputs(processor, image, prompt, device):
    """Build the official multimodal chat representation."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.py_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = processor.process_vision_info(messages)
    return processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
    ).to(device)


def response_text(response):
    """Extract text from the optional verbose response tuple."""
    return response[0] if isinstance(response, tuple) else response


def create_adapter(config):
    """Expose LocateAnything through the fixed evaluator contract."""
    required = {"model_path", "generation_mode", "coordinate_max"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"LocateAnything config is missing: {sorted(missing)}")
    device = assigned_device()
    tokenizer, processor, model = load_model(config, device)
    generation_mode = config["generation_mode"]
    coordinate_max = int(config["coordinate_max"])

    def predict_batch(records, prompts, max_new_tokens):
        outputs = []
        for record, prompt in zip(records, prompts):
            with Image.open(record["image_file"]) as source:
                image = source.convert("RGB")
            inputs = prepare_inputs(processor, image, prompt, device)
            with torch.no_grad():
                response = model.generate(
                    pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    generation_mode=generation_mode,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.0,
                    verbose=False,
                )
            raw_response = response_text(response)
            outputs.append({
                "predictions": decode(
                    raw_response,
                    image.width,
                    image.height,
                    coordinate_max,
                    2 if record.get("benchmark_protocol") == "point" else 4,
                ),
                "raw_response": raw_response,
                "hit_token_cap": not raw_response.endswith("<|im_end|>"),
            })
        return outputs

    return {
        "name": f"locateanything-native-{generation_mode}",
        "predict_batch": predict_batch,
    }
