"""Register lazily so vLLM can discover the model without initializing CUDA."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "ElliotVLForCausalLM", "elliot_vllm.model:ElliotVLForCausalLM"
    )


def configure_hf_config(config, max_pixels=None, min_pixels=65536):
    """Elliot's HF generate uses sequential positions, including image tokens."""
    # vLLM first probes overrides with a bare config to discover model_type.
    if not hasattr(config, "text_config"):
        return config
    if config.model_type != "elliot_vlm" or config.text_config.model_type != "qwen3_5_text":
        raise ValueError("This plugin supports Elliot with a Qwen3.5 text decoder only")
    if max_pixels is not None:
        config.eval_max_pixels = int(max_pixels)
        config.eval_min_pixels = int(min_pixels)
    rope = dict(config.text_config.rope_parameters)
    rope.pop("mrope_section", None)
    rope.pop("mrope_interleaved", None)
    config.text_config.rope_parameters = rope
    return config
