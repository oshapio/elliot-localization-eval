"""Elliot's HF vision modules feeding the native vLLM Qwen3.5 decoder.

Uses the vLLM multimodal interfaces demonstrated by its Llava implementation.
No HF language model is instantiated, and checkpoint parameter names are retained.
"""

from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal, IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from .processing import (
    ElliotProcessingInfo, ElliotMultiModalProcessor, ElliotDummyInputsBuilder,
)


@MULTIMODAL_REGISTRY.register_processor(
    ElliotMultiModalProcessor,
    info=ElliotProcessingInfo,
    dummy_inputs=ElliotDummyInputsBuilder,
)
class ElliotVLForCausalLM(nn.Module, SupportsMultiModal, IsHybrid):
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image inputs are supported")

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = config = vllm_config.model_config.hf_config
        if getattr(config, "encoder_data_balance", False):
            raise ValueError("Encoder data balancing must be disabled for inference")
        hf_class = get_class_from_dynamic_module(
            config.auto_map["AutoModelForCausalLM"], vllm_config.model_config.model,
        )
        self.configure_mm_token_handling(
            vocab_size=config.text_config.vocab_size,
            mm_token_ids=[config.image_pad_token_id],
        )
        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = hf_class._build_vision_model(hf_class, config)
            self.patch_merger = hf_class._build_projector(hf_class, config)
            if getattr(config, "eval_max_pixels", None):
                from .vision_attention import enable_sdpa
                enable_sdpa(self.vision_model)
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_multimodal(self, **kwargs):
        pixels = kwargs.get("pixel_values")
        if pixels is None:
            return []
        shapes = kwargs["spatial_shapes"]
        masks = kwargs["pixel_attention_mask"]
        # Encode images independently so visual features do not depend on the
        # scheduler's batch composition. Decoder requests are still batched.
        features = []
        for pixel, shape, mask in zip(pixels, shapes, masks):
            if pixel.ndim != 3:
                raise ValueError("Expected fixed-size CHW image pixels")
            vision = self.vision_model(
                pixel_values=pixel.unsqueeze(0).to(next(self.vision_model.parameters()).dtype),
                interpolate_pos_encoding=True,
                **({"spatial_shapes": shape.reshape(1, 2)}
                   if getattr(self.config, "eval_max_pixels", None) else {}),
            )
            features.append(self.patch_merger(
                vision.last_hidden_state, shape.reshape(1, 2), mask.reshape(1, -1),
                valid_only=True,
            ))
        return features

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)

    def get_mm_mapping(self):
        return MultiModelKeys.from_string_field(
            language_model="language_model", connector="patch_merger", tower_model="vision_model",
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
