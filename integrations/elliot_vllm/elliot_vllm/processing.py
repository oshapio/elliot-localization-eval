"""Use the checkpoint's image processor and image-placeholder expansion."""

import math

import torch
from transformers import BatchFeature
from vllm.multimodal.parse import ImageProcessorItems

from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    BaseMultiModalProcessor,
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdateDetails,
)


class ElliotProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self):
        return {"image": 1}

    def resize_shape(self, height, width):
        from qwen_vl_utils.vision_process import smart_resize
        config = self.get_hf_config()
        return smart_resize(height, width, factor=32,
                            min_pixels=config.eval_min_pixels, max_pixels=config.eval_max_pixels)

    def image_shape(self):
        config = self.get_hf_config()
        if getattr(config, "eval_max_pixels", None):
            side = math.isqrt(config.eval_max_pixels) // 32 * 32
            return side, side, (side // 32) ** 2
        processor = self.get_hf_processor().image_processor
        # Match the saved HF baseline, including its fixed resize. Do not infer
        # resolution from the checkpoint's name or unused max_num_patches field.
        if processor.__class__.__name__ != "SiglipImageProcessor":
            raise ValueError("Only the baseline SiglipImageProcessor is validated")
        size = processor.size
        patch = getattr(processor, "patch_size", 16)
        height, width = size["height"], size["width"]
        if height % (2 * patch) or width % (2 * patch):
            raise ValueError("Image dimensions must be divisible by the 2x2 patch merger")
        return height, width, (height // patch // 2) * (width // patch // 2)


class ElliotDummyInputsBuilder(BaseDummyInputsBuilder[ElliotProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        return "<image>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        height, width, _ = self.info.image_shape()
        return {"image": self._get_dummy_images(
            width=width, height=height, num_images=mm_counts.get("image", 0),
            overrides=mm_options.get("image"),
        )}


class ElliotMultiModalProcessor(BaseMultiModalProcessor[ElliotProcessingInfo]):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if getattr(self.info.get_hf_config(), "eval_max_pixels", None):
            processor = self.info.get_hf_processor()
            pixels, shapes, masks, counts = [], [], [], []
            for image in mm_data.get("images", []):
                height, width = self.info.resize_shape(image.height, image.width)
                resized = image.resize((width, height))
                pixel = processor.image_processor([resized], do_resize=False,
                                                  return_tensors="pt")["pixel_values"][0]
                pixels.append(pixel)
                shapes.append([height // 16, width // 16])
                masks.append(torch.ones((height // 16) * (width // 16), dtype=torch.int32))
                counts.append((height // 32) * (width // 32))
            expanded = processor._expand_text_with_images(prompt, counts)
            result = BatchFeature(dict(processor.tokenizer(expanded, return_tensors="pt", **tok_kwargs)))
            if pixels:
                result.update(pixel_values=pixels, spatial_shapes=torch.tensor(shapes),
                              pixel_attention_mask=masks)
            return result
        result = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        # vLLM owns sequence masks and positions after placeholder expansion.
        result.pop("image_mask", None)
        result.pop("image_grid_thw", None)
        return result

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {name: MultiModalFieldConfig.batched("image") for name in (
            "pixel_values", "spatial_shapes", "pixel_attention_mask",
        )}

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        config = self.info.get_hf_config()
        def replacement(index):
            if getattr(config, "eval_max_pixels", None):
                size = mm_items.get_items("image", ImageProcessorItems).get_image_size(index)
                height, width = self.info.resize_shape(size.height, size.width)
                count = (height // 32) * (width // 32)
            else:
                _, _, count = self.info.image_shape()
            tokens = ([config.vision_start_token_id]
                      + [config.image_pad_token_id] * count
                      + [config.vision_end_token_id])
            return PromptUpdateDetails.select_token_id(tokens, config.image_pad_token_id)
        return [PromptReplacement(
            modality="image", target="<image>",
            replacement=replacement,
        )]
