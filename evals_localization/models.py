"""Model adapters. make_rex()/make_qwen()/make_elliot() load a model and return

    {"name":          str,
     "question":      fn(entry) -> prompt str,                     # from tasks.py
     "predict_batch": fn(images, questions, entries, max_new_tokens)
                          -> [(preds, raw_text, hit_cap), ...],    # what eval.py calls
     "predict":       fn(image, question, entry) -> (preds, raw)}  # batch of 1, for notebooks

`preds` is {label: [[x1,y1,x2,y2] or [x,y], ...]} in ABSOLUTE pixels of the original
image -- the schema the vendored scorers read. `hit_cap` says the answer was cut off by
the token budget, which eval.py uses to decide what to retry.

Batch size changes bf16 reduction order, so greedy decoding drifts token-by-token
between batch sizes; measured effect on the SCORE is ~0.1-2 points (see the memory note),
so results are reproducible at the metric level, not byte level."""
import json
import re
from pathlib import Path

import torch

from paths import ELLIOT_PATH, QWEN_PATH, REX_ID   # cluster-specific paths live in paths.py
import tasks as T
import inference_text_prompt as rex_infer   # vendored: DEFAULT_PROMPT + parse_prediction

# Qwen3.5 (and Elliot, which embeds it) is a HYBRID model: most layers are gated-delta
# linear attention, which transformers runs via two optional accelerators. Neither works
# here, and both crash rather than degrade, so we disable them at import time.
#
# causal-conv1d: its prebuilt wheel needs GLIBC >= 2.32; this cluster is EL8 (2.28), so
# loading it aborts. Telling transformers it is unavailable selects the pure-torch conv.
from transformers.utils import import_utils
import_utils.is_causal_conv1d_available = lambda: False

# generation protocol, same for all three models: rex ships these (their eval defaults),
# so qwen/elliot use them too -- 512 truncated legitimate dense-detection answers, and
# without the penalty both models fall into box-repetition loops that burn the whole budget
MAX_NEW_TOKENS = 4096
REPETITION_PENALTY = 1.05


def _patch_fla():
    """Disable fla's Triton linear-attention kernels on pre-Ampere GPUs.

    The other accelerator (see the causal-conv1d note above) is `fla`
    (flash-linear-attention), which provides Triton kernels for the gated-delta layers:
    chunk_* for prefill, fused_recurrent_* for decode. They are compiled with bf16 PTX,
    which needs compute capability >= 8.0 -- on a V100 (sm_70) ptxas fails with
    "Feature '.bf16' requires .target sm_80 or higher" and generation dies.

    transformers calls these through module-level names and falls back to its own torch
    implementation when they are None, so setting them to None is the supported way to
    opt out. A100s and newer (capability >= 8) keep the fast kernels untouched.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
        m.chunk_gated_delta_rule = None
        m.fused_recurrent_gated_delta_rule = None


from localization_parsing import (
    _task_for, _question, _parse, extract_numbers, normalize_box,
    extract_labeled_boxes, extract_spatial_boxes, extract_points,
)


def _new_tokens(out, input_ids):
    """The generated part of a batch, dropping the echoed prompt.

    generate() has two return conventions: usually prompt+completion, but Elliot's
    multimodal path feeds inputs_embeds and gets back completion only. Slicing blindly
    would eat the answer in that case, so slice only when the prompt is really there.
    """
    n_in = input_ids.shape[1]
    if out.shape[1] > n_in and torch.equal(out[:, :n_in], input_ids):
        return out[:, n_in:]
    return out


def _finish(rows, tokenizer, entries, family, images, max_new_tokens):
    """Decode + parse a batch of generated rows -> [(preds, raw, hit_cap)] per sample.

    hit_cap means the model was still talking when the token budget ran out (no EOS in the
    row AND it used the whole budget), so the answer is cut off; eval.run_inference retries
    those with the full budget. Rows that stopped early are padded, but skip_special_tokens
    drops the padding, and a finished row contains EOS so it is never flagged.
    """
    eos = tokenizer.eos_token_id
    out = []
    for row, e, img in zip(rows, entries, images):
        if family == "elliot-loc":
            ignored = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
            ids = [int(token_id) for token_id in row if int(token_id) not in ignored]
            raw = tokenizer.decode(
                ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False).strip()
        else:
            raw = tokenizer.decode(row, skip_special_tokens=True).split("</think>")[-1].strip()
        hit = bool((row == eos).sum() == 0) and int(row.shape[0]) >= max_new_tokens
        w, h = img.size
        out.append((_parse(raw, e, family, w, h), raw, hit))
    return out


def make_rex(model_id=REX_ID, device="cuda"):
    """Rex-Omni-3B. Their own eval settings: their pixel budget, their system prompt,
    their greedy sampling, their parser. Predictions come back already labeled."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    min_px, max_px = 16 * 28 * 28, 2560 * 28 * 28          # their eval pixel budget
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_px, max_pixels=max_px)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()

    # Token-id remap shim. Rex-Omni added 1000 coordinate tokens <0>..<999> but reused the
    # base model's ids 150643-151642 for them WITHOUT resizing the embedding matrix (their
    # GitHub issue #22). transformers 5.x refuses to alias ids, so it appends all added
    # tokens at 151643+, past the 151936-row embedding -> CUDA device-side assert on lookup.
    # Their config.json means the original ids, so we translate at the boundary: shift the
    # tokenizer's ids down before the model sees them, and back up before decoding.
    def tok2model(ids):
        return torch.where(ids >= 151643, ids - 1000, ids)

    def model2tok(ids):
        return torch.where((ids >= 150643) & (ids < 151665), ids + 1000, ids)

    processor.tokenizer.padding_side = "left"      # required for batched generate

    def _messages(image, question):
        return [{"role": "system", "content": rex_infer.DEFAULT_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": image, "min_pixels": min_px, "max_pixels": max_px},
                    {"type": "text", "text": question}]}]

    @torch.inference_mode()
    def predict_batch(images, questions, entries, max_new_tokens=MAX_NEW_TOKENS):
        msgs = [_messages(im, q) for im, q in zip(images, questions)]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        vision = [process_vision_info(m)[0][0] for m in msgs]
        inputs = processor(text=texts, images=vision, padding=True, return_tensors="pt").to(device)
        inputs["input_ids"] = tok2model(inputs["input_ids"])
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, repetition_penalty=1.05)   # their greedy SamplingParams
        rows = _new_tokens(out, inputs["input_ids"])
        eos = processor.tokenizer.eos_token_id
        res = []
        for i, (e, im) in enumerate(zip(entries, images)):
            row = rows[i]
            end = (row == eos).nonzero()                  # rex keeps special tokens -> trim the
            row = row[: int(end[0, 0]) + 1] if len(end) else row   # right-padding by hand
            raw = processor.tokenizer.decode(model2tok(row), skip_special_tokens=False)
            in_h = int(inputs["image_grid_thw"][i][1]) * 14
            in_w = int(inputs["image_grid_thw"][i][2]) * 14
            w, h = im.size
            preds = rex_infer.parse_prediction(raw, w, h, in_h, in_w, "xyxy", e.get("task_name"))
            hit = bool((rows[i] == eos).sum() == 0) and int(rows[i].shape[0]) >= max_new_tokens
            res.append((preds, raw, hit))
        return res

    def predict(image, question, entry, max_new_tokens=MAX_NEW_TOKENS):
        preds, raw, _ = predict_batch([image], [question], [entry], max_new_tokens)[0]
        return preds, raw

    return {"name": "rex-omni", "question": _question("rex-omni"),
            "predict": predict, "predict_batch": predict_batch}


def make_qwen(path=QWEN_PATH, name="qwen3.5-2b", device="cuda"):
    """Qwen3.5-2B baseline. Thinking is disabled (the chat template pre-closes the
    <think> block) and detection is asked for in its trained JSON format."""
    _patch_fla()
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16).to(device).eval()

    processor.tokenizer.padding_side = "left"      # required for batched generate

    @torch.inference_mode()
    def predict_batch(images, questions, entries, max_new_tokens=MAX_NEW_TOKENS):
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": im},
                                          {"type": "text", "text": q}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for im, q in zip(images, questions)]
        inputs = processor(text=texts, images=list(images), padding=True, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=REPETITION_PENALTY)
        rows = _new_tokens(out, inputs["input_ids"])
        return _finish(rows, processor.tokenizer, entries, "qwen", images, max_new_tokens)

    def predict(image, question, entry, max_new_tokens=MAX_NEW_TOKENS):
        preds, raw, _ = predict_batch([image], [question], [entry], max_new_tokens)[0]
        return preds, raw

    return {"name": name, "question": _question("qwen"),
            "predict": predict, "predict_batch": predict_batch}


def make_elliot(path=ELLIOT_PATH, name="elliot-2b", device="cuda", mcore_path=None):
    """ElliotVL (Qwen3.5-2B LLM + SigLIP2-NaViT). Loaded like its own simple_infer.py:
    trust_remote_code, hand-built chat string, empty <think> prefill."""
    _patch_fla()                                            # elliot embeds stock qwen3_5 layers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=AutoConfig.from_pretrained(path, trust_remote_code=True),
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    family = "elliot"
    if mcore_path:
        import infer_mcore_overlay
        import infer_one
        payload = Path(mcore_path)
        if payload.is_dir():
            payload = payload / "mp_rank_00" / "model_optim_rng.pt"
        if not payload.is_file():
            raise FileNotFoundError(f"Megatron payload not found: {payload}")
        manifest_path = Path(path) / "coordinate_token_init.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            padded_size = int(manifest["padded_vocab_size"])
            current_size = int(model.get_input_embeddings().weight.shape[0])
            if current_size != padded_size:
                print(f"resizing HF vocabulary: {current_size} -> {padded_size}")
                model.resize_token_embeddings(padded_size, mean_resizing=False)
        infer_one.tie_language_model_embeddings(model)
        overlay = infer_mcore_overlay.overlay_mcore_checkpoint(model, payload)
        if overlay["missing_hf"] or overlay["missing_mcore"] or overlay["shape_mismatch"]:
            raise RuntimeError(f"incomplete Megatron overlay: {overlay}")
        print(f"Megatron overlay loaded {overlay['loaded_count']} tensors from {payload}")
        family = "elliot-loc"
    model = model.to(device).eval()
    # only keys ElliotVLForCausalLM.generate accepts (extra processor keys crash it)
    keys = {"input_ids", "attention_mask", "position_ids", "pixel_values",
            "image_mask", "spatial_shapes", "pixel_attention_mask"}
    tok = processor.tokenizer

    tok.padding_side = "left"                      # required for batched generate

    def _prompt(question):
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")     # their simple_infer.py format

    def _collate(prompts, images):
        """Batch of (prompt, image) pairs -> model inputs. Replaces ElliotVLProcessor.__call__.

        Why not just call the processor: given a LIST of texts it expands EVERY text with
        ALL the images' token counts (processing_elliot.py ~line 90), i.e. it assumes one
        conversation holding N images. Handed 16 texts + 16 images it demands 16 <image>
        placeholders per text and raises. So we repeat its steps, giving each text only its
        own image's token count.

        The steps, in its order: run the image processor over the batch; get each image's
        patch grid (spatial_shapes, or derived from the fixed patch size); mark real patches
        in pixel_attention_mask; replace each "<image>" with that image's number of merged
        tokens (2x2 pooling -> (h//2)*(w//2)); tokenize left-padded; flag the image-token
        positions so the model knows where to scatter the vision features.
        """
        img = processor.image_processor(list(images), return_tensors="pt")
        pixel_values = img["pixel_values"]
        patch = getattr(processor.image_processor, "patch_size", 16)
        shapes = img["spatial_shapes"].to(torch.long) if "spatial_shapes" in img else torch.tensor(
            [[pixel_values.shape[2] // patch, pixel_values.shape[3] // patch]] * pixel_values.shape[0])
        counts = ((shapes[:, 0] // 2) * (shapes[:, 1] // 2)).tolist()
        pam = torch.zeros((len(images), int((shapes[:, 0] * shapes[:, 1]).max())), dtype=torch.int32)
        for i, (hh, ww) in enumerate(shapes.tolist()):
            pam[i, : hh * ww] = 1
        texts = [processor._expand_text_with_images(p, [c]) for p, c in zip(prompts, counts)]
        enc = tok(texts, return_tensors="pt", padding=True)
        enc["pixel_values"], enc["spatial_shapes"], enc["pixel_attention_mask"] = pixel_values, shapes, pam
        enc["image_mask"] = enc["input_ids"] == tok.convert_tokens_to_ids(processor.v_pad)
        return {k: v.to(device) for k, v in enc.items() if k in keys and hasattr(v, "to")}

    @torch.inference_mode()
    def predict_batch(images, questions, entries, max_new_tokens=MAX_NEW_TOKENS):
        inputs = _collate([_prompt(q) for q in questions], images)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=REPETITION_PENALTY,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
        rows = _new_tokens(out, inputs["input_ids"])
        return _finish(rows, tok, entries, family, images, max_new_tokens)

    def predict(image, question, entry, max_new_tokens=MAX_NEW_TOKENS):
        preds, raw, _ = predict_batch([image], [question], [entry], max_new_tokens)[0]
        return preds, raw

    return {"name": name, "question": _question(family),
            "predict": predict, "predict_batch": predict_batch,
            "model": model, "tokenizer": tok,
            "prepare_inputs": lambda images, questions: _collate(
                [_prompt(question) for question in questions], images)}
