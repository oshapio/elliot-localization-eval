"""GPU comparison of HF and native vLLM using the same image and prompts.

Run HF and vLLM modes in separate processes to release model and CUDA memory.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--require-exact", action="store_true",
                        help="Also fail on HF or batch-dependent token differences")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluator"))
    from adapters.elliot_vllm import render_prompt
    questions = ["Locate every person in the image.", "Locate every object in the image."]
    report = {"model": args.model, "backend": args.backend, "image": args.image}
    start = time.monotonic()
    if args.backend == "hf":
        import torch
        from PIL import Image
        from transformers.utils import import_utils
        import_utils.is_causal_conv1d_available = lambda: False
        from transformers import AutoModelForCausalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16,
        ).to("cuda").eval()
        if model.config.text_config.tie_word_embeddings:
            model.language_model.get_output_embeddings().weight = model.language_model.get_input_embeddings().weight
        report["hf_embeddings_tied"] = (model.language_model.get_output_embeddings().weight.data_ptr()
                                        == model.language_model.get_input_embeddings().weight.data_ptr())
        image = Image.open(args.image).convert("RGB")
        if args.max_pixels:
            from elliot_vllm.vision_attention import enable_sdpa
            from elliot_vllm.vision_attention import sdpa_forward
            layer = model.vision_model.encoder.layers[0].self_attn
            dtype = next(layer.parameters()).dtype
            layer.float()
            probe = torch.randn(1, 35, layer.embed_dim, device="cuda", dtype=torch.float32)
            grid = torch.tensor([[5, 7]], device="cuda")
            with torch.inference_mode():
                eager = layer(probe, 5, 7, spatial_shapes=grid)[0]
                efficient = sdpa_forward(layer, probe, 5, 7, spatial_shapes=grid)[0]
            torch.testing.assert_close(eager, efficient, atol=1e-5, rtol=1e-5)
            report["vision_sdpa_max_error"] = (eager - efficient).abs().max().item()
            layer.to(dtype)
            with torch.inference_mode():
                eager = layer(probe.to(dtype), 5, 7, spatial_shapes=grid)[0]
                efficient = sdpa_forward(layer, probe.to(dtype), 5, 7, spatial_shapes=grid)[0]
            torch.testing.assert_close(eager, efficient, atol=0.01, rtol=0.01)
            report["vision_sdpa_bf16_max_error"] = (eager - efficient).abs().max().item()
            enable_sdpa(model.vision_model)
            from qwen_vl_utils.vision_process import smart_resize
            height, width = smart_resize(image.height, image.width, factor=32,
                                         min_pixels=65536, max_pixels=args.max_pixels)
            image = image.resize((width, height))
            processor.image_processor.do_resize = False
            report["resized_image"] = [width, height]
            report["image_tokens"] = (width // 32) * (height // 32)
        outputs = []
        for question in questions:
            inputs = processor(text=render_prompt(question), images=image, return_tensors="pt")
            inputs.pop("image_grid_thw", None)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                result = model.generate(**inputs, do_sample=False, repetition_penalty=1.05,
                                        max_new_tokens=args.max_tokens,
                                        eos_token_id=processor.tokenizer.eos_token_id,
                                        pad_token_id=processor.tokenizer.eos_token_id)[0].tolist()
            prompt_ids = inputs["input_ids"][0].tolist()
            if result[:len(prompt_ids)] == prompt_ids:
                result = result[len(prompt_ids):]
            outputs.append({"token_ids": result, "prompt_token_ids": prompt_ids,
                            "text": processor.tokenizer.decode(result, skip_special_tokens=False)})
        report["outputs"] = outputs
    else:
        from adapters.elliot_vllm import create_adapter
        adapter = create_adapter({"model_path": args.model, "max_model_len": 16384, "max_pixels": args.max_pixels,
                                  "max_num_seqs": 4})
        report["engine_startup_seconds"] = time.monotonic() - start
        records = [{"image_file": args.image}] * len(questions)
        generation_start = time.monotonic()
        singles = [adapter["predict_batch"]([record], [question], args.max_tokens)[0]
                   for record, question in zip(records, questions)]
        report["single_requests_seconds"] = time.monotonic() - generation_start
        generation_start = time.monotonic()
        batched = adapter["predict_batch"](records, questions, args.max_tokens)
        report["batched_requests_seconds"] = time.monotonic() - generation_start
        report.update(outputs=batched, singles=singles,
                      batch_matches_single=[a["token_ids"] == b["token_ids"]
                                            for a, b in zip(singles, batched)])
        baseline = json.loads(args.output.with_name("hf.json").read_text())
        report["hf_exact_token_match"] = [a["token_ids"] == b["token_ids"]
                                           for a, b in zip(baseline["outputs"], batched)]
        report["hf_prompt_token_match"] = [a["prompt_token_ids"] == b["prompt_token_ids"]
                                            for a, b in zip(baseline["outputs"], batched)]
    report["elapsed_seconds"] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("outputs", "singles")}, indent=2), flush=True)
    if args.backend == "vllm":
        if not all(report["hf_prompt_token_match"]):
            raise RuntimeError("HF/vLLM input tokenization differs")
        if not all(output["predictions"] for output in batched):
            raise RuntimeError("Smoke produced no decoded boxes")
        if args.require_exact and not all(report["batch_matches_single"] + report["hf_exact_token_match"]):
            raise RuntimeError("Smoke output differs; inspect the comparison report")


if __name__ == "__main__":
    main()
