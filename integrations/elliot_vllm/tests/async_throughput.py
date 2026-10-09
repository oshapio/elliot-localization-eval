"""Compare batch barriers with continuously replenished native AsyncLLM requests.

Both modes use one persistent engine and identical sampling settings. This isolates
request submission; it does not measure HTTP, multi-node routing or crash recovery.
"""

import argparse
import asyncio
from functools import partial
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evaluator"))
from adapters.elliot_vllm import render_prompt
from run import prompt_for


async def benchmark(args):
    from PIL import Image
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    from elliot_vllm import configure_hf_config

    config = json.loads(args.config.read_text())
    family = config.get("family", "elliot-loc")
    if family not in ("elliot-loc", "qwen"):
        raise ValueError(f"Unsupported prompt family: {family}")
    processor = None
    model_kwargs = {}
    sampling_kwargs = {}
    if family == "qwen":
        from transformers import AutoProcessor
        sys.path.insert(0, str(ROOT / "evals_localization"))
        from localization_parsing import _parse
        processor = AutoProcessor.from_pretrained(config["model_path"])
        model_kwargs["mm_processor_kwargs"] = config.get("mm_processor_kwargs", {})
        sampling_kwargs["repetition_penalty"] = config["repetition_penalty"]
    else:
        model_kwargs.update(
            hf_overrides=partial(configure_hf_config, max_pixels=config["max_pixels"],
                                 min_pixels=config.get("min_pixels", 65536)),
            logits_processors=["elliot_vllm.sampling:ElliotRepetitionPenalty"],
        )
        sampling_kwargs["extra_args"] = {"elliot_repetition_penalty": config["repetition_penalty"]}
    records = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=config["model_path"], trust_remote_code=True,
        dtype="bfloat16", limit_mm_per_prompt={"image": 1},
        max_model_len=config["max_model_len"], tensor_parallel_size=config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_num_seqs=config["max_num_seqs"], enforce_eager=False,
        enable_prefix_caching=False, async_scheduling=False,
        **model_kwargs,
    ))
    startup = time.perf_counter() - started
    params = SamplingParams(temperature=0, max_tokens=4096,
        skip_special_tokens=False, output_kind=RequestOutputKind.FINAL_ONLY,
        **sampling_kwargs)
    summary = {"startup_seconds": startup, "config": config,
               "max_new_tokens": 4096, "runs": []}

    async def request(index, tag):
        record = records[index]
        start = time.perf_counter()
        with Image.open(record["image_file"]) as source:
            image = source.convert("RGB")
        try:
            question = prompt_for(record, family)
            if processor is None:
                prompt = render_prompt(question)
            else:
                prompt = processor.apply_chat_template([{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": question},
                ]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            result = None
            async for result in engine.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image}},
                params, request_id=f"{tag}-{index}",
            ):
                pass
            assert result is not None and result.finished
            output = result.outputs[0]
            parsed = {}
            parse_error = None
            if family == "qwen":
                try:
                    parsed = _parse(output.text.split("</think>")[-1], record, family,
                                    image.width, image.height)
                except Exception as error:
                    parse_error = str(error)
            return {"index": index, "id": record["sample_id"],
                    "prompt": prompt, "predictions": parsed, "parse_error": parse_error,
                    "image_size": list(image.size),
                    "seconds": time.perf_counter() - start,
                    "prompt_token_ids": list(result.prompt_token_ids),
                    "token_ids": list(output.token_ids), "raw_response": output.text,
                    "finish_reason": output.finish_reason}
        finally:
            image.close()

    try:
        # Warm the engine with real images and ordinary EOS, not a tiny token cap.
        await asyncio.gather(*(request(i, "warmup") for i in range(min(4, len(records)))))
        reference = None
        # ABBA order reduces systematic warm-cache/order bias.
        for turn, mode in enumerate(["barrier32", "rolling64", "rolling64", "barrier32"]):
            tag = f"{turn}-{mode}"
            results = []
            started = time.perf_counter()
            with (args.output / f"{tag}.jsonl").open("w") as handle:
                async def save(index):
                    result = await request(index, tag)
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
                    results.append(result)

                if mode == "barrier32":
                    for start in range(0, len(records), 32):
                        await asyncio.gather(*(save(i) for i in
                            range(start, min(start + 32, len(records)))))
                else:
                    semaphore = asyncio.Semaphore(64)

                    async def limited(index):
                        async with semaphore:
                            await save(index)

                    await asyncio.gather(*(limited(i) for i in range(len(records))))
            elapsed = time.perf_counter() - started
            by_index = {result["index"]: result for result in results}
            assert len(by_index) == len(records)
            if reference is None:
                reference = by_index
            assert all(r["prompt_token_ids"] == reference[i]["prompt_token_ids"]
                       for i, r in by_index.items()), "Prompt tokens changed"
            tokens = sum(len(r["token_ids"]) for r in results)
            measurement = {"mode": mode, "seconds": elapsed, "requests": len(results),
                "requests_per_second": len(results) / elapsed,
                "output_tokens": tokens, "output_tokens_per_second": tokens / elapsed,
                "exact_outputs_vs_first": sum(r["token_ids"] == reference[i]["token_ids"]
                                               for i, r in by_index.items()),
                "capped": sum(r["finish_reason"] == "length" for r in results)}
            if family == "qwen":
                measurement["parse_errors"] = sum(r["parse_error"] is not None for r in results)
                measurement["nonempty_predictions"] = sum(bool(r["predictions"]) for r in results)
            summary["runs"].append(measurement)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(measurement), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(benchmark(parser.parse_args()))
