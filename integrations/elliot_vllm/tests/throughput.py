"""Compare request queue sizes and CUDA graphs on the same fixed eval subset."""
import argparse
import collections
import json
import time
from pathlib import Path
import sys


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eager8', 'eager64', 'graph64'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'profile start: {args.mode}', flush=True)
    import vllm
    import torch
    imports_done = time.monotonic()
    print(f'imports: {imports_done-start:.2f}s', flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'evaluator'))
    from adapters.elliot_vllm import create_adapter
    from run import prompt_for
    groups = collections.defaultdict(list)
    for line in args.manifest.open():
        row = json.loads(line)
        groups[row['benchmark_protocol'], row['dataset_name']].append(row)
    records = [group[i] for i in range(4) for _, group in sorted(groups.items())
               if i < len(group)]
    config = json.loads(args.config.read_text())
    batch_size = 8 if args.mode == 'eager8' else 64
    config.update(max_num_seqs=8 if args.mode == 'eager8' else 32,
                  enforce_eager=args.mode != 'graph64')
    adapter = create_adapter(config)
    engine_done = time.monotonic()
    # Keep warm-up separate; any later shape-specific compilation is timed.
    adapter['predict_batch'](records[:batch_size],
                             [prompt_for(r, 'elliot-loc') for r in records[:batch_size]], 32)
    warmup_done = time.monotonic()
    outputs = []
    batches = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        prompts = [prompt_for(r, 'elliot-loc') for r in batch]
        tick = time.monotonic()
        result = adapter['predict_batch'](batch, prompts, 512)
        batches.append(dict(samples=len(batch), seconds=time.monotonic()-tick,
                            generated_tokens=sum(r['generated_tokens'] for r in result),
                            capped=sum(r['hit_token_cap'] for r in result)))
        outputs.extend(dict(sample_id=r['sample_id'], **o) for r, o in zip(batch, result))
        print(json.dumps(batches[-1]), flush=True)
    elapsed = time.monotonic()-warmup_done
    tokens = sum(b['generated_tokens'] for b in batches)
    report = dict(mode=args.mode, samples=len(records), batch_size=batch_size,
                  max_new_tokens=512, retry_max_new_tokens=0,
                  config=config, torch=torch.__version__, vllm=vllm.__version__,
                  import_seconds=imports_done-start, engine_seconds=engine_done-imports_done,
                  warmup_seconds=warmup_done-engine_done, generation_seconds=elapsed,
                  generated_tokens=tokens, tokens_per_second=tokens/elapsed,
                  batches=batches, outputs=outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['outputs','batches','config']},indent=2),flush=True)


if __name__ == '__main__':
    main()
