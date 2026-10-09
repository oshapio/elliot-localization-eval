# Native asynchronous submission smoke

`async_throughput.py` uses vLLM's native `AsyncLLM`, the engine behind its API
server. One CPU client sends image requests directly to one persistent GPU
engine. There is no HTTP service or custom batching scheduler.

It compares a barrier after each 32 requests with a continuously replenished
pool of 64 requests. Both use the same engine, prompts, image processor, 4096
output-token limit, greedy decoding and output-only repetition penalty. The
engine's maximum active sequence count remains the checkpoint config's value.
The test uses barrier/rolling/rolling/barrier order after a four-request warmup.
Image opening and request processing are included in measured request time.

This isolates submission policy: the barrier mode emulates the current offline
runner's batch boundaries using the asynchronous frontend too. It is not a
direct timing comparison of `LLM.generate` against HTTP serving. Engine startup
is reported separately; module import time is excluded. Results include output
token throughput, token-cap counts and exact output agreement. Expanded prompt
tokens must match across modes. Different batch composition may change outputs.

Run with the isolated vLLM environment and the Elliot plugin enabled:

```bash
python integrations/elliot_vllm/tests/async_throughput.py \
  --manifest /path/to/fixed-smoke.jsonl \
  --config /path/to/elliot-config.json \
  --output /path/to/new-results-directory
```

The first Leonardo run is archived under
`outputs/async_throughput_28k_20260905`: 132 queries (six seeded samples from
each of the 22 existing protocol/dataset groups), Elliot run 022 step 28000,
1MP maximum image area, 8192 context, maximum 16 active sequences, TP=1.
Its frozen source, manifest hash, configuration and Slurm script are preserved.

This smoke does not implement or validate multi-GPU load balancing, durable
request recovery, or a production evaluation client. Its balanced small sample
also differs from the full suite's heavily LVIS-weighted request distribution.

## First measured result

Job `56175542` completed successfully on 2026-09-05. Engine construction took
247.05 seconds, excluding Python imports and the explicit request warmup.

| Round | Submission | Seconds | Requests/s | Output tokens/s | Capped outputs |
|---|---|---:|---:|---:|---:|
| 1 | Barrier 32 | 54.64 | 2.42 | 289.80 | 1 |
| 2 | Rolling 64 | 27.33 | 4.83 | 508.78 | 0 |
| 3 | Rolling 64 | 27.32 | 4.83 | 508.97 | 0 |
| 4 | Barrier 32 | 33.12 | 3.99 | 390.11 | 0 |

The conservative warm comparison is round 4 against rounds 2–3: approximately
21% higher request throughput and 30% higher output-token throughput. The first
barrier round generated more tokens, including one 4096-token capped output;
its roughly 2x wall-time difference should not be attributed solely to scheduling.
Both rolling rounds matched exactly on all 132 outputs. Rolling versus the final
barrier round matched on 72/132 outputs, although all expanded prompt tokens
matched. This validates execution and measures throughput, not score parity or
batch-invariant generation. It does not establish full-suite or multi-GPU speedup.
