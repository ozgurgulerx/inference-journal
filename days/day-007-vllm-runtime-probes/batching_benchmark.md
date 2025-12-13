# Day 007 – Micro-Batching vs Sequential Throughput

## Experiment Setup
- GPU:
- Model (SLM):
- vLLM version:
- Key server flags (including `max-model-len`, `max-num-seqs`, `gpu-memory-utilization`):
- Client request shape:
  - prompt length (tokens, approx):
  - `max_tokens`:
  - total requests (N):

## Sequential vs Concurrent Summary

```text
mode,sequential_s,concurrent_s,sequential_qps,concurrent_qps,sequential_tok_s,concurrent_tok_s,notes
baseline,,,,,,,
```

## Concurrency Sweep (Knee of the Curve)

```text
concurrency,mean_e2e_s,p95_e2e_s,qps,tok_s,gpu_util_pct,notes
1,,,,,,
2,,,,,,
4,,,,,,
8,,,,,,
16,,,,,,
32,,,,,,
```

## Observations
- Where does throughput stop scaling with concurrency?
- Where does p95 latency start to blow up (queuing)?
- Any changes in GPU utilization pattern as concurrency increases?

## Rules of Thumb (For This GPU + SLM)
- Sweet-spot concurrency range for this request shape:
- Concurrency level you’d avoid for latency-sensitive workloads:
- Any “gotchas” (e.g. small batches under-utilizing GPU, huge batches causing timeouts):

