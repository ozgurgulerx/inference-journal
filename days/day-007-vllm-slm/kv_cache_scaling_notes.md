# Day 007 – KV Cache Scaling Notes

## Experiment Setup
- GPU:
- Model (SLM):
- vLLM version:
- Fixed flags (other than `max-model-len`):

## KV Cache Scaling CSV
File: `kv_cache_scaling.csv`

Expected columns:

```text
max_model_len,gpu_mem_used_mb,delta_mb_from_prev,bytes_per_token_est,notes
```

## Bytes per KV Token (Back-of-Envelope)
- Rough slope from CSV (MB per 1K tokens):
- Approx bytes per KV token:
  - `bytes_per_token_est ≈ (delta_mb * 1024 * 1024) / delta_max_model_len`
- How does this compare to your theoretical expectation?

## Curve Shape
- Is the curve roughly linear?
- Any obvious step changes (allocation granularity, fragmentation, other effects)?
- Do you see a clear “knee” where increases in `max-model-len` start to cost a lot more VRAM?

## Headroom Envelope (Practical Choices)
- GPU total VRAM:
- Non-KV overhead you want to reserve (MB):
- Practical VRAM budget for KV cache (MB):
- For this GPU, what `max-model-len` would you pick if you want ≈N concurrent 4K requests?
- How would you tune `max-model-len` differently for:
  - Latency-focused interactive workloads:
  - Throughput-focused batch workloads:

## Takeaways
- One sentence on “what `max-model-len` really means” for this GPU.
- One configuration you’d recommend as a default for this card + SLM.

