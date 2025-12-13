# Day 007 – vLLM SLM: TTFT, Prefix Caching, KV Scaling
## Tier 3 – KV Scaling + Micro-Batching Under Load (Capacity Intuition)

> **Goal**:
> - Empirically connect `max-model-len` to VRAM usage (KV cache capacity commitment).
> - Quantify the throughput difference between sequential vs concurrent load.
>
> **Outcome**: A small KV scaling CSV + a batching benchmark note with numbers.

---

## Tier 3 – Stretch (Optional / Ambitious)

**Title** – KV scaling and batching regimes for an SLM vLLM server  
**Time Budget** – ~90–150 min

---

### A) KV cache scaling vs `max-model-len` (~45–75 min)

#### 1) Implement a KV scaling runner

Create:

- `days/day-007-vllm-slm/kv_scaling.sh`

Requirements:

- Loop over a small list of lengths (start conservative): `512 1024 2048 4096`
- For each length:
  - start server
  - wait for ready
  - capture `nvidia-smi --query-gpu=memory.used`
  - write a row to CSV
  - stop server cleanly

CSV:

- `days/day-007-vllm-slm/kv_cache_scaling.csv`

Columns:

- `max_model_len,gpu_mem_used_mb,delta_mb_from_prev,bytes_per_token_est,notes`

Tip:

- Include a small baseline row (e.g. with the smallest `max-model-len` you test) so `delta_mb_from_prev` and `bytes_per_token_est` are easy to compute.

#### 2) Interpret the curve

Create:

- `days/day-007-vllm-slm/kv_cache_scaling_notes.md`

Answer:

- Is it roughly linear?
- Are there step changes? (allocation granularity)
- Use the observed slope to back out an approximate **bytes per KV token**:
  - e.g. `bytes_per_token_est ≈ (delta_mb * 1024 * 1024) / delta_max_model_len`.
- Which `max-model-len` feels safe for this GPU *given you still want concurrency headroom*?
- Sketch a “headroom envelope”:
  - For this GPU, what `max-model-len` would you pick if you want ≈N concurrent 4K requests?
  - What trade-offs would you make for a latency-focused vs throughput-focused service?

---

### B) Micro-batching vs sequential throughput (~45–75 min)

#### 1) Write a tiny concurrency client

Create:

- `days/day-007-vllm-slm/batch_client.py`

Requirements:

- N requests total (e.g. 32)
- Run sequential and concurrent
- Print:
  - `sequential_s`
  - `concurrent_s`
  - `sequential_qps` / `concurrent_qps`
  - `sequential_tok_s` / `concurrent_tok_s` (if you can get token counts)

If you already have a client from earlier days, reuse it, but keep a Day 007 copy so this day is self-contained.

#### 2) Run the benchmark under a fixed server config

- Pick one `max-model-len` (e.g. 4096)
- Keep `max_tokens` constant (e.g. 64)
- Run 3 times and record the best/median (your call; just be consistent)

Create:

- `days/day-007-vllm-slm/batching_benchmark.md`

Include:

- numbers (sequential vs concurrent)
- a small table for a few concurrency levels:

```text
concurrency,mean_e2e_s,p95_e2e_s,qps,tok_s,gpu_util_pct,notes
1,,,,,,
4,,,,,,
8,,,,,,
16,,,,,,
```

- a short note on GPU utilization if you observed it (`nvidia-smi dmon`)
- 3–5 bullets that capture your **“rules of thumb”**:
  - e.g. “On this GPU + SLM, concurrency N–M is the sweet spot for 4K contexts.”
  - “Beyond concurrency K, p95 blows up without meaningful throughput gains.”

---

### C) Optional: Concurrency sweep (find “knee of the curve”)

If time permits, extend `batch_client.py` to sweep concurrency levels:

- 1, 2, 4, 8, 16, 32

Write results into:

- `days/day-007-vllm-slm/concurrency_sweep.csv`

Key insight to capture:

- Where does throughput stop scaling?
- Where does latency blow up? (queuing)

---

## Expected Artifact

- `days/day-007-vllm-slm/kv_scaling.sh`
- `days/day-007-vllm-slm/kv_cache_scaling.csv`
- `days/day-007-vllm-slm/kv_cache_scaling_notes.md`
- `days/day-007-vllm-slm/batch_client.py`
- `days/day-007-vllm-slm/batching_benchmark.md`
- (optional) `days/day-007-vllm-slm/concurrency_sweep.csv`

---

## What You Should Learn (Mental Models)

- `max-model-len` is not “just a limit” — it is a **VRAM reservation decision**.
- Continuous batching is not optional; it’s how the runtime reaches its advertised throughput.
- You can reason about safe operating points by combining:
  - KV memory commitment
  - measured throughput under concurrency
  - acceptable p95 latency for your target workload
