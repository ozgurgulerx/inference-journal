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

- `max_model_len,gpu_mem_used_mb,notes`

#### 2) Interpret the curve

Create:

- `days/day-007-vllm-slm/kv_cache_scaling_notes.md`

Answer:

- Is it roughly linear?
- Are there step changes? (allocation granularity)
- Which `max-model-len` feels safe for this GPU *given you still want concurrency headroom*?

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

If you already have a client from earlier days, reuse it, but keep a Day 007 copy so this day is self-contained.

#### 2) Run the benchmark under a fixed server config

- Pick one `max-model-len` (e.g. 4096)
- Keep `max_tokens` constant (e.g. 64)
- Run 3 times and record the best/median (your call; just be consistent)

Create:

- `days/day-007-vllm-slm/batching_benchmark.md`

Include:

- numbers (sequential vs concurrent)
- a short note on GPU utilization if you observed it (`nvidia-smi dmon`)

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
