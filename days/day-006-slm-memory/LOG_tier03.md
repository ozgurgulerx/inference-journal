# Day 006 – SLM + OS Memory & First-Token Path
## Tier 3 – vLLM KV Cache Scaling & Micro-Batching with SLM

> **Goal**:  
> - Understand how `max-model-len` affects vLLM KV cache footprint.  
> - Show how continuous batching boosts throughput even for a small model.

---

**Related theory**:

- `theory/kv_cache.md` – KV cache scaling laws and PagedAttention’s impact on capacity.  
- `theory/slms_as_probes.md` – why SLMs are perfect for KV and batching probes.

## Tier 3 – Stretch (Optional / Ambitious)

**Title** – vLLM KV Cache Scaling + Micro-Batching with SLM  
**Time Budget** – ~90–120 min

---

### A. KV Cache Scaling vs `max-model-len` (~45–60 min)

#### 1. Automate Runs over Multiple `max-model-len` Values

In `days/day-007-vllm-slm/kv_scaling.sh`:

```bash
#!/usr/bin/env bash
set -e

MODEL="microsoft/Phi-3-mini-4k-instruct"
echo "max_model_len,gpu_mem_used_mb" > kv_cache_scaling.csv

for L in 512 1024 2048 4096; do  # extend this list (e.g. 8192, 16384) as your GPU allows
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype auto \
    --max-model-len $L \
    --gpu-memory-utilization 0.92 > server_$L.log 2>&1 &
  PID=$!
  sleep 10  # wait for load

  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$L,$MEM" >> kv_cache_scaling.csv

  kill $PID
  sleep 5
done
```

Make it executable and run:

```bash
cd days/day-007-vllm-slm
chmod +x kv_scaling.sh
./kv_scaling.sh
```

#### 2. Inspect `kv_cache_scaling.csv`

Look for:

- Rough linearity between `max-model-len` and `gpu_mem_used_mb` (KV cache grows ~linearly with context).  
- Any unexpected jumps (e.g., page/block allocation thresholds, fragmentation artifacts).  
- At which `max-model-len` values GPU memory usage starts to leave too little headroom for batching and concurrency.

#### 3. Add Commentary

Create `days/day-007-vllm-slm/kv_cache_scaling_notes.md` (5–10 lines) answering:

- How memory scales with `max-model-len` on this GPU (does it match the expectations from `theory/kv_cache.md`?).  
- How much headroom remains for additional sequences / batching at each length.  
- Which `max-model-len` values you would recommend for real deployments (balance between long context and safe concurrency).

---

### B. Micro-Batching vs Sequential with SLM (~45–60 min)

#### 1. Create a Tiny Batch Client

`days/day-007-vllm-slm/batch_client.py` (aim for ≤20–25 lines):

```python
import asyncio, time, aiohttp

N = 32
URL = "http://localhost:8000/v1/completions"
payload = {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "Explain memory fragmentation in one short sentence.",
    "max_tokens": 32,
}


async def one_call(session: aiohttp.ClientSession) -> None:
  async with session.post(URL, json=payload) as r:
    await r.text()


async def run_sequential() -> float:
  async with aiohttp.ClientSession() as s:
    start = time.time()
    for _ in range(N):
      await one_call(s)
    return time.time() - start


async def run_concurrent() -> float:
  async with aiohttp.ClientSession() as s:
    start = time.time()
    await asyncio.gather(*[one_call(s) for _ in range(N)])
    return time.time() - start


async def main() -> None:
  seq = await run_sequential()
  conc = await run_concurrent()
  print("sequential_s", seq)
  print("concurrent_s", conc)


if __name__ == "__main__":
  asyncio.run(main())
```

#### 2. Run the Micro-Benchmark

With the vLLM server (using the same SLM config) running:

```bash
cd days/day-007-vllm-slm
python batch_client.py
```

Record `sequential_s` and `concurrent_s`.

#### 3. Optionally Observe GPU Utilization

```bash
nvidia-smi dmon -s pucv -d 1
```

Watch for utilization differences between sequential and concurrent modes.

#### 4. Write a Short Summary

Create `days/day-007-vllm-slm/batching_benchmark.md`:

- Numerical results: `sequential_s`, `concurrent_s`, rough tokens/sec for each.  
- A brief note on how much continuous batching helped even with an SLM.  
- Any qualitative GPU utilization notes.

---

### C. Concurrency vs Context Length & Tail Latency (Optional but High-Value)

To connect KV scaling and batching to **real concurrency limits** and tail latency:

1. For each `max-model-len` value you tested in `kv_scaling.sh`:

   - Use a simple client (your `batch_client.py` or the Day 003 `vllm_chat_bench.py`) to gradually increase concurrency (e.g., 4 → 8 → 16 → 32 …).  
   - Record when:
     - GPU runs out of memory (OOM / server crash), **or**  
     - p95/p99 latency becomes unacceptable for your target SLO.

2. Capture results in a small table, e.g. `days/day-007-vllm-slm/concurrency_vs_context.md`:

   ```markdown
   | max-model-len | max_concurrency | p95_ms | p99_ms | OOM? | Notes |
   |---------------|-----------------|--------|--------|------|-------|
   | 2048          | 32              | 250    | 400    | no   | stable |
   | 4096          | 16              | 320    | 600    | no   | tail growing |
   | 8192          | 8               | 500    | 900    | yes  | OOM @ >8 |
   ```

3. If your client supports it, vary any **batch window / request grouping** knob (or simply the concurrency level) to see:

   - When throughput gains from batching flatten out.  
   - How batching changes p95/p99, not just p50.

This step links the KV theory directly to **capacity planning**: it tells you, for each context length, how many users a single GPU can realistically support at your latency target.

---

### Tier 3 Artifacts

- `days/day-007-vllm-slm/kv_scaling.sh`  
- `days/day-007-vllm-slm/kv_cache_scaling.csv`  
- `days/day-007-vllm-slm/kv_cache_scaling_notes.md`  
- `days/day-007-vllm-slm/batch_client.py`  
- `days/day-007-vllm-slm/batching_benchmark.md`
