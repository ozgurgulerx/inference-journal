# Day 007 — vLLM Runtime Probes (Tier 1)
## TTFT Baseline, Cold/Warm Anatomy, and the Knobs That Actually Matter

This Tier 1 log is designed to be:
- **Runnable** end-to-end (a true probe runbook)
- **Interpretable** (a diagnosis framework, not just numbers)
- **Reusable** (a baseline you can repeat on any node/model)

It is intentionally layered:
1) **Runbook** (do this in one sitting)
2) **Measurement tables** (fill them, don’t “vibe”)
3) **Interpretation** (decision tree / root-cause decomposition)
4) **Appendices** (deep technical notes: page cache vs HBM vs CUDA graphs, KV/buffer allocations, batching mental model, knobs)

---

# 0) What you’re actually measuring (make this explicit)

### Definitions
- **TTFT** (time-to-first-token): request arrival → first token produced.  
  - Best measured with streaming/time-to-first-byte instrumentation.
- **E2E latency**: request arrival → full completion.
- **`wall_s` in this Tier 1 probe**: client-side wall clock from request send → response received.  
  - This is a **proxy** for TTFT+decode in non-streaming mode.  
  - It is still very useful for **cold vs warm comparisons**, **variance**, and **queueing effects**.

### Cold vs warm
- **Cold**: first request after server restart (includes one-time costs).
- **Warm**: subsequent requests (steady-state-ish).

---

# 1) Goal / Outcomes

### Goal
Establish a repeatable baseline for first-token behavior on vLLM using a single SLM:
- cold vs warm request timing
- warm steady-state variance
- minimal sanity metrics to explain where time went

### You’re “done” when these exist
- `days/day-007-vllm-runtime-probes/serve_slm_vllm.sh`
- `days/day-007-vllm-runtime-probes/ttft_probe.py`
- `days/day-007-vllm-runtime-probes/first_token_latency.md` (filled with tables + diagnosis paragraph)

---

# 2) Tier 1 Runbook (do this without inventing anything)

## 2.1 Freeze the SLM (do once)
Pick ONE model for the entire day and never change it mid-measurement:
- `microsoft/Phi-3-mini-4k-instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

Record here:
- MODEL = `...`

Why: changing model changes weight size, KV shapes, kernels, and “warm-up” behavior. Baselines become meaningless.

---

## 2.2 Baseline vLLM launch (known-good)
Create: `days/day-007-vllm-runtime-probes/serve_slm_vllm.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --port $PORT
```

Start:

```bash
chmod +x days/day-007-vllm-runtime-probes/serve_slm_vllm.sh
./days/day-007-vllm-runtime-probes/serve_slm_vllm.sh
```

---

## 2.3 Baseline environment snapshot (before/after matters)

Before launching vLLM (clean shell), record:

```bash
uname -a
python -c "import vllm; print('vLLM', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

After server is up (before any requests), record GPU mem again:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

---

## 2.4 Probe script (minimal and consistent)

Create: `days/day-007-vllm-runtime-probes/ttft_probe.py`

Requirements:

- measures `wall_s`  
- prints token counts if present  
- constant payload (do not change between runs)

Suggested payload (keep fixed):

- `max_tokens=64`  
- `temperature=0`  
- short prompt

Example:

```python
#!/usr/bin/env python3
import json
import time

import requests


URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "Say hello from a small language model."


def main() -> None:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 64,
        "temperature": 0.0,
    }
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=60)
    t1 = time.time()
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0].get("text", "").strip()
    usage = data.get("usage", {}) or {}
    out = {
        "wall_s": t1 - t0,
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "output_preview": text[:80],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
```

Sanity check endpoint:

```bash
curl -s http://localhost:8000/v1/models | head
python days/day-007-vllm-runtime-probes/ttft_probe.py
```

---

# 3) Measurements (fill tables, don’t narrate)

Create: `days/day-007-vllm-runtime-probes/first_token_latency.md`

Use the template below (copy as-is).

---

## 3.1 `first_token_latency.md` template

```markdown
# Day 007 — TTFT Baseline (Tier 1)

## Server Config
- MODEL =
- GPU =
- OS/kernel =
- vLLM version =
- Launch flags =
- Notes (anything non-default: quantization, env vars, etc.) =

## Baseline GPU Snapshot
- GPU mem used before server start (MiB) =
- GPU mem used after server start, before any request (MiB) =
- GPU mem used after cold_1 (MiB) =

## Cold vs Warm (restart required)
Procedure:
1) Restart server
2) Run probe once → cold_1
3) Run probe immediately again → warm_1

| run   | wall_s | total_tokens | notes |
|------|--------:|-------------:|------|
| cold_1 |        |              | first request after server start |
| warm_1 |        |              | immediate second request |

## Warm Variance (steady state)
Run 5 back-to-back probes (no restart).

| run | wall_s | total_tokens | notes |
|----:|-------:|-------------:|------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

Summary:
- min_wall_s =
- median_wall_s =
- max_wall_s =
- spread % = (max-min)/median

## Sanity Metrics During Warm Runs
- Typical GPU util band =
- Typical GPU power band =
- Typical CPU util band (vLLM pid) =
- Any interference (other GPU jobs, clock changes, etc.) =

## Interpretation (write 5–10 lines, evidence-based)
1) What dominated cold_1? (page cache vs HBM transfer/alloc vs runtime warmup vs KV/buffer alloc)
2) Is warm stable? (variance band)
3) Operational takeaway: what steady-state number would you treat as “baseline SLO anchor”?
4) What knob do you tune next (Tier 2/3) and why?
```

---

# 4) Interpretation (this is the “integrity” layer)

You should be able to explain cold vs warm as a sum of terms.

### TTFT (cold) mental equation

**Cold request** includes:

1. storage → RAM (page cache fill / major faults)  
2. RAM → HBM (weight staging + GPU allocations)  
3. runtime warm-up (kernel selection/autotune, CUDA graphs capture/instantiate, allocator pool setup)  
4. first-time KV / workspace buffer allocations  
5. first prefill compute  
6. queueing/scheduling delay (depends on concurrency + batching policy)

**Warm request** mostly includes:

- prefill compute + small overhead  
- queueing/scheduling delay  
- (and reuses prior allocations/caches)

---

## 4.1 Diagnosis decision table (use this, don’t guess)

### A) Page cache fill dominated

Signals:

- Disk read throughput spikes during cold_1.  
- `Cached/buffers` rises (host RAM used as cache).  
- GPU util low early.

Conclusion:

- cold_1 slow primarily because weights were fetched from storage into RAM.

### B) HBM transfer / alloc dominated

Signals:

- Disk relatively quiet (or already warm cache).  
- GPU memory jumps significantly during cold_1.  
- (Optional) PCIe/NVLink throughput high if measured.

Conclusion:

- cold_1 slow due to moving weights/buffers into GPU HBM and allocator work.

### C) Runtime warm-up dominated (CUDA graphs / autotune / pools)

Signals:

- Disk quiet.  
- cold_1 triggers “first run overhead”.  
- warm requests become more stable (lower variance) after 1–2 runs.

Conclusion:

- one-time runtime setup was paid on the first execution.

### D) First-time KV / buffer allocations dominated

Signals:

- Sharp GPU mem jump on first request (even if server already started).  
- warm_1 dramatically faster.  
- repeated restarts reproduce the jump.

Conclusion:

- first request forced allocation of KV pages + reusable scratch buffers.

---

# 5) Appendices (deep technical depth, properly placed)

## Appendix A — Page cache vs HBM (what “cold load” really means)

### Page cache (storage → RAM)

- Linux caches file pages in RAM.  
- First time you load model files after reboot/cache drop:
  - kernel reads weight shards from disk/network storage into RAM,  
  - this is “page cache fill”.
- Second time:
  - those file pages are already in RAM,  
  - file reads become memory-speed (warm load).

### HBM load (RAM → GPU)

- The runtime then allocates GPU buffers and copies/stages weights into HBM.  
- This is separate from page cache:
  - page cache = kernel/filesystem,  
  - HBM load = CUDA/runtime allocations + transfers.
- They can overlap (pipelining), but it still often “feels like phases”.

---

## Appendix B — “Initial CUDA graph warm-up” (what it actually is)

### What CUDA graphs are solving

Without graphs:

- CPU repeatedly launches many kernels per step (GEMMs, attention, layernorm, memcpy…).  
- kernel launch overhead and dispatcher overhead contribute to jitter and latency.

With CUDA graphs:

- runtime can **capture** a representative sequence of GPU ops,  
- **instantiate** the captured graph,  
- **replay** it with lower CPU overhead and more stable scheduling.

### Why warm-up is paid on cold_1

First execution may include:

- kernel selection/autotuning (cuBLASLt, CUTLASS, attention kernels),  
- allocator pool growth and initialization,  
- graph capture/instantiate work (if enabled in the stack),  
- sometimes JIT compilation or kernel loading depending on backend (PyTorch/Triton/inductor).

Practical: cold_1 often includes a “one-time bump” beyond weight I/O.

---

## Appendix C — “First-time KV / buffer allocations” (deep)

This phrase refers to **lazy allocations** that happen when the first real request runs.

### C1) KV cache allocations (PagedAttention)

- vLLM stores K/V for each active sequence.  
- KV cost scales roughly with:
  - layers × heads × head_dim × tokens × dtype.
- `max-model-len` is a **capacity commitment**:
  - higher `max-model-len` increases the worst-case KV footprint per sequence,  
  - reduces safe concurrency (fewer sequences fit in VRAM),  
  - can increase first-request allocation cost.

PagedAttention allocates KV in pages/blocks; these pages are often allocated on-demand as sequences are admitted.  
So the first request can trigger:

- allocation of KV pages,  
- metadata structures for paging,  
- device memory pool growth.

### C2) Workspace / scratch buffers

First execution can allocate persistent workspaces for:

- attention scratch,  
- GEMM workspaces (cuBLASLt),  
- reductions/softmax/layernorm intermediates.

These are frequently cached and reused, so they appear as “first request slow, later stable.”

### C3) Why this shows up as TTFT inflation

TTFT can’t be satisfied until:

- buffers exist,  
- KV exists,  
- kernels can execute.

So allocation work is on the critical path for token #1.

---

## Appendix D — Streams, concurrency, batching (the correct mental model)

### D1) Do attentions mix across streams?

No.

- each request has its own token history, attention mask, and KV pages,  
- batching does not merge contexts.

### D2) Then why do streams affect each other?

They don’t affect correctness, but they are operationally coupled by:

- shared GPU compute (SMs, tensor cores),  
- shared memory bandwidth,  
- shared KV capacity,  
- scheduler time slicing (queueing).

So: **semantically isolated, physically coupled**.

### D3) What batching *actually* is (and why “GPU has many cores” doesn’t remove the need)

It is not “one stream → one SM”.  
Instead:

- batching stacks token work from multiple sequences into a larger matrix operation,  
- a single kernel tiles that batched operation across all SMs,  
- bigger batches improve GPU occupancy and amortize overhead.

---

## Appendix E — Knobs (map knobs to the TTFT equation)

### Capacity / memory-shape knobs

- `--max-model-len`: biggest lever for KV commitment (affects concurrency headroom + allocation behavior).  
- `--gpu-memory-utilization`: how aggressively vLLM consumes VRAM headroom.

### Concurrency knobs

- client-side concurrency: how many requests you send in parallel (creates queueing and batching opportunity).  
- `--max-num-seqs`: server-side cap on active sequences inside vLLM (limits true in-engine concurrency).

### Batching policy knobs (per-tick “batch budget”)

- `--max-num-batched-tokens`: caps tokens processed per scheduler tick (prefill + decode). Higher → throughput up, TTFT up under load.  
- `--enable-chunked-prefill`: improves fairness under mixed long/short prompts by chunking prefill work so long prompts don’t monopolize.  
- prefix/prompt caching (if enabled): reduces repeated prefill cost; policy-adjacent but affects latency under repeated prefixes.

---

# 6) Tier 1: what you should conclude (so it’s complete)

After running Tier 1, you should be able to say:

- Cold penalty factor: `cold_1 / warm_median ≈ X×`.  
- Warm steady-state band: median ± spread.  
- Dominant cold term (page cache vs HBM vs warm-up vs KV/buffers) with evidence.  
- Next tuning direction:
  - if warm TTFT high but GPU underutilized → adjust batching/concurrency,  
  - if OOM / low concurrency headroom → revisit `max-model-len` / KV footprint,  
  - if cold penalty operationally unacceptable → design a pre-warm strategy.

