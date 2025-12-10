# Day 006 – SLM + OS Memory & First-Token Path

> **Phase**: 0 – OS & GPU Setup (Days 1–15)  
> **Theme**: THP, hugepages, allocators, page cache, and first‑token latency using a small language model (SLM) as a probe.  
> **Layers**:  
> - **Hardware/OS** → THP, explicit hugepages, allocators, page cache  
> - **Runtime (vLLM)** → SLM via vLLM, first-token latency, KV cache scaling, micro-batching  

---

## Snapshot (Today’s Focus)

- You’re still in **Phase 0 – OS & GPU Setup (Days 1–15)**, but the goal is to effectively burn through the substance of **Day 6 + Day 7** in one focused session.  
- So far: driver + CUDA are healthy, basic vLLM runs, NUMA pinning + OS hardening + initial OOM/capacity experiments are in the journal.  
- Today’s layers:
  - **Layer 1 – Hardware/OS** → THP, hugepages, allocators, page cache  
  - **Layer 3 – Runtime (vLLM)** → SLM via vLLM, first-token latency, KV cache scaling, micro-batching  
- Themes: **(1) long context & memory**, **(2) continuous batching**, **(7) throughput scaling**, **(10) reliability**.  
- Assumptions:
  - You can install `jemalloc`.  
  - You can pull an SLM like `microsoft/Phi-3-mini-4k-instruct` or `Qwen2.5-1.5B-Instruct`.  
  - vLLM already works on this node.

---

## Tier Breakdown

Think of this as:

- **Tier 1 = Day 6 core** – OS + SLM direct.  
- **Tier 2 = Day 6 stretch + Day 7 core** – allocator + vLLM first-token.  
- **Tier 3 = Day 7 stretch** – KV scaling + batching.

| Tier | Time | Scope |
|------|------|-------|
| Tier 1 | ~75–90 min | SLM + OS memory baseline (THP, hugepages, cold/warm load) |
| Tier 2 | ~90–120 min | Allocator impact + vLLM first-token latency using the same SLM |
| Tier 3 | ~90–120 min | vLLM KV cache scaling + micro-batching with SLM |

---

## Navigation

- **[Tier 1 – SLM + OS Memory Baseline](LOG_tier01.md)**  
  THP/hugepages configuration, cold vs warm load, memory footprint.

- **[Tier 2 – Allocators & vLLM First-Token](LOG_tier02.md)**  
  `glibc` vs `jemalloc` latency and first-token behavior via vLLM.

- **[Tier 3 – KV Scaling & Micro-Batching](LOG_tier03.md)**  
  `max-model-len` → KV cache footprint and continuous batching gains, even for SLMs.

---

## Cross-Day Context

- **Day 002 – GPU Node Bring-Up**: base OS/GPU setup and first CUDA/vLLM checks.  
- **Day 003 – vLLM Capacity & OOM**: capacity grids that will later run on this tuned memory baseline.  
- **Day 004 – Quantization vs BF16**: quantization experiments that benefit from predictable memory behavior.  
- **Day 005 – OS & NUMA Node Hardening**: CPU/NUMA topology; today extends that into **memory layout and page behavior**.

---

## Off-Hours Reading (Optional)

- vLLM paper – PagedAttention & KV cache sections (for interpreting `max-model-len` and GPU memory curves).  
- A short `jemalloc` vs `glibc` article (allocator behavior & fragmentation).  
- Any “first-token latency in LLM serving” doc (vLLM/TGI) to tie cold/warm behavior to production SLOs.

---

## Logging Template (For Tomorrow’s Write-Up)

Use this as a template in your day log (`LOG.md` or similar) once experiments are done:

```markdown
# Day 006–007 – SLM + OS Memory + vLLM

## SLM & Environment
- MODEL:
- GPU type:
- THP mode:
- nr_hugepages:

## Commands Run
- THP/hugepages config commands
- slm_load.py cold/warm runs (+ /usr/bin/time)
- slm_gen_latency.py (glibc vs jemalloc)
- vLLM server launches (different max-model-len values)
- curl cold/warm requests
- kv_scaling.sh
- batch_client.py

## Files Created/Updated
- days/day-006-slm-memory/slm_load.py
- days/day-006-slm-memory/slm_gen_latency.py
- days/day-006-slm-memory/README.md
- days/day-006-slm-memory/allocator_latency_comparison.csv
- days/day-007-vllm-slm/first_token_latency.md
- days/day-007-vllm-slm/kv_scaling.sh
- days/day-007-vllm-slm/kv_cache_scaling.csv
- days/day-007-vllm-slm/kv_cache_scaling_notes.md
- days/day-007-vllm-slm/batch_client.py
- days/day-007-vllm-slm/batching_benchmark.md

## Key Numbers / Metrics
- cold vs warm HF load times
- HF gen latency (glibc vs jemalloc)
- vLLM cold vs warm request times
- GPU memory vs max-model-len
- sequential_s vs concurrent_s (batch_client.py)

## Observations / Surprises
- Any allocator wins?
- How big is the cold→warm improvement in vLLM?
- KV memory scaling pattern?
- Throughput gain from continuous batching, even for SLM?
```

---

<p align="center">
  <a href="../day-005-OS-and-NUMA-node-hardening/">← Day 005</a> · 
  <a href="../README.md">Days Index</a>
</p>

