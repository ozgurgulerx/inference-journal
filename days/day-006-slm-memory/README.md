# Day 006 – SLM + OS Memory & First-Token Path

**Phase:** 0 – OS & GPU Setup (Days 1–15)  
**Theme:** THP, hugepages, allocators, page cache, and first-token latency using a small language model (SLM) as a probe.

**Layers**

- **Hardware/OS** → THP, explicit hugepages, allocators, page cache  
- **Runtime (vLLM)** → SLM via vLLM, first-token latency, KV cache scaling, micro-batching  

---

## Snapshot (Today’s Focus)

You’re still in **Phase 0 – OS & GPU Setup (Days 1–15)**, but the goal is to effectively burn through the substance of **Day 6 + Day 7** in one focused session.

So far:  
- Driver + CUDA are healthy  
- Basic vLLM runs  
- NUMA pinning + OS hardening + initial OOM/capacity experiments are in the journal  

Today’s layers:

- **Layer 1 – Hardware/OS** → THP, hugepages, allocators, page cache  
- **Layer 3 – Runtime (vLLM)** → SLM via vLLM, first-token latency, KV cache scaling, micro-batching  

Themes: **(1) long context & memory**, **(2) continuous batching**, **(7) throughput scaling**, **(10) reliability**.

Assumptions:

- You can install `jemalloc`.  
- You can pull an SLM like `microsoft/Phi-3-mini-4k-instruct` or `Qwen2.5-1.5B-Instruct`.  
- vLLM already works on this node.

---

## Tier Breakdown

Think of this as:

- **Tier 1** = Day 6 core – OS + SLM direct.  
- **Tier 2** = Day 6 stretch + Day 7 core – allocator + vLLM first-token.  
- **Tier 3** = Day 7 stretch – KV scaling + batching.

| Tier  | Time         | Scope                                                             |
|-------|-------------:|-------------------------------------------------------------------|
| Tier 1| ~75–90 min   | SLM + OS memory baseline (THP, hugepages, cold/warm load)        |
| Tier 2| ~90–120 min  | Allocator impact + vLLM first-token latency using the same SLM   |
| Tier 3| ~90–120 min  | vLLM KV cache scaling + micro-batching with SLM                  |

---

## Required Learnings / Mental Models

By the end of Day 006 you should **own** the following:

### 1. SLM as a Fast OS/Runtime Probe

- A **small model** (SLM) is the cheapest reliable way to probe:
  - OS memory config (THP mode, hugepages, page cache)
  - Load-time behavior (cold vs warm)
  - “Shape” of vLLM’s first-token + batching behavior  
- You should be able to spin up an SLM, run a few micro-benchmarks, and **quickly detect if a node is “in a good state”** before loading a big LLM.

### 2. Transparent Hugepages & Explicit Hugepages

- You understand the practical difference between **`always` / `madvise` / `never`** for THP and why `madvise` + explicit hugepages is a sane default for inference:
  - Avoids THP doing surprise global migrations/compactions.
  - Still allows **opt-in** hugepage usage for model weights / buffers.
- You know how to:
  - Inspect current config: `cat /sys/kernel/mm/transparent_hugepage/enabled`, `grep -i Huge /proc/meminfo`
  - Set it for inference workloads:
    - `echo madvise > /sys/kernel/mm/transparent_hugepage/enabled`
    - `echo never > /sys/kernel/mm/transparent_hugepage/defrag` (if available)
  - Reserve explicit hugepages with `nr_hugepages` and mount `hugetlbfs`:
    - `echo 1024 > /proc/sys/vm/nr_hugepages`
    - `mount -t hugetlbfs none /mnt/huge`

**Mental model:**  
> “THP is the kernel trying to be clever for me. For predictable inference, I want **explicit control**: small but stable hugepage reserves + `madvise`.”

### 3. Cold vs Warm Load & Page Cache

- You can **measure and interpret** cold vs warm load times for an SLM:
  - Cold: drop caches via `echo 3 > /proc/sys/vm/drop_caches` and time `slm_load.py`.
  - Warm: re-run `slm_load.py` and compare.
- You understand:
  - Cold load is dominated by **disk I/O + page cache population**.
  - Warm load is dominated by **Python graph construction / framework overhead**, with I/O mostly removed.
- You can read and log:
  - `cold_load_real_s` / `warm_load_real_s` from `/usr/bin/time`.
  - Rough host memory usage via `ps -C python -o pid,rss,vsz,cmd`.

**Mental model:**  
> “Page cache is my first ‘cache layer’ below GPU. If cold vs warm spreads are huge, storage or caching is likely the bottleneck.”

### 4. Allocator Choice (glibc vs jemalloc) Actually Matters

- You know how to swap allocators **without code changes**:
  - Default: glibc malloc.
  - Jemalloc: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python ...`
- You measured:
  - **`gen_latency_s`** for a short HF SLM generation under both allocators.
  - Wall time via `/usr/bin/time` and optionally RSS via `ps`.
- You understand what to look for:
  - Lower variance / slightly lower `gen_latency_s` with jemalloc.
  - Potentially lower RSS or better behavior under many allocations.

**Mental model:**  
> “Allocator is part of the inference stack. When I care about p99, I at least **try jemalloc** and capture numbers.”

### 5. First-Token Latency & Warm-Start in vLLM

- You can:
  - Start vLLM for the SLM with a clean config:
    - `python -m vllm.entrypoints.openai.api_server --model <SLM> --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.92`
  - Hit it with an OpenAI-compatible `/v1/completions` request (curl or Python).
- You have measured **cold vs warm request** wall time:
  - `cold_req_real_s` (first ever call after server start).
  - `warm_req_real_s` (immediate second call).
- You watched GPU memory with `nvidia-smi` during first requests and observed:
  - Memory jump when weights + KV allocs are set up.
  - Much smaller change for warm requests.

**Mental model:**  
> “Cold TTFT penalty = weight loading + initial JIT/graph warmup. In prod I either keep servers warm or hide this behind pre-warm requests.”

### 6. KV Cache Scaling vs `max-model-len`

- Via `kv_scaling.sh`, you’ve seen how:
  - `max-model-len` directly affects **GPU memory reserved for KV cache**.
  - The curve is roughly linear, with potential step changes due to block allocation.
- You can now reason about:
  - Why bumping `max-model-len` from 4K → 8K is not “free”.
  - How to pick `max-model-len` for a given GPU so you **don’t eat all headroom** for batching / extra models.

**Mental model:**  
> “`max-model-len` is a capacity commitment. Every extra token of context length has a **VRAM cost**, independent of whether every request uses it.”

### 7. Continuous Batching vs Sequential (Even for SLMs)

- With `batch_client.py` you have:
  - Measured `sequential_s` vs `concurrent_s` for N small prompts.
  - Seen real throughput gains when vLLM can **batch and schedule** multiple requests.
- You’ve optionally correlated this with:
  - GPU utilization via `nvidia-smi dmon`.

**Mental model:**  
> “If I don’t give the runtime concurrency, I force it into a low-utilization regime. Batching is not an optimization detail; it’s central to how these systems achieve their advertised throughput.”

---

## Navigation

- **[Tier 1 – SLM + OS Memory Baseline](./LOG_tier01.md)**  
  THP/hugepages configuration, cold vs warm load, memory footprint.

- **[Tier 2 – Allocators & vLLM First-Token](./LOG_tier02.md)**  
  glibc vs jemalloc latency and first-token behavior via vLLM.

- **[Tier 3 – KV Scaling & Micro-Batching](./LOG_tier03.md)**  
  `max-model-len` → KV cache footprint and continuous batching gains, even for SLMs.
- **[Theory – Huge Pages & DMA](./theory/day06_theory_huge_pages.md)**  
  MMUs, THP vs explicit hugepages, IOMMU, and their impact on LLM serving.
- **[Theory – Allocators & vLLM Serving](./theory/day06_theory_malloc.md)**  
  glibc vs jemalloc, RSS behavior, and allocator ownership for inference nodes.
- **[Theory – SLMs as Probes](./theory/slms_as_probes.md)**  
  Why 1–4B models are the best instruments for OS/runtime benchmarking and vLLM tuning.
- **[Theory – KV Cache & PagedAttention](./theory/kv_cache.md)**  
  How KV cache scales with context length, why naïve allocation wastes VRAM, and how PagedAttention changes capacity planning.

---

## Cross-Day Context

- **Day 002 – GPU Node Bring-Up**  
  Base OS/GPU setup and first CUDA/vLLM checks.

- **Day 003 – vLLM Capacity & OOM**  
  Capacity grids that will later run on this tuned memory baseline.

- **Day 004 – Quantization vs BF16**  
  Quantization experiments that benefit from predictable memory behavior.

- **Day 005 – OS & NUMA Node Hardening**  
  CPU/NUMA topology; today extends that into memory layout and page behavior.

Day 006 now acts as the **bridge** between:

- Pure OS/GPU tuning → and  
- Runtime-level behavior (vLLM first-token, KV cache, batching),

using an SLM as a **cheap, repeatable health probe**.

---

## Key Concepts & Learning Resources

The labs in Day 006 lean on a few foundational ideas. These resources are useful anchor points if you want to go deeper.

![Memory Access: CPU vs GPU DMA Paths](../assets/cpu_gpu_transfer.png)

### Transparent Huge Pages (THP) & Huge Pages

- **Concept**: Modern CPUs normally use 4KB pages; huge pages (e.g. 2MB on x86_64) reduce TLB pressure by mapping more memory per entry. Linux supports both **explicit hugepages** (HugeTLBfs) and **Transparent Huge Pages (THP)**, which automatically coalesces pages. THP can boost performance without code changes but may introduce latency or jitter for some workloads, which is why many latency-sensitive systems use `madvise` + explicit hugepages instead of `always`.
- **Reading**:
  - Red Hat “Huge Pages Overview” – 2MB pages, TLB misses, and THP vs explicit hugepages.  
  - Netdata “THP Benefits & Drawbacks” – why admins sometimes disable THP for stability/latency.

### Memory Allocators: glibc malloc vs jemalloc

- **Concept**: glibc’s default allocator (ptmalloc) can fragment and keep RSS high under certain allocation patterns. `jemalloc` is designed to bound fragmentation and return unused memory to the OS more aggressively. In long‑running, memory‑intensive services (databases, LLM servers), switching to `jemalloc` often reduces memory bloat and can tighten latency.
- **Reading**:
  - LinkedIn “Allocator Fragmentation” – jemalloc vs glibc; how jemalloc caps fragmentation and returns memory.  
  - “Battle of the Mallocators” – RocksDB case where glibc’s RSS was ~3.6× working set, jemalloc ~1.2×.  
  - Dev.to allocator comparison – benchmarks where glibc used ~3× more memory than jemalloc.

### OS Page Cache & Cold vs Warm Loads

- **Concept**: Linux caches file data in RAM. A **cold load** (after dropping caches or first boot) must hit disk to read model weights. A **warm load** reuses pages already in the page cache, making subsequent loads much faster even without touching GPU settings. In serving, cold starts often include pulling container layers + reading model files; warm starts reuse those cached bytes.
- **Reading**:
  - BentoML docs on cold vs warm start – model weights and containers as cached artifacts.  
  - General caching primers (“cold cache” vs “warm cache”) – explaining why first access is slow and later accesses are cheap.

### Small Language Models (SLMs) as Probes

- **Concept**: SLMs (1–4B parameters) fit comfortably on a single GPU and support non‑trivial context lengths (4K–32K tokens). They’re ideal **probe models** for OS and runtime experiments: you can measure allocator impact, THP modes, first‑token latency, and KV scaling without the overhead of giant models.
- **Examples**:
  - **Phi‑3 Mini 4K Instruct** (≈3.8B, 4K context) – lightweight, strong reasoning, good general probe.  
  - **Qwen2.5‑1.5B Instruct** (1.5B, up to 32K context) – small but long‑context, ideal for KV cache tests on modest GPUs.

### vLLM & PagedAttention (KV Cache Management)

- **Concept**: Transformer KV caches grow with context length and can dominate GPU memory. Naive systems pre‑allocate large contiguous buffers per sequence (e.g. for 4K tokens), wasting 60–80% of KV memory as unused padding when prompts are shorter. vLLM’s **PagedAttention** treats KV memory like a paged virtual memory system: it slices KV into fixed‑size pages that need not be contiguous, nearly eliminating internal fragmentation (<4% waste) and enabling efficient sharing.
- **Reading**:
  - vLLM blog – introduction to PagedAttention and how prior systems only used 20–38% of KV memory for real data.  
  - vLLM SOSP’23 paper – details on paged KV allocations, fragmentation, and throughput gains.

### KV Cache Scaling & Long Context Footprint

- **Concept**: KV cache size scales roughly linearly with sequence length and number of heads. Increasing `max-model-len` from 4K to 16K multiplies KV memory per sequence, and naive pre‑allocation reserves that entire chunk even if most requests are shorter. This is why high `max-model-len` can silently eat GPU memory on traditional engines.
- **Reading**:
  - vLLM evaluations showing KV memory vs sequence length (e.g. ~30% of a 40GB A100 used just for a 2048‑token KV cache on 13B).  
  - Analyses showing only ~20–38% of allocated KV memory holding real data when using fixed buffers.

### First-Token Latency (TTFT) in Serving

- **Concept**: **First‑token latency (TTFT)** is the time from request arrival until the first token is produced. It includes prompt encoding/prefill and any one‑time setup (model load, allocations, JIT). End‑to‑end latency adds the full decode time on top. TTFT is crucial for user‑perceived responsiveness, especially with streaming APIs: the user sees nothing before token 1.
- **Reading**:
  - LLM serving metric guides defining first‑token vs end‑to‑end latency and why TTFT is often the SLO.  
  - Articles on cold vs warm TTFT, covering how weight loading, kernel compilation, and cache priming impact the first call.

### Continuous Batching & Micro-Batching

- **Concept**: Batching amortizes compute and keeps GPUs busy. **Continuous batching** (like vLLM’s token‑level batching) constantly forms batches from incoming streams without fixed batch boundaries, balancing throughput against TTFT. The key trade‑off is: larger batches increase tokens/sec but waiting too long to fill them hurts first‑token latency.
- **Reading**:
  - LLM serving guides on dynamic batching vs latency – framing batch size/interval as a TTFT vs throughput trade‑off.  
  - vLLM blog posts showing how continuous batching + PagedAttention reach much higher throughput (often 10–20× vs naive per‑request HF loops) while still streaming first tokens quickly.

---

## Off-Hours Reading (Optional)

- vLLM paper – **PagedAttention & KV cache** sections  
  → For interpreting `max-model-len` vs GPU memory curves and why KV behaves as it does.

- A short **jemalloc vs glibc** article  
  → To reason about allocator behavior, fragmentation, and when jemalloc tends to help.

- Any **“first-token latency in LLM serving”** doc (vLLM / TGI / similar)  
  → To tie cold/warm behavior to real SLOs (TTFT, warm pools, pre-warming strategies).

For additional tuning ideas on cold vs warm behavior and page cache, see:

- **[Caching, Cold vs Warm Loads & Tunables](./theory/caching_cold_warm_loads.md)** – practical knobs across storage, OS memory, and serving patterns.

---

## Check Your Learning – Day 006 (20 Questions)

Use these as a quick self‑quiz after you’ve run the experiments and filled in the logs.

1. In your own words, why are **small language models (SLMs)** a good probe for OS and runtime behavior compared to jumping straight to a 70B model?
2. What THP modes (`always`, `madvise`, `never`) are available on your system, and which one did you settle on for inference? Why?
3. How would you explain the difference between **Transparent Hugepages** and **explicit hugepages** to a teammate who only knows “large pages make things faster”?
4. Which exact commands do you use to inspect current THP and hugepage settings on your node?
5. If you see large latency spikes during load or warm‑up, what’s one reason you might switch from `always` to `madvise` for THP?
6. What is the role of the **page cache** during model load, and how did your **cold vs warm** SLM load time measurements demonstrate this?
7. How would you design a quick experiment to verify that your model weights are actually benefiting from the page cache?
8. In your measurements, roughly what factor separated cold vs warm SLM load times, and what does that suggest about your storage stack?
9. What is the practical difference between running your SLM under **glibc malloc** vs **jemalloc** in terms of setup and expected behavior?
10. In your allocator comparison, did you see a meaningful difference in **gen latency** or **RSS**? If so, what pattern emerged?
11. When would you recommend to a team that they try swapping to jemalloc in production, and what metrics would you ask them to monitor?
12. How do you define **first‑token latency (TTFT)** vs end‑to‑end latency in your experiments for Day 006?
13. Which components do you believe dominate TTFT on your node (e.g. model load, graph/JIT warmup, KV allocations, network), and how did you infer that?
14. What specific vLLM flags or environment conditions (from Day 006 + 007) can affect TTFT even for an SLM?
15. How does increasing **`max-model-len`** change your KV cache memory footprint, and why can that reduce concurrency headroom even if you never hit the max length for most requests?
16. If you observed KV scaling behavior on this GPU, what rough “bytes per token” estimate did you derive, and how might that change your configuration choices?
17. How can continuous batching improve throughput for an SLM, and under what conditions might it hurt perceived latency?
18. If you were handed a **mysterious new GPU node**, what 3–5 quick checks from Day 006 would you run to decide if it’s “safe” for serving LLM traffic?
19. How would you explain to an SRE why THP, hugepages, and allocator choice are not just “tuning trivia” but directly affect latency SLOs?
20. Looking back at your Day 006 + 007 combined logging template, what would you change or add if you had to run the same investigations on a different cloud provider?

---

## Logging Template (For Day 006–007 Write-Up)

Use this as a template in your log files once experiments are done:

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
- days/day-007-vllm-runtime-probes/first_token_latency.md
- days/day-007-vllm-runtime-probes/kv_scaling.sh
- days/day-007-vllm-runtime-probes/kv_cache_scaling.csv
- days/day-007-vllm-runtime-probes/kv_cache_scaling_notes.md
- days/day-007-vllm-runtime-probes/batch_client.py
- days/day-007-vllm-runtime-probes/batching_benchmark.md

## Key Numbers / Metrics
- cold vs warm HF load times
- HF gen latency (glibc vs jemalloc)
- vLLM cold vs warm request times
- GPU memory vs max-model-len
- sequential_s vs concurrent_s (batch_client.py)

## Observations / Surprises
- Any allocator wins?
- How big is the cold → warm improvement in vLLM?
- KV memory scaling pattern?
- Throughput gain from continuous batching, even for SLM?
```

---

[← Day 005](../day-005-OS-and-NUMA-node-hardening/README.md) · [Days Index](../README.md)
