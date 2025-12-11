# Day 006 – Theory (Part 2): Memory Allocators & vLLM Serving

Modern LLM servers (vLLM, TGI, SGLang) run inside long‑lived Python processes that continuously allocate and free memory. Even though GPU memory is separately managed, a surprising amount of performance degradation comes from the **CPU‑side allocator**, not your model or GPU.

Most people ignore this layer, assuming “the engine handles it.” It doesn’t—and **cannot**. This is why allocator selection belongs in the Day 006 baseline.

---

## 1. How glibc malloc (ptmalloc) Breaks Down

glibc’s allocator is engineered for portability and safety, not efficiency under extreme, multi‑threaded, long‑lived memory churn. Three failure modes matter directly to vLLM.

### 1.1 Fragmentation → RSS Explosion

glibc uses many per‑thread arenas. Over time, freed blocks become stranded inside arenas, so even if your working set is small, **RSS remains inflated**.

Real‑world pattern:

- Working set: ~1 GB  
- glibc RSS: **3–4 GB**  
- jemalloc RSS: **1.1–1.2 GB**

This excess memory steals space from the page cache, competes with pinned buffers, and increases kernel pressure.

### 1.2 Poor Return Behavior to the OS

glibc rarely returns memory to the OS; it keeps arenas alive indefinitely. That means:

- The process grows and **never shrinks**.  
- Fragmentation accumulates.  
- Allocator latency spikes appear under pressure.

vLLM, being fully asynchronous and continuously batching, produces heavy small‑object churn (Python strings, metadata dicts, JSON parsing, logging, batching structures, tensor descriptors). This churn magnifies glibc’s weaknesses.

### 1.3 Latency Jitter

When glibc scavenges arenas or hits fragmentation thresholds, allocator operations take longer. This jitter shows up as:

- TTFT variance.  
- Request dispatch latency.  
- Overhead in Python‑level scheduling loops.

vLLM depends heavily on a fast, predictable control path—glibc quietly sabotages it at high load.

---

## 2. Why jemalloc Fits vLLM‑Style Workloads

jemalloc is built for **large‑memory, long‑lived, concurrent workloads**, exactly like an inference server.

### 2.1 Minimizes Fragmentation

Through size classes, slab/region design, and centralized extent management, jemalloc keeps memory tightly packed and prevents the multi‑GB RSS inflation seen in ptmalloc.

### 2.2 Actively Returns Memory to the OS

When jemalloc cannot reuse a run, it calls `madvise(DONTNEED)` to give pages back to the kernel:

- RSS drops.  
- Page cache breathing room increases.  
- Other services and pinned regions don’t starve.

For LLM servers that:

- Load large models (10–100 GB).  
- Pin additional buffers.  
- Rely on DRAM stability for DMA/IOMMU.  
- Need the page cache to stay warm for model files.

this behavior is a direct win over glibc.

### 2.3 Reduces Tail Latency

Allocator stalls decrease because there is:

- Less fragmentation.  
- Less arena contention.  
- Fewer “slow free” behaviors.  
- Faster multi‑threaded alloc/free.

Result: **tighter TTFT distributions**, especially under high‑QPS continuous batching.

---

## 3. Why vLLM Cannot Solve This for You

Many engineers assume:

> “vLLM is serious; surely they set jemalloc internally.”

They don’t—for good reasons:

- **Allocator choice is process‑wide**: switching affects Python, PyTorch CPU, HTTP server, logging, C extensions, everything in‑process.  
- **Portability and stability**: enforcing jemalloc from a library would break some environments, container images, and enterprise policies.  
- **Ownership**: allocator selection is a **platform decision**, like NUMA policy, THP policy, hugepage reservations, and IOMMU configuration.

vLLM sits on top of the node’s OS behavior; it cannot correct a bad allocator choice underneath it.

---

## 4. Division of Responsibilities

**Platform / Ops / Infra team (or you, on a single node)**:

- Set allocator policy and standardize jemalloc on inference nodes.  
- Validate RSS behavior under stress.  
- Provide a stable OS environment (THP, hugepages, IOMMU, NUMA tuned).

**Inference runtime (vLLM / TGI / SGLang)**:

- Efficient scheduling and batching.  
- GPU/CPU tensor allocation strategy.  
- PagedAttention and KV cache management.  
- Token generation pipeline and concurrency model.

**Why this separation?**

The allocator is a **global, node‑level subsystem**. The inference engine cannot safely replace it, but completely depends on its behavior.

---

## 5. One‑Sentence Summary

> glibc malloc fragments, hoards memory, and causes latency jitter in long‑running LLM servers; jemalloc bounds fragmentation, returns memory aggressively, stabilizes RSS, and reduces allocator‑driven tail latency. It is the platform’s responsibility—not vLLM’s—to choose and configure the allocator.

---

## 6. How to Operationalize It

### 6.1 For Experiments & Benchmarks

```bash
python slm_gen_latency.py
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python slm_gen_latency.py
```

Log:

- RSS.  
- `gen_latency_s`.  
- Variance.  
- Allocator CPU time (if collected).

If jemalloc tightens variance or reduces RSS (it usually does), adopt it.

### 6.2 For Production

In your service unit:

```ini
Environment="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
```

Monitor:

- RSS over days of uptime.  
- TTFT distribution.  
- p95/p99 latency.

You will almost always see stability improvements vs glibc.

