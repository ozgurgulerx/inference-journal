# Day 006 – Theory: Pages, Huge Pages, DMA & First-Token Latency

This note expands the Day 006 labs with a deeper look at:
- CPU pages, TLBs, and huge pages  
- THP vs explicit huge pages  
- How huge pages interact with GPU DMA / IOMMU  
- Practical takeaways for LLM serving with vLLM

---

## Memory Access Diagram – CPU vs GPU DMA Paths

![Memory Access: CPU vs GPU DMA Paths](../assets/cpu_gpu_transfer.png)

**Notes**:
- CPU MMU and IOMMU are **separate** translation systems.  
- DMA does **not** use CPU page tables; it uses the IOMMU.  
- Huge pages reduce the number of translations required in both MMUs.  
- NUMA affects **where** DRAM is located; huge pages affect **how** it is mapped.

---

## 1. Pages, TLBs, and Why Page Size Matters

- A **page** is the unit of virtual→physical mapping for the CPU MMU.
  - Typical size: **4 KB** on x86‑64.
  - Each page has a Page Table Entry (PTE) mapping `VA_page → PA_frame + flags`.
- The **TLB (Translation Lookaside Buffer)** caches recent VA→PA translations.
  - On a **TLB hit**, translation is a couple of cycles.  
  - On a **TLB miss**, the MMU walks multi‑level page tables (several memory loads) → tens of cycles and pipeline stalls.
- For large regions (e.g., 40–80 GB of model weights), 4 KB pages explode page count and drive:
  - Higher TLB miss rates.  
  - More page‑table walks.  
  - Extra CPU overhead during scans over weights, KV buffers, or activation tensors.

**Mental model**: Big models over many 4 KB pages = lots of translations. Reducing page count reduces translation overhead.

---

## 2. Huge Pages: What They Actually Are

- **Huge pages** are just larger hardware page sizes:
  - 2 MB (“huge”) and sometimes 1 GB (“gigantic”) on x86‑64.
- They are **native MMU pages**, not something layered on top of 4 KB pages:
  - The MMU interprets one PTE as covering 2 MB instead of 4 KB.

**Consequences**:

- 2 MB vs 4 KB → **512× fewer pages** for the same region.  
- Smaller page tables and fewer page‑table levels touched.  
- Fewer TLB entries required → lower TLB miss probability.  
- Lower overhead when scanning large contiguous arrays or mmapped model shards.

**Mental model**:

> Same TLB size, much larger “coverage” per entry → better for big, streaming workloads.

---

## 3. THP vs Explicit Huge Pages (HugeTLB)

### 3.1 Transparent Huge Pages (THP)

- Kernel attempts to **automatically coalesce** neighboring 4 KB pages into 2 MB pages.  
- Modes: `always`, `madvise`, `never`.
- Good for **throughput‑oriented** workloads with large, hot regions and less concern about latency.

From Red Hat / Netdata:

- THP can **reduce CPU overhead and improve throughput** for memory‑intensive workloads.  
- But THP uses background **compaction/migration** (`khugepaged`, defrag), which:
  - Burns CPU cycles.  
  - Can cause **latency spikes** and jitter when the kernel scans and moves pages.
- In practice, operators often:
  - Avoid `always` for low‑latency services.  
  - Prefer `madvise` or even `never` for latency‑sensitive databases, JVMs, and LLM serving.

### 3.2 Explicit Huge Pages (HugeTLB / hugetlbfs)

- Admin pre‑reserves a pool via `nr_hugepages` and optionally mounts `hugetlbfs`.  
- Applications or allocators explicitly request hugepage‑backed memory.  
- No background compaction; pages are carved out up front.

From Red Hat:

- Explicit huge pages fit workloads where:
  - Size and lifetime are well understood.  
  - Predictable latency matters more than automatic convenience.
- Downsides:
  - Memory is effectively **hard‑reserved**: unused huge pages are wasted RAM.  
  - Requires planning and capacity management.

**Inference‑friendly starting point**:

```bash
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo never   > /sys/kernel/mm/transparent_hugepage/defrag
echo <N>     > /proc/sys/vm/nr_hugepages    # small, deliberate pool
```

> Let THP help only where you opt in (`madvise`), and keep a small explicit hugepage pool for heavyweight regions (model files, pinned buffers).

---

## 4. How Huge Pages Interact with DMA & IOMMU (GPU Path)

You don’t just care about the CPU MMU; you also care about the **IOMMU** serving GPU **DMA**.

### 4.1 Two Translation Worlds

- **CPU path**  
  `VA → CPU MMU/TLB → PA → DRAM`

- **GPU DMA path**  
  `IO‑VA → IOMMU/IOTLB → PA → DRAM → PCIe/NVLink → GPU HBM`

DMA does **not** use CPU page tables. The IOMMU has its own page tables and **IOTLB**.

### 4.2 Pinned (Page‑Locked) Memory

- DMA requires **pinned memory**: physical frames the OS will not migrate or swap.  
- When you pin a large region, the kernel must:
  - Walk all pages in that region.  
  - Lock them.  
  - Install IOMMU mappings: `IO‑VA → PA`.

### 4.3 Where Huge Pages Help on the GPU Side

For a 16 GB pinned region:

- With 4 KB pages:  
  `16 GB / 4 KB ≈ 4,194,304` pages → ~4M IOMMU entries.  
- With 2 MB huge pages:  
  `16 GB / 2 MB = 8,192` pages → 8K entries.

**Benefits**:

- ~**512× fewer IOMMU mappings** to create during pinning → faster pin setup.  
- **Less IOTLB churn** during DMA → fewer IOMMU page walks.  
- Lower tail latency and less jitter during large transfers (model weight upload, big batch prefill).

**Net effect for vLLM / LLM serving**:

> Huge pages don’t change what you copy; they make **address translation + pinning cheaper**, for both CPU and GPU DMA.

---

## 5. Practical Takeaways for vLLM / LLM Serving

You can compress the theory into a few operational rules:

1. **Understand pages & TLBs**
   - 4 KB pages → high page count for big models → more TLB and IOTLB pressure.  
   - 2 MB pages mitigate this at both CPU MMU and IOMMU levels.

2. **Use THP carefully**
   - `always` can increase throughput but cause **latency spikes** due to compaction/migration.  
   - For latency‑sensitive inference, treat THP as an *opt‑in tool* (`madvise`), not a global magic switch.

3. **Reserve a small explicit hugepage pool**
   - Back model files or large pinned buffers with explicit hugepages when possible.  
   - Gains: consistent performance without background defrag jitter, at the cost of some reserved RAM.

4. **Remember the two MMUs**
   - CPU MMU ↔ CPU TLB (for CPU loads).  
   - IOMMU ↔ IOTLB (for GPU DMA).  
   - Both benefit from fewer, larger pages; neither shares page tables.

5. **Watch TTFT & tail latency, not just throughput**
   - Throughput can improve while **latency gets worse** if THP compaction or IOMMU churn kicks in.  
   - For LLM serving, your KPI is the **first‑token latency distribution**, especially under load; hugepages + sane THP settings are levers on the long tail.

---

## 6. Why Memory Allocators Matter for vLLM Serving (and Who Owns the Decision)

Modern LLM servers (vLLM, TGI, SGLang) run inside long‑lived Python processes that continuously allocate and free memory. Even though GPU memory is separately managed, a surprising amount of performance degradation comes from the **CPU‑side allocator**, not your model or GPU.

Most people ignore this layer, assuming “the engine handles it.” It doesn’t—and **cannot**. This is why allocator selection belongs in the Day 006 baseline.

### 6.1 How glibc malloc (ptmalloc) Breaks Down

glibc’s allocator is engineered for portability and safety, not efficiency under extreme, multi‑threaded, long‑lived memory churn. Three failure modes matter directly to vLLM:

#### Fragmentation → RSS Explosion

glibc uses many per‑thread arenas. Over time, freed blocks become stranded inside arenas, so even if your working set is small, **RSS remains inflated**.

Real‑world pattern:

- Working set: ~1 GB  
- glibc RSS: **3–4 GB**  
- jemalloc RSS: **1.1–1.2 GB**

This excess memory steals space from the page cache, competes with pinned buffers, and increases kernel pressure.

#### Poor Return Behavior to the OS

glibc rarely returns memory to the OS; it keeps arenas alive indefinitely. That means:

- The process grows and **never shrinks**.  
- Fragmentation accumulates.  
- Allocator latency spikes appear under pressure.

vLLM, being fully asynchronous and continuously batching, produces heavy small‑object churn (Python strings, metadata dicts, JSON parsing, logging, batching structures, tensor descriptors). This churn magnifies glibc’s weaknesses.

#### Latency Jitter

When glibc scavenges arenas or hits fragmentation thresholds, allocator operations take longer. This jitter shows up as:

- TTFT variance.  
- Request dispatch latency.  
- Overhead in Python‑level scheduling loops.

vLLM depends heavily on a fast, predictable control path—glibc quietly sabotages it at high load.

### 6.2 Why jemalloc Fits vLLM‑Style Workloads

jemalloc is built for **large‑memory, long‑lived, concurrent workloads**, exactly like an inference server.

#### Minimizes Fragmentation

Through size classes, slab/region design, and centralized extent management, jemalloc keeps memory tightly packed and prevents the multi‑GB RSS inflation seen in ptmalloc.

#### Actively Returns Memory to the OS

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

#### Reduces Tail Latency

Allocator stalls decrease because there is:

- Less fragmentation.  
- Less arena contention.  
- Fewer “slow free” behaviors.  
- Faster multi‑threaded alloc/free.

Result: **tighter TTFT distributions**, especially under high‑QPS continuous batching.

### 6.3 Why vLLM Cannot Solve This for You

Many engineers assume:

> “vLLM is serious; surely they set jemalloc internally.”

They don’t—for good reasons:

- **Allocator choice is process‑wide**: switching affects Python, PyTorch CPU, HTTP server, logging, C extensions, everything in‑process.  
- **Portability and stability**: enforcing jemalloc from a library would break some environments, container images, and enterprise policies.  
- **Ownership**: allocator selection is a **platform decision**, like NUMA policy, THP policy, hugepage reservations, and IOMMU configuration.

vLLM sits on top of the node’s OS behavior; it cannot correct a bad allocator choice underneath it.

### 6.4 Division of Responsibilities

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

### 6.5 One‑Sentence Summary

> glibc malloc fragments, hoards memory, and causes latency jitter in long‑running LLM servers; jemalloc bounds fragmentation, returns memory aggressively, stabilizes RSS, and reduces allocator‑driven tail latency. It is the platform’s responsibility—not vLLM’s—to choose and configure the allocator.

### 6.6 How to Operationalize It

**For experiments/benchmarks**:

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

**For production**:

In your service unit:

```ini
Environment="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
```

Monitor:

- RSS over days of uptime.  
- TTFT distribution.  
- p95/p99 latency.

You will almost always see stability improvements vs glibc.

---

Use this theory file as a deeper reference alongside the Day 006 labs:
- `LOG_tier01.md` – THP/hugepage experiments.  
- `LOG_tier02.md` – allocator + first‑token measurements.  
- `LOG_tier03.md` – KV scaling + micro‑batching.
