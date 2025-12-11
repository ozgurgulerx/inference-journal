# Day 006 – Theory (Part 4): KV Cache, Long Contexts & vLLM’s PagedAttention

> KV cache behavior is one of the main **hard limits** for LLM serving.  
> This note explains what the KV cache is, why naïve allocation wastes VRAM, and how vLLM’s PagedAttention changes the game.

---

## 1. What Is the KV Cache, Really?

In a Transformer, each self‑attention layer keeps two vectors per token per head:

- **K** (Key) – what this token “offers” to be attended to.  
- **V** (Value) – what information is retrieved when it is attended to.

During **prefill** (processing the prompt), for every token `t` you:

- Compute K\_t and V\_t for each attention head and layer.  
- Store them so that during **decode** (generating tokens) future tokens can attend back to all previous ones.

All of those stored K/V tensors across:

- `L` layers,  
- `H` heads,  
- `T` tokens in the sequence,

make up the **KV cache**.

Approximate memory per sequence:

```text
KV_mem_bytes ≈ 2 (K+V)
               × L (layers)
               × H (heads)
               × d_head (hidden per head)
               × T (tokens)
               × bytes_per_element (e.g. 2 for FP16)
```

Key points:

- KV cache **grows linearly with sequence length T**.  
- It scales with **layers × heads**, so bigger models grow KV even faster.  
- When serving many concurrent sequences, KV cache can easily exceed **weight memory**.

For large models, it’s common that KV memory, not model weights, becomes the **dominant VRAM consumer** at high concurrency.

---

## 2. Why Naïve KV Allocation Wastes Enormous GPU Memory

Traditional inference engines often allocate KV like this:

- One large **contiguous buffer per sequence**.  
- Sized for the **maximum context length** (e.g., always 4K or 8K tokens).  
- Allocation happens **up front**, before you know the final prompt length.

If:

- `max_seq_len = 4096`,  
- real prompt length = `200` tokens,

then roughly:

```text
used ≈ 200 / 4096 ≈ 5%
wasted ≈ 95%
```

Multiply that across many concurrent sequences and VRAM disappears very quickly.

Measurements from vLLM’s evaluations show:

- Only **20–38%** of fixed‑size KV allocations contained real data under realistic traffic.  
- The **rest was dead padding**.  
- This waste:
  - limits the model sizes you can serve,  
  - cuts the effective batch size,  
  - reduces concurrency,  
  - and throttles overall throughput.

This is why engines without paged KV caching hit OOM much earlier than intuition says they should.

---

## 3. vLLM’s Breakthrough: PagedAttention

PagedAttention treats KV memory like an OS treats RAM:

> The KV cache is broken into fixed‑size **pages** that can be allocated, reused, and freed independently.

Key properties:

- Pages are **not required to be contiguous** in GPU memory.  
- A sequence gets only as many pages as it actually needs.  
- When a sequence finishes, **only its pages** are freed, not some giant fixed buffer.  
- Reused pages dramatically cut internal fragmentation.  
- Empirically, KV waste drops from ~60–80% → **<4%**.

Conceptually:

```text
Naïve:
  [Seq 1: 4K buffer]
  [Seq 2: 4K buffer]
  [Seq 3: 4K buffer]

PagedAttention:
  Page 0 → part of Seq 1
  Page 1 → part of Seq 1
  Page 2 → part of Seq 2
  Page 3 → part of Seq 3
  Page 4 → part of Seq 2
  ...
```

The runtime:

- Allocates pages when new tokens are created.  
- Tracks which pages belong to which sequence.  
- Reclaims pages immediately when a request completes.

> It’s essentially a **GPU‑side virtual memory manager for KV**, built specifically for the decoder workload.

---

## 4. KV Cache Scaling & Long Context Footprint

Even with PagedAttention, the **physics** are the same:

```text
More context (T) → more KV pages → more VRAM per active sequence.
```

What changes is **how much of that VRAM is actually used**.

### 4.1 max-model-len as a Capacity Commitment

In vLLM, `--max-model-len` (or `max_model_len` in configs) controls the **maximum context** allowed for requests.

- In naïve systems, this value determines how big each sequence’s fixed buffer is.  
- You pay the memory cost **up front**, even when prompts are short.  
- Increasing max length (e.g. 4K → 16K) silently:
  - reserves far more KV memory per sequence,  
  - shrinks your safe concurrency region,  
  - and can cause OOMs even with short prompts.

PagedAttention allocates KV pages on demand; it doesn’t pre‑reserve the full worst case. But:

- More tokens → more pages → more VRAM per active sequence.  
- Very long contexts still push you up against VRAM limits at high concurrency.

Day 006/007’s `kv_scaling.sh` script is designed to **map this curve** on your hardware:

- For different `max-model-len` values (e.g. 512, 1024, 2048, 4096, 8192, …).  
- Record GPU memory usage after load.  
- Understand where your GPU hits uncomfortable thresholds.

---

## 5. Empirical Findings from vLLM’s Evaluations

From the vLLM paper/blog (13B‑scale model on a 40 GB A100):

- A 2048‑token KV cache could consume ~**30% of VRAM** by itself.  
- With naïve fixed buffers, only **20–38%** of that KV memory held real data; the rest was padding.  
- With PagedAttention:
  - Waste dropped to **<4%**.  
  - Effective concurrency and throughput increased sharply.  
  - Longer contexts became feasible without immediate OOM.

This is why vLLM can:

- Serve **larger contexts** on the same GPU.  
- Run **more concurrent requests**.  
- Maintain **higher tokens/sec** before hitting memory limits.

KV efficiency is not a micro‑optimization; it is a **primary scaling lever**.

---

## 6. Why This Matters for Inference Engineering

From an inference engineer’s perspective:

- You **cannot** set `max-model-len` blindly.
  - It determines how far up the KV curve each request pushes you.  
  - It shapes your safe concurrency region.

- KV footprint, not FLOPs, often dictates the **real concurrency limit**.
  - A GPU might have plenty of compute headroom but run out of KV memory.  
  - This manifests as OOMs or aggressive request shedding at high QPS.

- Fragmentation matters.
  - Naïve engines pay a huge “hidden tax” in unused KV padding.  
  - PagedAttention removes most of that tax, exposing the **true** capacity of your GPU.

- SLM probes (Day 006/007) are a **cheap way** to map the KV curve.
  - A 1–3B model has the same scaling law for KV vs context, just with smaller constants.  
  - You can safely explore 4K, 8K, 16K, 32K contexts without dragging 70B weights around.

- Choosing context lengths strategically can unlock batching.
  - Sometimes dropping from 32K → 16K context on a given GPU class
    - frees enough KV memory
    - to double safe concurrency
    - and significantly increase tokens/sec at your target SLO.

For production, your **KV cache policy** (max length, paging strategy, concurrency caps) is as important as:

- choosing the model,  
- setting quantization, or  
- picking the GPU SKU.

---

## 7. How Day 006/007 Use This in Practice

Day 006–007 labs turn this theory into concrete measurements:

- `kv_scaling.sh` – runs vLLM with different `max-model-len` values and records GPU memory usage.  
- `kv_cache_scaling.csv` – gives a per‑GPU curve: `max_model_len → gpu_mem_used_mb`.  
- `kv_cache_scaling_notes.md` – where you interpret:
  - how linear the curve is,  
  - where headroom disappears,  
  - how this influences safe concurrency / batching for that GPU class.

Combine this with:

- SLM probes (`slm_load.py`, `slm_gen_latency.py`).  
- allocator tuning (jemalloc).  
- hugepages/THP settings.

and you get a **full stack picture**:

- OS memory behavior.  
- allocator behavior.  
- KV cache scaling.  
- batching efficiency.

All before you ever benchmark a 13B–70B model on that node.

