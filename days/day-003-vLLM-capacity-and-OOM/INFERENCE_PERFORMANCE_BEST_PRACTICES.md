# LLM Inference Performance Engineering: Best Practices

If training is how your model learns, inference is how your business pays for it.

For large language models (LLMs), most of the real-world cost and user experience is dominated by **inference**: how fast you can serve tokens, how many users you can handle, and how efficiently you burn GPU/TPU time. This post lays out a **practical, systems-first playbook** for LLM inference performance engineering.

We'll build on ideas popularized by MosaicML/Databricks, NVIDIA, vLLM and others, and translate them into concrete decisions you can make about models, hardware, and serving architecture.

---

## 1. Mental Model: What Actually Happens During LLM Inference?

Before doing any optimization, you need a simple but correct mental model of how decoder-only LLMs generate text:

- **Prefill phase**  
  - The input prompt is ingested in parallel.  
  - This is more **compute-bound**: large matmuls over many tokens, typically achieving higher FLOP utilization.

- **Decode phase**  
  - Tokens are generated one by one in an autoregressive loop.  
  - Each step reuses past keys/values from attention via **KV caching**.  
  - This is mostly **memory-bandwidth-bound**: performance depends more on how quickly you can move weights + KV cache than on raw FLOPs.

**Implication**: Different stages want different optimizations and sometimes different hardware. Trying to optimize both as if they were the same is a classic mistake.

---

## 2. Metrics That Actually Matter

If you optimize without good metrics, you're just doing expensive cargo cult.

A clean set of serving metrics (following the MosaicML/Databricks framing):

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **TTFT** | Time To First Token | How quickly a user sees *any* output. Critical for interactive UX (chat, copilots). |
| **TPOT** | Time Per Output Token | Steady-state speed per user (e.g., 50ms/token = 20 tok/s). "The model feels fast/slow." |
| **E2E Latency** | `TTFT + TPOT × #output_tokens` | Total request time. For long answers, TPOT dominates. |
| **Throughput** | Total output tokens/sec across all requests | Your **capacity** and cost lever. |
| **Cost/1K tokens** | Infra cost ÷ tokens served | The business alignment metric. |
| **MBU** | Model Bandwidth Utilization | `(data moved per token) / (theoretical memory bandwidth)`. High = efficient; low = software/topology issues. |

**Best practice:**
1. Pick **one primary optimization target** per system (e.g., "minimize p95 latency for chat" or "maximize throughput/$ for offline batch").
2. Instrument **TTFT, TPOT, throughput, MBU, cost/1K tokens** and optimize against that set.

---

## 3. Hardware-Level Best Practices

### 3.1 Pick Hardware for Memory Bandwidth, Not Just FLOPs

For decode, LLMs are usually **memory-bandwidth bound**, so:

- Prefer GPUs/accelerators with **high HBM bandwidth** (e.g., H100 over A100 over older cards).
- When comparing multi-GPU setups, realize that doubling FLOPs often doesn't halve latency; interconnect + MBU kill the linearity.

**Rule of thumb:**
- For *smaller* models (e.g., 7B) and moderate context, a **single strong GPU** with high bandwidth often beats many weaker GPUs with poor links.
- For *very large* models (e.g., 70B+), you need multi-GPU but expect **sub-linear scaling**.

### 3.2 Understand Topology

Not all "8× GPU" servers are equal:

- NVLink full-mesh > partial NVLink > PCIe-only with cross-CPU hops.
- Tensor parallelism across poorly connected GPUs can murder MBU due to tiny, frequent cross-GPU transfers.

**Best practice:**
- Start by fitting the model on **as few GPUs as possible**, then scale out only when QPS demands it.
- Avoid over-sharding small models just to "use all the GPUs" – it can lower MBU and degrade performance.

---

## 4. Model-Level Best Practices

### 4.1 Right-Size the Model

A 70B model might be overkill for:
- Simple classification, routing, or extraction.
- "Copilot" flows where a 7B–14B model with good prompting and tools is enough.

Smaller models give you:
- Lower memory footprint → cheaper hardware, easier deployment.
- Better batch capacity → higher throughput.

**Strategy:** Benchmark **quality vs cost** across a few candidate sizes and **pick the smallest model that meets quality requirements**.

### 4.2 Quantization, Done Carefully

Quantization is the main lever to bring costs down:

| Type | Description |
|------|-------------|
| Weight-only | INT8, INT4 (GPTQ, AWQ, SmoothQuant) |
| Activation + KV cache | Compress KV cache for better batch sizes and longer contexts |

**Trade-off:** Every bit you drop reduces bandwidth and memory, but risks quality loss or instability on edge cases.

**Best practice:**
- Start with **8-bit** weight-only quantization on a smaller test workload.
- Validate with **task-specific evals**, not just generic benchmarks.
- Only push to **4-bit** when you really need consumer-grade hardware or extremely low cost.

---

## 5. Systems-Level Best Practices

### 5.1 Use a Real Inference Engine

Plain HF Transformers + naïve batching is usually not enough.

Modern inference engines give you:

| Feature | Benefit |
|---------|---------|
| **Continuous batching** | Requests join/leave batches every decode step. Up to 10× throughput vs naïve batching. |
| **Paged KV memory** | Like virtual memory for KV cache (vLLM's PagedAttention). Prevents fragmentation. |
| **Optimized kernels** | FlashAttention, fused layernorms, fused MLPs, CUDA graphs. |
| **Built-in quantization + parallelism** | GPTQ/AWQ/INT8 integrations and tensor/pipeline parallel primitives. |

> **Start with vLLM or TensorRT-LLM (served via Triton/NIM or similar), not raw Transformers.**

### 5.2 Batching: Static, Dynamic, Continuous

| Strategy | Use Case | Notes |
|----------|----------|-------|
| **Static** | Offline, homogeneous workloads | Near-optimal throughput if you control the batch. |
| **Dynamic** | Moderate QPS, similar lengths | Server groups arriving requests at fixed windows. |
| **Continuous** | Online shared services | Re-packs every decode iteration. **10–20× gains** over naïve decoding. |

**Practical strategy:**
- For **online shared services** (chat, agents): use **continuous batching** with configurable max batch size + queue delay budget.
- For **offline processing** (PDF summarization, log enrichment): use **large static batches** and maximize tokens/sec/$.

### 5.3 Prefill vs Decode Optimization

| Technique | What It Does |
|-----------|--------------|
| **Chunked prefill** | Split long prompts into chunks to avoid monopolizing the GPU. Allows overlapping prefill/decode. |
| **Disaggregated inference** | Run prefill on compute-heavy device, decode on bandwidth-optimized device. Advanced but powerful. |

If you don't want this complexity, choose an engine that **already implements chunked prefill & efficient scheduling**.

### 5.4 Parallelism Strategy

| Type | When to Use |
|------|-------------|
| **Data/replica scaling** | Model fits in one GPU. Simplest, scales linearly with QPS. **Prefer this first.** |
| **Tensor parallelism** | Model doesn't fit in one GPU. Keep degree low (2–4), topology-aware. |
| **Pipeline parallelism** | Very large models or custom deployments. Adds latency and complexity. |

---

## 6. Algorithmic & Decoding-Level Tricks

### 6.1 Limit Output Length and Stop Conditions

- Use **task-appropriate max tokens** – don't let chat completions default to 1024 when 128 is enough.
- Provide **domain-specific stop sequences** (e.g., `"\n\nUser:"`, `"</tool>"`) so generations end early.

**Impact:** Direct, immediate improvements on latency and cost with zero model changes.

### 6.2 Speculative and Assisted Decoding

| Technique | How It Works |
|-----------|--------------|
| **Speculative decoding** | Small draft model proposes multiple tokens; large model verifies. Boosts throughput if draft is accurate. |
| **Lookahead / tree-based** | Explore multiple token branches in parallel. Still evolving. |

**Speculative decoding is usually the safest first advanced technique to try.**

### 6.3 Cache Reuse and Prefix Caching

Many workloads repeat prefixes:
- System prompts and boilerplate instructions.
- Common context for a tenant or workspace.
- Shared retrieved documents across nearby requests.

| Technique | Benefit |
|-----------|---------|
| **Prefix caching** | Precompute KV cache for common prefix, reuse across requests. |
| **Session-level caching** | Reuse KV cache across conversation turns instead of re-prefilling. |

This reduces TTFT and overall cost, especially in agentic or multi-turn settings.

---

## 7. Observability, Benchmarking, and Feedback Loops

You can't optimize what you can't see.

### 7.1 What to Measure

| Layer | What to Track |
|-------|---------------|
| **Synthetic benchmarks** | Fixed prompts, fixed output length, controlled concurrency. Sweep batch size, context length, model size, quantization, GPU count. Record TTFT, TPOT, throughput, MBU, cost. |
| **Real-world traces** | Sample real traffic: actual prompt/output length distributions, user concurrency patterns. Replay against candidate configs. |
| **Production telemetry** | Per-request metrics (latency breakdowns, queue time, tokens in/out, errors). Track p50/p95/p99 vs time. Watch for MBU regressions. |
| **Quality guardrails** | For any perf tweak, run **automated quality evals**. Ideally task-specific, not just generic benchmarks. |

### 7.2 Automate It

**Best practice:** Build a **"perf pipeline"** similar to CI: when you change model, engine, or config, it runs a benchmark suite and gives a clear diff vs baseline.

---

## 8. A Practical Checklist

If you just want a minimal "do this first" list:

```
1. Clarify your main objective
   - Latency SLO? Throughput/$? Both with different tiers?

2. Pick an inference engine
   - Start with vLLM or TensorRT-LLM + Triton/NIM
   - Turn on continuous batching and paged KV memory

3. Right-size the model
   - Benchmark small, medium, large for your workload
   - Pick the smallest that meets quality

4. Quantize conservatively first
   - Try 8-bit weights only; validate quality with task-specific tests
   - Only push to 4-bit if you really need it

5. Tune batch size vs latency
   - Online: max batch size that keeps p95 within SLO under expected QPS
   - Offline: push batch size until throughput stops improving

6. Optimize decoding behavior
   - Set realistic max tokens and stop sequences
   - Enable prefix/session caching when possible
   - Experiment with speculative decoding if supported

7. Instrument MBU & end-to-end metrics
   - Track TTFT, TPOT, MBU, throughput, cost/1K tokens
   - Watch for regressions when changing hardware, topology, or models
```

---

## Summary

Done well, LLM inference performance engineering is less about clever hacks and more about **good models, good engines, correct mental models, and disciplined measurement**. Once those are in place, all the fancy tricks—paged attention, continuous batching, speculative decoding, disaggregated inference—become straightforward, composable tools instead of mysterious black magic.

---

*This guide mirrors and extends the original Databricks "LLM Inference Performance Engineering: Best Practices" blog and combines it with patterns from vLLM, TensorRT-LLM, and modern inference engines.*
