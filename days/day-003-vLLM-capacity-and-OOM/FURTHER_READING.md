# Further Reading: LLM Inference Benchmarking & Performance Engineering

This page collects the main references and patterns behind the metrics and instincts we've been using (TTFT, E2E, tokens/s, concurrency, queue vs compute, etc.).

---

## 1. Core Articles on LLM Inference & Benchmarking

📘 **[INFERENCE_PERFORMANCE_BEST_PRACTICES.md](INFERENCE_PERFORMANCE_BEST_PRACTICES.md)** — Local guide  
A comprehensive systems-first playbook covering hardware, model, systems, and algorithmic best practices. Synthesizes Databricks/MosaicML, NVIDIA, and vLLM patterns into actionable decisions.

**LLM Inference Performance Engineering: Best Practices — Databricks Mosaic AI**  
Great end-to-end overview of how output length, context length, batching, and hardware choices affect latency and throughput. Also sets up the basic "tokens/s per GPU at an SLO" mental model.  
👉 https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices

**LLM Inference Benchmarking: Fundamental Concepts — NVIDIA Technical Blog**  
Clean breakdown of latency vs throughput metrics, how to think about tokens/s, and how to design fair, comparable benchmarks for LLM inference.  
👉 https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/

**LLM Inference Benchmarking Guide: NVIDIA GenAI-Perf & NIM — NVIDIA**  
Shows how to operationalize those concepts with a concrete framework (genai-perf) and measure NIM/LLM deployments in a standardized way.  
👉 https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/

**LLM Inference Benchmarking: Performance Tuning with TensorRT-LLM — NVIDIA**  
Dives into how to tune TensorRT-LLM deployments: batching, kernels, and configuration for better latency/throughput on GPU.  
👉 https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/

---

## Key Concept: Load Control Parameters

Understanding how to control and interpret load is fundamental to benchmarking. These parameters directly map to what you configure in `vllm_chat_bench.py`.

### Concurrency

**Concurrency N** is the number of concurrent users, each having one active request, or equivalently the number of requests concurrently being served by an LLM service. As soon as each user's request receives a complete response, another request is sent to ensure that at any time the system has exactly N requests.

> **Tool difference**: LLMPerf sends requests in batches and drains between batches (concurrency drops to 0 at batch end). GenAI-Perf maintains exactly N active requests throughout. The Day 003 harness uses a semaphore to maintain steady concurrency.

### Concurrency vs Max Batch Size

**Max batch size** defines the maximum requests the inference engine can process simultaneously. If concurrency exceeds `max_batch_size × num_replicas`, requests queue—and you'll see TTFT increase due to queueing delay.

**Practical guidance**: Sweep concurrency from 1 up to slightly above max batch size. Beyond that, throughput saturates while latency keeps climbing.

### Request Rate (Alternative to Concurrency)

**Request rate** controls load by determining how fast new requests arrive (e.g., 10 req/s). Can be constant or Poisson-distributed.

> **Recommendation**: Prefer concurrency over request rate. With request rate, outstanding requests can grow unbounded if arrival exceeds throughput—harder to reason about.

### Other Parameters That Affect Benchmarks

| Parameter | What It Does | Benchmarking Guidance |
|-----------|--------------|----------------------|
| `ignore_eos` | Continue generating past EOS token | Set `True` for consistent output length |
| Sampling method | Greedy, top_p, top_k, temperature | Stay consistent within a benchmark run |

Greedy sampling (select highest logit) is fastest—skips probability normalization. Different methods have different compute costs.

---

## Mental Model: LLM Inference Benchmarking from First Principles

This section distills the core concepts into a framework you can use for any benchmarking exercise.

### 1. What Are We Actually Benchmarking?

At inference time, an LLM server is basically a **matmul engine + KV-cache factory + scheduler**.

Benchmarking = measuring how well that system turns input tokens into output tokens under constraints (latency, cost, QoS), for a defined workload.

From first principles, you care about:
- **How fast?** (latency, tokens/sec)
- **How many?** (requests/sec, concurrent users)
- **How expensive?** (cost/1M tokens, cost/user, GPUs needed)
- **At what quality/SLA?** (does it still pass accuracy thresholds / business constraints)

Everything else is detail.

### 2. Core Metrics (The Ones That Actually Matter)

#### 2.1 Latency Metrics

Split latency into **prefill** and **decode**:

| Metric | Definition | Dominated By |
|--------|------------|--------------|
| **TTFT** | Request sent → first output token | Prefill compute, scheduling/queuing, network |
| **ITL/TPOT** | Time between output tokens (decode loop) | Decode compute, KV-cache access |
| **E2E** | Request → full response complete | TTFT + (ITL × output_tokens) + overhead |

Always look at **P50 / P90 / P95 / P99** latency—not just averages.

#### 2.2 Throughput & Capacity

| Metric | Definition | Use Case |
|--------|------------|----------|
| **TPS** | Output tokens/sec across all users | "How much text can this cluster emit?" |
| **RPS** | Completed requests/sec | APIs with fixed prompt/output lengths |
| **Goodput** | Throughput subject to SLA (e.g., "RPS meeting P95 < 1s") | Production capacity planning |

TPS and RPS define capacity planning: *"How many GPUs do I need to serve X req/s at Y latency bound?"*

#### 2.3 Cost Metrics

Think in:
- Cost per 1M tokens (hardware + infra + license)
- Cost per request / per active user
- Cost per unit of goodput (e.g., "$/1000 SLA-compliant requests")

Most infra decisions (quantization, batching, KV offload) are about moving along the **latency ↔ throughput ↔ cost frontier**.

#### 2.4 Quality & Correctness Constraints

MLPerf Inference enforces "within X% of reference accuracy"—you can only publish performance numbers if you preserve task accuracy. For your experiments: **only compare configs at the same model + decoding params + quality bar**.

### 3. Fundamental Axes: What You Must Specify

A benchmark is meaningless unless you pin down:

| Axis | What to Specify |
|------|-----------------|
| **Model** | Size, architecture (dense/MoE), precision (FP16/FP8/NF4), quant method (AWQ/GPTQ), KV-cache behavior |
| **Hardware** | GPU type/count, memory, NVLink/PCIe, tensor/pipeline parallelism |
| **Software** | Runtime (vLLM, TRT-LLM, TGI), scheduler, serving mode, compiler optimizations |
| **Workload** | Prompt/output length distributions, concurrency, traffic pattern, streaming mode, decoding params |
| **Scenario** | Online (optimize TTFT+ITL), Offline (optimize TPS, tolerate latency), Hybrid (goodput under sharing) |

### 4. Benchmarking Methodology: How to Not Lie to Yourself

**Key principles:**

1. **Warmup & steady state** — Discard cold-start runs; ensure models loaded, caches primed
2. **Client vs server timing** — Server-side for kernel tuning, client-side for SLOs
3. **Realistic prompt mixes** — Use prompts that represent your product, not "Hello world"
4. **Distribution, not single numbers** — Report P50/P90/P95/P99 and latency-vs-concurrency curves
5. **Repeatability** — Fix seeds, decoding params, versions; run multiple trials

### 5. First-Principles Performance Model

Why the numbers look the way they do:

```
Prefill FLOPs  ≈ O(L_prompt × d × n_layers)
Decode FLOPs/token ≈ O(L_total × d × n_layers)  # L_total grows as you generate
```

**Implications:**
- TTFT increases ~linearly with prompt length
- Time per token increases ~linearly with context length (without sliding window tricks)
- Bigger models → more FLOPs/token

**Hardware ceilings:**
- Peak FLOPs of GPU(s)
- Memory bandwidth + KV-cache traffic
- Interconnect bandwidth (NVLink, PCIe, NIC) for multi-GPU

Record benchmark numbers (>1000 tok/s/user) are just pushing harder against FLOP + bandwidth ceilings with better kernels and scheduling.

### 6. Trade-offs to Always Think In

When looking at any benchmark chart, ask:

| Trade-off | Example |
|-----------|---------|
| **Latency ↔ Throughput** | Larger batch → higher TPS but worse TTFT |
| **Cost ↔ QoS** | Aggressive quant → cheaper, more TPS, but maybe worse quality/tail latency |
| **Infra ↔ Algorithm** | Better scheduling reduces GPU count without changing model |

A good benchmark report makes these curves explicit.

### 7. Common Pitfalls / Anti-Patterns

Red flags:
- Reporting only average latency with no concurrency / P95 / P99
- Comparing setups with different prompt/output lengths
- Comparing different decoding params without mentioning them
- Mixing cold start and warm state numbers
- Ignoring tokenization, network, or gateway overhead
- Using toy prompts when real workload is 2k-context RAG

### 8. Practical Checklist for Your Own Benchmarks

```
1. Fix: model + precision + decoding params
2. Choose 1–2 representative workloads (chat, RAG, batch)
3. Decide primary SLO (e.g., "P95 TTFT < 500ms at 200 RPS")
4. Sweep over: concurrency, batch size, quantization/KV policies
5. Record per point:
   - TTFT (P50/P95), E2E latency
   - TPS, RPS, goodput
   - GPU utilization, memory usage
   - Cost estimate
6. Plot:
   - Latency vs concurrency
   - TPS vs concurrency
   - Cost vs goodput
```

That's enough to make real engineering decisions.

---

## 2. Serving Systems & Runtime Design

**vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention**  
Explains why naive HF-style serving wastes KV cache and how PagedAttention plus better memory management unlock a big tokens/s jump at the same latency.  
👉 https://blog.vllm.ai/2023/06/20/vllm.html

**vLLM Joins the PyTorch Ecosystem (Byte-Sized AI)**  
High-level view of vLLM's role as a general-purpose serving runtime in the PyTorch ecosystem and how it integrates with real products like Amazon Rufus.  
👉 https://medium.com/byte-sized-ai/all?topic=vllm

**vLLM GitHub Repository**  
Worth reading just for the benchmark scripts and configuration patterns: how they structure runs, what they log, and how they parameterize batch sizes, sequence lengths, and hardware.  
👉 https://github.com/vllm-project/vllm  
Docs: https://docs.vllm.ai/

---

## 3. Benchmarking Frameworks & Surveys

**Introduction to LLM Inference Benchmarking — Yuchen Cheng**  
A focused survey of LLM inference benchmarking: defines the core metrics, discusses realistic workloads and traffic patterns, and compares open-source benchmarking tools (genai-perf, Guidellm, etc.).  
👉 https://rudeigerc.dev/posts/en/llm-inference-benchmarking/

**LLM Benchmarks Collections (Medium tag "LLM Benchmarks")**  
Useful for seeing how others run comparative GPU/model benchmarks and how they present throughput vs latency vs cost trade-offs.  
👉 https://medium.com/tag/llm-benchmarks

---

## 4. Practical Courses / Hands-on Material

**Efficiently Serving LLMs — Short Course (DeepLearning.AI, Travis Addair)**  
A compact, hands-on course on serving LLMs: batching, KV caching, quantization, LoRA, and basic deployment patterns. There is a public GitHub companion repo that mirrors the exercises.  
👉 https://learn.deeplearning.ai/courses/efficiently-serving-llms/

**Vendor Notebooks / Docs on Endpoint Benchmarking (example: Databricks)**  
Show you what a production-minded vendor considers "baseline" metrics and how they structure a benchmarking notebook for their own endpoints. A good starting point:  
👉 https://www.databricks.com/blog?keys=LLM%20inference

---

## 5. Learning Benchmarking-by-Coding (Suggested Path)

Use these readings as context, but actually learn by building a small harness:

### Step 1: Minimal synchronous runner
Single-threaded script that sends N requests to an OpenAI-compatible endpoint, measures E2E per request, counts tokens from the API usage, and writes a JSON/CSV summary (p50/p95 E2E, total tokens, tokens/s).

### Step 2: Add streaming + TTFT
Extend the client to use streaming: record time to first chunk (TTFT) and time to completion (E2E) separately and log both. At this point you can empirically see when TTFT ≈ E2E (no real streaming benefit).

### Step 3: Add concurrency
Move to asyncio (or your language of choice) with `n_requests` and `concurrency` knobs. Now you can reproduce the style of benchmark JSON you've been analyzing: `n_requests`, `concurrency`, `wall_clock_s`, `p50/p95 TTFT/E2E`, `throughput_tok_s`, `total_tokens`.

### Step 4: Add workload realism
Randomize prompts, control expected output length, and experiment with different arrival patterns (steady vs bursty). This lets you test how batching and queueing actually behave under "chatty users" vs "large offline jobs".

### Step 5: Compare runtimes and hardware
Use the same harness to compare:
- Different runtimes (plain HF, vLLM, TensorRT-LLM, etc.)
- Different GPUs or GPU counts
- Different configs (batch sizes, max tokens, quantization levels)

The readings above then become "explanations" for the curves you see.

---

## 6. How to Use This Page

- Link this from the main README under "Further Reading"
- As you go through each resource, append a one-two sentence "What this changed in my mental model" note under the relevant bullet
- This page evolves from a link list into your distilled inference-engineering knowledge

---

## Raw URL List

```
https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/
https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/
https://blog.vllm.ai/2023/06/20/vllm.html
https://medium.com/byte-sized-ai/all?topic=vllm
https://github.com/vllm-project/vllm
https://docs.vllm.ai/
https://rudeigerc.dev/posts/en/llm-inference-benchmarking/
https://medium.com/tag/llm-benchmarks
https://learn.deeplearning.ai/courses/efficiently-serving-llms/
https://www.databricks.com/blog?keys=LLM%20inference
```
