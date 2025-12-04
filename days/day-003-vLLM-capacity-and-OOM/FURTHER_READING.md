# Further Reading: LLM Inference Benchmarking & Performance Engineering

This page collects the main references and patterns behind the metrics and instincts we've been using (TTFT, E2E, tokens/s, concurrency, queue vs compute, etc.).

---

## 1. Core Articles on LLM Inference & Benchmarking

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
