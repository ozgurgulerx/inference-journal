# Chapter 1: Introduction to Inference Engineering

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Content is refined through hands-on study and experimentation.

---

## What is Inference Engineering?

**Inference engineering** is the discipline of designing and operating the stack that turns a trained model into a fast, cheap, and reliable API.

At its core, this means:

| Area | Description |
|------|-------------|
| **Engine Selection** | Choosing and optimizing engines (vLLM, TensorRT-LLM, TGI, SGLang, Triton) |
| **Hardware Exploitation** | Leveraging HBM, PCIe/NVLink, FP8/INT4, multi-GPU configurations |
| **Algorithmic Optimization** | Implementing PagedAttention, FlashAttention, speculative decoding, KV-cache tricks, batching/scheduling, quantization |
| **SLO Management** | Meeting p95 latency, throughput, cost, and failure mode requirements |

**Everything you study should map to one of these levers.**

---

## The Inference Stack

A complete inference stack consists of multiple layers, each with optimization opportunities:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│              (API Gateway, Load Balancer, Auth)              │
├─────────────────────────────────────────────────────────────┤
│                     Serving Framework                        │
│            (vLLM, TGI, TensorRT-LLM, SGLang)                │
├─────────────────────────────────────────────────────────────┤
│                      Runtime Layer                           │
│          (PyTorch, JAX, ONNX Runtime, TensorRT)             │
├─────────────────────────────────────────────────────────────┤
│                      Kernel Layer                            │
│        (FlashAttention, CUTLASS, cuBLAS, Triton)            │
├─────────────────────────────────────────────────────────────┤
│                      Hardware Layer                          │
│              (GPU, TPU, CPU, Custom ASICs)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Performance Levers

### 1. Throughput vs Latency

The fundamental trade-off in inference engineering:

| Metric | Definition | Optimization Focus |
|--------|------------|-------------------|
| **Throughput** | Tokens per second across all requests | Batch size, parallelism, memory efficiency |
| **Latency** | Time to generate a single token | Kernel fusion, caching, speculative decoding |
| **TTFT** | Time to First Token | Prefill optimization, prefix caching |
| **ITL** | Inter-Token Latency | Decode optimization, memory bandwidth |

### 2. Memory Hierarchy

Understanding memory is critical:

```
┌──────────────────┐
│   Registers      │  ← Fastest, smallest
├──────────────────┤
│   Shared Memory  │  ← ~100 TB/s bandwidth
├──────────────────┤
│   L2 Cache       │  ← Automatic caching
├──────────────────┤
│   HBM (GPU RAM)  │  ← 1-3 TB/s on modern GPUs
├──────────────────┤
│   System RAM     │  ← ~100 GB/s
├──────────────────┤
│   NVMe/SSD       │  ← ~7 GB/s
└──────────────────┘
```

### 3. The Compute vs Memory Bound Reality

| Phase | Characteristic | Bottleneck |
|-------|---------------|------------|
| **Prefill** | Process entire prompt at once | Compute-bound (matrix multiplications) |
| **Decode** | Generate one token at a time | Memory-bound (loading weights per token) |

This insight drives many optimizations:
- **Prefill**: Benefit from larger batch sizes, tensor parallelism
- **Decode**: Benefit from memory bandwidth, KV-cache efficiency

---

## Core Concepts You Must Know

### PagedAttention
vLLM's memory manager inspired by OS virtual memory paging. Near-zero KV waste, enabling larger effective batch sizes and longer contexts.

### Continuous Batching
Unlike static batching (wait for fixed batch), continuous batching adds new requests as slots free up. Maximizes GPU utilization.

### Speculative Decoding
Use a small, fast "draft" model to generate multiple tokens, then verify with the large model. Can achieve 2-3x speedups.

### Quantization
Reduce numerical precision (FP16 → INT8 → INT4) to:
- Decrease memory footprint
- Increase throughput
- Often with minimal accuracy loss

---

## Your First Lab Notebook Entry

**Goal**: "I can serve a 7-8B model with vLLM and understand every CLI flag that changes performance."

### Setup Experiment

1. **Spin up a GPU** (T4/A10/A100/H100 — whatever you have)

2. **Run vLLM** with an open-source model:
   ```bash
   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
     --port 8000 \
     --gpu-memory-utilization 0.9
   ```

3. **Measure baseline**:
   - Tokens/sec (prefill vs decode)
   - p50 / p95 latency for short (32), medium (512), and long (4k) prompts
   - GPU memory usage vs concurrent users

4. **Iterate with optimization flags**:
   ```bash
   vllm serve <model> \
     --max-num-seqs 32 \
     --max-num-batched-tokens 2048 \
     --tensor-parallel-size 1
   ```

5. **Log each change like a scientist**:
   
   | Knob | Before | After | Delta |
   |------|--------|-------|-------|
   | `max-num-seqs` 16→32 | 150 tok/s | 210 tok/s | +40% |
   | `quantization` FP16→INT8 | 8GB VRAM | 4GB VRAM | -50% |

---

## Essential Resources

### Primary Documentation
- **vLLM Docs**: Architecture overview, API, PagedAttention, continuous batching, CUDA graphs
- **vLLM Optimization Page**: Concrete flags and tuning guidelines
- **PagedAttention Paper**: The key conceptual leap in KV cache management ([arXiv](https://arxiv.org/abs/2309.06180))

### Must-Read Essays
| Source | Title | Focus |
|--------|-------|-------|
| NVIDIA | "Mastering LLM Techniques: Inference Optimization" | Core bottlenecks, parallelism, quantization |
| Together AI | "Best practices to accelerate inference" | Silicon, memory hierarchies, vertical integration |
| Transcendent AI | "Inference at Scale" | Cost, latency, robustness |
| Micron | "LLM Inference Engineering Report" | HBM bandwidth impact on throughput |

---

## What's Next

In the following chapters, we'll dive deep into each layer of the inference stack:

- **Chapter 2**: OS-level optimizations for maximum hardware utilization
- **Chapter 3**: vLLM internals — how PagedAttention and continuous batching actually work
- **Chapter 4**: Hands-on setup and your first production-ready server

---

<p align="center">
  <a href="../README.md">← Back to Table of Contents</a> | <a href="02-os-essentials.md">Next: OS Essentials →</a>
</p>
