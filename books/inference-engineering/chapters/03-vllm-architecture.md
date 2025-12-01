# Chapter 3: vLLM Architecture & Internals

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Understanding vLLM's internals is essential for effective optimization.

---

## In This Chapter

- [Overview](#overview)
- [Key Concepts](#key-concepts)
  - [1. Paged Attention](#1-paged-attention)
  - [2. Continuous Batching](#2-continuous-batching)
  - [3. Request Scheduling](#3-request-scheduling)
  - [4. Offline Batch Inferencing](#4-offline-batch-inferencing)
  - [5. Optimized Compute Path](#5-optimized-compute-path)
  - [6. Prefix Caching](#6-prefix-caching)
  - [7. Quantization Support](#7-quantization-support)
  - [8. Separation of API and Engine](#8-separation-of-api-and-engine)
- [Hardware-Aware Deployment](#hardware-aware-deployment)
- [Architecture Summary](#architecture-summary)

---

## Overview

vLLM is a high-throughput and memory-efficient inference engine for LLMs. Its core innovation is **PagedAttention**, which manages the KV cache like operating system virtual memory, enabling efficient memory utilization and high concurrency.

---

## Key Concepts

### 1. Paged Attention

#### The Problem: Memory Waste in Traditional KV Caching

Traditional LLM serving allocates fixed-size memory blocks for the KV cache, leading to:
- **Internal fragmentation**: Unused space within allocated blocks
- **External fragmentation**: Gaps between allocated blocks
- **Memory waste**: Up to 60-80% of memory wasted

#### The Solution: OS-Inspired Memory Management

PagedAttention divides the KV cache into fixed-size **pages** (blocks), similar to virtual memory paging:

```
┌─────────────────────────────────────────────────────────┐
│                    Physical KV Cache                     │
├─────────┬─────────┬─────────┬─────────┬─────────┬───────┤
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │  ...  │
└─────────┴─────────┴─────────┴─────────┴─────────┴───────┘
     ↑         ↑                   ↑         ↑
     │         │                   │         │
┌────┴────┬────┴────┐         ┌────┴────┬────┴────┐
│ Seq A-0 │ Seq A-1 │         │ Seq B-0 │ Seq B-1 │
└─────────┴─────────┘         └─────────┴─────────┘
    Sequence A                    Sequence B
    (Logical Blocks)              (Logical Blocks)
```

#### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Near-zero waste** | Only last block may have unused space |
| **Dynamic allocation** | Allocate blocks as sequences grow |
| **Memory sharing** | Multiple sequences can share blocks (for beam search, parallel sampling) |
| **Efficient swapping** | Blocks can be swapped to CPU memory |

#### 1.1 Key-Value Caching Deep Dive

The KV cache stores computed key and value vectors for all previous tokens, avoiding recomputation:

```
Without KV Cache:
Token 1: Compute K1, V1
Token 2: Compute K1, V1, K2, V2  ← Redundant!
Token 3: Compute K1, V1, K2, V2, K3, V3  ← More redundant!

With KV Cache:
Token 1: Compute and store K1, V1
Token 2: Load K1, V1; compute K2, V2; store K2, V2
Token 3: Load K1, V1, K2, V2; compute K3, V3; store K3, V3
```

**Memory requirement** per layer per token:
```
KV cache size = 2 × hidden_dim × num_layers × seq_len × batch_size × dtype_bytes
```

For Llama-3-70B (80 layers, 8192 hidden_dim, FP16):
- Per token: `2 × 8192 × 80 × 2 bytes = 2.6 MB`
- For 4K context: `~10.5 GB per sequence`

---

### 2. Continuous Batching

#### Static vs Continuous Batching

| Static Batching | Continuous Batching |
|-----------------|---------------------|
| Wait for batch to fill | Process requests as they arrive |
| All sequences finish together | Each sequence finishes independently |
| GPU idle during padding | Minimal idle time |
| Lower utilization | Near-optimal utilization |

#### How It Works

```
Time →
Static:    [Batch 1: A,B,C    wait...    ] [Batch 2: D,E,F]
Continuous: [A starts][B joins][C joins][A ends][D joins][B ends]...
```

vLLM's continuous batching:
1. Maintains a queue of pending requests
2. Adds new requests to the batch when slots open
3. Removes completed sequences immediately
4. New requests can join mid-generation

---

### 3. Request Scheduling

vLLM uses a sophisticated scheduler that balances:
- **Fairness**: All requests make progress
- **Efficiency**: Maximize GPU utilization
- **Memory**: Stay within memory limits

#### Scheduling Algorithm

```python
# Simplified scheduling logic
def schedule():
    running = []  # Currently generating
    waiting = []  # Queued requests
    
    while True:
        # Check if we can add more requests
        while waiting and memory_available():
            req = waiting.pop(0)
            allocate_kv_cache(req)
            running.append(req)
        
        # Generate one step for all running
        outputs = generate_step(running)
        
        # Handle completions
        for req in running:
            if req.is_complete():
                free_kv_cache(req)
                running.remove(req)
```

#### Key Parameters

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| `max_num_seqs` | Max concurrent sequences | Higher = better throughput, more memory |
| `max_num_batched_tokens` | Max tokens per iteration | Higher = better GPU utilization |
| `max_model_len` | Max sequence length | Higher = more memory per request |

---

### 4. Offline Batch Inferencing

For non-real-time workloads, vLLM supports efficient batch processing:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
]

sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

**Advantages of offline mode**:
- No server overhead
- Optimal batching across all prompts
- Better memory planning

---

### 5. Optimized Compute Path

vLLM optimizes the forward pass through:

#### CUDA Graphs

Pre-compile GPU operations to eliminate kernel launch overhead:

```bash
# Enable CUDA graphs (often default)
vllm serve model --enforce-eager=False
```

**Note**: CUDA graphs work best with fixed sequence lengths. For variable lengths, vLLM uses a hybrid approach.

#### Flash Attention

Fused attention kernel that's faster and uses less memory:

```python
# Automatically used when available
# Check: pip install flash-attn
```

#### Fused Operations

Multiple operations combined into single kernels:
- LayerNorm + Linear
- Attention + Projection
- GELU/SwiGLU activations

---

### 6. Prefix Caching

Reuse KV cache for shared prefixes across requests:

```
Request 1: "You are a helpful assistant. What is Python?"
Request 2: "You are a helpful assistant. Explain JavaScript."
                ↑ Shared prefix ↑
```

With prefix caching, the KV cache for "You are a helpful assistant." is computed once and shared.

```bash
# Enable prefix caching
vllm serve model --enable-prefix-caching
```

**Use cases**:
- System prompts
- Few-shot examples
- Multi-turn conversations

---

### 7. Quantization Support

vLLM supports multiple quantization formats:

| Format | Bits | Method | Use Case |
|--------|------|--------|----------|
| **FP16/BF16** | 16 | Native | Default, best quality |
| **INT8** | 8 | SmoothQuant | Good speed/quality balance |
| **FP8** | 8 | NVIDIA H100 | Best on Hopper GPUs |
| **GPTQ** | 4 | Post-training | Maximum compression |
| **AWQ** | 4 | Activation-aware | Better quality than GPTQ |
| **GGUF** | Various | llama.cpp format | Cross-platform |

```bash
# Serve a quantized model
vllm serve TheBloke/Llama-2-7B-GPTQ --quantization gptq

# Or AWQ
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq
```

---

### 8. Separation of API and Engine

vLLM cleanly separates:

```
┌─────────────────────────────────────────────────┐
│                  API Layer                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ OpenAI    │  │ Embeddings│  │ Async     │   │
│  │ Compatible│  │ API       │  │ Engine    │   │
│  └───────────┘  └───────────┘  └───────────┘   │
├─────────────────────────────────────────────────┤
│                Engine Layer                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ Scheduler │  │ KV Cache  │  │ Model     │   │
│  │           │  │ Manager   │  │ Executor  │   │
│  └───────────┘  └───────────┘  └───────────┘   │
└─────────────────────────────────────────────────┘
```

This separation allows:
- Multiple API frontends
- Engine reuse across deployments
- Easier testing and debugging

---

## Hardware-Aware Deployment

### GPU Selection Guide

| GPU | VRAM | Best For | Notes |
|-----|------|----------|-------|
| **T4** | 16GB | 7B models, dev/test | Cost-effective |
| **A10G** | 24GB | 7-13B models | Good price/performance |
| **A100 40GB** | 40GB | 13-30B models | High bandwidth |
| **A100 80GB** | 80GB | 30-70B models | Top tier |
| **H100** | 80GB | Any size, FP8 | Maximum throughput |

### Memory Estimation

```python
def estimate_memory_gb(params_billions, dtype="fp16", kv_cache_gb=2):
    """Estimate GPU memory needed for a model."""
    bytes_per_param = {"fp16": 2, "fp32": 4, "int8": 1, "int4": 0.5}
    model_memory = params_billions * bytes_per_param[dtype]
    overhead = 1.2  # 20% overhead for activations, etc.
    return (model_memory * overhead) + kv_cache_gb

# Example: Llama-3-8B in FP16
print(f"~{estimate_memory_gb(8):.1f} GB needed")  # ~11.6 GB
```

### TPU and Beyond

vLLM also supports:
- **TPU**: Via PyTorch XLA
- **AMD ROCm**: With compatible builds
- **CPU**: For testing (not production)

```bash
# TPU serving
vllm serve model --device tpu

# CPU (slow, for testing only)
vllm serve model --device cpu
```

---

## Architecture Summary

```
Request Flow:
                                                        
  Client Request                                         
       │                                                 
       ▼                                                 
  ┌─────────┐     ┌──────────┐     ┌─────────────┐      
  │   API   │────▶│Scheduler │────▶│   Model     │      
  │ Server  │     │          │     │  Executor   │      
  └─────────┘     └──────────┘     └─────────────┘      
       │               │                  │              
       │               ▼                  ▼              
       │         ┌──────────┐     ┌─────────────┐      
       │         │KV Cache  │◀───▶│   GPU(s)    │      
       │         │ Manager  │     │             │      
       │         └──────────┘     └─────────────┘      
       │                                  │              
       ◀──────────────────────────────────┘              
  Response (streaming or complete)                       
```

---

<p align="center">
  <a href="02-os-essentials.md">← Previous: OS Essentials</a> | <a href="../README.md">Table of Contents</a> | <a href="04-setup-basic-usage.md">Next: Setup & Basic Usage →</a>
</p>
