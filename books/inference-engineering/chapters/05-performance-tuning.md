# Chapter 5: Performance Tuning & Memory Trade-offs

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Performance tuning requires systematic experimentation and measurement.

---

## In This Chapter

- [Overview](#overview)
- [The Fundamental Trade-off](#the-fundamental-trade-off)
- [5.1 Latency Tuning and Acceleration Techniques](#51-latency-tuning-and-acceleration-techniques)
  - [Fused Kernels and Efficient Ops](#fused-kernels-and-efficient-ops)
  - [Speculative Decoding](#speculative-decoding)
  - [Prefill–Decode Separation](#prefilldecode-separation)
  - [Efficient Sampling & Caching](#efficient-sampling--caching)
- [5.2 Model Quantization and Compression](#52-model-quantization-and-compression)
- [5.3 Compiler Toolchains and Kernel Optimization](#53-compiler-toolchains-and-kernel-optimization)
- [Performance Tuning Workflow](#performance-tuning-workflow)
- [Key Takeaways](#key-takeaways)

---

## Overview

vLLM's claim to fame is high throughput and efficiency. This chapter covers:
- Measuring latency vs throughput
- Understanding batch size and concurrent request impact
- Using quantization to reduce memory
- Compiler toolchains for maximum performance

---

## The Fundamental Trade-off

In an ideal world, we want both low latency and high throughput. Reality forces trade-offs:

| Optimization | Effect on Throughput | Effect on Latency |
|-------------|---------------------|-------------------|
| Larger batch size | ↑ Increases | ↑ May increase per-request |
| More concurrent sequences | ↑ Increases | ↔ Depends on resources |
| Quantization | ↑ Increases | ↓ Usually decreases |
| Tensor parallelism | ↔ Varies | ↓ Decreases |
| Prefix caching | ↑ Increases | ↓ Decreases (cache hits) |

---

## 5.1 Latency Tuning and Acceleration Techniques

For small batch sizes (common in online inference), **memory access becomes the bottleneck** rather than compute. The DeepSpeed team notes that inference latency is "lower-bounded by the time to load model parameters from memory to registers."

### Fused Kernels and Efficient Ops

Instead of many tiny GPU kernels, fuse operations to minimize overhead:

```
Traditional:                     Fused:
┌─────────┐                     ┌─────────────────────┐
│LayerNorm│                     │                     │
├─────────┤                     │ LayerNorm + MatMul  │
│  MatMul │  → Kernel launches  │      + GELU         │
├─────────┤                     │                     │
│  GELU   │                     │ (Single kernel)     │
└─────────┘                     └─────────────────────┘
```

**Benefits**:
- Reduced kernel launch overhead
- Better memory locality
- Maximum tensor core utilization

**Hardware impact**:
- A100 → H100 (FP16 → FP8): ~4x speedup
- With optimized batching: 8x throughput gains

```bash
# Enable Flash Attention (if available)
pip install flash-attn

# vLLM uses it automatically when detected
```

### Speculative Decoding

Use a small, fast "draft" model to predict multiple tokens, then verify with the large model:

```
Traditional:
  Large Model: Token 1 → Token 2 → Token 3 → Token 4
               (slow)    (slow)    (slow)    (slow)

Speculative:
  Draft Model: Token 1,2,3,4 (fast, parallel)
  Large Model: Verify all 4 in one pass
               → Accept 3 tokens (reject 1)
```

**Results**: 2-3x speedups reported by OpenAI with GPT-4.

```bash
# Enable speculative decoding in vLLM
vllm serve main-model \
  --speculative-model draft-model \
  --num-speculative-tokens 5
```

### Prefill–Decode Separation

Separate the two phases for specialized handling:

| Phase | Characteristics | Optimization |
|-------|----------------|--------------|
| **Prefill** | Process entire prompt, compute-bound | Batch prompts, tensor parallel |
| **Decode** | Generate tokens, memory-bound | Optimize KV cache access |

vLLM can disaggregate these:

```bash
# Prefill on high-throughput GPU
# Decode on optimized GPU with large cache
# KV cache transferred between stages
```

### Efficient Sampling & Caching

**Prefix caching** reuses KV cache for shared prefixes:

```bash
# Enable prefix caching
vllm serve model --enable-prefix-caching
```

Example impact:
```
Request 1: "You are an assistant. What is Python?"  → 50ms
Request 2: "You are an assistant. What is Java?"    → 15ms (prefix cached)
```

---

## 5.2 Model Quantization and Compression

Quantization reduces numerical precision to decrease memory and increase speed.

### Quantization Landscape

| Format | Bits | Memory Reduction | Speed Impact | Quality |
|--------|------|------------------|--------------|---------|
| FP32 | 32 | Baseline | Baseline | 100% |
| FP16/BF16 | 16 | 2x | ~1.2x faster | ~100% |
| INT8 | 8 | 4x | ~1.5x faster | ~99% |
| FP8 | 8 | 4x | ~2x faster (H100) | ~99% |
| INT4/GPTQ | 4 | 8x | ~2x faster | ~97-99% |
| AWQ | 4 | 8x | ~2x faster | ~98-99% |

### 8-bit and 4-bit Weight Quantization

**LLM.int8()** (Tim Dettmers): 8-bit with per-channel scaling

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_8bit=True,
    device_map="auto"
)
```

**GPTQ** (4-bit post-training):

```bash
# Serve GPTQ model
vllm serve TheBloke/Llama-2-7B-GPTQ --quantization gptq
```

**AWQ** (Activation-aware, often better quality):

```bash
# Serve AWQ model
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq
```

### Memory Savings Example

Llama-3-8B on single GPU:

| Precision | VRAM Usage | Throughput | Quality |
|-----------|------------|------------|---------|
| FP16 | ~16 GB | 100% baseline | 100% |
| INT8 | ~8 GB | +30% | 99.5% |
| INT4 (GPTQ) | ~4 GB | +100% | 98% |
| INT4 (AWQ) | ~4 GB | +100% | 99% |

### FP8 on H100

H100 GPUs have native FP8 support:

```bash
# Use FP8 quantization (requires H100)
vllm serve model --quantization fp8
```

### Quantization Best Practices

1. **Start with FP16** - Baseline quality
2. **Try INT8 first** - Best quality/speed trade-off
3. **Use AWQ over GPTQ** - Usually better quality
4. **Benchmark on your tasks** - Quality impact varies by use case
5. **Consider GGUF** - Good for llama.cpp compatibility

---

## 5.3 Compiler Toolchains and Kernel Optimization

Beyond frameworks like vLLM, compiler toolchains can generate even faster code.

### Standard Runtimes vs Compiler Toolchains

| Standard Runtimes | Compiler Toolchains |
|-------------------|---------------------|
| PyTorch / vLLM / TGI | TVM / XLA / IREE / MLIR |
| General-purpose | Model-specific executable |
| Generic kernels | Fused operations globally |
| Framework overhead | Eliminated overhead |
| Safe, stable | Hardware-tailored binaries |

### Why Go Lower-Level?

**1. More Aggressive Fusion**
```
TensorRT-LLM fuses: Attention + Projection
TVM/MLIR can fuse: Attention + RMSNorm + Projection + more
```

**2. Model-Specific Optimization**
- Compile for exact hidden_size, seq_len, vocab
- Optimize for specific architecture (Mamba, MoE, etc.)
- Target specific hardware (H200 vs A100 vs TPU)

**3. Eliminate Framework Overhead**
- No dispatcher
- No autograd
- No Python interpreter
- Just fused kernels

### Deep Learning Compilers

#### Apache TVM / MLC

```python
# Example: Compile model with TVM
import tvm
from tvm import relay

# Load model, convert to Relay IR
mod, params = relay.frontend.from_pytorch(model, input_shapes)

# Tune and compile
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="cuda", params=params)
```

#### XLA (JAX/TensorFlow)

```python
import jax
import jax.numpy as jnp

@jax.jit  # XLA compilation
def forward(params, inputs):
    return model.apply(params, inputs)
```

#### NVIDIA TensorRT-LLM

```bash
# Build optimized engine
trtllm-build --model_dir ./model --output_dir ./engine

# Serve
trtllm-serve --engine_dir ./engine
```

### FlashInfer

A library of high-performance kernels that inference engines can plug in:

```python
import flashinfer

# Use FlashInfer's optimized attention
output = flashinfer.attention(query, key, value)
```

**Features**:
- JIT compilation for specific patterns
- Block-sparse KV cache format
- Inspector-executor pattern for optimal dispatch

### Real Examples in Production

| Project | Approach | Speedup |
|---------|----------|---------|
| FlashAttention 2/3 | Custom attention kernels | 2-4x |
| FasterTransformer | Fused CUDA kernels | 30-40% |
| MLC-LLM | TVM compilation | Runs on mobile |
| GPT-NeoX | Custom fused path | 30-40% |

---

## Performance Tuning Workflow

### Step 1: Baseline Measurement

```bash
# Start server with default settings
vllm serve model --port 8000

# Measure
python benchmark.py --endpoint http://localhost:8000/v1
```

### Step 2: Tune Batch Settings

```bash
# Experiment with batch sizes
vllm serve model \
  --max-num-seqs 16 \    # Try: 8, 16, 32, 64
  --max-num-batched-tokens 2048  # Try: 1024, 2048, 4096
```

### Step 3: Enable Optimizations

```bash
# Full optimization
vllm serve model \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.92 \
  --enforce-eager=False  # Enable CUDA graphs
```

### Step 4: Consider Quantization

```bash
# If memory-constrained or need more throughput
vllm serve quantized-model --quantization awq
```

### Step 5: Log Like a Scientist

| Experiment | Config Change | Throughput | Latency p50 | Latency p95 |
|------------|---------------|------------|-------------|-------------|
| Baseline | Default | 100 tok/s | 45ms | 120ms |
| Exp 1 | max_seqs=32 | 180 tok/s | 50ms | 150ms |
| Exp 2 | + prefix cache | 200 tok/s | 35ms | 100ms |
| Exp 3 | + AWQ quantization | 350 tok/s | 30ms | 85ms |

---

## Key Takeaways

1. **Measure first** - Don't optimize blindly
2. **Batch size is crucial** - Higher batches = more throughput
3. **Quantization is powerful** - 4-bit often retains 98%+ quality
4. **Prefix caching helps** - Especially for system prompts
5. **Know your bottleneck** - Compute vs memory bound matters
6. **Compiler toolchains exist** - For maximum performance

---

<p align="center">
  <a href="04-setup-basic-usage.md">← Previous: Setup & Basic Usage</a> | <a href="../README.md">Table of Contents</a> | <a href="06-serving-models.md">Next: Serving Different Models →</a>
</p>
