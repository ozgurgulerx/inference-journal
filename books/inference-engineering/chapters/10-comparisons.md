# Chapter 10: Comparisons & Further Reading

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Understanding trade-offs between engines helps you choose the right tool.

---

## In This Chapter

- [Overview](#overview)
- [Engine Comparison](#engine-comparison)
- [vLLM vs DeepSpeed](#vllm-vs-deepspeed)
- [vLLM vs TensorRT-LLM](#vllm-vs-tensorrt-llm)
- [vLLM vs Text Generation Inference (TGI)](#vllm-vs-text-generation-inference-tgi)
- [Other Engines](#other-engines)
- [Decision Framework](#decision-framework)
- [Further Reading](#further-reading)
- [Learning Path](#learning-path)

---

## Overview

This chapter compares vLLM with other popular inference engines and provides resources for continued learning.

---

## Engine Comparison

### Quick Reference

| Feature | vLLM | TGI | TensorRT-LLM | DeepSpeed |
|---------|------|-----|--------------|-----------|
| **PagedAttention** | ✅ Native | ✅ | ✅ | ❌ |
| **Continuous Batching** | ✅ | ✅ | ✅ | ❌ |
| **Tensor Parallelism** | ✅ | ✅ | ✅ | ✅ |
| **Quantization** | INT8/FP8/AWQ/GPTQ | INT8/GPTQ | FP8/INT8/INT4 | INT8 |
| **Multi-Node** | ✅ (Ray) | Limited | ✅ | ✅ |
| **OpenAI API** | ✅ | ✅ | Via wrapper | ❌ |
| **Ease of Use** | High | High | Medium | Medium |
| **Max Perf** | High | High | Highest | High |

---

## vLLM vs DeepSpeed

### DeepSpeed-Inference Strengths

- **ZeRO-Inference**: Partition model across GPUs with minimal memory
- **Custom Kernels**: DeepFusion optimized kernels
- **Microsoft Ecosystem**: Integrates with Azure, ONNX Runtime
- **Training+Inference**: Same framework for both

### vLLM Strengths

- **PagedAttention**: Superior memory efficiency for serving
- **Continuous Batching**: Better for variable-length requests
- **OpenAI Compatible**: Drop-in replacement
- **Simpler Setup**: Single command serving

### When to Choose

| Use Case | Best Choice |
|----------|-------------|
| High-concurrency serving | vLLM |
| Massive models (175B+) | DeepSpeed |
| Microsoft/Azure stack | DeepSpeed |
| OpenAI API compatibility | vLLM |
| Research/experimentation | DeepSpeed |
| Production chatbots | vLLM |

### Performance Comparison

```
Throughput (tokens/sec) - Llama 7B, A100
┌──────────────────────────────────────────┐
│ vLLM        ████████████████████  2,500  │
│ DeepSpeed   ████████████████     2,000   │
│ HF Pipeline ██████               750     │
└──────────────────────────────────────────┘

Memory Efficiency (concurrent requests)
┌──────────────────────────────────────────┐
│ vLLM        ████████████████████  64     │
│ DeepSpeed   ████████████         32      │
│ HF Pipeline ████                 8       │
└──────────────────────────────────────────┘
```

---

## vLLM vs TensorRT-LLM

### TensorRT-LLM Strengths

- **Maximum NVIDIA Performance**: Deep hardware integration
- **FP8 on Hopper**: Native H100 optimization
- **Kernel Fusion**: Aggressive operator fusion
- **Production Hardened**: NVIDIA support

### vLLM Strengths

- **Flexibility**: Works across GPU vendors
- **Rapid Development**: Faster to add new models
- **Open Source**: Community-driven development
- **Simpler Workflow**: No separate build step

### When to Choose

| Use Case | Best Choice |
|----------|-------------|
| Maximum H100/A100 perf | TensorRT-LLM |
| AMD ROCm support | vLLM |
| Quick prototyping | vLLM |
| Enterprise NVIDIA support | TensorRT-LLM |
| New model architectures | vLLM |
| FP8 quantization | TensorRT-LLM |

### Benchmark Example

```
Latency (ms) - Llama 70B, 4xH100
┌──────────────────────────────────────────┐
│ TensorRT-LLM  ██████████         25ms    │
│ vLLM          ████████████████   40ms    │
│                                          │
│ (TRT-LLM ~1.6x faster per-request)       │
└──────────────────────────────────────────┘

Setup Complexity (relative)
┌──────────────────────────────────────────┐
│ vLLM          ████                1x     │
│ TensorRT-LLM  ████████████████    4x     │
└──────────────────────────────────────────┘
```

---

## vLLM vs Text Generation Inference (TGI)

### TGI Strengths

- **Hugging Face Integration**: Native Hub support
- **Production Ready**: Battle-tested at scale
- **Rust Performance**: Core in Rust
- **Enterprise Features**: Auth, watermarking

### vLLM Strengths

- **PagedAttention**: More memory efficient
- **Higher Throughput**: Better at high concurrency
- **Python-Native**: Easier to extend
- **Speculative Decoding**: Better support

### When to Choose

| Use Case | Best Choice |
|----------|-------------|
| Hugging Face ecosystem | TGI |
| Maximum throughput | vLLM |
| Enterprise features | TGI |
| Memory-constrained | vLLM |
| Custom modifications | vLLM |

---

## Other Engines

### SGLang

- **Focus**: Structured generation, function calling
- **Strength**: Efficient constrained decoding
- **Use When**: Need JSON/schema output

### Triton Inference Server

- **Focus**: Multi-framework serving
- **Strength**: Ensemble models, batching
- **Use When**: Mixed model types

### llama.cpp

- **Focus**: CPU and edge inference
- **Strength**: Runs anywhere, GGUF format
- **Use When**: No GPU, mobile, edge

### Ollama

- **Focus**: Local LLM running
- **Strength**: Simplicity, model library
- **Use When**: Local development, demos

---

## Decision Framework

```
                    ┌─────────────────┐
                    │ What's your     │
                    │ priority?       │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    Maximum Perf        Ease of Use         Flexibility
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │NVIDIA?  │         │Production│         │AMD/TPU? │
    └────┬────┘         │or Dev?  │         └────┬────┘
         │              └────┬────┘              │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │ Yes  No │         │Prod  Dev│         │ Yes  No │
    │  │   │  │         │ │    │  │         │  │   │  │
    │  ▼   ▼  │         │ ▼    ▼  │         │  ▼   ▼  │
    │TRT  vLLM│         │TGI  vLLM│         │vLLM TRT │
    │-LLM    │         │         │         │        │
    └─────────┘         └─────────┘         └─────────┘
```

---

## Further Reading

### Academic Papers

| Paper | Topic | Link |
|-------|-------|------|
| PagedAttention | vLLM's core innovation | [arXiv](https://arxiv.org/abs/2309.06180) |
| FlashAttention 2 | Memory-efficient attention | [arXiv](https://arxiv.org/abs/2307.08691) |
| GPTQ | 4-bit quantization | [arXiv](https://arxiv.org/abs/2210.17323) |
| AWQ | Activation-aware quantization | [arXiv](https://arxiv.org/abs/2306.00978) |
| Speculative Decoding | Faster generation | [arXiv](https://arxiv.org/abs/2211.17192) |

### Blog Posts & Guides

| Source | Title | Focus |
|--------|-------|-------|
| NVIDIA Developer | "Mastering LLM Techniques" | Optimization overview |
| Together AI | "Accelerate Production Workloads" | Practical techniques |
| Hugging Face | "Text Generation Inference" | TGI deep dive |
| vLLM Blog | "Inside vLLM" | Architecture details |
| Anyscale | "Scaling LLM Inference" | Ray + vLLM |

### Documentation

- **vLLM**: [docs.vllm.ai](https://docs.vllm.ai)
- **TensorRT-LLM**: [NVIDIA Docs](https://nvidia.github.io/TensorRT-LLM/)
- **TGI**: [HF Docs](https://huggingface.co/docs/text-generation-inference)
- **DeepSpeed**: [DeepSpeed.ai](https://www.deepspeed.ai/inference/)

### Communities

- **vLLM Discord**: Active community support
- **Hugging Face Forums**: TGI discussions
- **NVIDIA Developer Forums**: TensorRT-LLM
- **r/LocalLLaMA**: General LLM inference

---

## Learning Path

### Week 1-2: Foundations
1. Run vLLM with a 7B model
2. Benchmark throughput and latency
3. Experiment with batch sizes

### Week 3-4: Optimization
1. Try quantization (AWQ, GPTQ)
2. Enable prefix caching
3. Compare with TGI on same model

### Week 5-6: Scaling
1. Set up multi-GPU serving
2. Deploy on Kubernetes
3. Implement monitoring

### Week 7-8: Production
1. Build observability stack
2. Set up alerting
3. Document runbooks

### Ongoing
- Follow engine releases
- Benchmark new models
- Contribute to open source

---

<p align="center">
  <a href="09-observability.md">← Previous: Observability</a> | <a href="../README.md">Table of Contents</a> | <a href="11-ecosystem-players.md">Next: Ecosystem Players →</a>
</p>
