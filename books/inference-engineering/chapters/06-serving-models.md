# Chapter 6: Serving Different Models

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Different model architectures require different serving strategies.

---

## In This Chapter

- [Overview](#overview)
- [6.1 Single GPU Inference](#61-single-gpu-inference)
- [6.2 Distributed and Scalable Inference](#62-distributed-and-scalable-inference)
  - [Tensor Parallelism (TP)](#tensor-parallelism-tp)
  - [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
  - [Multi-Node Deployment](#multi-node-deployment)
- [6.3 Weight Management Methods](#63-weight-management-methods)
  - [6.3.1 Weight Streaming](#631-weight-streaming)
  - [6.3.2 Other Weight Management Methods](#632-other-weight-management-methods)
- [6.4 Multi-Tenancy and Isolation](#64-multi-tenancy-and-isolation)
- [Model Serving Patterns](#model-serving-patterns)
- [Architecture Decision Guide](#architecture-decision-guide)

---

## Overview

This chapter covers serving various model types:
- Single GPU inference for standard models
- Distributed inference for large models
- Weight management techniques
- Multi-tenancy considerations

---

## 6.1 Single GPU Inference

### Model Size Guidelines

| Model Size | Minimum VRAM (FP16) | Recommended GPU |
|------------|---------------------|-----------------|
| 1-3B | 6 GB | T4, RTX 3060 |
| 7-8B | 16 GB | T4, A10G |
| 13B | 26 GB | A10G (quantized), A100 40GB |
| 30B | 60 GB | A100 80GB |
| 70B | 140 GB | Multiple GPUs required |

### Serving LLaMA Models

```bash
# Llama 3 8B
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

### Serving Mistral Models

```bash
# Mistral 7B
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

**Mistral-specific notes**:
- Uses Sliding Window Attention (4096 tokens)
- Efficient for long contexts within window

### Serving Mixtral (MoE)

Mixture of Experts models require special handling:

```bash
# Mixtral 8x7B (actually ~47B parameters, but sparse)
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9
```

**MoE considerations**:
- Only 2 experts active per token (out of 8)
- Requires more memory than active params suggest
- Throughput can be excellent due to sparsity

### Model-Specific Configurations

| Model | Special Flags | Notes |
|-------|---------------|-------|
| Llama 3 | Default works | RoPE, GQA |
| Mistral | Default works | Sliding window |
| Mixtral | `--tensor-parallel-size 2+` | MoE routing |
| Phi-3 | Default works | Small but capable |
| Qwen2 | Default works | Strong multilingual |

---

## 6.2 Distributed and Scalable Inference

For models that don't fit on a single GPU, use parallelism.

### Tensor Parallelism (TP)

Split each layer across GPUs:

```
┌─────────────────────────────────────────┐
│           Layer N Weight Matrix          │
│  ┌────────┬────────┬────────┬────────┐  │
│  │ GPU 0  │ GPU 1  │ GPU 2  │ GPU 3  │  │
│  │ 1/4    │ 1/4    │ 1/4    │ 1/4    │  │
│  └────────┴────────┴────────┴────────┘  │
│              ↓ All-Reduce ↓              │
└─────────────────────────────────────────┘
```

```bash
# Llama 70B on 4 GPUs
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

**When to use TP**:
- Within a single node (NVLink/NVSwitch)
- Low latency requirements
- High communication bandwidth available

### Pipeline Parallelism (PP)

Split layers across GPUs:

```
GPU 0: Layers 0-19  →  GPU 1: Layers 20-39  →  GPU 2: Layers 40-59
        ↓                      ↓                       ↓
    Activations           Activations             Output
```

```bash
# Combine TP and PP for very large models
vllm serve model \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2  # 8 GPUs total
```

**When to use PP**:
- Across nodes (InfiniBand)
- Lower communication bandwidth
- Very large models

### Multi-Node Deployment

```bash
# Node 0 (head)
ray start --head --port=6379

# Node 1+ (workers)
ray start --address=<head-ip>:6379

# Serve with distributed vLLM
vllm serve large-model \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray
```

### Scaling Recommendations

| Model Size | GPUs | Strategy |
|------------|------|----------|
| 7-13B | 1 | Single GPU |
| 30-40B | 2 | TP=2 |
| 65-70B | 4 | TP=4 |
| 70B+ | 8+ | TP=4, PP=2 or TP=8 |
| 180B+ | 16+ | Multi-node TP+PP |

---

## 6.3 Weight Management Methods

### 6.3.1 Weight Streaming

For models that can't fit in GPU memory even with parallelism:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│   Disk   │────▶│   CPU    │────▶│   GPU    │
│ (Model)  │     │ (Buffer) │     │ (Active) │
└──────────┘     └──────────┘     └──────────┘
         Stream weights on demand
```

**Use cases**:
- Research with limited hardware
- Very large models (hundreds of GB)
- CPU offloading

```bash
# Enable CPU offloading in vLLM
vllm serve large-model \
  --cpu-offload-gb 40  # Offload 40GB to CPU
```

### 6.3.2 Other Weight Management Methods

#### Model Sharding

Pre-shard models for faster loading:

```python
# Save sharded model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("large-model")
model.save_pretrained("./sharded-model", max_shard_size="5GB")
```

#### Quantization for Memory

Reduce precision to fit larger models:

```bash
# 70B in INT4 on 2x24GB GPUs
vllm serve TheBloke/Llama-2-70B-GPTQ \
  --quantization gptq \
  --tensor-parallel-size 2
```

#### LoRA Serving

Serve base model + multiple LoRA adapters:

```bash
# Enable LoRA support
vllm serve base-model \
  --enable-lora \
  --lora-modules adapter1=./lora1 adapter2=./lora2
```

Request specific adapter:

```python
response = client.chat.completions.create(
    model="adapter1",  # Use specific LoRA
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

## 6.4 Multi-Tenancy and Isolation

When multiple users or models share infrastructure.

### Security Isolation

| Level | Method | Trade-off |
|-------|--------|-----------|
| **Process** | Separate processes | Moderate isolation |
| **Container** | Docker/Kubernetes | Good isolation |
| **VM** | Separate VMs | Strong isolation |
| **Hardware** | Dedicated GPUs | Complete isolation |

### Container Isolation

```yaml
# Kubernetes pod with isolated GPU
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: model-a
    resources:
      limits:
        nvidia.com/gpu: 1
    securityContext:
      runAsNonRoot: true
      readOnlyRootFilesystem: true
```

### MIG for Multi-Tenancy

Use Multi-Instance GPU for isolated slices:

```bash
# Create isolated MIG instances
sudo nvidia-smi mig -cgi 9,9,9,9 -C  # 4 x 20GB instances on A100 80GB

# Each tenant gets one MIG instance
```

### Data Isolation

```
┌──────────────────────────────────────────┐
│               Load Balancer               │
├──────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │Tenant A │  │Tenant B │  │Tenant C │  │
│  │Container│  │Container│  │Container│  │
│  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │
│  └─────────┘  └─────────┘  └─────────┘  │
│           No cross-tenant access          │
└──────────────────────────────────────────┘
```

### Performance Isolation

Prevent noisy neighbors:

```yaml
# Resource limits in Kubernetes
resources:
  limits:
    nvidia.com/gpu: 1
    memory: 32Gi
    cpu: 8
  requests:
    memory: 24Gi
    cpu: 4
```

### Model Format Security

Use safe model formats:

```python
# Use safetensors instead of pickle
from safetensors.torch import load_file
weights = load_file("model.safetensors")
```

vLLM and Hugging Face prefer `safetensors` format for security.

---

## Model Serving Patterns

### Single Model, Single GPU

```bash
vllm serve model --port 8000
```

### Single Model, Multi-GPU

```bash
vllm serve large-model --tensor-parallel-size 4
```

### Multiple Models, Multiple GPUs

Run separate servers:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 vllm serve model-a --port 8001

# Terminal 2
CUDA_VISIBLE_DEVICES=1 vllm serve model-b --port 8002
```

### Multiple LoRAs, Single Base

```bash
vllm serve base-model \
  --enable-lora \
  --max-loras 8 \
  --lora-modules \
    adapter1=./loras/adapter1 \
    adapter2=./loras/adapter2
```

---

## Architecture Decision Guide

```
                    ┌─────────────────┐
                    │ Model Size?     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         < 13B          13-70B          > 70B
              │              │              │
              ▼              ▼              ▼
        Single GPU      TP=2-4         TP=4-8
         FP16/INT8    + Quantization   + PP=2+
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────┴────────┐
                    │ Memory still    │
                    │ insufficient?   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Add quantization  CPU offload  More GPUs
```

---

<p align="center">
  <a href="05-performance-tuning.md">← Previous: Performance Tuning</a> | <a href="../README.md">Table of Contents</a> | <a href="07-scaling.md">Next: Scaling →</a>
</p>
