# Chapter 11: Ecosystem Players

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Understanding the ecosystem helps you choose the right platform for your needs.

---

## Overview

The LLM inference ecosystem includes various platforms, from bare-metal GPU providers to fully-managed inference services. This chapter maps the landscape.

---

## Landscape Overview

```
                        Abstraction Level
                              ↑
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   Managed APIs          Serverless            Self-Managed
        │                     │                     │
   ┌────┴────┐           ┌────┴────┐           ┌────┴────┐
   │ OpenAI  │           │ Modal   │           │ RunPod  │
   │ Anthropic│           │ Replicate│           │ Lambda  │
   │ Cohere  │           │ Baseten │           │ Vast.ai │
   └─────────┘           │ Fireworks│           │ CoreWeave│
                         │ Together │           └─────────┘
                         └─────────┘
                              ↓
                        Control Level
```

---

## Cloud Platforms

### AWS

| Service | Description | Best For |
|---------|-------------|----------|
| **SageMaker** | Managed ML platform | Enterprise, integrated |
| **EC2 + GPU** | Raw GPU instances | Full control |
| **Inferentia** | Custom inference chips | Cost optimization |
| **Bedrock** | Managed LLM APIs | Quick start |

```bash
# SageMaker deployment example
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_data="s3://bucket/model.tar.gz",
    role="arn:aws:iam::...",
    transformers_version="4.28",
    pytorch_version="2.0"
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1
)
```

### Google Cloud

| Service | Description | Best For |
|---------|-------------|----------|
| **Vertex AI** | Managed ML platform | GCP ecosystem |
| **GKE + GPU** | Kubernetes with GPUs | Scalable deployments |
| **TPU** | Tensor Processing Units | High throughput |

### Azure

| Service | Description | Best For |
|---------|-------------|----------|
| **Azure ML** | Managed ML platform | Microsoft ecosystem |
| **Azure OpenAI** | Managed OpenAI models | Enterprise OpenAI |
| **AKS + GPU** | Kubernetes with GPUs | Self-managed |

---

## Inference Providers

### RunPod

**Focus**: GPU cloud, serverless inference

**Strengths**:
- Competitive GPU pricing
- Serverless endpoints
- Easy scaling
- Community templates

```bash
# RunPod serverless example
runpodctl deploy template \
  --name vllm-llama \
  --gpu "NVIDIA A100" \
  --image "runpod/vllm:latest"
```

**Pricing**: Pay-per-second GPU usage

### Replicate

**Focus**: Model hosting, easy deployment

**Strengths**:
- Simple deployment (Cog format)
- Version control for models
- Public model library
- Streaming support

```python
import replicate

output = replicate.run(
    "meta/llama-2-70b-chat",
    input={"prompt": "Hello, world!"}
)
```

**Pricing**: Per-second compute

### Modal

**Focus**: Serverless Python, GPU compute

**Strengths**:
- Pythonic interface
- Fast cold starts
- GPU scheduling
- Development-friendly

```python
import modal

app = modal.App()

@app.function(gpu="A100")
def inference(prompt: str):
    # Load model and generate
    return response

# Deploy
modal deploy app.py
```

**Pricing**: Pay-per-second, scale to zero

### Together AI

**Focus**: High-performance inference

**Strengths**:
- Optimized inference stack
- Competitive pricing
- Open-source models
- Research partnerships

**Specialties**:
- Speculative decoding
- Quantization
- Custom fine-tuning

### Fireworks AI

**Focus**: Speed, developer experience

**Strengths**:
- Very fast inference
- Function calling
- Easy API
- JSON mode

```python
import fireworks.client

response = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/llama-v3-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Baseten

**Focus**: Production model serving

**Strengths**:
- Truss framework for packaging
- Auto-scaling
- Enterprise features
- Multi-GPU support

```python
import truss

# Package model
truss.init("my-model")

# Deploy
truss.push("my-model")
```

### Anyscale

**Focus**: Ray-based scaling

**Strengths**:
- Ray integration
- Distributed computing
- Aviary (LLM platform)
- Enterprise Ray support

---

## Comparison Matrix

| Provider | Min Cost | Scale to Zero | Multi-GPU | OpenAI API | Best For |
|----------|----------|---------------|-----------|------------|----------|
| RunPod | $ | ✅ | ✅ | ✅ | Cost-effective |
| Replicate | $$ | ✅ | ✅ | ❌ | Easy deployment |
| Modal | $$ | ✅ | ✅ | Via wrapper | Python developers |
| Together | $$ | ✅ | ✅ | ✅ | Performance |
| Fireworks | $$ | ✅ | ✅ | ✅ | Speed |
| Baseten | $$$ | ✅ | ✅ | Via wrapper | Enterprise |
| Anyscale | $$$ | ❌ | ✅ | ✅ | Ray users |
| SageMaker | $$$+ | ❌ | ✅ | Via wrapper | AWS shops |

---

## Self-Managed Options

### Lambda Labs

**Focus**: Simple GPU cloud

**Strengths**:
- Easy instance launching
- Persistent storage
- Reasonable pricing
- Good availability

### Vast.ai

**Focus**: GPU marketplace

**Strengths**:
- Cheapest GPUs
- Spot-like pricing
- Community machines
- Flexible options

**Considerations**: Variable reliability

### CoreWeave

**Focus**: GPU-first cloud

**Strengths**:
- Enterprise-grade GPUs
- Kubernetes native
- Networking optimized
- Large allocations available

### Paperspace / DigitalOcean

**Focus**: Developer-friendly GPU cloud

**Strengths**:
- Simple interface
- Gradient platform
- Notebooks included

---

## Choosing a Provider

### Decision Tree

```
                    ┌─────────────────┐
                    │ What do you     │
                    │ need?           │
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    ▼                        ▼                        ▼
Maximum Control        Easy Deployment           Managed API
    │                        │                        │
    ▼                        ▼                        ▼
┌───────────┐           ┌───────────┐           ┌───────────┐
│RunPod/    │           │Replicate/ │           │Together/  │
│Vast.ai/   │           │Modal/     │           │Fireworks/ │
│CoreWeave  │           │Baseten    │           │OpenAI     │
└───────────┘           └───────────┘           └───────────┘
```

### By Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Prototyping | Modal, Replicate | Quick iteration |
| Cost-sensitive | Vast.ai, RunPod | Low pricing |
| Enterprise | Baseten, SageMaker | Support, compliance |
| Maximum speed | Fireworks, Together | Optimized stacks |
| Custom models | RunPod, Modal | Full control |
| Batch processing | RunPod, Lambda | Persistent instances |

### Cost Comparison (Approximate)

```
A100 80GB hourly cost (approximate)
┌──────────────────────────────────────────────────┐
│ Vast.ai (spot)     ████               $1.50      │
│ RunPod             ██████             $2.00      │
│ Lambda Labs        ████████           $2.50      │
│ CoreWeave          ██████████         $3.00      │
│ AWS p4d            ████████████████   $5.00+     │
│ GCP A2             ████████████████   $5.00+     │
└──────────────────────────────────────────────────┘
Note: Prices change frequently. Check current rates.
```

---

## Emerging Trends

### Edge Inference

- **Groq**: LPU-based ultra-fast inference
- **Cerebras**: Wafer-scale inference
- **SambaNova**: Dataflow architecture

### Specialized Hardware

| Company | Technology | Focus |
|---------|------------|-------|
| Groq | LPU | Lowest latency |
| Cerebras | Wafer-scale | Largest models |
| Graphcore | IPU | Training + inference |
| Tenstorrent | RISC-V AI | Scalable |

### Open Source Infrastructure

- **vLLM**: Community inference engine
- **Ray**: Distributed computing
- **Kubernetes + GPU Operator**: Container orchestration
- **llm-d**: Kubernetes LLM stack

---

## Practical Recommendations

### Starting Out

1. **Prototype**: Modal or Replicate
2. **Benchmark**: RunPod serverless
3. **Production**: Evaluate based on needs

### Scaling Up

1. **Validate**: Run cost/performance tests
2. **Compare**: At least 2-3 providers
3. **Negotiate**: Volume discounts available

### Enterprise

1. **Compliance**: Check SOC2, HIPAA, etc.
2. **Support**: SLA requirements
3. **Hybrid**: Consider on-prem for sensitive data

---

<p align="center">
  <a href="10-comparisons.md">← Previous: Comparisons</a> | <a href="../README.md">Table of Contents</a> | <a href="../appendices/references.md">Next: References →</a>
</p>
