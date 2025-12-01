# Chapter 4: Setup & Basic Usage

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Hands-on practice is essential for mastering inference engineering.

---

## Overview

This chapter walks through setting up vLLM, serving your first model, and integrating with the OpenAI-compatible API. By the end, you'll have a working inference server.

---

## Installation

### Prerequisites

1. **GPU with CUDA support** (compute capability 7.0+)
2. **Python 3.8+**
3. **CUDA 11.8 or 12.x**

### Verify GPU Access

```bash
# Check GPU visibility
nvidia-smi

# Optional: check CUDA version
nvcc --version 2>/dev/null || echo "nvcc not installed (fine for serving)"
```

### Install vLLM

```bash
# Create virtual environment
python3 -m venv vllm-env
source vllm-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install vLLM
pip install "vllm>=0.6.0"

# Install OpenAI client (for testing)
pip install "openai>=1.40.0"
```

### Set Up Hugging Face Token

For gated models (e.g., Llama):

```bash
export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"
```

Or login via CLI:

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Serve an SLM on a Single GPU with vLLM

### Quick Test with Small Model

First, verify everything works with a tiny model:

```bash
source vllm-env/bin/activate
vllm serve facebook/opt-125m --port 8000 --dry-run
```

The `--dry-run` flag loads the model and exits without starting the server. If you see weight loading logs without errors, proceed.

### Serve a Production Model

For a proper 7-8B model:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --dtype auto \
  --enforce-eager \
  --model-name llama3-8b
```

### Key Serving Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--gpu-memory-utilization` | Fraction of GPU memory to use | 0.85-0.95 |
| `--max-model-len` | Maximum sequence length | Match your use case |
| `--max-num-seqs` | Maximum concurrent sequences | Start with 16-32 |
| `--max-num-batched-tokens` | Maximum tokens per batch | 2048-4096 |
| `--dtype` | Data type (auto, float16, bfloat16) | `auto` |
| `--enforce-eager` | Disable CUDA graphs | For debugging |
| `--tensor-parallel-size` | Number of GPUs for parallelism | 1 for single GPU |

### Monitor GPU During Serving

In a separate terminal:

```bash
# Log GPU metrics every second
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits \
  -l 1 > nvidia_smi_log.csv
```

---

## Testing the Server

### Basic cURL Request

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### Expected Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "llama3-8b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The capital of France is Paris."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 8,
    "total_tokens": 22
  }
}
```

---

## Enable Streaming Responses

Streaming returns tokens as they're generated, reducing time-to-first-token perception:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [
      {"role": "user", "content": "Explain relativity in one sentence."}
    ],
    "stream": true
  }'
```

### Streamed Output Format

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"In"},...}]}
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":" essence"},...}]}
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":","},...}]}
...
data: [DONE]
```

### Why Streaming Matters

| Metric | Without Streaming | With Streaming |
|--------|-------------------|----------------|
| User sees first token | After full generation | Immediately |
| Perceived latency | High | Low |
| UX for long outputs | Poor | Excellent |

---

## Use the OpenAI Python SDK

vLLM is OpenAI API-compatible, so you can use the official SDK:

### Basic Usage

```python
from openai import OpenAI

# Point to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't require auth by default
)

response = client.chat.completions.create(
    model="llama3-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming with SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

stream = client.chat.completions.create(
    model="llama3-8b",
    messages=[
        {"role": "user", "content": "Write a haiku about coding."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Newline at end
```

### Completions API (Non-Chat)

```python
response = client.completions.create(
    model="llama3-8b",
    prompt="The meaning of life is",
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].text)
```

---

## Common Serving Configurations

### Development/Testing

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --enforce-eager  # Easier debugging
```

### Production (Single GPU)

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --disable-log-requests  # Reduce logging overhead
```

### Multi-GPU (Tensor Parallel)

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

### Quantized Model

```bash
# GPTQ quantized
vllm serve TheBloke/Llama-2-7B-GPTQ \
  --quantization gptq \
  --port 8000

# AWQ quantized
vllm serve TheBloke/Llama-2-7B-AWQ \
  --quantization awq \
  --port 8000
```

---

## Benchmarking Your Setup

### Simple Throughput Test

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

prompts = [
    "Explain quantum computing.",
    "What is machine learning?",
    "Describe the solar system.",
] * 10  # 30 requests

start = time.time()
for prompt in prompts:
    response = client.chat.completions.create(
        model="llama3-8b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
elapsed = time.time() - start

print(f"Processed {len(prompts)} requests in {elapsed:.2f}s")
print(f"Throughput: {len(prompts)/elapsed:.2f} requests/sec")
```

### Concurrent Load Test

```python
import asyncio
from openai import AsyncOpenAI

async def make_request(client, prompt):
    return await client.chat.completions.create(
        model="llama3-8b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

async def benchmark(num_concurrent=10):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    prompts = ["Explain AI briefly."] * num_concurrent
    
    start = time.time()
    await asyncio.gather(*[make_request(client, p) for p in prompts])
    elapsed = time.time() - start
    
    print(f"{num_concurrent} concurrent requests: {elapsed:.2f}s")
    print(f"Throughput: {num_concurrent/elapsed:.2f} req/s")

asyncio.run(benchmark(10))
asyncio.run(benchmark(32))
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Error | Model too large | Reduce `--gpu-memory-utilization`, use quantization |
| Slow first request | Model loading | Use `--enforce-eager=False` for CUDA graphs |
| Connection refused | Server not ready | Wait for "Running on http://..." message |
| CUDA version mismatch | Driver/toolkit mismatch | Align CUDA versions |

### Debug Mode

```bash
# Enable verbose logging
VLLM_LOGGING_LEVEL=DEBUG vllm serve model --port 8000
```

### Check Server Health

```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

---

## What's Next

Now that you have a working server:
- **Chapter 5**: Learn to optimize throughput and latency
- **Chapter 6**: Serve different model architectures
- **Chapter 7**: Scale to multi-GPU deployments

---

<p align="center">
  <a href="03-vllm-architecture.md">← Previous: vLLM Architecture</a> | <a href="../README.md">Table of Contents</a> | <a href="05-performance-tuning.md">Next: Performance Tuning →</a>
</p>
