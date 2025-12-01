# Day 002 – Tier 3: Quantization & Advanced Configuration

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) and [Tier 2](LOG_tier02.md) first  
> **Goal**: Push performance further with quantization and understand vLLM's tuning knobs  
> **End State**: AWQ-quantized model running 2x faster, production-ready configuration

---

## Pre-Reading (30 min)

| Resource | Why | Time |
|----------|-----|------|
| [AWQ Paper](https://arxiv.org/abs/2306.00978) | Understand activation-aware quantization | 10 min |
| [vLLM Quantization Docs](https://docs.vllm.ai/en/latest/quantization/supported_hardware.html) | Which formats work on which GPUs | 10 min |
| [TheBloke's Quantization Guide](https://huggingface.co/TheBloke) | Practical quantization overview | 10 min |

#### Videos
- [Quantization Explained](https://www.youtube.com/watch?v=IxrlHAJtqKE) - MIT lecture on quantization basics (20 min)
- [AWQ Deep Dive](https://www.youtube.com/watch?v=3GULXE7U7mY) - Practical AWQ walkthrough (15 min)

#### GitHub Repos to Explore
- [vllm-project/vllm](https://github.com/vllm-project/vllm) - Source code
- [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) - AWQ implementation
- [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) - GPTQ implementation

---

## Tier 3 – Deep Work Block (~2-3 hours)

**Objective**: Serve quantized models, compare FP16 vs INT4, understand memory-throughput tradeoffs.

---

### Task 3.1: Serve AWQ-Quantized Llama-3-8B
**Tags**: `[Inference–Runtime]` `[Phase2-Quantization]`  
**Time**: 30 min  
**Win**: 4-bit model running, using ~50% less memory

#### Learn First
- **AWQ (Activation-aware Weight Quantization)**: Preserves important weights, quantizes less important ones
- **INT4**: 4-bit integers instead of FP16 (16-bit), ~4x smaller model
- **Why it works**: LLM weights have varying importance; smart quantization keeps accuracy

#### Lab Instructions

```bash
# Stop any running servers
pkill -f "vllm serve"

# Serve AWQ-quantized model
# Note: Model downloads automatically, ~4GB instead of ~16GB
vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee ~/artifacts/vllm_awq_startup.log &

# Wait for model to load
sleep 30

# Check memory usage
nvidia-smi | tee ~/artifacts/nvidia_smi_awq.txt
```

```bash
# Test the quantized model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Llama-2-7B-Chat-AWQ",
    "messages": [{"role": "user", "content": "Explain quantization in ML briefly."}],
    "max_tokens": 100
  }' | python3 -m json.tool
```

#### Success Criteria
- [ ] AWQ model loads successfully
- [ ] Memory usage: ~4-5GB instead of ~14GB
- [ ] Response quality is coherent (not gibberish)

---

### Task 3.2: FP16 vs AWQ Benchmark
**Tags**: `[Inference–Runtime]` `[Phase2-Quantization]`  
**Time**: 30 min  
**Win**: Quantified speedup and quality comparison

#### Lab Instructions

First, let's also start the FP16 model for comparison:

```bash
# Start FP16 model on different port (if you have enough VRAM)
# Or do sequential testing
pkill -f "vllm serve"

# We'll benchmark AWQ first, then FP16
```

Create comprehensive benchmark:

```bash
cat > ~/quantization_benchmark.py << 'EOF'
import requests
import time
import json
import subprocess

def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

def benchmark_model(base_url, model_name, num_requests=10, max_tokens=100):
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What is the difference between AI and ML?",
        "How does gradient descent work?",
        "Explain neural networks briefly.",
        "What is overfitting?",
        "Describe the transformer architecture.",
        "What is attention in deep learning?",
        "Explain backpropagation.",
        "What are embeddings?",
        "Describe transfer learning."
    ][:num_requests]
    
    results = []
    for prompt in prompts:
        start = time.time()
        r = requests.post(f"{base_url}/v1/chat/completions", json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        })
        elapsed = time.time() - start
        
        data = r.json()
        tokens = data["usage"]["completion_tokens"]
        results.append({
            "elapsed": elapsed,
            "tokens": tokens,
            "tok_per_sec": tokens / elapsed
        })
    
    avg_tok_per_sec = sum(r["tok_per_sec"] for r in results) / len(results)
    gpu_mem = get_gpu_memory()
    
    return {
        "model": model_name,
        "avg_tok_per_sec": avg_tok_per_sec,
        "gpu_memory_mb": gpu_mem,
        "results": results
    }

# Benchmark AWQ model
print("=" * 60)
print("QUANTIZATION BENCHMARK")
print("=" * 60)

awq_results = benchmark_model(
    "http://localhost:8000",
    "TheBloke/Llama-2-7B-Chat-AWQ"
)
print(f"\nAWQ (INT4):")
print(f"  Throughput: {awq_results['avg_tok_per_sec']:.1f} tok/s")
print(f"  GPU Memory: {awq_results['gpu_memory_mb']} MB")

# Save AWQ results
with open("/root/artifacts/awq_benchmark.json", "w") as f:
    json.dump(awq_results, f, indent=2)

print("\nAWQ results saved. Now restart with FP16 model to compare.")
print("Run: pkill -f 'vllm serve' && vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000")
EOF
```

```bash
# Benchmark AWQ
python3 ~/quantization_benchmark.py
```

Now test FP16 (requires stopping AWQ server):

```bash
# Stop AWQ server
pkill -f "vllm serve"

# Start FP16 model
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee ~/artifacts/vllm_fp16_startup.log &

sleep 45

# Benchmark FP16
cat > ~/benchmark_fp16.py << 'EOF'
import requests
import time
import json
import subprocess

def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

prompts = [
    "Explain the concept of machine learning in simple terms.",
    "What is the difference between AI and ML?",
    "How does gradient descent work?",
    "Explain neural networks briefly.",
    "What is overfitting?",
]

results = []
for prompt in prompts:
    start = time.time()
    r = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    })
    elapsed = time.time() - start
    data = r.json()
    tokens = data["usage"]["completion_tokens"]
    results.append({"elapsed": elapsed, "tokens": tokens, "tok_per_sec": tokens / elapsed})

avg_tok = sum(r["tok_per_sec"] for r in results) / len(results)
gpu_mem = get_gpu_memory()

print(f"\nFP16:")
print(f"  Throughput: {avg_tok:.1f} tok/s")
print(f"  GPU Memory: {gpu_mem} MB")

# Load AWQ results and compare
with open("/root/artifacts/awq_benchmark.json") as f:
    awq = json.load(f)

print(f"\n{'='*60}")
print("COMPARISON: AWQ vs FP16")
print(f"{'='*60}")
print(f"AWQ (INT4): {awq['avg_tok_per_sec']:.1f} tok/s, {awq['gpu_memory_mb']} MB")
print(f"FP16:       {avg_tok:.1f} tok/s, {gpu_mem} MB")
print(f"\nMemory savings: {(1 - awq['gpu_memory_mb']/gpu_mem)*100:.0f}%")
speedup = awq['avg_tok_per_sec'] / avg_tok if avg_tok > 0 else 0
print(f"Throughput change: {speedup:.2f}x")

# Save comparison
with open("/root/artifacts/quantization_comparison.json", "w") as f:
    json.dump({
        "awq": awq,
        "fp16": {"avg_tok_per_sec": avg_tok, "gpu_memory_mb": gpu_mem, "results": results},
        "memory_savings_pct": (1 - awq['gpu_memory_mb']/gpu_mem)*100,
        "speedup": speedup
    }, f, indent=2)
EOF

python3 ~/benchmark_fp16.py
```

#### Success Criteria
- [ ] Both models benchmarked
- [ ] AWQ uses ~50-70% less memory
- [ ] AWQ similar or better throughput (memory-bound → faster)

---

### Task 3.3: vLLM Configuration Deep Dive
**Tags**: `[Inference–Runtime]` `[Phase3-Optimization]`  
**Time**: 45 min  
**Win**: Understand key vLLM flags and their impact

#### Learn First
- [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/models/engine_args.html)

Key parameters to understand:
- `--gpu-memory-utilization`: How much VRAM to use (0.0-1.0)
- `--max-model-len`: Maximum sequence length
- `--max-num-seqs`: Maximum concurrent sequences
- `--enforce-eager`: Disable CUDA graphs (for debugging)

#### Lab Instructions

```bash
pkill -f "vllm serve"

# Experiment 1: Low memory utilization
echo "Test 1: gpu-memory-utilization=0.5"
vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.5 \
  --max-num-seqs 8 &
sleep 30

curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TheBloke/Llama-2-7B-Chat-AWQ", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}' > /dev/null

nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee ~/artifacts/config_test1.txt
pkill -f "vllm serve"
sleep 5

# Experiment 2: High memory utilization
echo "Test 2: gpu-memory-utilization=0.95"
vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 &
sleep 30

curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TheBloke/Llama-2-7B-Chat-AWQ", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}' > /dev/null

nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee ~/artifacts/config_test2.txt
```

Create a configuration test script:

```bash
cat > ~/config_benchmark.py << 'EOF'
import requests
import time
import json
import concurrent.futures

def concurrent_test(num_requests=20):
    """Test throughput with concurrent requests"""
    def single_request(i):
        start = time.time()
        r = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "TheBloke/Llama-2-7B-Chat-AWQ",
            "messages": [{"role": "user", "content": f"Question {i}: Explain briefly."}],
            "max_tokens": 50
        })
        return time.time() - start
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as ex:
        latencies = list(ex.map(single_request, range(num_requests)))
    total_time = time.time() - start
    
    return {
        "total_time": total_time,
        "throughput_req_per_sec": num_requests / total_time,
        "mean_latency": sum(latencies) / len(latencies)
    }

result = concurrent_test(20)
print(f"Throughput: {result['throughput_req_per_sec']:.2f} req/s")
print(f"Mean latency: {result['mean_latency']:.2f}s")
EOF

python3 ~/config_benchmark.py | tee ~/artifacts/high_util_benchmark.txt
```

#### Success Criteria
- [ ] Understand how `gpu-memory-utilization` affects memory
- [ ] Higher utilization → more KV cache → more concurrent requests
- [ ] Document findings in notes

---

### Task 3.4: Production-Ready Configuration
**Tags**: `[Inference–Runtime]` `[Phase3-Optimization]` `[Business]`  
**Time**: 30 min  
**Win**: A documented, optimized configuration for a real use case

#### Lab Instructions

Create two production configs:

```bash
mkdir -p ~/configs

# Config 1: Latency-optimized (for chat applications)
cat > ~/configs/latency_optimized.sh << 'EOF'
#!/bin/bash
# Latency-Optimized Configuration
# Use case: Chat applications, interactive use
# Priority: Low TTFT, consistent response times

vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-model-len 2048 \
  --disable-log-requests
EOF

# Config 2: Throughput-optimized (for batch processing)
cat > ~/configs/throughput_optimized.sh << 'EOF'
#!/bin/bash
# Throughput-Optimized Configuration
# Use case: Batch processing, offline tasks
# Priority: Maximum requests per second

vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 \
  --max-model-len 4096 \
  --disable-log-requests
EOF

chmod +x ~/configs/*.sh
```

Document your findings:

```bash
cat > ~/artifacts/day02_config_guide.md << 'EOF'
# vLLM Configuration Guide - Day 02 Findings

## Hardware
- GPU: RTX 4090 (24GB)
- Model: Llama-2-7B-Chat-AWQ (INT4)

## Key Parameters Tested

### gpu-memory-utilization
- 0.5: ~12GB used, conservative, good for debugging
- 0.85: ~20GB used, balanced for production
- 0.95: ~22GB used, maximum throughput

### max-num-seqs
- 8: Low concurrency, lower latency
- 16: Balanced
- 32-64: High throughput, higher latency per request

## Recommended Configurations

### Chat/Interactive (latency-sensitive)
```
--gpu-memory-utilization 0.85
--max-num-seqs 16
--max-model-len 2048
```

### Batch Processing (throughput-sensitive)
```
--gpu-memory-utilization 0.95
--max-num-seqs 64
--max-model-len 4096
```

## Benchmark Results
- AWQ vs FP16: ~60% memory savings
- Throughput: [YOUR NUMBERS] tok/s
- Concurrent requests: [YOUR NUMBERS] req/s at 20 concurrent
EOF
```

#### Success Criteria
- [ ] Two production configs created
- [ ] Configuration guide documented
- [ ] You can explain when to use each config

---

## Tier 3 Summary

| Task | Status | Key Finding |
|------|--------|-------------|
| 3.1 AWQ Model | ⬜ | ~4-5GB memory usage |
| 3.2 FP16 vs AWQ | ⬜ | XX% memory savings |
| 3.3 Config Deep Dive | ⬜ | Understand key params |
| 3.4 Production Config | ⬜ | 2 documented configs |

**Key Learning**: Quantization gives massive memory savings with minimal quality loss. Configuration tuning depends on use case (latency vs throughput).

**Commit after Tier 3:**
```bash
cd ~/artifacts
git add .
git commit -m "day02-tier3: AWQ quantization (XX% memory savings) + production configs"
```

---

**→ Continue to [Tier 4](LOG_tier04.md) for the Boss Challenge: Full Case Study**
