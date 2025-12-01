# Day 002 – GPU Node Bring-Up on RunPod

> **Goal**: Go from zero to serving a real LLM on RunPod, understanding every layer of the stack.  
> **End State**: Serve Llama-3-8B with vLLM, measure baseline metrics, and understand why it's fast.

---

## 📚 Pre-Reading (15 min before you start)

Read these to understand what you're about to do:

| Resource | Why | Time |
|----------|-----|------|
| [RunPod Quick Start](https://docs.runpod.io/pods/overview) | Understand pod types, billing, storage | 5 min |
| [vLLM README](https://github.com/vllm-project/vllm) | See what vLLM claims to do | 5 min |
| [PagedAttention Paper Abstract](https://arxiv.org/abs/2309.06180) | 1-paragraph context on why vLLM exists | 5 min |

---

## Tier 1 – Must-Do Core Block (~2 hours)

**Objective**: Get a GPU pod running, verify the stack, serve your first model.

---

### ✅ Task 1.1: Create RunPod Account & Launch First Pod
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 20 min  
**Win**: You have SSH access to a GPU machine

#### 📖 Learn First
- [RunPod GPU Types & Pricing](https://www.runpod.io/gpu-instance/pricing)
- Understand: RTX 3090 (24GB) vs RTX 4090 (24GB) vs A100 (40/80GB)

#### 🔧 Lab Instructions

1. **Create account** at [runpod.io](https://runpod.io)

2. **Add credits** ($10-25 is enough for a full day of experimentation)

3. **Deploy a GPU Pod**:
   - Go to **Pods** → **Deploy**
   - Select template: `RunPod Pytorch 2.1` (has CUDA + PyTorch pre-installed)
   - GPU: **RTX 4090** (24GB VRAM, ~$0.44/hr) or **RTX 3090** (~$0.31/hr)
   - Container Disk: **20 GB**
   - Volume Disk: **50 GB** (for model weights)
   - Click **Deploy**

4. **Connect via SSH or Web Terminal**:

```bash
# Option 1: Web Terminal (click "Connect" → "Start Web Terminal")
# Option 2: SSH (copy the SSH command from pod details)
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

5. **Verify GPU access**:

```bash
nvidia-smi
```

Expected output shows your GPU (e.g., RTX 4090, 24GB).

#### 🏆 Success Criteria
- [ ] `nvidia-smi` shows GPU with ~24GB VRAM
- [ ] You can run commands in the terminal

#### 📁 Artifact
```bash
nvidia-smi | tee ~/artifacts/nvidia_smi_initial.txt
```

---

### ✅ Task 1.2: Verify CUDA & PyTorch Stack  
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 10 min  
**Win**: Confirmed GPU compute works end-to-end

#### 🔧 Lab Instructions

```bash
# Check CUDA version
nvcc --version

# Check PyTorch sees GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

```bash
# Quick GPU compute test
python3 << 'EOF'
import torch
import time

# Matrix multiplication benchmark
size = 10000
a = torch.randn(size, size, device='cuda')
b = torch.randn(size, size, device='cuda')

torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

tflops = (2 * size**3) / elapsed / 1e12
print(f"Matrix multiply ({size}x{size}): {elapsed*1000:.1f}ms, {tflops:.1f} TFLOPS")
EOF
```

#### 🏆 Success Criteria
- [ ] PyTorch reports `CUDA available: True`
- [ ] Matrix multiply runs without errors
- [ ] You see TFLOPS output (RTX 4090 should hit ~80+ TFLOPS for FP32)

---

### ✅ Task 1.3: Install vLLM & Serve First Model (GPT-2)
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 15 min  
**Win**: vLLM server running, responding to requests

#### 📖 Learn First
- [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

#### 🔧 Lab Instructions

```bash
# Install vLLM
pip install vllm

# Verify installation
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

```bash
# Start vLLM server with tiny model first (GPT-2, 124M params)
vllm serve gpt2 --port 8000 &

# Wait for server to start
sleep 30

# Test with curl
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "The future of AI inference is",
    "max_tokens": 50
  }'
```

#### 🏆 Success Criteria
- [ ] vLLM server starts without errors
- [ ] curl returns a completion
- [ ] You see "The future of AI inference is..." + generated text

```bash
# Stop the server for next task
pkill -f "vllm serve"
```

---

### ✅ Task 1.4: Serve a Real Model – Llama-3-8B
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 30 min (includes download time)  
**Win**: Production-grade 8B model serving on your GPU

#### 📖 Learn First
- [Llama 3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

#### 🔧 Lab Instructions

```bash
# Login to HuggingFace (needed for Llama access)
pip install huggingface_hub
huggingface-cli login
# Paste your HF token (get from https://huggingface.co/settings/tokens)
```

```bash
# Serve Llama-3-8B-Instruct
# Note: First run downloads ~16GB, takes 5-10 min
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee ~/artifacts/vllm_llama3_startup.log &

# Wait for model to load (watch the log)
tail -f ~/artifacts/vllm_llama3_startup.log
# Wait until you see "Uvicorn running on http://0.0.0.0:8000"
```

```bash
# Test the model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain PagedAttention in one paragraph."}
    ],
    "max_tokens": 200
  }' | python3 -m json.tool
```

#### 🏆 Success Criteria
- [ ] Model loads without OOM errors
- [ ] Chat completion returns coherent response about PagedAttention
- [ ] `nvidia-smi` shows ~15-16GB VRAM used

```bash
# Capture GPU state
nvidia-smi | tee ~/artifacts/nvidia_smi_llama3_loaded.txt
```

---

### ✅ Task 1.5: First Benchmark – Measure Baseline Throughput
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 20 min  
**Win**: You have real numbers: tokens/sec, latency

#### 📖 Learn First
- [vLLM Benchmarking Guide](https://docs.vllm.ai/en/latest/performance/benchmarks.html)
- Key metrics: **TTFT** (time to first token), **TPOT** (time per output token), **Throughput** (tokens/sec)

#### 🔧 Lab Instructions

Create a benchmark script:

```bash
cat > ~/benchmark_llama3.py << 'EOF'
import requests
import time
import json
import statistics

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

def benchmark_single_request(prompt, max_tokens=100):
    start = time.time()
    response = requests.post(BASE_URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    })
    elapsed = time.time() - start
    
    data = response.json()
    output_tokens = data["usage"]["completion_tokens"]
    tokens_per_sec = output_tokens / elapsed
    
    return {
        "elapsed_sec": elapsed,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec
    }

# Run 10 requests
prompts = [
    "Write a haiku about GPU computing.",
    "Explain transformers in simple terms.",
    "What is the capital of France?",
    "List 5 benefits of LLM inference optimization.",
    "Write a short poem about CUDA cores.",
    "Explain batch processing in one sentence.",
    "What is KV cache?",
    "Describe PagedAttention briefly.",
    "Why is memory bandwidth important for LLMs?",
    "What is speculative decoding?"
]

results = []
print("Running 10 single-request benchmarks...")
for i, prompt in enumerate(prompts):
    r = benchmark_single_request(prompt)
    results.append(r)
    print(f"  Request {i+1}: {r['tokens_per_sec']:.1f} tok/s, {r['elapsed_sec']:.2f}s")

# Summary
tok_per_sec = [r['tokens_per_sec'] for r in results]
print(f"\n=== BASELINE RESULTS ===")
print(f"Mean throughput: {statistics.mean(tok_per_sec):.1f} tokens/sec")
print(f"Median throughput: {statistics.median(tok_per_sec):.1f} tokens/sec")
print(f"Min/Max: {min(tok_per_sec):.1f} / {max(tok_per_sec):.1f} tokens/sec")

# Save results
with open("/root/artifacts/benchmark_baseline.json", "w") as f:
    json.dump({"results": results, "summary": {
        "mean_tok_per_sec": statistics.mean(tok_per_sec),
        "median_tok_per_sec": statistics.median(tok_per_sec),
    }}, f, indent=2)
print("\nResults saved to ~/artifacts/benchmark_baseline.json")
EOF
```

```bash
python3 ~/benchmark_llama3.py
```

#### 🏆 Success Criteria
- [ ] All 10 requests complete successfully
- [ ] You have baseline numbers (expect ~30-60 tok/s for single requests on RTX 4090)
- [ ] Results saved to JSON

---

## Tier 1 Summary

| Task | Status | Key Metric |
|------|--------|------------|
| 1.1 RunPod Setup | ⬜ | SSH access confirmed |
| 1.2 CUDA Verify | ⬜ | PyTorch sees GPU |
| 1.3 vLLM + GPT-2 | ⬜ | Server responds |
| 1.4 Llama-3-8B | ⬜ | Model loaded, ~16GB VRAM |
| 1.5 Baseline Benchmark | ⬜ | XX tok/s measured |

**Commit after Tier 1:**
```bash
cd ~/artifacts
git init
git add .
git commit -m "day02-tier1: RunPod setup + Llama-3-8B baseline (~XX tok/s)"
```

---

**→ Continue to [Tier 2](LOG_tier02.md) for concurrent request benchmarking and HF vs vLLM comparison**

