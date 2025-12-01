# Day 002 – Tier 2: Concurrent Benchmarks & HF Comparison

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) first  
> **Goal**: Understand why vLLM is faster than naive HuggingFace, measure concurrent request handling  
> **End State**: Side-by-side benchmark showing vLLM's continuous batching advantage

---

## 📚 Pre-Reading (20 min)

| Resource | Why | Time |
|----------|-----|------|
| [Continuous Batching Explained](https://www.anyscale.com/blog/continuous-batching-llm-inference) | Core concept that makes vLLM fast | 10 min |
| [vLLM Blog: How it works](https://blog.vllm.ai/2023/06/20/vllm.html) | Official explanation of PagedAttention | 10 min |

#### 🎥 Video (watch during breaks)
- [vLLM Talk at Ray Summit](https://www.youtube.com/watch?v=80bIUggRJf4) - Deep dive into architecture (30 min)

---

## Tier 2 – Extension Block (~2 hours)

**Objective**: Compare vLLM vs HuggingFace, understand continuous batching with concurrent requests.

---

### ✅ Task 2.1: Set Up HuggingFace Baseline Server
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 25 min  
**Win**: HF server running same model, ready for comparison

#### 📖 Learn First
- [HuggingFace Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- Why HF is slower: No continuous batching, no PagedAttention, sequential processing

#### 🔧 Lab Instructions

First, stop any running vLLM server:
```bash
pkill -f "vllm serve"
```

Create a simple HF-based server:

```bash
cat > ~/hf_server.py << 'EOF'
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

app = Flask(__name__)

print("Loading Llama-3-8B with HuggingFace...")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)
print("Model loaded!")

@app.route("/v1/completions", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "Hello")
    max_tokens = data.get("max_tokens", 100)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_tokens = len(outputs[0]) - len(inputs.input_ids[0])
    
    return jsonify({
        "text": generated,
        "usage": {"completion_tokens": output_tokens},
        "elapsed_sec": elapsed
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True)
EOF
```

```bash
pip install flask

# Run HF server on port 8001
python3 ~/hf_server.py 2>&1 | tee ~/artifacts/hf_server.log &

# Wait for model to load
sleep 60

# Test it
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain KV cache:", "max_tokens": 50}'
```

#### 🏆 Success Criteria
- [ ] HF server starts and loads model
- [ ] Test request returns completion
- [ ] Note: Will use ~16GB VRAM (same as vLLM)

---

### ✅ Task 2.2: Head-to-Head Single Request Comparison
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 20 min  
**Win**: Direct comparison showing vLLM is faster even for single requests

#### 🔧 Lab Instructions

Start vLLM server on different port:
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.45 \
  2>&1 | tee ~/artifacts/vllm_comparison.log &

# Wait for startup
sleep 45
```

Create comparison benchmark:

```bash
cat > ~/compare_single_request.py << 'EOF'
import requests
import time
import json

def benchmark_hf(prompt, max_tokens=100):
    start = time.time()
    r = requests.post("http://localhost:8001/v1/completions", json={
        "prompt": prompt,
        "max_tokens": max_tokens
    })
    elapsed = time.time() - start
    data = r.json()
    return {
        "engine": "HuggingFace",
        "elapsed": elapsed,
        "tokens": data["usage"]["completion_tokens"],
        "tok_per_sec": data["usage"]["completion_tokens"] / elapsed
    }

def benchmark_vllm(prompt, max_tokens=100):
    start = time.time()
    r = requests.post("http://localhost:8000/v1/completions", json={
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens
    })
    elapsed = time.time() - start
    data = r.json()
    tokens = data["usage"]["completion_tokens"]
    return {
        "engine": "vLLM",
        "elapsed": elapsed,
        "tokens": tokens,
        "tok_per_sec": tokens / elapsed
    }

prompts = [
    "Write a detailed explanation of how transformers work:",
    "Explain the concept of attention mechanism in neural networks:",
    "What are the key differences between GPT and BERT architectures?"
]

print("=" * 60)
print("SINGLE REQUEST COMPARISON: HuggingFace vs vLLM")
print("=" * 60)

results = []
for i, prompt in enumerate(prompts):
    print(f"\nPrompt {i+1}: {prompt[:50]}...")
    
    hf_result = benchmark_hf(prompt, max_tokens=100)
    print(f"  HF:   {hf_result['tok_per_sec']:.1f} tok/s ({hf_result['elapsed']:.2f}s)")
    
    vllm_result = benchmark_vllm(prompt, max_tokens=100)
    print(f"  vLLM: {vllm_result['tok_per_sec']:.1f} tok/s ({vllm_result['elapsed']:.2f}s)")
    
    speedup = hf_result['elapsed'] / vllm_result['elapsed']
    print(f"  → vLLM is {speedup:.1f}x faster")
    
    results.append({"prompt": prompt[:50], "hf": hf_result, "vllm": vllm_result, "speedup": speedup})

# Save results
with open("/root/artifacts/single_request_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"Average speedup: {sum(r['speedup'] for r in results)/len(results):.1f}x")
print("=" * 60)
EOF
```

```bash
python3 ~/compare_single_request.py
```

#### 🏆 Success Criteria
- [ ] Both servers respond successfully
- [ ] vLLM shows speedup (typically 1.5-3x for single requests)
- [ ] Results saved to JSON

---

### ✅ Task 2.3: Concurrent Requests – Where vLLM Really Shines
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 30 min  
**Win**: See continuous batching in action with 10+ concurrent requests

#### 📖 Learn First
This is the key insight:
- **HF**: Processes requests one-by-one, GPU idle between batches
- **vLLM**: Continuous batching - dynamically batches all active requests, GPU always busy

#### 🔧 Lab Instructions

```bash
cat > ~/concurrent_benchmark.py << 'EOF'
import requests
import time
import json
import concurrent.futures
import statistics

def single_request_hf(prompt):
    start = time.time()
    try:
        r = requests.post("http://localhost:8001/v1/completions", json={
            "prompt": prompt,
            "max_tokens": 50
        }, timeout=120)
        elapsed = time.time() - start
        return {"success": True, "elapsed": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e)}

def single_request_vllm(prompt):
    start = time.time()
    try:
        r = requests.post("http://localhost:8000/v1/completions", json={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "prompt": prompt,
            "max_tokens": 50
        }, timeout=120)
        elapsed = time.time() - start
        return {"success": True, "elapsed": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_concurrent_benchmark(request_func, num_requests, label):
    prompts = [f"Question {i}: Explain concept {i} briefly." for i in range(num_requests)]
    
    print(f"\n{label}: Sending {num_requests} concurrent requests...")
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(request_func, prompts))
    
    total_time = time.time() - start
    successful = [r for r in results if r["success"]]
    latencies = [r["elapsed"] for r in successful]
    
    return {
        "total_time": total_time,
        "successful": len(successful),
        "failed": num_requests - len(successful),
        "throughput_req_per_sec": len(successful) / total_time,
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "p95_latency": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else 0
    }

print("=" * 70)
print("CONCURRENT REQUEST BENCHMARK: HuggingFace vs vLLM")
print("=" * 70)

all_results = {}
for num_concurrent in [1, 5, 10, 20]:
    print(f"\n{'='*70}")
    print(f"Testing with {num_concurrent} concurrent requests")
    print(f"{'='*70}")
    
    # Test HF
    hf_result = run_concurrent_benchmark(single_request_hf, num_concurrent, "HuggingFace")
    print(f"  HF:   {hf_result['throughput_req_per_sec']:.2f} req/s, "
          f"mean latency: {hf_result['mean_latency']:.2f}s, "
          f"p95: {hf_result['p95_latency']:.2f}s")
    
    # Test vLLM
    vllm_result = run_concurrent_benchmark(single_request_vllm, num_concurrent, "vLLM")
    print(f"  vLLM: {vllm_result['throughput_req_per_sec']:.2f} req/s, "
          f"mean latency: {vllm_result['mean_latency']:.2f}s, "
          f"p95: {vllm_result['p95_latency']:.2f}s")
    
    if hf_result['throughput_req_per_sec'] > 0:
        speedup = vllm_result['throughput_req_per_sec'] / hf_result['throughput_req_per_sec']
        print(f"  → vLLM throughput is {speedup:.1f}x higher")
    
    all_results[num_concurrent] = {"hf": hf_result, "vllm": vllm_result}

# Save results
with open("/root/artifacts/concurrent_benchmark.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print("KEY INSIGHT: vLLM's advantage grows with concurrency!")
print("This is continuous batching in action.")
print("=" * 70)
EOF
```

```bash
python3 ~/concurrent_benchmark.py 2>&1 | tee ~/artifacts/concurrent_benchmark.log
```

#### 🏆 Success Criteria
- [ ] Both engines handle concurrent requests
- [ ] vLLM shows increasing advantage at higher concurrency (5x+ at 20 concurrent)
- [ ] You understand WHY: continuous batching vs sequential processing

---

### ✅ Task 2.4: Visualize the Results
**Tags**: `[Inference–Runtime]` `[Business]`  
**Time**: 15 min  
**Win**: A chart you can show to demonstrate vLLM's value

#### 🔧 Lab Instructions

```bash
pip install matplotlib

cat > ~/visualize_results.py << 'EOF'
import json
import matplotlib.pyplot as plt

# Load data
with open("/root/artifacts/concurrent_benchmark.json") as f:
    data = json.load(f)

concurrency = sorted([int(k) for k in data.keys()])
hf_throughput = [data[str(c)]["hf"]["throughput_req_per_sec"] for c in concurrency]
vllm_throughput = [data[str(c)]["vllm"]["throughput_req_per_sec"] for c in concurrency]

# Create chart
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(concurrency))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], hf_throughput, width, label='HuggingFace', color='#ff6b6b')
bars2 = ax.bar([i + width/2 for i in x], vllm_throughput, width, label='vLLM', color='#4ecdc4')

ax.set_xlabel('Concurrent Requests')
ax.set_ylabel('Throughput (requests/sec)')
ax.set_title('vLLM vs HuggingFace: Throughput at Different Concurrency Levels\nLlama-3-8B on RTX 4090')
ax.set_xticks(x)
ax.set_xticklabels(concurrency)
ax.legend()

# Add speedup labels
for i, (hf, vllm) in enumerate(zip(hf_throughput, vllm_throughput)):
    if hf > 0:
        speedup = vllm / hf
        ax.annotate(f'{speedup:.1f}x', xy=(i, vllm), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/root/artifacts/hf_vs_vllm_throughput.png', dpi=150)
print("Chart saved to ~/artifacts/hf_vs_vllm_throughput.png")
EOF
```

```bash
python3 ~/visualize_results.py
```

#### 🏆 Success Criteria
- [ ] Chart generated showing clear vLLM advantage
- [ ] Speedup numbers visible on chart

---

## Tier 2 Summary

| Task | Status | Key Finding |
|------|--------|-------------|
| 2.1 HF Server Setup | ⬜ | HF baseline running |
| 2.2 Single Request | ⬜ | vLLM ~Xx faster |
| 2.3 Concurrent | ⬜ | vLLM ~Xx faster at 20 concurrent |
| 2.4 Visualization | ⬜ | Chart shows scaling |

**Key Learning**: vLLM's continuous batching gives **increasing returns** as concurrency grows. This is why it's used in production.

**Commit after Tier 2:**
```bash
cd ~/artifacts
git add .
git commit -m "day02-tier2: HF vs vLLM comparison (Xx speedup at 20 concurrent)"
```

**Cleanup:**
```bash
pkill -f "hf_server"
pkill -f "vllm serve"
```

---

**→ Continue to [Tier 3](LOG_tier03.md) for vLLM configuration tuning and quantization**

