# Day 002 – Tier 4: BOSS CHALLENGE 🏆

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md), [Tier 2](LOG_tier02.md), and [Tier 3](LOG_tier03.md)  
> **Goal**: Create a complete, publishable case study showing your optimization journey  
> **End State**: A professional case study document with before/after metrics, ready to share

---

## 🎯 The Challenge

**Create a consulting-style case study**: "Optimizing LLM Inference: From 30 tok/s to 150+ tok/s"

This is what separates a hobbyist from a professional. You will:
1. Define a realistic client scenario
2. Document baseline performance
3. Apply optimizations systematically
4. Quantify improvements
5. Package findings professionally

---

## 📚 Pre-Reading (15 min)

| Resource | Why | Time |
|----------|-----|------|
| [How to Write a Case Study](https://blog.hubspot.com/marketing/how-to-write-case-study) | Structure for business impact | 5 min |
| [Anyscale LLM Perf Blog](https://www.anyscale.com/blog/llm-performance-benchmark) | Example of professional benchmarking | 10 min |

#### 📂 Example Case Studies to Model
- [vLLM Performance Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [TensorRT-LLM Benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/H100vsA100.md)

---

## Tier 4 – Boss Challenge (~2-3 hours)

---

### ✅ Task 4.1: Define the Client Scenario
**Tags**: `[Business]` `[Phase3-Optimization]`  
**Time**: 20 min  
**Win**: Clear problem statement that sounds like a real consulting engagement

#### 🔧 Lab Instructions

```bash
cat > ~/artifacts/case_study.md << 'EOF'
# Case Study: Optimizing LLM Inference for Production Chat

## Executive Summary
[FILL IN AT END]

## Client Scenario

**Company**: TechStartup Inc. (fictional)  
**Use Case**: Customer support chatbot  
**Requirements**:
- Model: Llama-2-7B equivalent quality
- Latency: < 2 seconds time-to-first-token
- Throughput: Support 50+ concurrent users
- Budget: Single RTX 4090 GPU ($2000)
- Uptime: 99.9%

**Current State** (Before Optimization):
- Using HuggingFace Transformers with default settings
- Throughput: ~X req/s at 10 concurrent users
- Memory: 14GB VRAM
- Cost: Can only serve ~Y concurrent users

**Goal**:
- 5x throughput improvement
- 50% memory reduction
- Same or better quality

---

## Methodology

### Phase 1: Baseline Measurement
[FILL IN]

### Phase 2: Runtime Optimization (vLLM)
[FILL IN]

### Phase 3: Quantization (AWQ)
[FILL IN]

### Phase 4: Configuration Tuning
[FILL IN]

---

## Results

### Before vs After Comparison

| Metric | Before (HF) | After (vLLM+AWQ) | Improvement |
|--------|-------------|------------------|-------------|
| Single Request Latency | X sec | Y sec | Z% faster |
| Throughput (10 concurrent) | X req/s | Y req/s | Zx |
| Throughput (20 concurrent) | X req/s | Y req/s | Zx |
| GPU Memory | X GB | Y GB | Z% less |
| Max Concurrent Users | X | Y | Zx |

### Key Findings
1. [FILL IN]
2. [FILL IN]
3. [FILL IN]

---

## Recommendations

### Immediate Actions
1. Switch from HuggingFace to vLLM
2. Use AWQ quantization
3. Configure for throughput: `--gpu-memory-utilization 0.95 --max-num-seqs 64`

### Future Optimizations
1. Consider speculative decoding for latency-sensitive paths
2. Evaluate tensor parallelism if scaling to multiple GPUs
3. Implement proper monitoring with Prometheus/Grafana

---

## ROI Analysis

**Hardware Cost**: $2000 (RTX 4090)
**Throughput Improvement**: Xx
**Effective Cost Reduction**: From $Y/1M tokens to $Z/1M tokens

---

## Appendix

### A. Benchmark Scripts
See `~/artifacts/` for all benchmark code.

### B. Configuration Files
See `~/configs/` for production-ready configurations.

### C. Raw Data
See `~/artifacts/*.json` for all benchmark results.
EOF
```

#### 🏆 Success Criteria
- [ ] Case study template created
- [ ] Client scenario is realistic and relatable

---

### ✅ Task 4.2: Run Complete Benchmark Suite
**Tags**: `[Inference–Runtime]` `[Phase3-Optimization]`  
**Time**: 45 min  
**Win**: All numbers collected for case study

#### 🔧 Lab Instructions

Create a comprehensive benchmark that tests everything:

```bash
cat > ~/full_benchmark_suite.py << 'EOF'
#!/usr/bin/env python3
"""
Complete Benchmark Suite for Day 02 Case Study
Tests: HF baseline, vLLM FP16, vLLM AWQ at multiple concurrency levels
"""

import requests
import time
import json
import subprocess
import concurrent.futures
import statistics
import os

RESULTS = {}

def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

def run_concurrent_test(url, model, num_requests, max_tokens=50):
    """Run concurrent requests and measure throughput"""
    
    def single_request(i):
        prompt = f"Question {i}: Explain this concept briefly in 2-3 sentences."
        start = time.time()
        try:
            r = requests.post(f"{url}/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }, timeout=120)
            elapsed = time.time() - start
            data = r.json()
            tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
            return {"success": True, "elapsed": elapsed, "tokens": tokens}
        except Exception as e:
            return {"success": False, "error": str(e), "elapsed": time.time() - start}
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as ex:
        results = list(ex.map(single_request, range(num_requests)))
    total_time = time.time() - start
    
    successful = [r for r in results if r["success"]]
    if not successful:
        return None
    
    latencies = [r["elapsed"] for r in successful]
    total_tokens = sum(r["tokens"] for r in successful)
    
    return {
        "num_requests": num_requests,
        "successful": len(successful),
        "failed": num_requests - len(successful),
        "total_time": total_time,
        "throughput_req_per_sec": len(successful) / total_time,
        "throughput_tok_per_sec": total_tokens / total_time,
        "mean_latency": statistics.mean(latencies),
        "p50_latency": statistics.median(latencies),
        "p95_latency": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[0],
        "gpu_memory_mb": get_gpu_memory()
    }

def test_configuration(name, url, model, concurrency_levels=[1, 5, 10, 20, 30]):
    """Test a configuration at multiple concurrency levels"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    results = {"name": name, "model": model, "tests": {}}
    
    for n in concurrency_levels:
        print(f"  Concurrency {n}...", end=" ", flush=True)
        result = run_concurrent_test(url, model, n)
        if result:
            results["tests"][n] = result
            print(f"{result['throughput_req_per_sec']:.2f} req/s, "
                  f"{result['throughput_tok_per_sec']:.1f} tok/s, "
                  f"p95={result['p95_latency']:.2f}s")
        else:
            print("FAILED")
    
    return results

# Main benchmark execution
if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE BENCHMARK SUITE - Day 02 Case Study")
    print("=" * 70)
    
    all_results = {}
    
    # Test 1: vLLM with AWQ (assuming it's running on port 8000)
    print("\n[1/2] Testing vLLM + AWQ...")
    try:
        awq_results = test_configuration(
            "vLLM + AWQ (INT4)",
            "http://localhost:8000",
            "TheBloke/Llama-2-7B-Chat-AWQ",
            [1, 5, 10, 20]
        )
        all_results["vllm_awq"] = awq_results
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Save intermediate results
    with open("/root/artifacts/full_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE - Results saved to ~/artifacts/full_benchmark_results.json")
    print("=" * 70)
    
    # Generate summary table
    print("\nSUMMARY TABLE:")
    print("-" * 70)
    print(f"{'Config':<25} {'Concurrency':<12} {'Throughput':<15} {'P95 Latency':<12}")
    print("-" * 70)
    
    for config_name, config_data in all_results.items():
        for conc, test in config_data.get("tests", {}).items():
            print(f"{config_name:<25} {conc:<12} "
                  f"{test['throughput_req_per_sec']:.2f} req/s{'':<5} "
                  f"{test['p95_latency']:.2f}s")
EOF

chmod +x ~/full_benchmark_suite.py
```

Make sure vLLM with AWQ is running, then execute:

```bash
# Ensure AWQ model is running
pkill -f "vllm serve"
vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
  --port 8000 \
  --quantization awq \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 &

sleep 45

# Run full benchmark
python3 ~/full_benchmark_suite.py 2>&1 | tee ~/artifacts/full_benchmark.log
```

#### 🏆 Success Criteria
- [ ] All benchmark configurations tested
- [ ] Results saved to JSON
- [ ] Summary table generated

---

### ✅ Task 4.3: Create Visualization Dashboard
**Tags**: `[Business]` `[Phase3-Optimization]`  
**Time**: 30 min  
**Win**: Professional charts for your case study

#### 🔧 Lab Instructions

```bash
cat > ~/create_charts.py << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("/root/artifacts/full_benchmark_results.json") as f:
    data = json.load(f)

# Also load comparison data if available
try:
    with open("/root/artifacts/concurrent_benchmark.json") as f:
        comparison_data = json.load(f)
except:
    comparison_data = None

# Chart 1: Throughput comparison at different concurrency
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# If we have HF vs vLLM comparison data
if comparison_data:
    concurrency = sorted([int(k) for k in comparison_data.keys()])
    hf_throughput = [comparison_data[str(c)]["hf"]["throughput_req_per_sec"] for c in concurrency]
    vllm_throughput = [comparison_data[str(c)]["vllm"]["throughput_req_per_sec"] for c in concurrency]
    
    ax = axes[0]
    x = np.arange(len(concurrency))
    width = 0.35
    ax.bar(x - width/2, hf_throughput, width, label='HuggingFace', color='#e74c3c')
    ax.bar(x + width/2, vllm_throughput, width, label='vLLM', color='#2ecc71')
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('HuggingFace vs vLLM Throughput')
    ax.set_xticks(x)
    ax.set_xticklabels(concurrency)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

# Throughput scaling chart
if "vllm_awq" in data:
    awq_data = data["vllm_awq"]["tests"]
    concurrency = sorted([int(k) for k in awq_data.keys()])
    throughput = [awq_data[str(c)]["throughput_req_per_sec"] for c in concurrency]
    latency = [awq_data[str(c)]["p95_latency"] for c in concurrency]
    
    ax = axes[1]
    ax.plot(concurrency, throughput, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('vLLM + AWQ: Throughput Scaling')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/artifacts/case_study_charts.png', dpi=150, bbox_inches='tight')
print("Charts saved to ~/artifacts/case_study_charts.png")

# Chart 2: Memory comparison
fig, ax = plt.subplots(figsize=(8, 5))

memory_data = {
    'HuggingFace\n(FP16)': 14000,  # Approximate
    'vLLM\n(FP16)': 14000,  # Approximate
    'vLLM + AWQ\n(INT4)': 5000,  # From our tests
}

# Try to get actual memory from benchmarks
try:
    with open("/root/artifacts/quantization_comparison.json") as f:
        quant_data = json.load(f)
    memory_data['vLLM\n(FP16)'] = quant_data['fp16']['gpu_memory_mb']
    memory_data['vLLM + AWQ\n(INT4)'] = quant_data['awq']['gpu_memory_mb']
except:
    pass

colors = ['#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(memory_data.keys(), memory_data.values(), color=colors)
ax.set_ylabel('GPU Memory (MB)')
ax.set_title('Memory Usage Comparison')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, memory_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
            f'{val/1000:.1f}GB', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/root/artifacts/memory_comparison.png', dpi=150, bbox_inches='tight')
print("Memory chart saved to ~/artifacts/memory_comparison.png")
EOF

python3 ~/create_charts.py
```

#### 🏆 Success Criteria
- [ ] Throughput comparison chart generated
- [ ] Memory comparison chart generated
- [ ] Charts are professional quality

---

### ✅ Task 4.4: Complete the Case Study Document
**Tags**: `[Business]` `[Phase4-Ship]`  
**Time**: 45 min  
**Win**: A complete, shareable case study

#### 🔧 Lab Instructions

Update your case study with real numbers:

```bash
cat > ~/finalize_case_study.py << 'EOF'
import json

# Load all benchmark data
try:
    with open("/root/artifacts/full_benchmark_results.json") as f:
        full_results = json.load(f)
except:
    full_results = {}

try:
    with open("/root/artifacts/concurrent_benchmark.json") as f:
        comparison = json.load(f)
except:
    comparison = {}

try:
    with open("/root/artifacts/quantization_comparison.json") as f:
        quant = json.load(f)
except:
    quant = {}

# Extract key metrics
hf_baseline = comparison.get("10", {}).get("hf", {})
vllm_optimized = comparison.get("10", {}).get("vllm", {})

print("=" * 70)
print("CASE STUDY KEY METRICS")
print("=" * 70)

print("\nBEFORE (HuggingFace):")
if hf_baseline:
    print(f"  Throughput at 10 concurrent: {hf_baseline.get('throughput_req_per_sec', 'N/A'):.2f} req/s")
    print(f"  Mean latency: {hf_baseline.get('mean_latency', 'N/A'):.2f}s")

print("\nAFTER (vLLM + AWQ):")
if vllm_optimized:
    print(f"  Throughput at 10 concurrent: {vllm_optimized.get('throughput_req_per_sec', 'N/A'):.2f} req/s")
    print(f"  Mean latency: {vllm_optimized.get('mean_latency', 'N/A'):.2f}s")

if hf_baseline and vllm_optimized:
    speedup = vllm_optimized.get('throughput_req_per_sec', 1) / max(hf_baseline.get('throughput_req_per_sec', 1), 0.01)
    print(f"\nSPEEDUP: {speedup:.1f}x throughput improvement")

if quant:
    print(f"\nMEMORY SAVINGS: {quant.get('memory_savings_pct', 'N/A'):.0f}%")

print("\n" + "=" * 70)
print("Copy these numbers into your case_study.md!")
print("=" * 70)
EOF

python3 ~/finalize_case_study.py
```

Now manually update `~/artifacts/case_study.md` with your real numbers, then create the final version:

```bash
# Add executive summary
cat >> ~/artifacts/case_study.md << 'EOF'

---

## Final Notes

### What We Learned
1. **vLLM's continuous batching** provides massive throughput improvements at high concurrency
2. **AWQ quantization** reduces memory by ~60% with minimal quality loss
3. **Configuration tuning** (gpu-memory-utilization, max-num-seqs) is critical for production

### Tools & Technologies Used
- RunPod (GPU cloud)
- vLLM (inference engine)
- AWQ (quantization)
- Python (benchmarking)
- Matplotlib (visualization)

### Time Investment
- Tier 1: ~2 hours (setup + baseline)
- Tier 2: ~2 hours (HF vs vLLM comparison)
- Tier 3: ~2-3 hours (quantization + config)
- Tier 4: ~2-3 hours (case study)
- **Total: 8-10 hours**

### Cost
- RunPod RTX 4090: ~$0.44/hr × 10 hrs = ~$4.40
- Total learning cost: < $5

---

*Case study created as part of 100 Days of Inference Engineering*
*Day 002 - GPU Node Bring-Up on RunPod*
EOF

echo "Case study finalized! See ~/artifacts/case_study.md"
```

#### 🏆 Success Criteria
- [ ] All metrics filled in with real numbers
- [ ] Executive summary written
- [ ] Recommendations are actionable
- [ ] Document is ready to share

---

## 🏆 Day 02 Complete!

### Final Checklist

| Tier | Status | Key Achievement |
|------|--------|-----------------|
| Tier 1 | ⬜ | RunPod setup, Llama-3-8B running, baseline benchmark |
| Tier 2 | ⬜ | HF vs vLLM comparison, continuous batching understood |
| Tier 3 | ⬜ | AWQ quantization, production configs |
| Tier 4 | ⬜ | Complete case study with charts |

### Artifacts Created
```
~/artifacts/
├── nvidia_smi_*.txt          # GPU state snapshots
├── benchmark_baseline.json    # Tier 1 benchmarks
├── single_request_comparison.json
├── concurrent_benchmark.json  # HF vs vLLM
├── hf_vs_vllm_throughput.png # Chart
├── awq_benchmark.json        # Quantization results
├── quantization_comparison.json
├── full_benchmark_results.json
├── case_study_charts.png
├── memory_comparison.png
├── case_study.md             # THE FINAL DELIVERABLE
└── day02_config_guide.md

~/configs/
├── latency_optimized.sh
└── throughput_optimized.sh
```

### Final Commit
```bash
cd ~/artifacts
git add .
git commit -m "day02-complete: Full case study - HF to vLLM+AWQ optimization (Xx speedup)"
git push  # If you have a remote
```

### What You Achieved Today

You went from **zero** to:
1. ✅ Running a production LLM on cloud GPU
2. ✅ Understanding why vLLM beats HuggingFace (continuous batching)
3. ✅ Applying quantization for 60%+ memory savings
4. ✅ Creating production-ready configurations
5. ✅ Writing a professional case study

**This is exactly what a top 1% inference engineer does.**

---

## 📚 Evening Reading (Optional)

| Resource | Why |
|----------|-----|
| [vLLM Paper](https://arxiv.org/abs/2309.06180) | Deep dive into PagedAttention |
| [FlashAttention Paper](https://arxiv.org/abs/2205.14135) | Next optimization to understand |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) | Advanced latency optimization |

---

**→ Tomorrow (Day 003)**: Speculative Decoding & Streaming Optimization
