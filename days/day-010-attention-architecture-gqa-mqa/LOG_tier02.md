# Day 010 – Attention Architecture Variants: MHA, GQA, MQA
## Tier 2: Hands-On Labs (Microtasks 1-9)

> **Prerequisites**: Read [Tier 1](LOG_tier01.md) theory  
> **Goal**: Measure and compare KV cache behavior across different attention architectures  
> **Time**: ~3 hours hands-on

---

## Microtask Overview

| # | Task | Focus | Key Artifact |
|---|------|-------|--------------|
| 1 | Setup & Model Selection | Environment | `scripts/model_config.sh` |
| 2 | KV Cache Calculator | Theory validation | `scripts/kv_cache_calculator.py` |
| 3 | Baseline Memory Profiling | Measurement | `logs/baseline_memory.csv` |
| 4 | GQA Model Memory Profiling | Comparison | `logs/gqa_memory.csv` |
| 5 | Sequence Length Sweep | Scaling behavior | `reports/seq_len_kv_scaling.csv` |
| 6 | Concurrency Headroom Test | Practical limits | `reports/max_concurrency.md` |
| 7 | Throughput Comparison | Performance | `reports/throughput_comparison.csv` |
| 8 | vLLM KV Cache Inspection | Runtime internals | `logs/vllm_kv_stats.json` |
| 9 | Architecture Fingerprint Script | Reusable tool | `scripts/arch_fingerprint.py` |

---

## Microtask 1: Setup & Model Selection

**Objective**: Prepare environment and select comparison models.

**Time**: 20 min

### 1.0 Prerequisites & Installation

Before starting, ensure you have the required packages installed:

```bash
# Create/activate virtual environment (recommended)
python -m venv venv && source venv/bin/activate

# Install required packages
pip install vllm>=0.5.0 transformers torch

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**vLLM version notes**:
- v0.5.0+: Stable GQA support, recommended for these exercises
- v0.6.0+: Improved prefix caching, better memory efficiency
- Always check [vLLM releases](https://github.com/vllm-project/vllm/releases) for compatibility

### 1.1 Create Day 10 Directory Structure

```bash
cd ~/Developer/Projects/inference-journal/days/day-010-attention-architecture-gqa-mqa

mkdir -p scripts data/prompts logs/nvidia_smi reports archive
```

### 1.2 Select Comparison Models

Choose models that differ primarily in attention architecture:

| Model | Attention | n_heads | n_kv_heads | Size | HF ID |
|-------|-----------|---------|------------|------|-------|
| **Model A** (MHA-like) | GQA (32/32) | 32 | 32 | 7B | `meta-llama/Llama-2-7b-hf` |
| **Model B** (GQA) | GQA (32/8) | 32 | 8 | 7-8B | `mistralai/Mistral-7B-v0.1` |

> **Note**: Llama-2-7B has n_kv_heads=32 (same as n_heads), making it effectively MHA. Mistral-7B has n_kv_heads=8, true GQA.

### 1.3 Create Model Config Script

```bash
cat > scripts/model_config.sh << 'EOF'
#!/usr/bin/env bash
# Day 10 - Model Configuration for Architecture Comparison

# Model A: MHA-equivalent (Llama-2-7B style)
export MODEL_A_ID="meta-llama/Llama-2-7b-hf"
export MODEL_A_NAME="llama2-7b-mha"
export MODEL_A_HEADS=32
export MODEL_A_KV_HEADS=32
export MODEL_A_LAYERS=32
export MODEL_A_D_HEAD=128

# Model B: GQA (Mistral-7B)
export MODEL_B_ID="mistralai/Mistral-7B-v0.1"
export MODEL_B_NAME="mistral-7b-gqa"
export MODEL_B_HEADS=32
export MODEL_B_KV_HEADS=8
export MODEL_B_LAYERS=32
export MODEL_B_D_HEAD=128

# Alternative Model B: Qwen2.5-1.5B (smaller, faster iteration)
export MODEL_B_ALT_ID="Qwen/Qwen2.5-1.5B-Instruct"
export MODEL_B_ALT_NAME="qwen25-1.5b-gqa"
export MODEL_B_ALT_HEADS=12
export MODEL_B_ALT_KV_HEADS=2
export MODEL_B_ALT_LAYERS=28
export MODEL_B_ALT_D_HEAD=128

# Common settings
export MAX_MODEL_LEN=4096
export GPU_MEM_UTIL=0.85
export DTYPE="float16"
export DTYPE_BYTES=2

echo "Model configs loaded."
EOF

chmod +x scripts/model_config.sh
source scripts/model_config.sh
```

### 1.4 Create Test Prompts

```bash
cat > data/prompts/test_prompts.txt << 'EOF'
Explain the concept of attention in transformers in 3 sentences.
Write a Python function to calculate fibonacci numbers.
What are the trade-offs between model size and inference speed?
Summarize the key benefits of grouped-query attention.
How does KV cache affect memory usage during LLM inference?
EOF
```

### Success Criteria
- [ ] Directory structure created
- [ ] Model config script sourced without errors
- [ ] Test prompts file exists

---

## Microtask 2: KV Cache Calculator

**Objective**: Build a calculator to predict KV cache sizes, then validate against measurements.

**Time**: 25 min

### 2.1 Create KV Cache Calculator

```bash
cat > scripts/kv_cache_calculator.py << 'EOF'
#!/usr/bin/env python3
"""
KV Cache Size Calculator for Different Attention Architectures.
Day 10 - Attention Architecture Comparison
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    n_layers: int
    n_heads: int          # Query heads
    n_kv_heads: int       # KV heads (= n_heads for MHA, < n_heads for GQA/MQA)
    d_head: int           # Head dimension
    d_model: Optional[int] = None  # Hidden size (computed if not provided)
    
    def __post_init__(self):
        if self.d_model is None:
            self.d_model = self.n_heads * self.d_head
    
    @property
    def attention_type(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        elif self.n_kv_heads == 1:
            return "MQA"
        else:
            return f"GQA-{self.n_heads // self.n_kv_heads}"
    
    @property
    def kv_reduction_factor(self) -> float:
        return self.n_heads / self.n_kv_heads


# Pre-defined model configs
MODELS = {
    # Llama family
    "llama2-7b": ModelConfig("Llama-2-7B", n_layers=32, n_heads=32, n_kv_heads=32, d_head=128),
    "llama2-70b": ModelConfig("Llama-2-70B", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128),
    "llama3-8b": ModelConfig("Llama-3-8B", n_layers=32, n_heads=32, n_kv_heads=8, d_head=128),
    "llama3-70b": ModelConfig("Llama-3-70B", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128),
    "llama3.1-8b": ModelConfig("Llama-3.1-8B", n_layers=32, n_heads=32, n_kv_heads=8, d_head=128),
    "llama3.1-70b": ModelConfig("Llama-3.1-70B", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128),
    "llama3.2-3b": ModelConfig("Llama-3.2-3B", n_layers=28, n_heads=24, n_kv_heads=8, d_head=128),
    # Mistral family (note: SWA capped at 4096 for Mistral-7B)
    "mistral-7b": ModelConfig("Mistral-7B", n_layers=32, n_heads=32, n_kv_heads=8, d_head=128),
    "mistral-large": ModelConfig("Mistral-Large", n_layers=88, n_heads=96, n_kv_heads=8, d_head=128),
    "mixtral-8x7b": ModelConfig("Mixtral-8x7B", n_layers=32, n_heads=32, n_kv_heads=8, d_head=128),
    # Qwen family
    "qwen25-1.5b": ModelConfig("Qwen2.5-1.5B", n_layers=28, n_heads=12, n_kv_heads=2, d_head=128),
    "qwen25-7b": ModelConfig("Qwen2.5-7B", n_layers=28, n_heads=28, n_kv_heads=4, d_head=128),
    "qwen25-72b": ModelConfig("Qwen2.5-72B", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128),
    # Other architectures
    "falcon-7b": ModelConfig("Falcon-7B", n_layers=32, n_heads=71, n_kv_heads=1, d_head=64),
    "gemma2-9b": ModelConfig("Gemma-2-9B", n_layers=42, n_heads=16, n_kv_heads=8, d_head=256),
}


def calc_kv_cache_bytes(config: ModelConfig, seq_len: int, batch_size: int = 1, 
                        dtype_bytes: int = 2) -> int:
    """
    Calculate KV cache size in bytes.
    
    KV = 2 (K+V) × n_layers × n_kv_heads × d_head × seq_len × batch × dtype_bytes
    """
    return (2 * config.n_layers * config.n_kv_heads * config.d_head * 
            seq_len * batch_size * dtype_bytes)


def calc_mha_equivalent_bytes(config: ModelConfig, seq_len: int, batch_size: int = 1,
                               dtype_bytes: int = 2) -> int:
    """Calculate what KV cache would be if this model used MHA."""
    return (2 * config.n_layers * config.n_heads * config.d_head * 
            seq_len * batch_size * dtype_bytes)


def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"


def print_model_comparison(seq_len: int, batch_size: int = 1, dtype_bytes: int = 2):
    """Print KV cache comparison table for all models."""
    print(f"\n{'='*80}")
    print(f"KV Cache Comparison (seq_len={seq_len}, batch={batch_size}, dtype={dtype_bytes}B)")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Type':<8} {'Heads':<6} {'KV Heads':<9} {'KV Cache':<12} {'Reduction':<10}")
    print(f"{'-'*80}")
    
    for name, config in MODELS.items():
        kv_bytes = calc_kv_cache_bytes(config, seq_len, batch_size, dtype_bytes)
        mha_bytes = calc_mha_equivalent_bytes(config, seq_len, batch_size, dtype_bytes)
        reduction = config.kv_reduction_factor
        
        print(f"{config.name:<15} {config.attention_type:<8} {config.n_heads:<6} "
              f"{config.n_kv_heads:<9} {format_bytes(kv_bytes):<12} {reduction:.1f}×")
    
    print(f"{'='*80}\n")


def print_sequence_scaling(model_name: str, dtype_bytes: int = 2):
    """Print how KV cache scales with sequence length for a model."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        return
    
    config = MODELS[model_name]
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    print(f"\n{'='*60}")
    print(f"KV Cache Scaling: {config.name} ({config.attention_type})")
    print(f"{'='*60}")
    print(f"{'Seq Length':<12} {'KV Cache (1 seq)':<18} {'KV Cache (8 seqs)':<18}")
    print(f"{'-'*60}")
    
    for seq_len in seq_lengths:
        kv_1 = calc_kv_cache_bytes(config, seq_len, batch_size=1, dtype_bytes=dtype_bytes)
        kv_8 = calc_kv_cache_bytes(config, seq_len, batch_size=8, dtype_bytes=dtype_bytes)
        print(f"{seq_len:<12} {format_bytes(kv_1):<18} {format_bytes(kv_8):<18}")
    
    print(f"{'='*60}\n")


def estimate_max_concurrent(config: ModelConfig, available_vram_gb: float, 
                            seq_len: int, dtype_bytes: int = 2) -> int:
    """Estimate max concurrent sequences given available VRAM."""
    available_bytes = available_vram_gb * 1024**3
    kv_per_seq = calc_kv_cache_bytes(config, seq_len, batch_size=1, dtype_bytes=dtype_bytes)
    return int(available_bytes * 0.9 / kv_per_seq)  # 90% safety factor


def main():
    parser = argparse.ArgumentParser(description="KV Cache Calculator")
    parser.add_argument("--model", type=str, help="Model name (e.g., llama3-8b)")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--dtype-bytes", type=int, default=2, help="Bytes per element (2=fp16)")
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument("--scaling", action="store_true", help="Show sequence scaling")
    parser.add_argument("--available-vram", type=float, help="Available VRAM in GB for concurrency estimate")
    args = parser.parse_args()
    
    if args.compare:
        print_model_comparison(args.seq_len, args.batch, args.dtype_bytes)
    elif args.scaling and args.model:
        print_sequence_scaling(args.model, args.dtype_bytes)
    elif args.model:
        if args.model not in MODELS:
            print(f"Unknown model. Available: {list(MODELS.keys())}")
            return
        config = MODELS[args.model]
        kv = calc_kv_cache_bytes(config, args.seq_len, args.batch, args.dtype_bytes)
        print(f"\n{config.name} ({config.attention_type})")
        print(f"  Seq length: {args.seq_len}")
        print(f"  Batch size: {args.batch}")
        print(f"  KV cache: {format_bytes(kv)}")
        print(f"  KV reduction vs MHA: {config.kv_reduction_factor:.1f}×")
        
        if args.available_vram:
            max_seqs = estimate_max_concurrent(config, args.available_vram, 
                                               args.seq_len, args.dtype_bytes)
            print(f"  Max concurrent seqs ({args.available_vram}GB): ~{max_seqs}")
    else:
        print_model_comparison(args.seq_len, args.batch, args.dtype_bytes)


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/kv_cache_calculator.py
```

### 2.2 Test the Calculator

```bash
# Compare all models at 4K context
python scripts/kv_cache_calculator.py --compare --seq-len 4096

# Show scaling for a specific model
python scripts/kv_cache_calculator.py --model llama3-8b --scaling

# Estimate max concurrency with 10GB available VRAM
python scripts/kv_cache_calculator.py --model mistral-7b --seq-len 4096 --available-vram 10
```

### Expected Output

```
================================================================================
KV Cache Comparison (seq_len=4096, batch=1, dtype=2B)
================================================================================
Model           Type     Heads  KV Heads  KV Cache     Reduction 
--------------------------------------------------------------------------------
Llama-2-7B      MHA      32     32        2.00 GB      1.0×
Llama-3-8B      GQA-4    32     8         512.00 MB    4.0×
Mistral-7B      GQA-4    32     8         512.00 MB    4.0×
Qwen2.5-1.5B    GQA-6    12     2         114.69 MB    6.0×
Falcon-7B       MQA      71     1         32.00 MB     71.0×
================================================================================
```

### Success Criteria
- [ ] Calculator runs without errors
- [ ] Output matches expected KV cache sizes
- [ ] Understand the reduction factors

---

## Microtask 3: Baseline Memory Profiling (MHA-equivalent)

**Objective**: Measure actual GPU memory usage for an MHA-equivalent model.

**Time**: 30 min

### 3.1 Create Memory Profiling Script

```bash
cat > scripts/memory_profile.py << 'EOF'
#!/usr/bin/env python3
"""
GPU Memory Profiler for vLLM Model Loading.
Measures VRAM at different stages: idle, model loaded, after inference.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime


def get_gpu_memory_mb() -> dict:
    """Get current GPU memory usage via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    used, total, free = map(int, result.stdout.strip().split(", "))
    return {"used_mb": used, "total_mb": total, "free_mb": free}


def profile_vllm_memory(model_id: str, max_model_len: int = 4096, 
                        gpu_mem_util: float = 0.85) -> dict:
    """Profile memory usage through vLLM model lifecycle."""
    from vllm import LLM, SamplingParams
    
    results = {"model_id": model_id, "max_model_len": max_model_len, 
               "timestamp": datetime.now().isoformat()}
    
    # Stage 1: Before loading
    results["stage_0_idle"] = get_gpu_memory_mb()
    print(f"[Stage 0] Idle: {results['stage_0_idle']['used_mb']} MB")
    
    # Stage 2: After model load
    print(f"[Stage 1] Loading model: {model_id}...")
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
    )
    time.sleep(2)  # Allow memory to settle
    results["stage_1_model_loaded"] = get_gpu_memory_mb()
    print(f"[Stage 1] Model loaded: {results['stage_1_model_loaded']['used_mb']} MB")
    
    # Stage 3: After warmup inference
    print("[Stage 2] Running warmup inference...")
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    prompts = ["Hello, how are you today?"] * 4
    _ = llm.generate(prompts, sampling_params)
    time.sleep(1)
    results["stage_2_after_warmup"] = get_gpu_memory_mb()
    print(f"[Stage 2] After warmup: {results['stage_2_after_warmup']['used_mb']} MB")
    
    # Stage 4: Peak usage with longer generation
    print("[Stage 3] Running peak usage test...")
    long_prompts = ["Explain the concept of machine learning in detail. " * 10] * 8
    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    _ = llm.generate(long_prompts, sampling_params)
    time.sleep(1)
    results["stage_3_peak_usage"] = get_gpu_memory_mb()
    print(f"[Stage 3] Peak usage: {results['stage_3_peak_usage']['used_mb']} MB")
    
    # Calculate deltas
    results["model_weight_estimate_mb"] = (
        results["stage_1_model_loaded"]["used_mb"] - 
        results["stage_0_idle"]["used_mb"]
    )
    results["kv_cache_estimate_mb"] = (
        results["stage_3_peak_usage"]["used_mb"] - 
        results["stage_1_model_loaded"]["used_mb"]
    )
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    results = profile_vllm_memory(args.model, args.max_model_len, args.gpu_mem_util)
    
    print("\n" + "="*60)
    print("Memory Profile Summary")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Model Weight Estimate: {results['model_weight_estimate_mb']} MB")
    print(f"KV Cache Estimate: {results['kv_cache_estimate_mb']} MB")
    print("="*60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/memory_profile.py
```

### 3.2 Profile MHA-equivalent Model

```bash
# Profile Llama-2-7B (or smaller model for faster iteration)
# For quick testing, use Qwen2.5-1.5B first:

python scripts/memory_profile.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --max-model-len 4096 \
    --output logs/memory_profile_qwen25_1.5b.json

# If you have access to larger models:
# python scripts/memory_profile.py \
#     --model "meta-llama/Llama-2-7b-hf" \
#     --max-model-len 4096 \
#     --output logs/memory_profile_llama2_7b.json
```

### 3.3 Record Baseline

Create baseline record:

```bash
cat > logs/baseline_memory.csv << 'EOF'
model,attention_type,n_heads,n_kv_heads,max_model_len,idle_mb,loaded_mb,peak_mb,weights_mb,kv_estimate_mb
EOF

# After running profiler, add a row like:
# qwen25-1.5b,GQA-6,12,2,4096,500,1800,2400,1300,600
```

### Success Criteria
- [ ] Memory profiler runs successfully
- [ ] Baseline measurements recorded
- [ ] Understanding of memory breakdown (weights vs KV)

---

## Microtask 4: GQA Model Memory Profiling

**Objective**: Compare memory profile of GQA model against baseline.

**Time**: 25 min

### 4.1 Profile GQA Model

```bash
# Profile Mistral-7B (GQA) or alternative GQA model
python scripts/memory_profile.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --max-model-len 4096 \
    --output logs/memory_profile_mistral_7b.json

# Alternative: Use Qwen2.5-7B for aggressive GQA (n_kv_heads=4)
# python scripts/memory_profile.py \
#     --model "Qwen/Qwen2.5-7B-Instruct" \
#     --max-model-len 4096 \
#     --output logs/memory_profile_qwen25_7b.json
```

### 4.2 Side-by-Side Comparison

```bash
cat > scripts/compare_profiles.py << 'EOF'
#!/usr/bin/env python3
"""Compare memory profiles between two models."""

import json
import sys

def load_profile(path):
    with open(path) as f:
        return json.load(f)

def compare(profile_a, profile_b):
    print("\n" + "="*70)
    print("Memory Profile Comparison")
    print("="*70)
    
    models = [profile_a, profile_b]
    metrics = ["stage_0_idle", "stage_1_model_loaded", "stage_2_after_warmup", "stage_3_peak_usage"]
    
    print(f"{'Metric':<25} {profile_a['model_id'][:20]:<22} {profile_b['model_id'][:20]:<22}")
    print("-"*70)
    
    for metric in metrics:
        a_val = profile_a.get(metric, {}).get("used_mb", "N/A")
        b_val = profile_b.get(metric, {}).get("used_mb", "N/A")
        print(f"{metric:<25} {a_val:<22} {b_val:<22}")
    
    print("-"*70)
    print(f"{'Model Weights (est)':<25} {profile_a.get('model_weight_estimate_mb', 'N/A'):<22} {profile_b.get('model_weight_estimate_mb', 'N/A'):<22}")
    print(f"{'KV Cache (est)':<25} {profile_a.get('kv_cache_estimate_mb', 'N/A'):<22} {profile_b.get('kv_cache_estimate_mb', 'N/A'):<22}")
    
    # Calculate KV savings
    kv_a = profile_a.get('kv_cache_estimate_mb', 0)
    kv_b = profile_b.get('kv_cache_estimate_mb', 0)
    if kv_a and kv_b:
        savings = (kv_a - kv_b) / kv_a * 100 if kv_a > kv_b else (kv_b - kv_a) / kv_b * 100
        print(f"\nKV Cache difference: {abs(kv_a - kv_b)} MB ({savings:.1f}%)")
    
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_profiles.py <profile_a.json> <profile_b.json>")
        sys.exit(1)
    
    profile_a = load_profile(sys.argv[1])
    profile_b = load_profile(sys.argv[2])
    compare(profile_a, profile_b)
EOF

chmod +x scripts/compare_profiles.py

# Run comparison
python scripts/compare_profiles.py \
    logs/memory_profile_qwen25_1.5b.json \
    logs/memory_profile_mistral_7b.json
```

### Success Criteria
- [ ] GQA model profiled successfully
- [ ] Side-by-side comparison shows KV cache difference
- [ ] KV savings align with theoretical predictions

---

## Microtask 5: Sequence Length Sweep

**Objective**: Measure how KV cache scales with sequence length for different architectures.

**Time**: 30 min

### 5.1 Create Sequence Length Sweep Script

```bash
cat > scripts/seq_len_sweep.py << 'EOF'
#!/usr/bin/env python3
"""
Sweep sequence lengths and measure KV cache memory usage.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime


def get_gpu_memory_mb() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def sweep_sequence_lengths(model_id: str, seq_lengths: list, 
                           base_gpu_mem_util: float = 0.9) -> list:
    """Measure memory at different sequence lengths."""
    from vllm import LLM, SamplingParams
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n[Sweep] Testing seq_len={seq_len}...")
        
        try:
            # Load model with this max_model_len
            llm = LLM(
                model=model_id,
                max_model_len=seq_len,
                gpu_memory_utilization=base_gpu_mem_util,
                trust_remote_code=True,
            )
            
            mem_loaded = get_gpu_memory_mb()
            
            # Generate some tokens to trigger KV allocation
            sampling_params = SamplingParams(temperature=0, max_tokens=min(64, seq_len // 4))
            prompts = ["Test prompt for memory measurement."] * 4
            _ = llm.generate(prompts, sampling_params)
            time.sleep(1)
            
            mem_peak = get_gpu_memory_mb()
            
            results.append({
                "model_id": model_id,
                "seq_len": seq_len,
                "mem_loaded_mb": mem_loaded,
                "mem_peak_mb": mem_peak,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"  Loaded: {mem_loaded} MB, Peak: {mem_peak} MB")
            
            # Clean up for next iteration
            del llm
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            time.sleep(2)
            
        except Exception as e:
            print(f"  Error at seq_len={seq_len}: {e}")
            results.append({
                "model_id": model_id,
                "seq_len": seq_len,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-lengths", type=int, nargs="+", 
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    results = sweep_sequence_lengths(args.model, args.seq_lengths)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*50)
    print("Sequence Length Sweep Summary")
    print("="*50)
    for r in results:
        if "error" not in r:
            print(f"seq_len={r['seq_len']:<6} loaded={r['mem_loaded_mb']:<6} peak={r['mem_peak_mb']}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/seq_len_sweep.py
```

### 5.2 Run Sweep

```bash
# Sweep on smaller model first
python scripts/seq_len_sweep.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --seq-lengths 512 1024 2048 4096 \
    --output reports/seq_len_sweep_qwen25.json
```

### 5.3 Convert to CSV

```bash
cat > scripts/json_to_csv.py << 'EOF'
#!/usr/bin/env python3
import json
import sys
import csv

with open(sys.argv[1]) as f:
    data = json.load(f)

writer = csv.DictWriter(sys.stdout, fieldnames=["model_id", "seq_len", "mem_loaded_mb", "mem_peak_mb"])
writer.writeheader()
for row in data:
    if "error" not in row:
        writer.writerow({k: row.get(k) for k in writer.fieldnames})
EOF

python scripts/json_to_csv.py reports/seq_len_sweep_qwen25.json > reports/seq_len_kv_scaling.csv
```

### Success Criteria
- [ ] Sweep completes for multiple sequence lengths
- [ ] Memory scales approximately linearly with seq_len
- [ ] CSV report generated

---

## Microtask 6: Concurrency Headroom Test

**Objective**: Find maximum concurrent sequences before OOM for different architectures.

**Time**: 30 min

### 6.1 Create Concurrency Test Script

```bash
cat > scripts/concurrency_test.py << 'EOF'
#!/usr/bin/env python3
"""
Test maximum concurrent sequences before OOM.
Binary search to find the limit.
"""

import argparse
import subprocess
import time


def get_gpu_memory_mb() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def test_concurrency(model_id: str, n_concurrent: int, max_model_len: int,
                     max_tokens: int = 128) -> tuple:
    """
    Test if model can handle n_concurrent sequences.
    Returns (success: bool, peak_memory_mb: int, error: str or None)
    """
    from vllm import LLM, SamplingParams
    import gc
    import torch
    
    try:
        llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.92,
            max_num_seqs=n_concurrent + 4,  # headroom
            trust_remote_code=True,
        )
        
        # Generate prompts
        prompts = [f"Write a short story about topic {i}." for i in range(n_concurrent)]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
        
        # Run inference
        _ = llm.generate(prompts, sampling_params)
        peak_mem = get_gpu_memory_mb()
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        
        return True, peak_mem, None
        
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        return False, 0, str(e)


def find_max_concurrency(model_id: str, max_model_len: int, 
                         low: int = 1, high: int = 64) -> dict:
    """Binary search for max concurrency."""
    print(f"\nSearching max concurrency for {model_id}...")
    print(f"max_model_len={max_model_len}, search range=[{low}, {high}]")
    
    best_success = 0
    best_mem = 0
    
    while low <= high:
        mid = (low + high) // 2
        print(f"\n  Testing n_concurrent={mid}...", end=" ")
        
        success, mem, error = test_concurrency(model_id, mid, max_model_len)
        
        if success:
            print(f"✓ (peak={mem}MB)")
            best_success = mid
            best_mem = mem
            low = mid + 1
        else:
            print(f"✗ ({error[:50] if error else 'OOM'})")
            high = mid - 1
    
    return {
        "model_id": model_id,
        "max_model_len": max_model_len,
        "max_concurrent": best_success,
        "peak_memory_mb": best_mem
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--search-low", type=int, default=1)
    parser.add_argument("--search-high", type=int, default=32)
    args = parser.parse_args()
    
    result = find_max_concurrency(
        args.model, args.max_model_len,
        args.search_low, args.search_high
    )
    
    print("\n" + "="*50)
    print("Concurrency Test Result")
    print("="*50)
    print(f"Model: {result['model_id']}")
    print(f"Max Model Len: {result['max_model_len']}")
    print(f"Max Concurrent Sequences: {result['max_concurrent']}")
    print(f"Peak Memory: {result['peak_memory_mb']} MB")
    print("="*50)


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/concurrency_test.py
```

### 6.2 Run Concurrency Test

```bash
# Test on smaller model first
python scripts/concurrency_test.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --max-model-len 4096 \
    --search-low 4 \
    --search-high 32
```

### 6.3 Document Results

```bash
cat > reports/max_concurrency.md << 'EOF'
# Max Concurrency Results (Day 10)

## Test Configuration
- GPU: [YOUR GPU]
- VRAM: [YOUR VRAM] GB
- max_model_len: 4096
- max_tokens: 128

## Results

| Model | Attention | n_kv_heads | Max Concurrent | Peak VRAM (MB) |
|-------|-----------|------------|----------------|----------------|
| Qwen2.5-1.5B | GQA-6 | 2 | [FILL] | [FILL] |
| [Model B] | [Type] | [heads] | [FILL] | [FILL] |

## Analysis

### KV Cache Impact
- Model A with [X] KV heads: max [N] concurrent
- Model B with [Y] KV heads: max [M] concurrent
- Ratio: [M/N]× → matches theoretical KV reduction of [ratio]×

### Practical Implication
[Your observations about how architecture affects serving capacity]
EOF
```

### Success Criteria
- [ ] Max concurrency found for at least one model
- [ ] Results documented in markdown
- [ ] Understanding of architecture → concurrency relationship

---

## Microtask 7: Throughput Comparison

**Objective**: Compare tokens/sec throughput between architectures.

**Time**: 30 min

### 7.1 Create Throughput Benchmark

```bash
cat > scripts/throughput_bench.py << 'EOF'
#!/usr/bin/env python3
"""
Throughput benchmark comparing different models.
"""

import argparse
import json
import time
from datetime import datetime


def benchmark_throughput(model_id: str, n_requests: int = 32, 
                         concurrency: int = 8, max_tokens: int = 128,
                         max_model_len: int = 4096) -> dict:
    """Measure throughput for a model."""
    from vllm import LLM, SamplingParams
    
    print(f"\nBenchmarking: {model_id}")
    print(f"  n_requests={n_requests}, concurrency={concurrency}, max_tokens={max_tokens}")
    
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.88,
        max_num_seqs=concurrency + 4,
        trust_remote_code=True,
    )
    
    # Warmup
    warmup_prompts = ["Warmup prompt."] * 4
    _ = llm.generate(warmup_prompts, SamplingParams(max_tokens=16))
    
    # Benchmark
    prompts = [f"Write a detailed explanation about topic number {i}." 
               for i in range(n_requests)]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    
    wall_time = t1 - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    result = {
        "model_id": model_id,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "wall_time_s": round(wall_time, 2),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall_time, 2),
        "avg_tokens_per_req": round(total_tokens / n_requests, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {result['throughput_tok_s']} tok/s")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    result = benchmark_throughput(
        args.model, args.n_requests, args.concurrency,
        args.max_tokens, args.max_model_len
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/throughput_bench.py
```

### 7.2 Run Throughput Tests

```bash
# Test Model A
python scripts/throughput_bench.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --n-requests 32 --concurrency 8 --max-tokens 128 \
    --output reports/throughput_qwen25_1.5b.json

# Test Model B (if different model available)
# python scripts/throughput_bench.py \
#     --model "mistralai/Mistral-7B-v0.1" \
#     --n-requests 32 --concurrency 8 --max-tokens 128 \
#     --output reports/throughput_mistral_7b.json
```

### 7.3 Create Comparison CSV

```bash
echo "model,attention_type,n_requests,concurrency,max_tokens,wall_time_s,throughput_tok_s" > reports/throughput_comparison.csv
# Add rows from benchmark results
```

### Success Criteria
- [ ] Throughput measured for at least one model
- [ ] Results in JSON and CSV format
- [ ] Understanding of architecture → throughput relationship

---

## Microtask 8: vLLM KV Cache Inspection

**Objective**: Inspect vLLM's internal KV cache statistics.

**Time**: 20 min

### 8.1 Create KV Stats Inspector

```bash
cat > scripts/vllm_kv_stats.py << 'EOF'
#!/usr/bin/env python3
"""
Inspect vLLM's KV cache statistics and block allocation.
"""

import argparse
import json


def inspect_kv_cache(model_id: str, max_model_len: int = 4096) -> dict:
    """Get KV cache configuration from vLLM."""
    from vllm import LLM
    
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    
    # Access model config
    model_config = llm.llm_engine.model_config
    cache_config = llm.llm_engine.cache_config
    
    stats = {
        "model_id": model_id,
        "max_model_len": max_model_len,
        "model_config": {
            "num_attention_heads": getattr(model_config.hf_config, 'num_attention_heads', None),
            "num_key_value_heads": getattr(model_config.hf_config, 'num_key_value_heads', None),
            "hidden_size": getattr(model_config.hf_config, 'hidden_size', None),
            "num_hidden_layers": getattr(model_config.hf_config, 'num_hidden_layers', None),
            "head_dim": getattr(model_config.hf_config, 'head_dim', None),
        },
        "cache_config": {
            "block_size": cache_config.block_size,
            "num_gpu_blocks": cache_config.num_gpu_blocks,
            "num_cpu_blocks": cache_config.num_cpu_blocks,
        }
    }
    
    # Calculate derived stats
    n_heads = stats["model_config"]["num_attention_heads"]
    n_kv_heads = stats["model_config"]["num_key_value_heads"] or n_heads
    
    if n_heads and n_kv_heads:
        stats["derived"] = {
            "attention_type": "MHA" if n_heads == n_kv_heads else f"GQA-{n_heads // n_kv_heads}",
            "kv_reduction_factor": n_heads / n_kv_heads,
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    stats = inspect_kv_cache(args.model, args.max_model_len)
    
    print("\n" + "="*60)
    print("vLLM KV Cache Statistics")
    print("="*60)
    print(json.dumps(stats, indent=2))
    print("="*60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/vllm_kv_stats.py
```

### 8.2 Run Inspection

```bash
python scripts/vllm_kv_stats.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --output logs/vllm_kv_stats.json
```

### Success Criteria
- [ ] KV stats extracted from vLLM
- [ ] Attention type correctly identified
- [ ] Block allocation visible

---

## Microtask 9: Architecture Fingerprint Script

**Objective**: Create a reusable tool to fingerprint any model's attention architecture.

**Time**: 20 min

### 9.1 Create Fingerprint Script

```bash
cat > scripts/arch_fingerprint.py << 'EOF'
#!/usr/bin/env python3
"""
Fingerprint a model's attention architecture from HuggingFace config.
No GPU required - just reads config.json.
"""

import argparse
import json
from transformers import AutoConfig


def fingerprint_model(model_id: str) -> dict:
    """Extract attention architecture details from model config."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Common attribute names across different model families
    n_heads = (
        getattr(config, 'num_attention_heads', None) or
        getattr(config, 'n_head', None) or
        getattr(config, 'num_heads', None)
    )
    
    n_kv_heads = (
        getattr(config, 'num_key_value_heads', None) or
        getattr(config, 'num_kv_heads', None) or
        n_heads  # Default to MHA
    )
    
    n_layers = (
        getattr(config, 'num_hidden_layers', None) or
        getattr(config, 'n_layer', None) or
        getattr(config, 'num_layers', None)
    )
    
    hidden_size = (
        getattr(config, 'hidden_size', None) or
        getattr(config, 'd_model', None) or
        getattr(config, 'n_embd', None)
    )
    
    head_dim = (
        getattr(config, 'head_dim', None) or
        (hidden_size // n_heads if hidden_size and n_heads else None)
    )
    
    # Determine attention type
    if n_heads and n_kv_heads:
        if n_kv_heads == n_heads:
            attention_type = "MHA"
        elif n_kv_heads == 1:
            attention_type = "MQA"
        else:
            attention_type = f"GQA-{n_heads // n_kv_heads}"
    else:
        attention_type = "Unknown"
    
    # Calculate KV reduction
    kv_reduction = n_heads / n_kv_heads if n_heads and n_kv_heads else 1.0
    
    return {
        "model_id": model_id,
        "architecture": getattr(config, 'architectures', ['Unknown'])[0] if hasattr(config, 'architectures') else 'Unknown',
        "attention_type": attention_type,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "num_hidden_layers": n_layers,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "kv_reduction_factor": kv_reduction,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
        "vocab_size": getattr(config, 'vocab_size', None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Model IDs to fingerprint")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    results = []
    
    print("\n" + "="*80)
    print("Model Architecture Fingerprints")
    print("="*80)
    
    for model_id in args.models:
        try:
            fp = fingerprint_model(model_id)
            results.append(fp)
            
            print(f"\n{model_id}")
            print(f"  Architecture: {fp['architecture']}")
            print(f"  Attention: {fp['attention_type']} (heads={fp['num_attention_heads']}, kv_heads={fp['num_key_value_heads']})")
            print(f"  Layers: {fp['num_hidden_layers']}, Hidden: {fp['hidden_size']}, Head dim: {fp['head_dim']}")
            print(f"  KV Reduction: {fp['kv_reduction_factor']:.1f}×")
            print(f"  Max context: {fp['max_position_embeddings']}")
            
        except Exception as e:
            print(f"\n{model_id}: Error - {e}")
    
    print("\n" + "="*80)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/arch_fingerprint.py
```

### 9.2 Test Fingerprint Tool

```bash
# Fingerprint multiple models
python scripts/arch_fingerprint.py \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "mistralai/Mistral-7B-v0.1" \
    "meta-llama/Llama-2-7b-hf" \
    --output reports/model_fingerprints.json
```

### Expected Output

```
================================================================================
Model Architecture Fingerprints
================================================================================

Qwen/Qwen2.5-1.5B-Instruct
  Architecture: Qwen2ForCausalLM
  Attention: GQA-6 (heads=12, kv_heads=2)
  Layers: 28, Hidden: 1536, Head dim: 128
  KV Reduction: 6.0×
  Max context: 32768

mistralai/Mistral-7B-v0.1
  Architecture: MistralForCausalLM
  Attention: GQA-4 (heads=32, kv_heads=8)
  Layers: 32, Hidden: 4096, Head dim: 128
  KV Reduction: 4.0×
  Max context: 32768
```

### Success Criteria
- [ ] Fingerprint tool works on multiple models
- [ ] Correctly identifies attention types
- [ ] JSON output saved for reference

---

## Bonus Microtask: Sliding Window Attention (SWA) Exploration

**Objective**: Understand and measure Mistral's Sliding Window Attention behavior.

**Time**: 25 min (optional)

### B.1 Enhanced Fingerprint with SWA Detection

```bash
cat > scripts/arch_fingerprint_extended.py << 'EOF'
#!/usr/bin/env python3
"""
Extended fingerprint including SWA and other advanced attention features.
"""

import argparse
import json
from transformers import AutoConfig


def fingerprint_extended(model_id: str) -> dict:
    """Extract attention architecture including SWA, MoE indicators."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Base attention params
    n_heads = getattr(config, 'num_attention_heads', None)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    n_layers = getattr(config, 'num_hidden_layers', None)
    hidden_size = getattr(config, 'hidden_size', None)
    head_dim = hidden_size // n_heads if hidden_size and n_heads else None

    # Sliding Window Attention
    sliding_window = getattr(config, 'sliding_window', None)

    # MoE indicators
    num_experts = getattr(config, 'num_local_experts', None) or getattr(config, 'num_experts', None)
    num_experts_per_tok = getattr(config, 'num_experts_per_tok', None)

    # Attention type
    if n_heads and n_kv_heads:
        if n_kv_heads == n_heads:
            attn_type = "MHA"
        elif n_kv_heads == 1:
            attn_type = "MQA"
        else:
            attn_type = f"GQA-{n_heads // n_kv_heads}"
    else:
        attn_type = "Unknown"

    return {
        "model_id": model_id,
        "architecture": getattr(config, 'architectures', ['Unknown'])[0],
        "attention_type": attn_type,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "num_hidden_layers": n_layers,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "kv_reduction_factor": n_heads / n_kv_heads if n_heads and n_kv_heads else 1.0,
        "sliding_window": sliding_window,
        "is_moe": num_experts is not None,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    results = []
    print("\n" + "="*80)
    print("Extended Model Architecture Fingerprints")
    print("="*80)

    for model_id in args.models:
        try:
            fp = fingerprint_extended(model_id)
            results.append(fp)

            print(f"\n{model_id}")
            print(f"  Attention: {fp['attention_type']} (heads={fp['num_attention_heads']}, kv_heads={fp['num_key_value_heads']})")
            print(f"  KV Reduction: {fp['kv_reduction_factor']:.1f}×")

            if fp['sliding_window']:
                print(f"  Sliding Window: {fp['sliding_window']} tokens ★")

            if fp['is_moe']:
                print(f"  MoE: {fp['num_experts']} experts, {fp['num_experts_per_tok']} active/token ★")

        except Exception as e:
            print(f"\n{model_id}: Error - {e}")

    print("\n" + "="*80)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/arch_fingerprint_extended.py
```

### B.2 Test SWA Detection

```bash
# Test on Mistral and Mixtral (both have SWA)
python scripts/arch_fingerprint_extended.py \
    "mistralai/Mistral-7B-v0.1" \
    "mistralai/Mixtral-8x7B-v0.1" \
    "meta-llama/Llama-3.1-8B" \
    --output reports/extended_fingerprints.json
```

### Expected Output

```
================================================================================
Extended Model Architecture Fingerprints
================================================================================

mistralai/Mistral-7B-v0.1
  Attention: GQA-4 (heads=32, kv_heads=8)
  KV Reduction: 4.0×
  Sliding Window: 4096 tokens ★

mistralai/Mixtral-8x7B-v0.1
  Attention: GQA-4 (heads=32, kv_heads=8)
  KV Reduction: 4.0×
  Sliding Window: 4096 tokens ★
  MoE: 8 experts, 2 active/token ★

meta-llama/Llama-3.1-8B
  Attention: GQA-4 (heads=32, kv_heads=8)
  KV Reduction: 4.0×
```

### B.3 SWA Memory Implications

With Sliding Window Attention, KV cache is bounded:

```python
# Standard: KV grows with sequence length
kv_standard = 2 * n_layers * n_kv_heads * d_head * seq_len * dtype_bytes

# SWA: KV capped at window size
kv_swa = 2 * n_layers * n_kv_heads * d_head * min(seq_len, window) * dtype_bytes

# Example: Mistral-7B with 32K input
# Without SWA: 512 MB × (32K / 4K) = 4 GB
# With SWA (4K window): 512 MB (constant!)
```

### Success Criteria
- [ ] Extended fingerprint detects sliding_window
- [ ] MoE models correctly identified
- [ ] Understand SWA memory implications

---

## RunPod Cloud GPU Exercises

> **These exercises validate Day 10 concepts with real cloud billing data.**
> Budget: ~$5-15 total for all exercises.

### RunPod Setup

Before running these exercises, set up your RunPod environment:

```bash
# 1. Create RunPod account at https://runpod.io
# 2. Add credits ($10-20 recommended for all exercises)
# 3. Create API key: Settings → API Keys → Create

# 4. Set environment variables
export RUNPOD_API_KEY="your_api_key_here"

# 5. Install RunPod SDK
pip install runpod openai aiohttp

# 6. Verify connection
python -c "import runpod; print(runpod.get_pods())"
```

**Recommended GPU tiers**:
- **L4** ($0.44/hr): Budget-friendly, good for 7B models
- **A10G** ($0.69/hr): Better throughput, 24GB VRAM
- **A100** ($1.99/hr): Production testing, 80GB VRAM

---

## Bonus B2: GQA Cost Savings Validation (RunPod)

**Objective**: Prove that GQA reduces cloud costs by measuring actual RunPod billing.

**Time**: 45 min | **Budget**: ~$2-3

### B2.1 Deploy Endpoints

Deploy two serverless endpoints on RunPod:

1. **MHA-equivalent**: `meta-llama/Llama-2-7b-chat-hf` (n_kv_heads=32)
2. **GQA model**: `mistralai/Mistral-7B-Instruct-v0.2` (n_kv_heads=8)

Use the same GPU tier (A10G recommended) for fair comparison.

### B2.2 Create Cost Benchmark Script

```bash
cat > scripts/runpod_cost_bench.py << 'EOF'
#!/usr/bin/env python3
"""
RunPod Cost Benchmark: Compare GQA vs MHA-equivalent models.
Measures actual execution time and calculates cost difference.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from openai import AsyncOpenAI


async def run_benchmark(
    endpoint_url: str,
    api_key: str,
    n_requests: int = 50,
    max_tokens: int = 128,
    model_name: str = "model"
) -> dict:
    """Run benchmark and measure timing."""

    client = AsyncOpenAI(
        base_url=f"{endpoint_url}/v1",
        api_key=api_key,
    )

    prompts = [
        f"Explain concept {i} about machine learning in detail."
        for i in range(n_requests)
    ]

    print(f"\n[{model_name}] Starting {n_requests} requests...")

    start_time = time.perf_counter()

    # Run requests concurrently in batches
    batch_size = 10
    total_tokens = 0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tasks = [
            client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": p}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            for p in batch
        ]
        responses = await asyncio.gather(*tasks)
        total_tokens += sum(r.usage.completion_tokens for r in responses)
        print(f"  Completed {min(i+batch_size, len(prompts))}/{n_requests}")

    end_time = time.perf_counter()
    wall_time = end_time - start_time

    return {
        "model_name": model_name,
        "endpoint_url": endpoint_url,
        "n_requests": n_requests,
        "max_tokens": max_tokens,
        "total_output_tokens": total_tokens,
        "wall_time_seconds": round(wall_time, 2),
        "throughput_tok_s": round(total_tokens / wall_time, 2),
        "timestamp": datetime.now().isoformat(),
    }


def calculate_costs(results: list, gpu_cost_per_hour: float) -> list:
    """Add cost calculations to results."""
    for r in results:
        gpu_seconds = r["wall_time_seconds"]
        r["gpu_cost_usd"] = round(gpu_seconds * (gpu_cost_per_hour / 3600), 4)
        r["cost_per_1k_tokens"] = round(
            (r["gpu_cost_usd"] / r["total_output_tokens"]) * 1000, 6
        ) if r["total_output_tokens"] > 0 else 0
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-a", required=True, help="First endpoint URL")
    parser.add_argument("--endpoint-b", required=True, help="Second endpoint URL")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--name-a", default="Model-A", help="Name for first model")
    parser.add_argument("--name-b", default="Model-B", help="Name for second model")
    parser.add_argument("--n-requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-cost", type=float, default=0.69, help="GPU $/hour")
    parser.add_argument("--output", type=str, default="reports/runpod_cost_comparison.json")
    args = parser.parse_args()

    results = []

    # Benchmark Model A
    result_a = await run_benchmark(
        args.endpoint_a, args.api_key,
        args.n_requests, args.max_tokens, args.name_a
    )
    results.append(result_a)

    # Benchmark Model B
    result_b = await run_benchmark(
        args.endpoint_b, args.api_key,
        args.n_requests, args.max_tokens, args.name_b
    )
    results.append(result_b)

    # Calculate costs
    results = calculate_costs(results, args.gpu_cost)

    # Print comparison
    print("\n" + "="*70)
    print("RunPod Cost Comparison: GQA vs MHA")
    print("="*70)
    print(f"{'Metric':<25} {args.name_a:<20} {args.name_b:<20}")
    print("-"*70)
    print(f"{'Wall time (s)':<25} {results[0]['wall_time_seconds']:<20} {results[1]['wall_time_seconds']:<20}")
    print(f"{'Total tokens':<25} {results[0]['total_output_tokens']:<20} {results[1]['total_output_tokens']:<20}")
    print(f"{'Throughput (tok/s)':<25} {results[0]['throughput_tok_s']:<20} {results[1]['throughput_tok_s']:<20}")
    print(f"{'GPU cost ($)':<25} {results[0]['gpu_cost_usd']:<20} {results[1]['gpu_cost_usd']:<20}")
    print(f"{'Cost per 1K tokens ($)':<25} {results[0]['cost_per_1k_tokens']:<20} {results[1]['cost_per_1k_tokens']:<20}")
    print("="*70)

    # Calculate savings
    if results[0]['gpu_cost_usd'] > results[1]['gpu_cost_usd']:
        savings_pct = (1 - results[1]['gpu_cost_usd'] / results[0]['gpu_cost_usd']) * 100
        print(f"\n{args.name_b} saves {savings_pct:.1f}% vs {args.name_a}")
    else:
        savings_pct = (1 - results[0]['gpu_cost_usd'] / results[1]['gpu_cost_usd']) * 100
        print(f"\n{args.name_a} saves {savings_pct:.1f}% vs {args.name_b}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x scripts/runpod_cost_bench.py
```

### B2.3 Run the Comparison

```bash
# Example: Compare Llama-2-7B (MHA) vs Mistral-7B (GQA)
python scripts/runpod_cost_bench.py \
    --endpoint-a "https://api.runpod.ai/v2/YOUR_LLAMA_ENDPOINT_ID" \
    --endpoint-b "https://api.runpod.ai/v2/YOUR_MISTRAL_ENDPOINT_ID" \
    --api-key "$RUNPOD_API_KEY" \
    --name-a "Llama-2-7B-MHA" \
    --name-b "Mistral-7B-GQA" \
    --n-requests 50 \
    --gpu-cost 0.69
```

### Expected Results

```
======================================================================
RunPod Cost Comparison: GQA vs MHA
======================================================================
Metric                    Llama-2-7B-MHA       Mistral-7B-GQA
----------------------------------------------------------------------
Wall time (s)             45.2                 38.1
Total tokens              6400                 6400
Throughput (tok/s)        141.6                168.0
GPU cost ($)              0.0087               0.0073
Cost per 1K tokens ($)    0.001359             0.001141
======================================================================

Mistral-7B-GQA saves 16.1% vs Llama-2-7B-MHA
```

### Success Criteria
- [ ] Both endpoints deployed and responding
- [ ] Cost comparison completed
- [ ] GQA model shows measurable cost savings
- [ ] Results saved to JSON

---

## Bonus B3: Cold Start Analysis (RunPod)

**Objective**: Measure how attention architecture affects cold start vs warm latency.

**Time**: 30 min | **Budget**: ~$1-2

### B3.1 Create Cold Start Probe

```bash
cat > scripts/runpod_cold_start.py << 'EOF'
#!/usr/bin/env python3
"""
RunPod Cold Start Probe: Measure container startup + model load time.
Compares cold start vs warm request latency.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from openai import AsyncOpenAI


async def measure_latency(
    client: AsyncOpenAI,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 32,
) -> dict:
    """Measure single request latency with timing breakdown."""

    t_start = time.perf_counter()

    # Make request
    response = await client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
    )

    t_first_token = None
    tokens = 0

    async for chunk in response:
        if t_first_token is None and chunk.choices[0].delta.content:
            t_first_token = time.perf_counter()
        if chunk.choices[0].delta.content:
            tokens += 1

    t_end = time.perf_counter()

    return {
        "ttft_ms": round((t_first_token - t_start) * 1000, 2) if t_first_token else None,
        "total_time_ms": round((t_end - t_start) * 1000, 2),
        "tokens": tokens,
    }


async def cold_start_test(
    endpoint_url: str,
    api_key: str,
    model_name: str,
    wait_for_cold: int = 300,  # 5 minutes
) -> dict:
    """Test cold start vs warm latency."""

    client = AsyncOpenAI(
        base_url=f"{endpoint_url}/v1",
        api_key=api_key,
        timeout=120.0,  # Long timeout for cold start
    )

    results = {
        "model_name": model_name,
        "endpoint_url": endpoint_url,
        "timestamp": datetime.now().isoformat(),
        "measurements": [],
    }

    print(f"\n[{model_name}] Cold Start Test")
    print("="*50)

    # First request (likely cold)
    print("Request 1 (potentially cold)...")
    try:
        m1 = await measure_latency(client)
        results["measurements"].append({"type": "initial", **m1})
        print(f"  TTFT: {m1['ttft_ms']}ms, Total: {m1['total_time_ms']}ms")
    except Exception as e:
        print(f"  Error: {e}")
        results["measurements"].append({"type": "initial", "error": str(e)})

    # Warm requests (immediate follow-up)
    for i in range(3):
        print(f"Request {i+2} (warm)...")
        try:
            m = await measure_latency(client)
            results["measurements"].append({"type": "warm", **m})
            print(f"  TTFT: {m['ttft_ms']}ms, Total: {m['total_time_ms']}ms")
        except Exception as e:
            print(f"  Error: {e}")
        await asyncio.sleep(1)

    # Calculate cold vs warm difference
    initial = results["measurements"][0]
    warm_avg = sum(
        m["ttft_ms"] for m in results["measurements"][1:]
        if m.get("ttft_ms")
    ) / max(1, len([m for m in results["measurements"][1:] if m.get("ttft_ms")]))

    if initial.get("ttft_ms") and warm_avg:
        results["cold_warm_ratio"] = round(initial["ttft_ms"] / warm_avg, 2)
        results["cold_overhead_ms"] = round(initial["ttft_ms"] - warm_avg, 2)
        print(f"\nCold/Warm ratio: {results['cold_warm_ratio']}x")
        print(f"Cold overhead: {results['cold_overhead_ms']}ms")

    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="RunPod endpoint URL")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--name", default="Model", help="Model name for labeling")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    results = await cold_start_test(
        args.endpoint, args.api_key, args.name
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x scripts/runpod_cold_start.py
```

### B3.2 Run Cold Start Tests

```bash
# Test GQA model
python scripts/runpod_cold_start.py \
    --endpoint "https://api.runpod.ai/v2/YOUR_MISTRAL_ENDPOINT_ID" \
    --api-key "$RUNPOD_API_KEY" \
    --name "Mistral-7B-GQA" \
    --output reports/cold_start_gqa.json

# Test MHA-equivalent model
python scripts/runpod_cold_start.py \
    --endpoint "https://api.runpod.ai/v2/YOUR_LLAMA_ENDPOINT_ID" \
    --api-key "$RUNPOD_API_KEY" \
    --name "Llama-2-7B-MHA" \
    --output reports/cold_start_mha.json
```

### Expected Insights

| Metric | Cold Start | Warm | Difference |
|--------|------------|------|------------|
| TTFT (GQA) | ~3000ms | ~200ms | 15× slower |
| TTFT (MHA) | ~3500ms | ~220ms | 16× slower |

**Key observation**: Cold start dominated by container + model load, not attention type.

### Success Criteria
- [ ] Cold start measured for at least one model
- [ ] Warm latency baseline established
- [ ] Cold/warm ratio documented
- [ ] Understanding: when does cold start matter vs steady-state?

---

## Tier 2 Summary

| Microtask | Status | Key Finding |
|-----------|--------|-------------|
| 1. Setup | ⬜ | Models selected, structure created |
| 2. KV Calculator | ⬜ | Theoretical predictions ready |
| 3. MHA Memory | ⬜ | Baseline memory measured |
| 4. GQA Memory | ⬜ | GQA memory measured, savings confirmed |
| 5. Seq Len Sweep | ⬜ | Linear scaling observed |
| 6. Concurrency | ⬜ | Max concurrent found |
| 7. Throughput | ⬜ | Throughput compared |
| 8. KV Stats | ⬜ | vLLM internals inspected |
| 9. Fingerprint | ⬜ | Reusable tool created |
| B1. SWA Exploration | ⬜ | (Optional) SWA + MoE detection |
| B2. RunPod Cost | ⬜ | (Cloud) GQA cost savings validated |
| B3. Cold Start | ⬜ | (Cloud) Cold vs warm latency measured |

### Artifacts Created

```
scripts/
├── model_config.sh
├── kv_cache_calculator.py
├── memory_profile.py
├── compare_profiles.py
├── seq_len_sweep.py
├── json_to_csv.py
├── concurrency_test.py
├── throughput_bench.py
├── vllm_kv_stats.py
├── arch_fingerprint.py
├── arch_fingerprint_extended.py  # Bonus: SWA + MoE detection
├── runpod_cost_bench.py          # RunPod: cost comparison
└── runpod_cold_start.py          # RunPod: cold start analysis

logs/
├── memory_profile_*.json
├── vllm_kv_stats.json
└── nvidia_smi/

reports/
├── seq_len_kv_scaling.csv
├── max_concurrency.md
├── throughput_comparison.csv
├── model_fingerprints.json
├── extended_fingerprints.json    # Bonus: with SWA/MoE info
├── runpod_cost_comparison.json   # RunPod: GQA vs MHA costs
├── cold_start_gqa.json           # RunPod: cold start data
└── cold_start_mha.json           # RunPod: cold start data
```

---

**→ Continue to [Tier 3](LOG_tier03.md)**: Analysis, quality check, and model selection framework
