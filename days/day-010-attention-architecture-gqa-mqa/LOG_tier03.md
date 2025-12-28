# Day 010 – Attention Architecture Variants: MHA, GQA, MQA
## Tier 3: Analysis, Quality & Decision Framework (Microtasks 10-18)

> **Prerequisites**: Complete [Tier 2](LOG_tier02.md) measurements  
> **Goal**: Analyze results, validate quality, build model selection framework  
> **Time**: ~2.5 hours

---

## Microtask Overview

| # | Task | Focus | Key Artifact |
|---|------|-------|--------------|
| 10 | Memory Analysis | Data interpretation | `reports/memory_analysis.md` |
| 11 | Throughput-Latency Curves | Performance characterization | `reports/perf_curves.md` |
| 12 | Quality Spot Check | Output validation | `reports/quality_comparison.md` |
| 13 | Cost-Per-Token Analysis | Economics | `reports/cost_analysis.md` |
| 14 | vLLM Source: KV Head Detection | Code understanding | `reports/vllm_kv_internals.md` |
| 15 | Architecture Selection Matrix | Decision framework | `reports/selection_matrix.md` |
| 16 | Long-Context Projection | Scaling analysis | `reports/long_context_projection.md` |
| 17 | Production Checklist | Operationalization | `reports/production_checklist.md` |
| 18 | Day 10 Synthesis | Final summary | `reports/day10_synthesis.md` |

---

## Microtask 10: Memory Analysis

**Objective**: Analyze Tier 2 memory measurements and validate against theory.

**Time**: 20 min

### 10.1 Create Analysis Template

```bash
cat > reports/memory_analysis.md << 'EOF'
# Memory Analysis: MHA vs GQA (Day 10)

## Raw Data Summary

### Model A: [MODEL_NAME] ([ATTENTION_TYPE])
- n_heads: [X], n_kv_heads: [Y]
- Theoretical KV reduction: [ratio]×

| Metric | Value |
|--------|-------|
| Idle VRAM | [X] MB |
| Model loaded | [X] MB |
| After warmup | [X] MB |
| Peak usage | [X] MB |
| Weight estimate | [X] MB |
| KV estimate | [X] MB |

### Model B: [MODEL_NAME] ([ATTENTION_TYPE])
- n_heads: [X], n_kv_heads: [Y]
- Theoretical KV reduction: [ratio]×

| Metric | Value |
|--------|-------|
| Idle VRAM | [X] MB |
| Model loaded | [X] MB |
| After warmup | [X] MB |
| Peak usage | [X] MB |
| Weight estimate | [X] MB |
| KV estimate | [X] MB |

## Theory vs Measurement

### KV Cache Prediction Accuracy

| Model | Theoretical KV | Measured KV | Error % |
|-------|----------------|-------------|---------|
| Model A | [X] MB | [Y] MB | [Z]% |
| Model B | [X] MB | [Y] MB | [Z]% |

### Explanation of Discrepancy
[Why measured differs from theoretical - overhead, block allocation, etc.]

## Key Findings

1. **KV Reduction Validated**: GQA model showed [X]× reduction vs MHA, matching theoretical [Y]×
2. **Memory Overhead**: vLLM adds ~[X] MB overhead beyond raw KV cache
3. **Block Allocation**: vLLM allocates in [X]-token blocks, causing step-function behavior

## Implications for Serving

- **Concurrency headroom**: GQA model can serve [X]× more concurrent sequences
- **Break-even point**: At [X] concurrent users, GQA savings equal [Y] GB
- **Recommendation**: For [use case], prefer [architecture] because [reason]
EOF
```

### 10.2 Fill In Analysis

Using your Tier 2 measurements, fill in the template above.

### 10.3 Create Comparison Chart Script

```bash
cat > scripts/plot_memory_comparison.py << 'EOF'
#!/usr/bin/env python3
"""
Generate memory comparison visualization.
Requires: pip install matplotlib
"""

import json
import matplotlib.pyplot as plt
import sys


def plot_comparison(profile_paths: list, output_path: str):
    """Create bar chart comparing memory profiles."""
    
    profiles = []
    for path in profile_paths:
        with open(path) as f:
            profiles.append(json.load(f))
    
    # Extract data
    models = [p['model_id'].split('/')[-1][:15] for p in profiles]
    weights = [p.get('model_weight_estimate_mb', 0) for p in profiles]
    kv_cache = [p.get('kv_cache_estimate_mb', 0) for p in profiles]
    
    # Create grouped bar chart
    x = range(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], weights, width, label='Model Weights', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], kv_cache, width, label='KV Cache (est)', color='coral')
    
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Breakdown: Model Weights vs KV Cache')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_memory_comparison.py <profile1.json> <profile2.json> ... <output.png>")
        sys.exit(1)
    
    profiles = sys.argv[1:-1]
    output = sys.argv[-1]
    plot_comparison(profiles, output)
EOF

# Generate plot if matplotlib available
# python scripts/plot_memory_comparison.py \
#     logs/memory_profile_qwen25_1.5b.json \
#     logs/memory_profile_mistral_7b.json \
#     reports/memory_comparison.png
```

### Success Criteria
- [ ] Memory analysis document completed
- [ ] Theory vs measurement comparison done
- [ ] Discrepancies explained

---

## Microtask 11: Throughput-Latency Curves

**Objective**: Characterize performance trade-offs across concurrency levels.

**Time**: 25 min

### 11.1 Create Concurrency Sweep Script

```bash
cat > scripts/concurrency_sweep.py << 'EOF'
#!/usr/bin/env python3
"""
Sweep concurrency levels and measure throughput + latency.
Produces data for throughput-latency curves.
"""

import argparse
import json
import time
from datetime import datetime


def run_at_concurrency(model_id: str, concurrency: int, n_requests: int,
                       max_tokens: int, max_model_len: int) -> dict:
    """Run benchmark at specific concurrency level."""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.88,
        max_num_seqs=concurrency + 4,
        trust_remote_code=True,
    )
    
    # Warmup
    _ = llm.generate(["Warmup."] * 2, SamplingParams(max_tokens=16))
    
    # Actual test
    prompts = [f"Explain topic {i} in detail." for i in range(n_requests)]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    
    latencies = []
    total_tokens = 0
    
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.perf_counter()
    
    wall_time = t_end - t_start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    # Approximate per-request latency (simplified)
    avg_latency_ms = (wall_time / n_requests) * 1000
    
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall_time, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "requests_per_s": round(n_requests / wall_time, 2),
    }


def sweep_concurrency(model_id: str, concurrencies: list, n_requests: int,
                      max_tokens: int, max_model_len: int) -> list:
    """Sweep across concurrency levels."""
    results = []
    
    for conc in concurrencies:
        print(f"\n[Sweep] Testing concurrency={conc}...")
        
        try:
            result = run_at_concurrency(model_id, conc, n_requests, 
                                        max_tokens, max_model_len)
            result["model_id"] = model_id
            results.append(result)
            print(f"  Throughput: {result['throughput_tok_s']} tok/s, "
                  f"Latency: {result['avg_latency_ms']:.0f} ms")
            
            # Cleanup between runs
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"concurrency": conc, "error": str(e)})
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    results = sweep_concurrency(
        args.model, args.concurrencies, args.n_requests,
        args.max_tokens, args.max_model_len
    )
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/concurrency_sweep.py
```

### 11.2 Run Sweep and Document

```bash
# Run sweep
python scripts/concurrency_sweep.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --concurrencies 1 2 4 8 16 \
    --output reports/concurrency_sweep_qwen25.json

# Create analysis document
cat > reports/perf_curves.md << 'EOF'
# Throughput-Latency Analysis (Day 10)

## Concurrency Sweep Results

### Model: [MODEL_NAME]

| Concurrency | Throughput (tok/s) | Avg Latency (ms) | Requests/s |
|-------------|-------------------|------------------|------------|
| 1 | [X] | [X] | [X] |
| 2 | [X] | [X] | [X] |
| 4 | [X] | [X] | [X] |
| 8 | [X] | [X] | [X] |
| 16 | [X] | [X] | [X] |

## Observations

### Throughput Scaling
- Linear scaling up to concurrency=[X]
- Saturation begins at concurrency=[X]
- Max throughput: [X] tok/s at concurrency=[X]

### Latency Behavior
- Single-request latency: [X] ms
- Latency at max throughput: [X] ms ([X]% increase)
- Latency knee point: concurrency=[X]

### Optimal Operating Point
For this model on this GPU:
- **Latency-optimized**: concurrency=[X], throughput=[X] tok/s
- **Throughput-optimized**: concurrency=[X], latency=[X] ms

## Architecture Impact

How attention type affects the curves:
- GQA models can push to higher concurrency before latency degrades
- MHA models saturate earlier due to KV cache pressure
- [Your observations]
EOF
```

### Success Criteria
- [ ] Concurrency sweep completed
- [ ] Throughput and latency documented
- [ ] Optimal operating point identified

---

## Microtask 12: Quality Spot Check

**Objective**: Verify that GQA doesn't degrade output quality vs MHA.

**Time**: 25 min

### 12.1 Create Quality Comparison Script

```bash
cat > scripts/quality_comparison.py << 'EOF'
#!/usr/bin/env python3
"""
Side-by-side quality comparison between models.
Generates outputs for manual inspection.
"""

import argparse
import json
from datetime import datetime

# Test prompts covering different capabilities
TEST_PROMPTS = [
    # Factual
    "What is the capital of France? Answer in one sentence.",
    
    # Reasoning
    "If I have 15 apples and give away 7, then buy 3 more, how many do I have? Show your work.",
    
    # Code
    "Write a Python function to check if a string is a palindrome.",
    
    # Explanation
    "Explain what grouped-query attention is and why it's used in modern LLMs. Keep it under 100 words.",
    
    # Creative
    "Write a haiku about machine learning.",
    
    # Summarization
    "Summarize in one sentence: The transformer architecture revolutionized NLP by introducing self-attention mechanisms that allow models to process all tokens in parallel rather than sequentially, leading to faster training and better capture of long-range dependencies.",
    
    # Instruction following
    "List exactly 3 benefits of using GQA over MHA. Use bullet points.",
    
    # Edge case: Long reasoning
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Think carefully before answering.",
]


def generate_outputs(model_id: str, prompts: list, max_tokens: int = 256) -> list:
    """Generate outputs for all prompts."""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=model_id,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for prompt, output in zip(prompts, outputs):
        results.append({
            "prompt": prompt,
            "output": output.outputs[0].text.strip(),
            "tokens": len(output.outputs[0].token_ids),
        })
    
    return results


def compare_models(model_a: str, model_b: str, output_path: str):
    """Generate side-by-side comparison."""
    print(f"\nGenerating outputs for Model A: {model_a}")
    outputs_a = generate_outputs(model_a, TEST_PROMPTS)
    
    # Cleanup before loading second model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nGenerating outputs for Model B: {model_b}")
    outputs_b = generate_outputs(model_b, TEST_PROMPTS)
    
    # Combine results
    comparison = {
        "model_a": model_a,
        "model_b": model_b,
        "timestamp": datetime.now().isoformat(),
        "comparisons": []
    }
    
    for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        comparison["comparisons"].append({
            "id": i + 1,
            "prompt": a["prompt"],
            "model_a_output": a["output"],
            "model_a_tokens": a["tokens"],
            "model_b_output": b["output"],
            "model_b_tokens": b["tokens"],
        })
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("Quality Comparison Summary")
    print("="*80)
    
    for comp in comparison["comparisons"]:
        print(f"\n[{comp['id']}] {comp['prompt'][:60]}...")
        print(f"  Model A ({comp['model_a_tokens']} tok): {comp['model_a_output'][:100]}...")
        print(f"  Model B ({comp['model_b_tokens']} tok): {comp['model_b_output'][:100]}...")
    
    print(f"\nFull results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    compare_models(args.model_a, args.model_b, args.output)


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/quality_comparison.py
```

### 12.2 Run Comparison

```bash
# Compare two models (if both available)
# python scripts/quality_comparison.py \
#     --model-a "meta-llama/Llama-2-7b-hf" \
#     --model-b "mistralai/Mistral-7B-v0.1" \
#     --output reports/quality_comparison_raw.json

# For faster iteration, compare same model family different sizes
python scripts/quality_comparison.py \
    --model-a "Qwen/Qwen2.5-1.5B-Instruct" \
    --model-b "Qwen/Qwen2.5-1.5B-Instruct" \
    --output reports/quality_comparison_raw.json
```

### 12.3 Create Quality Report

```bash
cat > reports/quality_comparison.md << 'EOF'
# Quality Comparison: MHA vs GQA (Day 10)

## Test Methodology
- 8 diverse prompts covering: factual, reasoning, code, explanation, creative, summarization, instruction, edge cases
- Temperature=0 for reproducibility
- max_tokens=256

## Model Comparison

| Model | Attention Type | Parameters |
|-------|---------------|------------|
| Model A | [MHA/GQA] | [size] |
| Model B | [GQA] | [size] |

## Results by Category

### 1. Factual Questions
- Model A: [Pass/Fail] - [brief note]
- Model B: [Pass/Fail] - [brief note]

### 2. Reasoning
- Model A: [Pass/Fail] - [brief note]
- Model B: [Pass/Fail] - [brief note]

### 3. Code Generation
- Model A: [Pass/Fail] - [brief note]
- Model B: [Pass/Fail] - [brief note]

### 4. Explanation
- Model A: [Pass/Fail] - [brief note]
- Model B: [Pass/Fail] - [brief note]

### 5. Creative
- Model A: [Quality 1-5] - [brief note]
- Model B: [Quality 1-5] - [brief note]

### 6. Summarization
- Model A: [Pass/Fail] - [brief note]
- Model B: [Pass/Fail] - [brief note]

### 7. Instruction Following
- Model A: [Pass/Fail] - [brief note]
- Model B: [Pass/Fail] - [brief note]

### 8. Edge Case (Tricky Math)
- Model A: [Pass/Fail] - [answer given]
- Model B: [Pass/Fail] - [answer given]

## Summary

### Quality Score
- Model A: [X]/8 passed
- Model B: [X]/8 passed

### Key Observations
1. [Observation about GQA quality]
2. [Any degradation noticed?]
3. [Recommendation]

### Conclusion
[Does GQA degrade quality noticeably for these tasks? Is it acceptable for production?]
EOF
```

### Success Criteria
- [ ] Side-by-side outputs generated
- [ ] Manual quality inspection completed
- [ ] Quality report filled out

---

## Microtask 13: Cost-Per-Token Analysis

**Objective**: Calculate economic impact of architecture choice.

**Time**: 20 min

### 13.1 Create Cost Analysis Template

```bash
cat > reports/cost_analysis.md << 'EOF'
# Cost-Per-Token Analysis (Day 10)

## GPU Economics Baseline

| Parameter | Value |
|-----------|-------|
| GPU Type | [e.g., A10, L4, RTX 4090] |
| GPU Cost/Hour | $[X] |
| VRAM | [X] GB |

## Model Comparison

### Model A: [NAME] ([ATTENTION_TYPE])
- Throughput at optimal concurrency: [X] tok/s
- Max concurrent sequences: [X]
- Cost per 1M tokens: $[X]

### Model B: [NAME] ([ATTENTION_TYPE])  
- Throughput at optimal concurrency: [X] tok/s
- Max concurrent sequences: [X]
- Cost per 1M tokens: $[X]

## Cost Calculations

### Tokens per Dollar
```
tokens_per_dollar = throughput_tok_s × 3600 / hourly_cost

Model A: [X] tok/s × 3600 / $[Y] = [Z] tokens/$
Model B: [X] tok/s × 3600 / $[Y] = [Z] tokens/$
```

### Monthly Cost at Scale

Assuming 1B tokens/month:

| Model | Tokens/Month | Hours Needed | Monthly Cost |
|-------|--------------|--------------|--------------|
| Model A | 1B | [X] hrs | $[Y] |
| Model B | 1B | [X] hrs | $[Y] |

**Monthly Savings with GQA**: $[X] ([Y]%)

## Concurrency Value

GQA's memory efficiency enables higher concurrency:
- Model A: [X] concurrent users per GPU
- Model B: [Y] concurrent users per GPU
- **User capacity increase**: [Z]%

### Value of Extra Capacity

If serving [X] users requires:
- Model A: [N] GPUs
- Model B: [M] GPUs
- **GPU savings**: [N-M] GPUs × $[cost] = $[savings]/month

## TCO Recommendation

For a [use case] workload:
- **If latency-critical**: [recommendation]
- **If cost-critical**: [recommendation]
- **Balanced**: [recommendation]

## Break-Even Analysis

GQA overhead (if any quality/latency cost):
- Quality acceptable for: [use cases]
- Quality NOT acceptable for: [use cases]
- Break-even point: [X] requests/day makes GQA worth it
EOF
```

### 13.2 Cost Calculator Script

```bash
cat > scripts/cost_calculator.py << 'EOF'
#!/usr/bin/env python3
"""
Calculate cost per token for different models/architectures.
"""

import argparse


def calculate_costs(throughput_tok_s: float, gpu_cost_per_hour: float,
                    max_concurrent: int, model_name: str):
    """Calculate various cost metrics."""
    
    tokens_per_hour = throughput_tok_s * 3600
    cost_per_1m_tokens = (1_000_000 / tokens_per_hour) * gpu_cost_per_hour
    tokens_per_dollar = tokens_per_hour / gpu_cost_per_hour
    
    # Monthly estimates (1B tokens)
    hours_for_1b = 1_000_000_000 / tokens_per_hour
    monthly_cost_1b = hours_for_1b * gpu_cost_per_hour
    
    print(f"\n{'='*50}")
    print(f"Cost Analysis: {model_name}")
    print(f"{'='*50}")
    print(f"Throughput: {throughput_tok_s:.1f} tok/s")
    print(f"GPU cost: ${gpu_cost_per_hour:.2f}/hr")
    print(f"Max concurrent: {max_concurrent}")
    print(f"\nMetrics:")
    print(f"  Tokens/hour: {tokens_per_hour:,.0f}")
    print(f"  Cost per 1M tokens: ${cost_per_1m_tokens:.4f}")
    print(f"  Tokens per dollar: {tokens_per_dollar:,.0f}")
    print(f"\nAt 1B tokens/month:")
    print(f"  Hours needed: {hours_for_1b:.1f}")
    print(f"  Monthly cost: ${monthly_cost_1b:.2f}")
    print(f"{'='*50}")
    
    return {
        "model": model_name,
        "throughput_tok_s": throughput_tok_s,
        "cost_per_1m_tokens": cost_per_1m_tokens,
        "tokens_per_dollar": tokens_per_dollar,
        "monthly_cost_1b": monthly_cost_1b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--throughput", type=float, required=True, help="Tokens/sec")
    parser.add_argument("--gpu-cost", type=float, required=True, help="$/hour")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--model-name", type=str, default="Model")
    args = parser.parse_args()
    
    calculate_costs(args.throughput, args.gpu_cost, args.max_concurrent, args.model_name)


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/cost_calculator.py

# Example usage:
# python scripts/cost_calculator.py --throughput 500 --gpu-cost 1.50 --model-name "Qwen2.5-1.5B"
```

### Success Criteria
- [ ] Cost analysis document completed
- [ ] Cost per 1M tokens calculated
- [ ] Monthly savings quantified

---

## Microtask 14: vLLM Source Dive – KV Head Detection

**Objective**: Understand how vLLM detects and handles different attention architectures.

**Time**: 25 min

### 14.1 Source Code Pointers

Key files to examine in vLLM source:

```bash
cat > reports/vllm_kv_internals.md << 'EOF'
# vLLM KV Cache Internals (Day 10)

## Key Source Files

### 1. Model Config Detection
**File**: `vllm/config.py`

vLLM reads `num_key_value_heads` from HuggingFace config:
```python
# Simplified logic
num_kv_heads = getattr(hf_config, 'num_key_value_heads', num_attention_heads)
```

### 2. KV Cache Allocation  
**File**: `vllm/worker/cache_engine.py`

KV cache is allocated based on:
- `num_kv_heads` (not `num_attention_heads`)
- `head_dim`
- `num_layers`
- `block_size`
- `dtype`

### 3. Attention Implementation
**File**: `vllm/attention/backends/flash_attn.py` (or other backends)

GQA is handled by:
- Repeating KV heads to match query heads during attention computation
- Memory-efficient: stores reduced KV, expands on-the-fly

### 4. Block Manager
**File**: `vllm/core/block_manager.py`

Block allocation is KV-head-aware:
- Fewer KV heads → smaller blocks → more blocks fit in memory
- This enables higher `max_num_seqs`

## How GQA Saves Memory

### Standard MHA Path
```
KV per layer = 2 × num_heads × head_dim × seq_len × dtype_bytes
```

### GQA Path  
```
KV per layer = 2 × num_kv_heads × head_dim × seq_len × dtype_bytes
```

### vLLM's Optimization
1. Detects `num_kv_heads < num_attention_heads`
2. Allocates KV cache for `num_kv_heads` only
3. During attention, broadcasts KV to match query heads
4. Saves memory proportional to `num_heads / num_kv_heads`

## Verification Commands

### Check detected config
```python
from vllm import LLM
llm = LLM(model="...")
config = llm.llm_engine.model_config.hf_config
print(f"Query heads: {config.num_attention_heads}")
print(f"KV heads: {config.num_key_value_heads}")
```

### Check cache allocation
```python
cache_config = llm.llm_engine.cache_config
print(f"Block size: {cache_config.block_size}")
print(f"GPU blocks: {cache_config.num_gpu_blocks}")
```

## Key Insight

vLLM's PagedAttention + GQA = multiplicative memory savings:
- PagedAttention: eliminates fragmentation → better utilization
- GQA: reduces raw KV size → more sequences fit

Together they enable serving many more concurrent users than naive implementations.
EOF
```

### 14.2 Hands-On Verification

```bash
cat > scripts/verify_kv_detection.py << 'EOF'
#!/usr/bin/env python3
"""Verify vLLM correctly detects attention architecture."""

import sys
from vllm import LLM


def verify_detection(model_id: str):
    print(f"\nLoading: {model_id}")
    llm = LLM(
        model=model_id,
        max_model_len=2048,
        gpu_memory_utilization=0.5,  # Low for quick loading
        trust_remote_code=True,
    )
    
    hf_config = llm.llm_engine.model_config.hf_config
    cache_config = llm.llm_engine.cache_config
    
    n_heads = getattr(hf_config, 'num_attention_heads', None)
    n_kv_heads = getattr(hf_config, 'num_key_value_heads', n_heads)
    
    print(f"\n{'='*50}")
    print(f"Model: {model_id}")
    print(f"{'='*50}")
    print(f"num_attention_heads: {n_heads}")
    print(f"num_key_value_heads: {n_kv_heads}")
    print(f"Attention type: {'MHA' if n_heads == n_kv_heads else f'GQA-{n_heads//n_kv_heads}'}")
    print(f"KV reduction: {n_heads / n_kv_heads:.1f}×")
    print(f"\nCache config:")
    print(f"  block_size: {cache_config.block_size}")
    print(f"  num_gpu_blocks: {cache_config.num_gpu_blocks}")
    print(f"{'='*50}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    verify_detection(model)
EOF

# Run verification
python scripts/verify_kv_detection.py "Qwen/Qwen2.5-1.5B-Instruct"
```

### Success Criteria
- [ ] vLLM source pointers documented
- [ ] KV head detection verified
- [ ] Understanding of how GQA saves memory in vLLM

---

## Microtask 15: Architecture Selection Matrix

**Objective**: Create a decision framework for model selection based on attention architecture.

**Time**: 20 min

### 15.1 Create Selection Matrix

```bash
cat > reports/selection_matrix.md << 'EOF'
# Model Selection Matrix: Attention Architecture (Day 10)

## Decision Factors

| Factor | MHA Favored | GQA Favored | MQA Favored |
|--------|-------------|-------------|-------------|
| **Quality priority** | ✓✓ | ✓ | |
| **Memory constrained** | | ✓ | ✓✓ |
| **High concurrency** | | ✓ | ✓✓ |
| **Long context** | | ✓✓ | ✓✓ |
| **Latency sensitive** | ✓ | ✓ | ✓ |
| **Cost sensitive** | | ✓ | ✓✓ |

## Selection Flowchart

```
                    ┌─────────────────────┐
                    │ Quality critical?    │
                    │ (reasoning, code)   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
               Yes  │  Consider MHA or    │  No
              ┌─────│  conservative GQA   │─────┐
              │     └─────────────────────┘     │
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │ Memory/cost     │               │ Memory/cost     │
    │ constrained?    │               │ constrained?    │
    └────────┬────────┘               └────────┬────────┘
             │                                  │
        Yes  │  No                         Yes  │  No
        ┌────┴────┐                       ┌────┴────┐
        ▼         ▼                       ▼         ▼
   ┌────────┐ ┌────────┐            ┌────────┐ ┌────────┐
   │ GQA    │ │ MHA    │            │ MQA    │ │ GQA    │
   │ (4-8×) │ │        │            │ or     │ │        │
   └────────┘ └────────┘            │ GQA    │ └────────┘
                                    └────────┘
```

## Use Case Recommendations

### 1. Interactive Chat (Low Latency, Many Users)
**Recommendation**: GQA (4× reduction)
- Reason: Balances quality with concurrency
- Models: Llama-3-8B, Mistral-7B, Qwen2.5-7B

### 2. Batch Processing (High Throughput)
**Recommendation**: GQA or MQA
- Reason: Maximize GPU utilization
- Models: Falcon-7B (MQA), Qwen2.5 series

### 3. Long-Context Applications
**Recommendation**: GQA (8× reduction) or MQA
- Reason: KV cache dominates at long context
- Models: Llama-3-70B (8× GQA), specialized long-context models

### 4. Complex Reasoning / Code
**Recommendation**: MHA or conservative GQA
- Reason: Quality matters more than efficiency
- Models: Llama-2-7B, high-quality fine-tuned models

### 5. Edge / Limited Memory
**Recommendation**: Aggressive GQA or MQA
- Reason: Must fit in small VRAM
- Models: Qwen2.5-1.5B (6× GQA), small MQA models

## Model Recommendations by GPU

| GPU (VRAM) | MHA Models | GQA Models | Notes |
|------------|------------|------------|-------|
| 8GB | Small only | 1.5-3B GQA | Qwen2.5-1.5B ideal |
| 16GB | 7B possible | 7-8B comfortable | Mistral-7B, Llama-3-8B |
| 24GB | 7B easy | 13B possible | Higher concurrency |
| 40GB+ | 13B easy | 70B possible | Production-scale |

## Quick Reference Card

When selecting a model:

1. **Check**: `num_key_value_heads` in model config
2. **Calculate**: KV reduction = `num_heads / num_kv_heads`
3. **Estimate**: Max concurrent = (Available VRAM / KV per seq) × safety_factor
4. **Decide**: Based on quality requirements vs efficiency needs
EOF
```

### Success Criteria
- [ ] Selection matrix created
- [ ] Decision flowchart documented
- [ ] Use case recommendations provided

---

## Microtask 16: Long-Context Projection

**Objective**: Project how architecture choice affects long-context serving.

**Time**: 20 min

### 16.1 Create Long-Context Analysis

```bash
cat > reports/long_context_projection.md << 'EOF'
# Long-Context Projection (Day 10)

## Why Long Context Amplifies Architecture Differences

KV cache scales with sequence length:
```
KV_size ∝ seq_len × n_kv_heads
```

At 4K context, architecture differences are noticeable.
At 32K+ context, architecture differences are **critical**.

## Projection Table

### 7B-class Model (32 layers, d_head=128)

| Context | MHA (32 KV heads) | GQA-4 (8 KV heads) | GQA-8 (4 KV heads) |
|---------|-------------------|--------------------|--------------------|
| 4K | 2.0 GB | 512 MB | 256 MB |
| 8K | 4.0 GB | 1.0 GB | 512 MB |
| 16K | 8.0 GB | 2.0 GB | 1.0 GB |
| 32K | 16.0 GB | 4.0 GB | 2.0 GB |
| 64K | 32.0 GB | 8.0 GB | 4.0 GB |
| 128K | 64.0 GB | 16.0 GB | 8.0 GB |

### Implications

#### At 32K Context on 24GB GPU

| Architecture | KV per seq | Model (7B fp16) | Available | Max Seqs |
|--------------|------------|-----------------|-----------|----------|
| MHA | 16 GB | 14 GB | 0 GB | **OOM** |
| GQA-4 | 4 GB | 14 GB | ~6 GB | 1-2 |
| GQA-8 | 2 GB | 14 GB | ~8 GB | 3-4 |

**Conclusion**: MHA cannot serve 32K context on 24GB; GQA makes it possible.

#### At 128K Context

Even aggressive GQA requires:
- GQA-8 (4 KV heads): 8 GB per sequence
- Single sequence uses significant VRAM
- Multi-user serving requires multi-GPU or careful scheduling

## Long-Context Model Selection

### For 32K+ Context
**Requirement**: At least GQA-4 (4× reduction)
**Recommended**: 
- Llama-3-70B (GQA-8)
- Mistral-Large (GQA)
- Qwen2.5-72B (GQA)

### For 128K+ Context
**Requirement**: Aggressive GQA or specialized architecture
**Recommended**:
- Models with sliding window attention
- Models with sparse attention patterns
- GQA-8 or higher reduction

## Calculator

```python
# Quick calculation for your setup
def can_serve_context(
    available_vram_gb: float,
    model_weights_gb: float,
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    target_context: int,
    dtype_bytes: int = 2
) -> dict:
    kv_per_seq = 2 * n_layers * n_kv_heads * d_head * target_context * dtype_bytes
    kv_per_seq_gb = kv_per_seq / 1024**3
    
    available_for_kv = available_vram_gb - model_weights_gb - 1.0  # 1GB overhead
    max_seqs = int(available_for_kv / kv_per_seq_gb) if kv_per_seq_gb > 0 else 0
    
    return {
        "kv_per_seq_gb": kv_per_seq_gb,
        "available_for_kv_gb": available_for_kv,
        "max_concurrent_seqs": max(0, max_seqs),
        "feasible": max_seqs >= 1
    }
```

## Takeaway

**Architecture choice becomes MORE important as context length increases.**

- At 4K: MHA works, GQA is nice-to-have
- At 16K: GQA recommended
- At 32K+: GQA required
- At 128K+: Aggressive GQA or specialized attention required
EOF
```

### Success Criteria
- [ ] Long-context projections calculated
- [ ] Feasibility analysis for different GPU sizes
- [ ] Clear recommendations documented

---

## Microtask 17: Production Checklist

**Objective**: Create operational checklist for deploying models with different attention architectures.

**Time**: 15 min

### 17.1 Create Production Checklist

```bash
cat > reports/production_checklist.md << 'EOF'
# Production Deployment Checklist: Attention Architecture (Day 10)

## Pre-Deployment Verification

### 1. Architecture Identification
- [ ] Confirmed `num_attention_heads` from model config
- [ ] Confirmed `num_key_value_heads` from model config
- [ ] Calculated KV reduction factor
- [ ] Documented attention type (MHA/GQA/MQA)

### 2. Memory Planning
- [ ] Calculated model weight memory requirement
- [ ] Calculated KV cache per sequence at target context length
- [ ] Verified total fits in GPU VRAM with safety margin
- [ ] Determined max concurrent sequences

### 3. Quality Validation
- [ ] Ran quality spot-check on representative prompts
- [ ] Compared outputs to MHA baseline (if GQA/MQA)
- [ ] Documented any quality degradation
- [ ] Approved for target use case

### 4. Performance Baseline
- [ ] Measured single-request latency
- [ ] Measured throughput at target concurrency
- [ ] Identified optimal operating point
- [ ] Set SLO targets based on measurements

## Deployment Configuration

### vLLM Recommended Settings

```bash
# For GQA model (e.g., Mistral-7B, Llama-3-8B)
vllm serve $MODEL_ID \
    --max-model-len 4096 \           # Adjust based on use case
    --gpu-memory-utilization 0.90 \  # Can be higher with GQA
    --max-num-seqs 32 \              # Higher than MHA equivalent
    --enable-prefix-caching \        # Always enable
    --dtype auto
```

### Concurrency Guidelines

| Attention Type | Suggested max_num_seqs (24GB GPU, 4K context) |
|----------------|-----------------------------------------------|
| MHA | 4-8 |
| GQA-4 | 16-32 |
| GQA-8 | 32-64 |
| MQA | 64+ |

## Monitoring

### Key Metrics to Track
- [ ] KV cache utilization (%)
- [ ] GPU memory utilization (%)
- [ ] Request queue depth
- [ ] p50/p95/p99 latency
- [ ] Throughput (tokens/sec)

### Alert Thresholds
- KV cache > 90%: Warning (may reject requests)
- GPU memory > 95%: Critical (OOM risk)
- p99 latency > 2× p50: Investigate queueing

## Rollback Criteria

Consider rollback to different architecture if:
- [ ] Quality degradation reported by users
- [ ] Unexpected OOM under normal load
- [ ] Latency SLO consistently violated
- [ ] Throughput significantly below projections

## Documentation

- [ ] Model architecture documented in runbook
- [ ] Capacity limits documented
- [ ] Known limitations documented
- [ ] On-call escalation path defined
EOF
```

### Success Criteria
- [ ] Checklist created
- [ ] All items actionable
- [ ] Ready for production use

---

## Microtask 18: Day 10 Synthesis

**Objective**: Synthesize all learnings into a final summary document.

**Time**: 20 min

### 18.1 Create Synthesis Document

```bash
cat > reports/day10_synthesis.md << 'EOF'
# Day 10 Synthesis: Attention Architecture Variants

## Executive Summary

**Key Learning**: Attention architecture (MHA vs GQA vs MQA) is a fundamental trade-off between model quality and serving efficiency. GQA has emerged as the practical sweet spot for production LLM serving.

## What I Learned

### 1. The Memory Math
- KV cache scales with `n_kv_heads`, not `n_attention_heads`
- GQA reduces KV cache by `n_heads / n_kv_heads` factor
- At 4K context, this means [X] GB savings for 7B model
- At 32K context, GQA becomes **required**, not optional

### 2. Measured Reality
- Theoretical predictions matched measured values within [X]%
- GQA model served [X]× more concurrent users
- Throughput improved by [X]% at high concurrency
- Quality degradation was [minimal/noticeable/significant]

### 3. Production Implications
- Model selection should start with attention architecture
- GQA is the default choice for most production workloads
- MHA only for quality-critical, low-concurrency use cases
- MQA for extreme efficiency needs where quality can be sacrificed

## Connecting to Previous Days

| Day | Topic | Connection to Day 10 |
|-----|-------|---------------------|
| Day 03 | Capacity & OOM | Architecture determines OOM threshold |
| Day 04 | Quantization | Quantization + GQA = multiplicative savings |
| Day 07 | Runtime Probes | KV allocation affects cold start |
| Day 09 | KV Cache Deep Dive | Architecture is the KV size multiplier |

## Key Artifacts Created

1. **KV Cache Calculator** (`scripts/kv_cache_calculator.py`)
   - Reusable tool for any model
   - Includes all major model families

2. **Architecture Fingerprint** (`scripts/arch_fingerprint.py`)
   - Quick model architecture detection
   - No GPU required

3. **Selection Matrix** (`reports/selection_matrix.md`)
   - Decision framework for model selection
   - Use case recommendations

4. **Production Checklist** (`reports/production_checklist.md`)
   - Deployment verification steps
   - Monitoring guidelines

## Mental Model Update

Before Day 10:
> "KV cache is big and scales with context length"

After Day 10:
> "KV cache scales with `n_kv_heads × context_length`. Architecture choice is the first-order lever for serving efficiency. GQA models can serve 4-8× more users with minimal quality loss."

## Open Questions for Future Days

1. **Speculative Decoding**: How does architecture affect speculation efficiency?
2. **Prefix Caching**: Does GQA affect prefix cache hit rates?
3. **Quantization + GQA**: What's the compound effect on quality?
4. **Long Context**: At what point do we need sparse attention?

## Recommendations

### For Learning
- Always check `num_key_value_heads` when evaluating a model
- Use the fingerprint tool on any new model
- Run memory profiler before committing to production

### For Production
- Default to GQA models (Llama-3, Mistral, Qwen2.5)
- Size `max_num_seqs` based on architecture, not just model size
- Monitor KV cache utilization, not just GPU memory

### For Consulting
- Architecture comparison is a valuable differentiator
- Cost savings from GQA can be [X]% monthly
- Quality validation is essential before recommending GQA

## Next Steps

Potential Day 11 topics (building on Day 10):
- [ ] **Speculative Decoding**: Theme 6 - decoding acceleration
- [ ] **Distributed Serving**: Theme 7 - tensor parallelism with GQA
- [ ] **Multi-tenancy**: Theme 8 - per-tenant model selection

---

*Day 10 Complete. Architecture-aware model selection is now part of my inference engineering toolkit.*
EOF
```

### Success Criteria
- [ ] Synthesis document completed
- [ ] Key learnings articulated
- [ ] Mental model updated
- [ ] Next steps identified

---

## Tier 3 Summary

| Microtask | Status | Key Output |
|-----------|--------|------------|
| 10. Memory Analysis | ⬜ | Theory vs measurement validated |
| 11. Perf Curves | ⬜ | Throughput-latency characterized |
| 12. Quality Check | ⬜ | Quality impact assessed |
| 13. Cost Analysis | ⬜ | Economic impact quantified |
| 14. vLLM Source | ⬜ | Internals understood |
| 15. Selection Matrix | ⬜ | Decision framework created |
| 16. Long-Context | ⬜ | Scaling projections done |
| 17. Production Checklist | ⬜ | Operational readiness verified |
| 18. Synthesis | ⬜ | Day 10 complete |

### Final Artifacts

```
reports/
├── memory_analysis.md
├── perf_curves.md
├── quality_comparison.md
├── cost_analysis.md
├── vllm_kv_internals.md
├── selection_matrix.md
├── long_context_projection.md
├── production_checklist.md
└── day10_synthesis.md
```

---

## Reading List (Day 10)

### Core Papers
1. **GQA Paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)
2. **MQA Paper**: "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)

### Model References
3. **Llama 2**: Technical Report (Touvron et al., 2023)
4. **Llama 3**: Model Card and documentation
5. **Mistral 7B**: Technical Report

### Implementation
6. **vLLM Source**: `config.py`, `cache_engine.py`, attention backends
7. **FlashAttention-2**: GQA-aware implementation details

### Blog Posts
8. **"The KV Cache"**: Any good explanation of KV cache mechanics
9. **"Why GQA?"**: Architecture decision explanations from model authors

---

**Day 10 Complete!**

You now understand how attention architecture affects LLM serving at a fundamental level. This knowledge directly impacts model selection, capacity planning, and cost optimization in production.

**→ Continue to Day 11**: [Next topic based on gap analysis]
