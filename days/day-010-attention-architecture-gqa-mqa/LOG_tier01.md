# Day 010 – Attention Architecture Variants: MHA, GQA, MQA
## Tier 1: Theory & Foundations

> **Prerequisites**: Complete [Day 009](../day-009-latency-metrics-and-slo-lab/) (KV Cache Deep Dive)  
> **Goal**: Understand how attention architecture choices (MHA vs GQA vs MQA) affect KV cache size, memory footprint, and serving performance  
> **End State**: Mental model for architecture-aware model selection; baseline measurements comparing models with different attention patterns  
> **Time**: ~2 hours theory + setup

---

## Why This Matters (Day 09 → Day 10 Bridge)

Day 09 taught you that **KV cache is the dominant memory consumer** during inference:
- KV cache scales with `layers × heads × head_dim × seq_len × batch × 2 (K+V) × dtype_bytes`
- At high concurrency or long context, KV cache can exceed model weights in memory

**The architectural lever you haven't pulled yet**: The `heads` term in that equation.

Different attention architectures **trade off quality vs KV cache size**:
- **MHA** (Multi-Head Attention): Full KV per head → maximum quality, maximum memory
- **GQA** (Grouped-Query Attention): Shared KV across query head groups → reduced memory
- **MQA** (Multi-Query Attention): Single KV set for all query heads → minimum memory

This day teaches you to:
1. Understand the memory math behind each architecture
2. Measure the actual impact on vLLM serving
3. Make informed model selection decisions for production

---

## Core Concepts

### 1. Multi-Head Attention (MHA) – The Baseline

Standard transformer attention (Vaswani et al., 2017):

```
For each head h ∈ [1, n_heads]:
    Q_h = X · W_q^h    # Query projection
    K_h = X · W_k^h    # Key projection  
    V_h = X · W_v^h    # Value projection
    
    Attention_h = softmax(Q_h · K_h^T / √d_head) · V_h
    
Output = Concat(Attention_1, ..., Attention_n) · W_o
```

**KV Cache per layer (MHA)**:
```
KV_bytes = 2 × n_heads × d_head × seq_len × dtype_bytes
         = 2 × n_heads × (d_model / n_heads) × seq_len × dtype_bytes
         = 2 × d_model × seq_len × dtype_bytes
```

**Example (Llama-2-7B style)**:
- `n_heads = 32`, `d_model = 4096`, `d_head = 128`
- Per layer: `2 × 4096 × seq_len × 2 bytes (fp16)`
- For 4K context: `2 × 4096 × 4096 × 2 = 67 MB per layer`
- 32 layers: **~2.1 GB KV cache** for a single 4K sequence

---

### 2. Multi-Query Attention (MQA) – The Memory Saver

Introduced by Shazeer (2019) for faster inference:

```
For each head h ∈ [1, n_heads]:
    Q_h = X · W_q^h    # Separate query per head (unchanged)
    
K = X · W_k            # SINGLE key projection (shared)
V = X · W_v            # SINGLE value projection (shared)

For each head h:
    Attention_h = softmax(Q_h · K^T / √d_head) · V
```

**KV Cache per layer (MQA)**:
```
KV_bytes = 2 × 1 × d_head × seq_len × dtype_bytes
         = 2 × d_head × seq_len × dtype_bytes
```

**Memory reduction**: `n_heads × d_head` → `1 × d_head` = **n_heads× smaller**

**Example (same model dimensions)**:
- Per layer: `2 × 128 × 4096 × 2 = 2.1 MB` (vs 67 MB for MHA)
- 32 layers: **~67 MB KV cache** (vs 2.1 GB for MHA)
- **32× memory reduction** for KV cache

**Trade-off**: Quality can degrade, especially for complex reasoning tasks.

---

### 3. Grouped-Query Attention (GQA) – The Sweet Spot

Introduced by Ainslie et al. (2023) to balance quality and efficiency:

```
Define n_kv_heads < n_heads (e.g., n_heads=32, n_kv_heads=8)
Group size = n_heads / n_kv_heads (e.g., 32/8 = 4 query heads per KV group)

For each KV group g ∈ [1, n_kv_heads]:
    K_g = X · W_k^g    # One K per group
    V_g = X · W_v^g    # One V per group
    
For each query head h in group g:
    Q_h = X · W_q^h
    Attention_h = softmax(Q_h · K_g^T / √d_head) · V_g
```

**KV Cache per layer (GQA)**:
```
KV_bytes = 2 × n_kv_heads × d_head × seq_len × dtype_bytes
```

**Memory reduction factor**: `n_heads / n_kv_heads`

**Example (Llama-3-8B: n_heads=32, n_kv_heads=8)**:
- Per layer: `2 × 8 × 128 × 4096 × 2 = 16.8 MB` (vs 67 MB for MHA)
- 32 layers: **~537 MB KV cache** (vs 2.1 GB for MHA)
- **4× memory reduction** while retaining most quality

---

## Architecture Comparison Table

| Model Family | Attention Type | n_heads | n_kv_heads | KV Reduction | Notes |
|--------------|----------------|---------|------------|--------------|-------|
| Llama-2-7B | GQA | 32 | 32 | 1× (full MHA) | Actually MHA despite GQA support |
| Llama-2-70B | GQA | 64 | 8 | 8× | First major GQA deployment |
| Llama-3-8B | GQA | 32 | 8 | 4× | GQA by default |
| Llama-3-70B | GQA | 64 | 8 | 8× | |
| Mistral-7B | GQA | 32 | 8 | 4× | + Sliding window attention |
| Qwen2.5-7B | GQA | 28 | 4 | 7× | Aggressive KV reduction |
| Falcon-7B | MQA | 71 | 1 | 71× | Original MQA model |
| Falcon-40B | GQA | 128 | 8 | 16× | |

---

## Memory Math: Why This Matters for Serving

### The Concurrency Equation

From Day 09, you know:
```
Available_KV_VRAM = Total_VRAM - Model_Weights - Activations - Overhead
Max_Concurrent_Seqs ≈ Available_KV_VRAM / KV_per_seq
```

**Impact of attention architecture on max concurrency**:

| Attention | KV per 4K seq (7B model) | Max seqs in 10GB headroom |
|-----------|-------------------------|---------------------------|
| MHA (32 KV heads) | ~2.1 GB | 4-5 |
| GQA (8 KV heads) | ~537 MB | 18-19 |
| MQA (1 KV head) | ~67 MB | 149 |

**Serving implication**: GQA models can serve **4× more concurrent users** than equivalent MHA models with the same VRAM.

---

### The Throughput Equation

KV cache bandwidth affects decode speed:
```
KV_bandwidth_per_token = 2 × n_kv_heads × d_head × seq_len × dtype_bytes
Decode_time ∝ KV_bandwidth_per_token / Memory_BW
```

Smaller KV cache → faster memory reads → higher tokens/sec at same batch size.

---

## What You'll Measure Today

### Tier 1 Artifacts (Theory Understanding)
- [ ] Annotated architecture comparison for 3 models
- [ ] KV cache size predictions (calculated) vs observed (nvidia-smi)

### Tier 2 Labs (Microtasks 1-9)
- [ ] Compare Llama-2-7B (MHA-like) vs Mistral-7B (GQA) on same workload
- [ ] Measure KV cache memory at varying sequence lengths
- [ ] Compare max concurrency before OOM

### Tier 3 Analysis (Microtasks 10-18)
- [ ] Throughput vs concurrency curves for MHA vs GQA
- [ ] Quality spot-check (does GQA degrade output?)
- [ ] Model selection decision framework

---

## Environment Assumptions

Same as Day 09:
- **GPU**: Single NVIDIA GPU with 24GB VRAM (e.g., RTX 4090, A10, L4)
- **OS**: Ubuntu 22.04/24.04
- **Runtime**: vLLM ≥ 0.6.x with V1 engine
- **Python**: 3.10+

### Folder Layout

```
days/day-010-attention-architecture-gqa-mqa/
├── LOG_tier01.md          # This file (theory)
├── LOG_tier02.md          # Microtasks 1-9 (Core + Instrumentation)
├── LOG_tier03.md          # Microtasks 10-18 (Analysis + Framework)
├── scripts/
│   ├── kv_cache_calculator.py
│   ├── attention_arch_bench.py
│   └── model_comparison.sh
├── data/
│   └── prompts/
├── logs/
│   └── nvidia_smi/
├── reports/
│   ├── kv_cache_measurements.csv
│   └── architecture_comparison.md
└── archive/
```

---

## Key Formulas Reference

### KV Cache Size (per sequence, full model)

```python
def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    """Calculate KV cache size for one sequence."""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * dtype_bytes

# Examples:
# Llama-2-7B (MHA): kv_cache_bytes(32, 32, 128, 4096) = 2,147,483,648 (~2.1 GB)
# Llama-3-8B (GQA): kv_cache_bytes(32, 8, 128, 4096) = 536,870,912 (~537 MB)
# Mistral-7B (GQA): kv_cache_bytes(32, 8, 128, 4096) = 536,870,912 (~537 MB)
```

### Memory Reduction Factor

```python
def kv_reduction_factor(n_heads, n_kv_heads):
    """How much smaller is GQA/MQA vs MHA."""
    return n_heads / n_kv_heads

# Llama-3-8B: 32 / 8 = 4× reduction
# Qwen2.5-7B: 28 / 4 = 7× reduction
# Falcon-7B MQA: 71 / 1 = 71× reduction
```

### Max Concurrent Sequences

```python
def max_concurrent_seqs(available_vram_bytes, kv_per_seq_bytes, safety_factor=0.9):
    """Estimate max sequences that fit in available VRAM."""
    return int((available_vram_bytes * safety_factor) / kv_per_seq_bytes)
```

---

## Reading List

### Required (Before Labs)
1. **"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** (Ainslie et al., 2023) – The GQA paper
2. **"Fast Transformer Decoding: One Write-Head is All You Need"** (Shazeer, 2019) – Original MQA paper

### Recommended (Deep Dive)
3. **Llama 2 Technical Report** (Touvron et al., 2023) – Sections on model architecture
4. **Llama 3 Model Card** – GQA configuration details
5. **vLLM Documentation: Model Support** – Which architectures are optimized

### Reference
6. **"Attention Is All You Need"** (Vaswani et al., 2017) – Original MHA
7. **FlashAttention-2** (Dao, 2023) – How attention is actually computed

---

## Self-Check Questions

Before proceeding to Tier 2, you should be able to answer:

1. **Why does GQA reduce memory but not compute?**
   - Hint: Query projections are unchanged; only K/V are shared.

2. **If a model has 32 query heads and 8 KV heads, how many query heads share each KV head?**
   - Answer: 4 (group size = 32/8)

3. **Why can MQA degrade quality more than GQA?**
   - Hint: All queries attend to the same K/V → less expressive attention patterns.

4. **For a 128K context window, how does GQA vs MHA affect serving capacity?**
   - Calculate: At 128K, MHA KV cache is ~68 GB for 7B model; GQA (4×) is ~17 GB.

5. **Why do newer models (Llama-3, Mistral) prefer GQA over MQA?**
   - Trade-off: GQA retains quality while still reducing memory significantly.

---

## Tier 1 Summary

| Concept | Key Insight |
|---------|-------------|
| **MHA** | Full KV per head; best quality, highest memory |
| **GQA** | Shared KV across query groups; balance of quality/memory |
| **MQA** | Single KV for all queries; minimum memory, quality risk |
| **KV Reduction** | `n_heads / n_kv_heads` factor |
| **Serving Impact** | Lower KV → higher concurrency → better throughput |

---

**→ Continue to [Tier 2](LOG_tier02.md)**: Hands-on measurement of KV cache and architecture comparison
