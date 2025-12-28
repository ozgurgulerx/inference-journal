# Day 009 – KV Cache Deep Dive: Anatomy, Prefix Reuse & Block Tuning

> **Goal**: Master vLLM's PagedAttention KV cache internals—understand memory allocation patterns, measure prefix reuse benefits, and tune block size for your hardware.

---

## Snapshot (Today's Focus)

- **Day 9/100**: Runtime layer deep-dive after establishing measurement discipline
- **Recent artifacts**: SLM memory profiling, NUMA tuning, quantization benchmarks, measurement fundamentals
- **Layer**: Inference Runtimes
- **Theme**: KV cache design, partitioning & cross-request reuse (Theme #3)
- **Assumption**: vLLM running with basic measurement setup from Day 1

---

## 1) KV Cache Fundamentals (Why This Matters)

### The Memory Bottleneck Problem

In LLM inference, the **KV cache** stores computed key and value vectors from previous tokens to avoid redundant computation during autoregressive decoding. However, traditional implementations suffer from severe memory inefficiency:

- **Only 20–38% of allocated KV memory is actually used** in conventional systems
- The rest is reserved but empty (pre-allocated for max sequence length)
- This creates **external fragmentation** (gaps between allocations) and **internal fragmentation** (waste within allocations)

### PagedAttention: The Solution

vLLM's PagedAttention treats the KV cache like **virtual memory in an operating system**:

| Concept | OS Analogy | PagedAttention Implementation |
|---------|------------|-------------------------------|
| Page | Fixed-size memory unit | KV Block (e.g., 16 tokens) |
| Page Table | Virtual→Physical mapping | Sequence→Block mapping |
| On-demand allocation | malloc/mmap | Allocate blocks as tokens generate |
| Fragmentation | Memory waste | Near-zero with block-based allocation |

**Key insight**: Instead of reserving one large contiguous slab per sequence, PagedAttention breaks the cache into **fixed-size KV blocks** that can be allocated on demand and placed anywhere in GPU memory.

### Memory Per Token Formula

Each new token introduces key and value vectors in each transformer layer:

```
Memory per token ≈ 2 × (bytes per value) × (num_layers) × (num_heads) × (head_dim)
```

**Example calculation** (Phi-3-mini, bf16):
- 2 bytes per value (bf16)
- 32 layers
- 32 attention heads  
- 96 head dimension
- **Result**: `2 × 2 × 32 × 32 × 96 = 393,216 bytes (~384 KB) per token`

For a 4k context: `4000 × 384 KB ≈ 1.5 GB` just for KV cache!

**Reference**: Hugging Face estimates that storing the KV cache for 10,000 tokens on LLaMA-2 7B consumes ~5 GB—roughly one-third the size of the model weights.

---

## 2) Block-Based Allocation Deep Dive

### How Blocks Work

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU KV Cache Memory                       │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────┤
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5  │
│ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok   │
└─────────┴─────────┴─────────┴─────────┴─────────┴──────────┘
         ↑           ↑         ↑
         │           │         │
    Seq A: tok 0-15  │    Seq B: tok 0-15
              Seq A: tok 16-31
```

**Block allocation rules**:
- Each block holds KV for exactly `block_size` tokens (default: 16)
- Sequences maintain a mapping: logical position → physical block
- Blocks allocated **on demand** as tokens generate
- At most `block_size - 1` token slots wasted per sequence (final block only)

### Fragmentation Comparison

| Allocation Strategy | Memory Utilization | Fragmentation |
|--------------------|--------------------|---------------|
| Traditional (contiguous) | 20-38% | High (external + internal) |
| PagedAttention (blocks) | ~100% | Near-zero (only final block) |

**For N tokens with block size B**: exactly `ceil(N/B)` blocks allocated.

---

## 3) Prefix Caching (Automatic Prefix Reuse)

### The Problem: Redundant Computation

Many inference scenarios share common prefixes:
- Same system prompt across users
- Few-shot examples repeated
- RAG context templates

Without caching: **each request recomputes the entire prefix KV cache**.

### vLLM's Solution: Content-Addressed Blocks

vLLM implements **Automatic Prefix Caching** by globally indexing KV blocks:

```
Block Hash = hash(token_content + preceding_prefix_hash)
```

**How it works**:
1. First request computes prefix → blocks stored with content hash
2. Second request with same prefix → hash matches → reuse existing blocks
3. Reference count tracks sharing → blocks freed when unused

```
Request 1: [System Prompt (2k tokens)] + [User Query A]
                    ↓
           Compute & store blocks 0-124 (2k/16)
                    ↓
Request 2: [System Prompt (2k tokens)] + [User Query B]
                    ↓
           Hash match! Reuse blocks 0-124, only compute new query
```

### Expected Benefits

| Metric | Without Prefix Cache | With Prefix Cache | Improvement |
|--------|---------------------|-------------------|-------------|
| Memory (10 requests, 2k shared) | 10× prefix storage | 1× prefix storage | ~90% savings |
| TTFT (repeat prompts) | Full prompt compute | Skip prefix compute | 70-80% faster |
| Throughput | Limited by redundancy | Higher effective batch | 2-4× increase |

---

## 4) Block Size Trade-offs

### The Tuning Knob

Block size is the "page size" of PagedAttention—a critical tuning parameter:

| Block Size | Internal Fragmentation | Overhead | Prefix Sharing |
|------------|----------------------|----------|----------------|
| Small (16) | Low (≤15 wasted) | Higher (more blocks) | Fine-grained |
| Large (64) | High (≤63 wasted) | Lower (fewer blocks) | Coarse-grained |

### Detailed Trade-off Analysis

**Smaller blocks (16 tokens)**:
- ✅ Minimal waste per sequence
- ✅ Better prefix sharing granularity
- ✅ Higher memory utilization under pressure
- ❌ More kernel launches per generation step
- ❌ Higher pointer chasing overhead

**Larger blocks (64 tokens)**:
- ✅ Fewer memory operations per token
- ✅ More efficient GPU kernel execution
- ❌ Up to 63 wasted token slots per sequence
- ❌ Coarser prefix matching (lower cache hit rate)

### vLLM Author Findings

The vLLM paper reports:
- PagedAttention incurs ~20-26% overhead in attention kernel due to indirections
- **But**: end-to-end throughput more than doubled due to better memory usage
- **Optimal**: ~16 tokens/block provides excellent trade-off in general

---

## 5) Key Metrics to Capture

### From nvidia-smi
- `memory.used` – total GPU memory consumption
- `memory.total` – available GPU memory
- Memory growth rate during generation

### From vLLM /metrics endpoint
```
vllm_cache_config_info          # Cache configuration
vllm_num_preemptions_total      # Preemptions due to memory pressure
vllm_gpu_cache_usage_perc       # Cache utilization percentage
vllm_cpu_cache_usage_perc       # CPU offload cache usage
```

### Derived Metrics
- **Blocks allocated**: `ceil(tokens / block_size)`
- **Memory per token**: `total_kv_memory / active_tokens`
- **Fragmentation ratio**: `(allocated - used) / allocated`
- **Prefix cache hit rate**: `reused_blocks / total_prefix_blocks`

---

## References & Reading (Off-Hours)

1. **Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (SOSP 2023)**
   - Sections 3-4: KV cache design, fragmentation issues, block-based solution
   - Why: Foundation paper for understanding vLLM internals

2. **Data Science Dojo, "Memory Is the Real Bottleneck: How Paged Attention Powers vLLM (Part 2)"**
   - Intuitive explanation of KV cache fragmentation and PagedAttention benefits
   - Block size trade-offs with examples

3. **Hugging Face Blog, "KV Caching Explained"**
   - Memory per token formula derivation
   - Practical memory calculations for common models

4. **vLLM Documentation: Automatic Prefix Caching**
   - Implementation details and configuration options
   - LRU eviction policy for cache management

