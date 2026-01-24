# Day 009 – KV Cache Deep Dive: Anatomy, Prefix Reuse & Block Tuning

> **Goal**: Master vLLM's PagedAttention KV cache internals—understand memory allocation patterns, measure prefix reuse benefits, tune block size, and translate findings into actionable SLOs.

---

## Snapshot (Today's Focus)

- **Day 9/100**: Runtime layer deep-dive after establishing measurement discipline
- **Layer**: Inference Runtimes
- **Theme**: KV cache design, partitioning & cross-request reuse (Theme #3)
- **Outcome**: Concrete SLO + admission policy backed by measured data

> **Audit addendum:** Extend this day to log **$ per 1M tokens** and **per-tenant latency/cost** from the same runs. Capture p95/p99 for mixed workloads, and note any guardrail or prefix-cache effects on quality/behavior. Use the measurement-first template for new additions.

---

## A) Assumptions

- **Hardware**: Single-GPU RunPod VM (Ubuntu + NVIDIA drivers) with ≥24 GB VRAM
- **Software**: Python 3.x, vLLM installed (or will be in Task 1)
- **Model**: HuggingFace transformer (e.g., `meta-llama/Llama-2-7b-chat-hf`), max context ~2048 tokens
- **vLLM Version**: Engine V1 (`VLLM_USE_V1=1`), automatic prefix caching ON by default
- **Environment**: Ephemeral (wiped on shutdown) — all outputs saved under `day009/` and archived
- **Tools**: `tmux` or `screen` for long-running processes, basic Linux CLI familiarity

---

## B) Folder Layout

All Day 009 artifacts live under `day009/` for easy packaging:

```
day009/
├── scripts/                          # Automation and experiment scripts
│   ├── setup_env.sh                  # Installs vLLM and dependencies
│   ├── sanity_check.py               # Verifies model loads and generates
│   ├── run_baseline.sh               # Baseline profiling (prefix cache OFF)
│   ├── run_prefix_off.sh             # Shared prefix scenario, caching OFF
│   ├── run_prefix_on.sh              # Shared prefix scenario, caching ON
│   ├── run_block_sweep.sh            # Block size sweep (16, 32, 64)
│   ├── sample_gpu_mem.sh             # High-frequency GPU memory logging
│   ├── measure_ttft.py               # TTFT vs E2E latency measurement
│   ├── calc_kv_mem.py                # Theoretical KV memory calculator
│   └── profile_nsys.sh               # Nsight Systems capture utility
│
├── data/
│   └── prompts_shared_prefix.txt     # Example prompts with common prefix
│
├── logs/                             # Raw logs and output files
│   ├── setup.txt                     # Environment setup log
│   ├── sanity_check.log              # Model load verification
│   ├── baseline.json                 # Baseline metrics
│   ├── prefix_off.json               # Prefix caching OFF metrics
│   ├── prefix_on.json                # Prefix caching ON metrics
│   ├── block_sweep_16.json           # Block size=16 metrics
│   ├── block_sweep_32.json           # Block size=32 metrics
│   ├── block_sweep_64.json           # Block size=64 metrics (or error)
│   ├── gpu_mem_usage.csv             # High-resolution memory trace
│   ├── ttft_measurement.txt          # TTFT/E2E timing data
│   └── nsys_profile.qdrep            # Nsight Systems capture
│
├── reports/                          # Markdown analysis reports
│   ├── baseline_report.md            # Baseline profiling summary
│   ├── prefix_caching_report.md      # Prefix ON vs OFF comparison
│   ├── block_tuning_report.md        # Block size sweep analysis
│   ├── memory_analysis.md            # Theoretical vs measured KV memory
│   ├── SLO_analysis.md               # SLO table and admission policy
│   └── readme_day009.md              # Top-level Day 009 summary
│
└── archive/
    └── day009_results.tar.gz         # Compressed results for download
```

---

## C) Microtask Overview

| Task | Title | Time | Path |
|------|-------|------|------|
| 1 | Environment Setup & Package Install | 10 min | Fast |
| 2 | Sanity Check Model Load & Generation | 5 min | Fast |
| 3 | Baseline Profiling (No Prefix, Default Block) | 15 min | Fast |
| 4 | Prefix Caching OFF – Shared Prefix Scenario | 15 min | Fast |
| 5 | Prefix Caching ON – Shared Prefix Scenario | 15 min | Fast |
| 6 | Block Size Sweep – 16 vs 32 | 20 min | Fast |
| 7 | Block Size Sweep – 64 (Exploratory) | 10 min | Stretch |
| 8 | High-Resolution GPU Memory Logging | 15 min | Stretch |
| 9 | Inspect KV Block Allocation Logs | 10 min | Stretch |
| 10 | Latency Breakdown (TTFT vs E2E) | 20 min | Stretch |
| 11 | Compute Theoretical KV Memory per Token | 15 min | Stretch |
| 12 | Source Dive – PagedAttention & KV Cache Code | 30 min | Stretch |
| 13 | Verify Prefix Reuse Conditions | 20 min | Stretch |
| 14 | Compare Theoretical vs Measured Memory | 20 min | Stretch |
| 15 | Latency & Throughput under Concurrency | 20 min | Stretch |
| 16 | Define SLO and Workload Envelope | 15 min | Fast |
| 17 | Design Admission Control & Policy | 15 min | Fast |
| 18 | Wrap-up – Persist & Archive Results | 10 min | Fast |

**Fast Path** (~2 hours): Tasks 1, 2, 3, 4, 5, 6, 16, 17, 18  
**Stretch Path** (+4-6 hours): All tasks including 7-15

---

## D) KV Cache Fundamentals

### The Memory Bottleneck Problem

In LLM inference, the **KV cache** stores computed key and value vectors from previous tokens. Traditional implementations suffer from severe memory inefficiency:

- **Only 20–38% of allocated KV memory is actually used** in conventional systems
- The rest is reserved but empty (pre-allocated for max sequence length)
- Creates **external fragmentation** (gaps) and **internal fragmentation** (waste within allocations)

### PagedAttention: The Solution

vLLM's PagedAttention treats the KV cache like **virtual memory in an OS**:

| Concept | OS Analogy | PagedAttention Implementation |
|---------|------------|-------------------------------|
| Page | Fixed-size memory unit | KV Block (e.g., 16 tokens) |
| Page Table | Virtual→Physical mapping | Sequence→Block mapping |
| On-demand allocation | malloc/mmap | Allocate blocks as tokens generate |

**Key insight**: Instead of reserving one large contiguous slab per sequence, PagedAttention breaks the cache into **fixed-size KV blocks** that can be allocated on demand and placed anywhere in GPU memory.

### Memory Per Token Formula

```
Memory per token = 2 × (bytes per value) × (num_layers) × (num_heads) × (head_dim)
```

**Example** (Llama-2-7B, bf16):
- 2 bytes/value × 32 layers × 32 heads × 128 head_dim = **16 KB per token**
- For 50k tokens: 50,000 × 16 KB ≈ **800 MB** just for KV cache

### Block Allocation

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU KV Cache Memory                       │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────┤
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5  │
│ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok   │
└─────────┴─────────┴─────────┴─────────┴─────────┴──────────┘
```

- Each block holds KV for exactly `block_size` tokens (default: 16 on GPU, max 32)
- Blocks allocated **on demand** as tokens generate
- At most `block_size - 1` token slots wasted per sequence (final block only)
- For N tokens: exactly `ceil(N/block_size)` blocks allocated

---

## E) Prefix Caching Mechanics

### Content-Addressed Blocks

vLLM implements **Automatic Prefix Caching** by globally indexing KV blocks:

```
Block Hash = hash(token_content + preceding_prefix_hash)
```

**Critical constraint**: Only **full blocks** are cached. If a prefix ends mid-block, that partial block won't be reused.

### How It Works

1. First request computes prefix → blocks stored with content hash
2. Second request with same prefix → hash matches → reuse existing blocks
3. Reference count tracks sharing → blocks freed via LRU when unused

### Expected Benefits

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| Memory (10 req, 2k shared) | 10× prefix | 1× prefix | ~90% savings |
| TTFT (repeat prompts) | Full compute | Skip prefix | 70-80% faster |
| Throughput | Redundant work | Batching gains | 2-4× increase |

---

## F) Block Size Trade-offs

| Block Size | Fragmentation | Overhead | Prefix Sharing | GPU Support |
|------------|---------------|----------|----------------|-------------|
| 16 | Low (≤15 waste) | Higher | Fine-grained | ✅ Yes |
| 32 | Medium (≤31 waste) | Medium | Medium | ✅ Yes (max on GPU) |
| 64 | High (≤63 waste) | Lower | Coarse | ❌ CPU only |

**vLLM defaults**: Block size 16 on GPU provides optimal balance of memory efficiency vs kernel overhead.

---

## G) Reading List (Off-Hours)

1. **vLLM Design Doc – PagedAttention**
   - https://docs.vllm.ai/en/stable/design/paged_attention/
   - Understand blocks, warps, and attention kernel design
   - Key: vLLM organizes KV cache into fixed-size blocks (16/32 tokens)

2. **vLLM Design Doc – Automatic Prefix Caching**
   - https://docs.vllm.ai/en/stable/design/prefix_caching/
   - How prefix caching works with block hashes
   - Critical: "Only full blocks are cached"

3. **Inside vLLM: Anatomy of a High-Throughput LLM Inference System** (Aleksa Gordić)
   - https://www.aleksagordic.com/blog/vllm
   - Architecture overview, KV cache manager, scheduler
   - Confirms default block size (16) and CUDA graphs usage

4. **GitHub Issues on vLLM**
   - https://github.com/vllm-project/vllm/issues
   - Why prefix sharing fails if single token differs
   - GPU blocks vs max_seq_len relationship

5. **vLLM Source Code** (targeted reading)
   - https://github.com/vllm-project/vllm
   - `block_table.py`: Block allocation logic
   - `prefix_caching.py`: Hash computation for prefix reuse
