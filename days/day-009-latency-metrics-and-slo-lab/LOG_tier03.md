# Day 009 – Analysis, SLO & Policy (Microtasks 10-18)

> This tier is "how you think like a production inference engineer": percentiles are not vibes — they are the output of **queueing + workload mix + system policy**.

---

## Task 10: Latency Breakdown (TTFT vs E2E)

**Objective**: Measure Time-To-First-Token (TTFT) separately from total E2E time, especially under concurrency.

**Prerequisites**: A running vLLM server or streaming capability.

**Time**: ~20 min

### Create `scripts/measure_ttft.py`

```python
#!/usr/bin/env python3
"""Measure TTFT vs E2E latency with streaming."""
import time
import requests
import json
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000/v1"
MODEL = "meta-llama/Llama-2-7b-chat-hf"

def measure_single_request(prompt, request_id):
    """Measure TTFT and E2E for a single streaming request."""
    start = time.time()
    first_token_time = None
    tokens = 0
    
    try:
        response = requests.post(
            f"{BASE_URL}/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": True
            },
            stream=True,
            timeout=60
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != 'data: [DONE]':
                    if first_token_time is None:
                        first_token_time = time.time()
                    tokens += 1
        
        end = time.time()
        ttft = (first_token_time - start) * 1000 if first_token_time else None
        e2e = (end - start) * 1000
        
        return {
            'request_id': request_id,
            'ttft_ms': ttft,
            'e2e_ms': e2e,
            'tokens': tokens,
            'error': None
        }
    except Exception as e:
        return {
            'request_id': request_id,
            'ttft_ms': None,
            'e2e_ms': None,
            'tokens': 0,
            'error': str(e)
        }

def run_concurrency_test(concurrency, num_requests):
    """Run concurrent requests and measure latencies."""
    prompt = "Explain the concept of machine learning in simple terms."
    results = []
    
    print(f"\n=== Concurrency: {concurrency}, Requests: {num_requests} ===")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(measure_single_request, prompt, i)
            for i in range(num_requests)
        ]
        for future in futures:
            results.append(future.result())
    
    # Analyze
    ttfts = [r['ttft_ms'] for r in results if r['ttft_ms']]
    e2es = [r['e2e_ms'] for r in results if r['e2e_ms']]
    errors = sum(1 for r in results if r['error'])
    
    if ttfts:
        ttfts_sorted = sorted(ttfts)
        print(f"  TTFT p50: {statistics.median(ttfts):.1f}ms")
        print(f"  TTFT p95: {ttfts_sorted[int(len(ttfts)*0.95)]:.1f}ms")
        print(f"  TTFT p99: {ttfts_sorted[int(len(ttfts)*0.99)]:.1f}ms")
    
    if e2es:
        e2es_sorted = sorted(e2es)
        print(f"  E2E p50:  {statistics.median(e2es):.1f}ms")
        print(f"  E2E p95:  {e2es_sorted[int(len(e2es)*0.95)]:.1f}ms")
    
    print(f"  Errors: {errors}/{num_requests}")
    
    return results

if __name__ == "__main__":
    all_results = {}
    
    # Test different concurrency levels
    for conc in [1, 2, 4, 8]:
        all_results[conc] = run_concurrency_test(conc, 20)
    
    # Save results
    with open('logs/ttft_measurement.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n=== Results saved to logs/ttft_measurement.json ===")
```

### Commands

```bash
cd ~/day009

# Start vLLM server
export MODEL="meta-llama/Llama-2-7b-chat-hf"
tmux new -d -s vllm_server "vllm serve --model $MODEL --host 0.0.0.0 --port 8000"
sleep 30  # Wait for load

# Run TTFT measurement
python scripts/measure_ttft.py 2>&1 | tee logs/ttft_measurement.txt

# Stop server
tmux kill-session -t vllm_server
```

**Expected Artifacts**:
- `logs/ttft_measurement.json`
- `logs/ttft_measurement.txt`

**Success Criteria**:
- Concrete TTFT measurements at different concurrency levels
- Observe: p99 TTFT grows with concurrency (some requests wait in queue)

**Key Insight**: Document in `reports/SLO_analysis.md`:
> "In concurrency=5 test, p50 TTFT = X ms, p90 TTFT = Y ms (some requests waited ~Z ms in queue). 
> Once started, token throughput remained ~N tokens/sec per request."

---

## Task 11: Compute Theoretical KV Memory per Token

**Objective**: Calculate how much GPU memory one token's KV cache should consume, using actual model config.

**Prerequisites**: Know model config (layers, heads, head_dim, dtype).

**Time**: ~15 min

### Create `scripts/calc_kv_mem.py`

```python
#!/usr/bin/env python3
"""Calculate theoretical KV cache memory per token."""
import os
from transformers import AutoConfig

MODEL = os.environ.get("MODEL", "meta-llama/Llama-2-7b-chat-hf")

print(f"=== KV Memory Calculator ===")
print(f"Model: {MODEL}")

# Load config
conf = AutoConfig.from_pretrained(MODEL)

# Extract parameters
L = conf.num_hidden_layers
H = conf.num_attention_heads
d = conf.hidden_size // conf.num_attention_heads  # head dimension
bytes_per_val = 2  # bf16/fp16

print(f"\nModel Architecture:")
print(f"  Layers (L): {L}")
print(f"  Attention heads (H): {H}")
print(f"  Head dimension (d): {d}")
print(f"  Hidden size: {conf.hidden_size}")

# Calculate per-token memory
# KV cache stores Key and Value for each layer
# Memory = 2 (K+V) × L × H × d × bytes_per_val
per_token_bytes = 2 * L * H * d * bytes_per_val
per_token_kb = per_token_bytes / 1024

print(f"\nKV Memory per Token:")
print(f"  Formula: 2 × L × H × d × bytes = 2 × {L} × {H} × {d} × {bytes_per_val}")
print(f"  Result: {per_token_bytes:,} bytes ({per_token_kb:.2f} KB)")

# Project for various context lengths
for tokens in [1000, 4000, 10000, 50000]:
    total_mb = (tokens * per_token_bytes) / 1024 / 1024
    total_gb = total_mb / 1024
    print(f"  {tokens:,} tokens: {total_mb:.1f} MB ({total_gb:.2f} GB)")

# Compare with vLLM block allocation
BLOCK_SIZE = 16
print(f"\nBlock-based Storage (block_size={BLOCK_SIZE}):")
per_block_bytes = BLOCK_SIZE * per_token_bytes
print(f"  Per block: {per_block_bytes:,} bytes ({per_block_bytes/1024:.2f} KB)")

# GPU memory budget calculation
GPU_MEM_GB = 24  # Example: RTX 4090
MODEL_MEM_GB = 14  # Approximate model weights
KV_BUDGET_GB = (GPU_MEM_GB - MODEL_MEM_GB) * 0.9  # 90% util

max_tokens = int((KV_BUDGET_GB * 1024 * 1024 * 1024) / per_token_bytes)
max_blocks = max_tokens // BLOCK_SIZE

print(f"\nCapacity Estimate ({GPU_MEM_GB}GB GPU, {MODEL_MEM_GB}GB model):")
print(f"  KV budget: {KV_BUDGET_GB:.1f} GB")
print(f"  Max tokens: {max_tokens:,}")
print(f"  Max blocks: {max_blocks:,}")
```

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"
python scripts/calc_kv_mem.py 2>&1 | tee logs/kv_memory_calc.txt
```

**Expected Artifacts**:
- `logs/kv_memory_calc.txt`
- Include calculations in `reports/memory_analysis.md`

**Success Criteria**:
- For Llama-2-7B: ~16 KB per token
- Can predict: 50k tokens ≈ 800 MB KV cache

---

## Task 12: Source Dive – PagedAttention & KV Cache Code

**Objective**: Read key portions of vLLM source to understand KV caching implementation.

**Prerequisites**: Basic familiarity with Python/C++ code.

**Time**: ~30 min

### Focus Areas

1. **KV Cache Manager** – `vllm/core/block.py` or `vllm/v1/worker/gpu/block_table.py`
   - Look for: `BlockTables` class, `num_blocks`, block allocation/free

2. **Prefix Caching** – Search for "prefix" or hash computation
   - Look for: Hash function using `parent_hash + block_tokens`
   - Verify: "Only full blocks are cached"

3. **Attention Kernel** – `vllm/engine/paged_attention.cu` or docs
   - Note: Block size as template parameter (why 64 isn't supported on GPU)

### Commands

```bash
cd ~/day009

# Find vLLM installation
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
echo "vLLM path: $VLLM_PATH"

# Search for block-related code
grep -r "block_size" $VLLM_PATH --include="*.py" | head -20
grep -r "prefix" $VLLM_PATH --include="*.py" | head -20
grep -r "BlockTable" $VLLM_PATH --include="*.py" | head -10

# Look at specific files
cat $VLLM_PATH/core/block.py | head -100 || echo "File not found"
```

### Document Findings

Create note in `reports/block_tuning_report.md`:

```markdown
## Source Code Findings

### Block Allocation
- BlockTables class preallocates fixed number of blocks at init
- Blocks are managed in a pool, allocated/freed per sequence
- Block size is a compile-time constant for CUDA kernels

### Prefix Caching
- Uses SHA256 hash of (parent_prefix_hash + block_tokens)
- Only full blocks are cached (partial blocks excluded)
- Hash collision = cache hit → reuse KV block

### Block Size Constraints
- GPU attention kernels compiled with specific block sizes
- Block sizes >32 not supported on CUDA without kernel recompile
- Smaller blocks = more kernel iterations per forward pass
```

**Success Criteria**:
- Connect code to experimental results
- Understand why block_size=64 failed (kernel not compiled)
- Confirm partial blocks aren't cached

---

## Task 13: Verify Prefix Reuse Conditions

**Objective**: Validate exact condition under which prefix caching reuses a block vs not.

**Prerequisites**: Task 12 understanding of prefix hashing.

**Time**: ~20 min

### Experiment: Single Token Difference

```python
#!/usr/bin/env python3
"""Test prefix reuse with single token difference."""
import time
import requests

BASE_URL = "http://localhost:8000/v1"
MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Two prompts that differ by ONE token in middle of block
# Assuming block_size=16, tokens 0-15 are in first block
PROMPT_A = "I love to eat ice cream on sunny days because it is delicious and refreshing"
PROMPT_B = "I hate to eat ice cream on sunny days because it is delicious and refreshing"
# "love" vs "hate" at position ~2-3 (within first block)

def measure_ttft(prompt, label):
    start = time.time()
    first_token = None
    
    response = requests.post(
        f"{BASE_URL}/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 50, "stream": True},
        stream=True
    )
    
    for line in response.iter_lines():
        if line and first_token is None:
            first_token = time.time()
            break
    
    ttft = (first_token - start) * 1000 if first_token else None
    print(f"{label}: TTFT = {ttft:.1f}ms")
    return ttft

# With prefix caching enabled:
print("=== Test: Single Token Difference ===")
print("Prompt A first (populates cache):")
ttft_a = measure_ttft(PROMPT_A, "Prompt A")

print("\nPrompt B (differs by 'love' -> 'hate'):")
ttft_b = measure_ttft(PROMPT_B, "Prompt B")

print(f"\nAnalysis:")
if abs(ttft_a - ttft_b) < ttft_a * 0.2:
    print("Both similar TTFT -> Block containing diff token was NOT reused")
else:
    print("Different TTFT -> Partial reuse occurred")
```

### Expected Outcome

Document in `reports/prefix_caching_report.md`:

```markdown
## Prefix Reuse Conditions

### Experiment: Single Token Difference
- Prompt A: "I love to eat ice cream..."
- Prompt B: "I hate to eat ice cream..."
- Difference: "love" vs "hate" at position 2

### Results
- Prompt A TTFT: X ms (cold cache)
- Prompt B TTFT: Y ms (expected similar to A)

### Analysis
The second prompt's TTFT was almost as high as the first, confirming that:
1. The divergent block and everything after it was recomputed
2. Only blocks BEFORE the divergent token were reused
3. This aligns with vLLM's design: cache keys include block's token content

### Implication
- Smaller block sizes = more fine-grained prefix sharing
- But also more overhead per token
- Trade-off: block size 16 is optimal for most cases
```

---

## Task 14: Compare Theoretical vs Measured Memory Usage

**Objective**: Reconcile theory with reality for KV memory consumption.

**Prerequisites**: Data from Tasks 8, 9, 11.

**Time**: ~20 min

### Analysis Template

Create `reports/memory_analysis.md`:

```markdown
# Memory Analysis: Theoretical vs Measured

## Theoretical Calculations (Task 11)

| Parameter | Value |
|-----------|-------|
| Model | Llama-2-7B |
| Layers | 32 |
| Heads | 32 |
| Head dim | 128 |
| Precision | bf16 (2 bytes) |
| Per-token memory | 16 KB |

## Engine Allocation (Task 9)

| Metric | Observed |
|--------|----------|
| GPU blocks | [FROM LOGS] |
| CPU blocks | [FROM LOGS] |
| Block size | 16 |
| GPU token capacity | [GPU_BLOCKS × 16] |

## Memory Budget Calculation

```
GPU: 24 GB
Model weights: ~14 GB
Available for KV: ~10 GB × 0.9 = 9 GB

Expected capacity: 9 GB / 16 KB = ~562,500 tokens
Observed capacity: [FROM LOGS] tokens
```

## Discrepancy Analysis

**Why might observed < expected?**

1. **max_model_len constraint**: Engine reserves space for worst-case sequences
   - If max_seq_len=2048, engine plans for concurrent sequences of that length
   
2. **CPU swap allocation**: Some blocks reserved for CPU offload
   
3. **CUDA overhead**: Graphs, scratch space, alignment padding

4. **Fragmentation buffer**: Engine may not allocate to 100% to avoid OOM

## Memory Trace Observation (Task 8)

- Memory at startup: X MiB
- Memory during generation: ~flat (pre-allocated)
- Conclusion: vLLM allocates KV pool upfront, not per-token

## Key Insight

> vLLM uses ~2x theoretical memory in practice due to pre-allocation, 
> fragmentation headroom, and max_seq_len planning. This ensures 
> predictable performance and no runtime allocation delays.
```

---

## Task 15: Latency & Throughput under Concurrency (Queuing Effects)

**Objective**: Interpret how p50/p95/p99 latencies scale with concurrency, using queueing theory intuition.

**Prerequisites**: Task 10 data (TTFT at different concurrency levels).

**Time**: ~20 min

### Analysis

Using data from Task 10, create analysis in `reports/SLO_analysis.md`:

```markdown
# Concurrency Analysis

## Observed Latency Scaling

| Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
|-------------|----------|----------|----------|------------|
| 1 | X ms | X ms | X ms | X tok/s |
| 2 | X ms | X ms | X ms | X tok/s |
| 4 | X ms | X ms | X ms | X tok/s |
| 8 | X ms | X ms | X ms | X tok/s |

## Queueing Theory Interpretation

**Light load (concurrency=1)**:
- p99 ≈ p50 (low variability)
- No queueing delay
- TTFT = pure compute time

**Medium load (concurrency=4)**:
- p95/p99 start growing
- Some requests wait for earlier ones
- Scheduler interleaves but can't parallelize everything

**Heavy load (concurrency=8)**:
- p99 >> p50 (high tail latency)
- Queueing dominates
- TTFT includes ~1 scheduling round delay (~hundreds ms)

## Capacity Threshold

From data: **concurrency=4** appears to be the knee where p99 starts degrading significantly.

Beyond this:
- Requests queue longer
- p99 TTFT exceeds acceptable threshold
- Error rate may increase
```

---

## Task 16: Define SLO and Workload Envelope

**Objective**: Formulate concrete Service Level Objectives based on Day 009 findings.

**Prerequisites**: Results from Tasks 3-5 and 15.

**Time**: ~15 min

### Create SLO Table in `reports/SLO_analysis.md`

```markdown
# Service Level Objectives (SLO)

## Workload Envelope

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max input tokens | 100 | Typical chat prompt |
| Max output tokens | 50 | Reasonable response |
| Max concurrency | 4 | Before p99 degrades |
| Sampling | temperature=0.7 | Standard generation |

## SLO Targets

| Metric | Target (p99) | Conditions |
|--------|--------------|------------|
| TTFT | ≤ 1.5 s | prompt ≤100 tok, ≤4 concurrent |
| E2E Latency | ≤ 3.0 s | 50 tok output |
| Stream Rate | ≥ 20 tok/s | After first token |
| Error Rate | < 0.1% | Under normal load |

## Justification

Based on Day 009 experiments:
- Baseline TTFT p95: X ms at concurrency=1
- Prefix caching reduced TTFT by Y% for repeated prompts
- At concurrency=4, TTFT p99 was Z ms (within target)
- At concurrency=8, TTFT p99 exceeded target

## Caveats

1. **Long prompts** (>100 tokens): TTFT will be higher unless prefix cached
2. **Cold cache**: First request with new prefix pays full compute cost
3. **Burst traffic**: May need queue buffering for short bursts
```

---

## Task 17: Design Admission Control & Policy

**Objective**: Translate SLO into concrete admission control policy for production.

**Prerequisites**: SLO from Task 16.

**Time**: ~15 min

### Policy Document in `reports/SLO_analysis.md`

```markdown
# Admission Control Policy

## Concurrency Control

```
MAX_CONCURRENT_REQUESTS = 4
QUEUE_MAX_LENGTH = 10
QUEUE_TIMEOUT_MS = 5000
```

**Rules**:
1. If active_requests < MAX_CONCURRENT: admit immediately
2. If active_requests >= MAX_CONCURRENT AND queue_length < QUEUE_MAX: enqueue
3. If queue_length >= QUEUE_MAX: reject with 503 (Service Unavailable)
4. If queue_wait > QUEUE_TIMEOUT: reject with 408 (Timeout)

## Scheduling Policy

**FIFO** (First-In-First-Out) with considerations:
- All requests treated equally
- Future enhancement: priority queue for short prompts

## Prefix Caching Policy

**MUST enable** (`--enable-prefix-caching`):
- Day 009 showed X% TTFT improvement
- Without caching, long prompts violate SLO
- Common system prompts cached and reused

**Cache warming** (optional):
- On server start, send dummy request with common system prompt
- Prepopulates cache before user traffic

## Request Validation

Reject requests that exceed envelope:
- Input tokens > 200: reject (would violate TTFT SLO)
- Output tokens > 500: reject or cap

## Monitoring & Alerts

| Metric | Warning | Critical |
|--------|---------|----------|
| TTFT p99 | > 1.2s | > 1.5s |
| Queue length | > 5 | > 10 |
| Error rate | > 0.05% | > 0.1% |
| GPU memory | > 90% | > 95% |

## Connection to Day 009 Data

| Decision | Evidence |
|----------|----------|
| Max concurrency=4 | At 5, p99 TTFT exceeded 1.5s target |
| Prefix caching required | 70% TTFT reduction observed |
| Queue length=10 | Allows short bursts without rejection |
```

---

## Task 18: Wrap-up – Persist & Archive Results

**Objective**: Save all Day 009 artifacts for future reference.

**Prerequisites**: All tasks completed (or as much as needed).

**Time**: ~10 min

### Commands

```bash
cd ~/day009

# Ensure all reports are complete
ls -la reports/

# Create final README
cat > reports/readme_day009.md << 'EOF'
# Day 009 Summary: KV Cache Deep Dive

## Completed Experiments
- [x] Baseline profiling (Task 3)
- [x] Prefix caching ON vs OFF (Tasks 4-5)
- [x] Block size sweep 16/32/64 (Tasks 6-7)
- [x] Memory profiling (Tasks 8-9)
- [x] Latency breakdown under concurrency (Task 10)
- [x] Theoretical vs measured analysis (Tasks 11, 14)
- [x] SLO definition (Task 16)
- [x] Admission policy (Task 17)

## Key Learnings
1. **PagedAttention**: Treats KV cache like virtual memory with fixed-size blocks
2. **Prefix Caching**: Yields ~70% TTFT improvement for shared prompts
3. **Block Size**: 16-32 optimal on GPU; 64 not supported
4. **Memory**: vLLM pre-allocates KV pool; ~16KB per token for 7B model
5. **Concurrency**: p99 TTFT degrades beyond 4 concurrent requests

## SLO Derived
- TTFT p99 ≤ 1.5s
- Max concurrency: 4
- Prefix caching: required

## Files
- `logs/`: Raw experiment outputs
- `reports/`: Analysis documents
- `scripts/`: Automation scripts
EOF

# Archive everything
cd ~
tar -czvf day009_results.tar.gz day009/
mv day009_results.tar.gz day009/archive/

echo "=== Archive created: day009/archive/day009_results.tar.gz ==="
ls -lh day009/archive/
```

**Expected Artifacts**:
- `day009/archive/day009_results.tar.gz`
- `reports/readme_day009.md`

**Success Criteria**:
- All key data preserved
- Reports are coherent and reproducible
- Environment can be safely shut down

---

## Fast Path Summary (~2 hours)

Focus on: **Tasks 1, 2, 3, 4, 5, 6, 16, 17, 18**

| Phase | Tasks | Time |
|-------|-------|------|
| Setup | 1, 2 | 15 min |
| Core experiments | 3, 4, 5, 6 | 65 min |
| SLO & Policy | 16, 17 | 30 min |
| Archive | 18 | 10 min |

**Outcome**: Baseline, prefix caching comparison, block size check, and actionable SLO + admission policy.

---

## Stretch Path Summary (+4-6 hours)

Add: **Tasks 7, 8, 9, 10, 11, 12, 13, 14, 15**

**Outcome**: Deep understanding of vLLM internals, theoretical validation, source code familiarity, and production-ready analysis.

---

## Consulting-Quality Latency Table

Every benchmark you publish should include:

| Field | Example |
|-------|---------|
| Model | Llama-2-7B-chat |
| Precision | bf16 |
| max_model_len | 2048 |
| max_tokens | 50 |
| Prompt mix | Short (50 tok avg) |
| Concurrency | 4 |
| Request count | 200 |
| TTFT p50/p95/p99 | 120/280/450 ms |
| E2E p50/p95/p99 | 800/1200/1800 ms |
| Error rate | 0% |
| GPU util | 85% |

This is enough to **reproduce, compare, and justify an SLO + admission policy**.
