# Day 009 – KV Cache Lab (Hands-On)

> **This tier is purely procedural**: run the experiments, generate artifacts, and keep the report honest.

---

## Tier 1 – Must Do: KV Cache Anatomy & Memory Profiling

**Title**: KV Cache Baseline Profiling  
**Goal**: Understand how vLLM's PagedAttention manages KV cache and measure baseline memory patterns  
**Time Budget**: ~45-60 min  
**Outcome**: Quantified memory-per-token, block allocation patterns, and fragmentation metrics

---

### Step 1: Start vLLM with Debug Metrics

```bash
# From day-009 folder
MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

vllm serve "$MODEL" \
  --port $PORT \
  --disable-log-requests \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.85
```

Verify server is up:
```bash
curl -s http://localhost:$PORT/health
```

### Step 2: Capture Baseline GPU Memory

Start memory monitoring in a separate terminal:
```bash
# Log GPU memory every 100ms during experiment
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,nounits \
  -lms 100 > gpu_memory_baseline.csv &
GPU_MONITOR_PID=$!
```

### Step 3: Run Single Long-Context Request (4k tokens)

Create the test script `kv_baseline_probe.py`:

```python
#!/usr/bin/env python3
"""Single request to measure KV cache memory growth."""
import time
import requests
import json

BASE_URL = "http://localhost:8000/v1"
MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Long prompt to fill context
LONG_PROMPT = """You are a detailed technical writer. Write an extremely comprehensive 
guide about distributed systems, covering consensus algorithms, replication strategies,
partition tolerance, CAP theorem, eventual consistency, vector clocks, Paxos, Raft,
and real-world implementations. Include code examples, diagrams descriptions, and
performance considerations. Be as thorough as possible."""

def run_long_generation():
    start = time.time()
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": 3500,  # Target ~4k total context
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True}
        },
        stream=True
    )
    
    tokens_generated = 0
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: ') and line != 'data: [DONE]':
                try:
                    data = json.loads(line[6:])
                    if data.get('choices', [{}])[0].get('delta', {}).get('content'):
                        tokens_generated += 1
                    if 'usage' in data:
                        print(f"\nUsage: {data['usage']}")
                except:
                    pass
    
    elapsed = time.time() - start
    print(f"Generated ~{tokens_generated} tokens in {elapsed:.2f}s")
    print(f"Tokens/sec: {tokens_generated/elapsed:.2f}")

if __name__ == "__main__":
    run_long_generation()
```

Run it:
```bash
python3 kv_baseline_probe.py
```

### Step 4: Scrape vLLM KV Cache Metrics

```bash
# Capture cache metrics
curl -s http://localhost:8000/metrics | grep -E "vllm_(cache|gpu|num)" > vllm_metrics_baseline.txt

# Key metrics to extract:
cat vllm_metrics_baseline.txt | grep -E "(cache_config|gpu_cache_usage|num_preemptions)"
```

### Step 5: Calculate Memory Per Token

Create `analyze_kv_memory.py`:

```python
#!/usr/bin/env python3
"""Analyze KV cache memory from experiment data."""
import pandas as pd

# Model config for Phi-3-mini
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 96
BYTES_PER_VALUE = 2  # bf16

# Theoretical memory per token
theoretical_per_token = 2 * BYTES_PER_VALUE * NUM_LAYERS * NUM_HEADS * HEAD_DIM
print(f"Theoretical memory per token: {theoretical_per_token:,} bytes ({theoretical_per_token/1024:.2f} KB)")

# For 4k tokens
tokens = 4000
total_kv_memory = tokens * theoretical_per_token
print(f"Expected KV cache for {tokens} tokens: {total_kv_memory/1024/1024/1024:.2f} GB")

# Block calculations (assuming block_size=16)
BLOCK_SIZE = 16
blocks_needed = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
wasted_slots = (blocks_needed * BLOCK_SIZE) - tokens
fragmentation_pct = (wasted_slots / (blocks_needed * BLOCK_SIZE)) * 100

print(f"\nBlock allocation (block_size={BLOCK_SIZE}):")
print(f"  Blocks needed: {blocks_needed}")
print(f"  Wasted slots: {wasted_slots}")
print(f"  Fragmentation: {fragmentation_pct:.2f}%")
```

Run analysis:
```bash
python3 analyze_kv_memory.py
```

### Step 6: Stop Monitoring & Document Results

```bash
kill $GPU_MONITOR_PID

# Analyze GPU memory growth
python3 -c "
import pandas as pd
df = pd.read_csv('gpu_memory_baseline.csv')
print('GPU Memory Stats:')
print(f'  Start: {df[\"memory.used [MiB]\"].iloc[0]} MiB')
print(f'  Peak:  {df[\"memory.used [MiB]\"].max()} MiB')
print(f'  End:   {df[\"memory.used [MiB]\"].iloc[-1]} MiB')
print(f'  Delta: {df[\"memory.used [MiB]\"].max() - df[\"memory.used [MiB]\"].iloc[0]} MiB')
"
```

### Expected Artifact: `baseline_memory.md`

Document your findings:
```markdown
# KV Cache Baseline Memory Profile

## Configuration
- Model: microsoft/Phi-3-mini-4k-instruct
- Precision: bf16
- Block size: 16 tokens
- GPU: [your GPU]

## Theoretical Calculations
- Memory per token: X KB
- Expected KV for 4k tokens: X GB
- Blocks needed: X
- Fragmentation: X%

## Measured Results
- GPU memory start: X MiB
- GPU memory peak: X MiB  
- GPU memory delta: X MiB
- Blocks allocated (from metrics): X
- Cache utilization: X%

## Observations
- [Memory growth pattern]
- [Fragmentation vs theoretical]
- [Any surprises]
```

---

## Tier 2 – Deepen: Prefix Cache Reuse Lab

**Title**: Measure Prefix Caching Impact  
**Goal**: Quantify memory savings and latency improvements from shared prefixes  
**Time Budget**: ~30-40 min  
**Outcome**: Side-by-side comparison of prefix caching ON vs OFF

---

### Step 1: Create Shared Prefix Test

Create `prefix_cache_test.py`:

```python
#!/usr/bin/env python3
"""Test prefix caching with shared system prompts."""
import time
import requests
import json
import statistics

BASE_URL = "http://localhost:8000/v1"
MODEL = "microsoft/Phi-3-mini-4k-instruct"

# 2k-token shared system prompt (simulate RAG context or few-shot)
SHARED_SYSTEM_PROMPT = """You are an expert technical consultant. Here is your knowledge base:

[CONTEXT BLOCK 1 - Distributed Systems]
Distributed systems are collections of independent computers that appear to users as a single
coherent system. Key challenges include: network partitions, clock synchronization, consensus,
replication, and fault tolerance. The CAP theorem states that a distributed system cannot
simultaneously provide Consistency, Availability, and Partition tolerance...
""" + "Additional context. " * 400  # Pad to ~2k tokens

# Different user queries
USER_QUERIES = [
    "Explain the Raft consensus algorithm in simple terms.",
    "What are the trade-offs between strong and eventual consistency?",
    "How does vector clocks help with conflict resolution?",
    "Describe a real-world implementation of Paxos.",
    "What is the difference between leader-based and leaderless replication?",
    "How do distributed databases handle network partitions?",
    "Explain the concept of quorum in distributed systems.",
    "What are the advantages of CRDTs over traditional locking?",
    "How does Apache Kafka achieve high throughput?",
    "Describe the consensus mechanism in etcd.",
]

def measure_request(system_prompt, user_query, request_num):
    """Measure TTFT and generation time for a single request."""
    start = time.time()
    ttft = None
    tokens = 0
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: ') and line != 'data: [DONE]':
                try:
                    data = json.loads(line[6:])
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content')
                    if content:
                        if ttft is None:
                            ttft = time.time() - start
                        tokens += 1
                except:
                    pass
    
    e2e = time.time() - start
    return {
        'request': request_num,
        'ttft_ms': ttft * 1000 if ttft else None,
        'e2e_ms': e2e * 1000,
        'tokens': tokens
    }

def run_test_batch(name, num_requests=10):
    """Run batch of requests and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    results = []
    for i, query in enumerate(USER_QUERIES[:num_requests]):
        result = measure_request(SHARED_SYSTEM_PROMPT, query, i+1)
        results.append(result)
        print(f"  Request {i+1}: TTFT={result['ttft_ms']:.1f}ms, E2E={result['e2e_ms']:.1f}ms")
    
    ttfts = [r['ttft_ms'] for r in results if r['ttft_ms']]
    print(f"\nSummary:")
    print(f"  TTFT p50: {statistics.median(ttfts):.1f}ms")
    print(f"  TTFT p95: {sorted(ttfts)[int(len(ttfts)*0.95)]:.1f}ms")
    print(f"  TTFT mean: {statistics.mean(ttfts):.1f}ms")
    
    return results

if __name__ == "__main__":
    results = run_test_batch("Prefix Cache Test", 10)
    
    # Save results
    with open('prefix_cache_results.json', 'w') as f:
        json.dump(results, f, indent=2)
```

### Step 2: Test WITHOUT Prefix Caching

```bash
# Restart vLLM WITHOUT prefix caching
pkill -f "vllm serve"
sleep 5

vllm serve "$MODEL" \
  --port 8000 \
  --disable-log-requests \
  --gpu-memory-utilization 0.85 &

sleep 30  # Wait for model load

# Capture baseline metrics
curl -s http://localhost:8000/metrics | grep vllm_gpu_cache > metrics_no_prefix.txt

# Run test
python3 prefix_cache_test.py
mv prefix_cache_results.json prefix_results_OFF.json

# Capture post-test metrics
curl -s http://localhost:8000/metrics | grep vllm_gpu_cache >> metrics_no_prefix.txt
```

### Step 3: Test WITH Prefix Caching

```bash
# Restart vLLM WITH prefix caching
pkill -f "vllm serve"
sleep 5

vllm serve "$MODEL" \
  --port 8000 \
  --disable-log-requests \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 &

sleep 30

# Capture baseline metrics
curl -s http://localhost:8000/metrics | grep vllm_gpu_cache > metrics_with_prefix.txt

# Run test
python3 prefix_cache_test.py
mv prefix_cache_results.json prefix_results_ON.json

# Capture post-test metrics
curl -s http://localhost:8000/metrics | grep vllm_gpu_cache >> metrics_with_prefix.txt
```

### Step 4: Compare Results

Create `compare_prefix_results.py`:

```python
#!/usr/bin/env python3
"""Compare prefix caching ON vs OFF results."""
import json
import statistics

def load_results(filename):
    with open(filename) as f:
        return json.load(f)

def analyze(results, label):
    ttfts = [r['ttft_ms'] for r in results if r['ttft_ms']]
    e2es = [r['e2e_ms'] for r in results]
    
    print(f"\n{label}:")
    print(f"  TTFT - p50: {statistics.median(ttfts):.1f}ms, mean: {statistics.mean(ttfts):.1f}ms")
    print(f"  E2E  - p50: {statistics.median(e2es):.1f}ms, mean: {statistics.mean(e2es):.1f}ms")
    
    # First vs subsequent (shows cache warm-up effect)
    print(f"  First request TTFT: {ttfts[0]:.1f}ms")
    print(f"  Subsequent avg TTFT: {statistics.mean(ttfts[1:]):.1f}ms")
    
    return ttfts, e2es

off_results = load_results('prefix_results_OFF.json')
on_results = load_results('prefix_results_ON.json')

ttft_off, e2e_off = analyze(off_results, "Prefix Caching OFF")
ttft_on, e2e_on = analyze(on_results, "Prefix Caching ON")

# Calculate improvements
ttft_improvement = (statistics.mean(ttft_off[1:]) - statistics.mean(ttft_on[1:])) / statistics.mean(ttft_off[1:]) * 100
print(f"\n{'='*60}")
print(f"TTFT improvement (subsequent requests): {ttft_improvement:.1f}%")
print(f"{'='*60}")
```

Run comparison:
```bash
python3 compare_prefix_results.py
```

### Expected Artifact: `prefix_cache_results.md`

```markdown
# Prefix Cache Reuse Lab Results

## Test Configuration
- Shared prefix: ~2k tokens (system prompt)
- Unique queries: 10 different user questions
- Max output tokens: 100

## Results: Prefix Caching OFF
| Metric | First Request | Subsequent (avg) |
|--------|---------------|------------------|
| TTFT   | X ms          | X ms             |
| E2E    | X ms          | X ms             |

## Results: Prefix Caching ON  
| Metric | First Request | Subsequent (avg) |
|--------|---------------|------------------|
| TTFT   | X ms          | X ms             |
| E2E    | X ms          | X ms             |

## Improvements
- TTFT improvement (subsequent): X%
- Memory savings: ~90% for prefix portion
- Effective throughput increase: ~X%

## Key Observations
- First request pays full prefix computation cost
- Subsequent requests skip prefix entirely (hash match)
- Memory footprint significantly reduced
```

---

## Tier 3 – Stretch: Cache Block Size Tuning

**Title**: Find Optimal Block Size for Your Hardware  
**Goal**: Compare block sizes (16, 32, 64) and measure trade-offs  
**Time Budget**: ~20-30 min  
**Outcome**: Block size recommendation for your workload

---

### Step 1: Create Block Size Sweep Script

Create `block_size_sweep.sh`:

```bash
#!/bin/bash
MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

for BLOCK_SIZE in 16 32 64; do
    echo "=========================================="
    echo "Testing block_size=$BLOCK_SIZE"
    echo "=========================================="
    
    # Restart vLLM with specific block size
    pkill -f "vllm serve" 2>/dev/null
    sleep 5
    
    vllm serve "$MODEL" \
      --port $PORT \
      --disable-log-requests \
      --block-size $BLOCK_SIZE \
      --gpu-memory-utilization 0.85 &
    
    sleep 30  # Wait for model load
    
    # Capture config
    curl -s http://localhost:$PORT/metrics | grep vllm_cache_config > "metrics_block${BLOCK_SIZE}.txt"
    
    # Run standardized test (multiple short sequences)
    python3 block_size_test.py --block-size $BLOCK_SIZE
    
    # Capture final metrics
    curl -s http://localhost:$PORT/metrics | grep -E "(vllm_cache|vllm_gpu)" >> "metrics_block${BLOCK_SIZE}.txt"
done
```

### Step 2: Create Test Script

Create `block_size_test.py`:

```python
#!/usr/bin/env python3
"""Test different block sizes with varied sequence lengths."""
import argparse
import time
import requests
import json
import statistics

BASE_URL = "http://localhost:8000/v1"
MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Test with varying sequence lengths to expose fragmentation
PROMPTS = [
    ("short", "What is 2+2?"),  # ~10 tokens
    ("medium", "Explain quantum computing in three paragraphs."),  # ~50 tokens output
    ("long", "Write a detailed essay about climate change."),  # ~200 tokens output
]

def run_test(block_size):
    print(f"\nBlock size: {block_size}")
    results = []
    
    for name, prompt in PROMPTS:
        for _ in range(5):  # 5 iterations each
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 250,
                    "temperature": 0.7
                }
            )
            elapsed = (time.time() - start) * 1000
            
            data = response.json()
            tokens = data['usage']['completion_tokens']
            results.append({
                'type': name,
                'tokens': tokens,
                'latency_ms': elapsed,
                'ms_per_token': elapsed / tokens if tokens > 0 else 0
            })
    
    # Summarize
    for prompt_type in ['short', 'medium', 'long']:
        subset = [r for r in results if r['type'] == prompt_type]
        avg_latency = statistics.mean([r['latency_ms'] for r in subset])
        avg_ms_per_tok = statistics.mean([r['ms_per_token'] for r in subset])
        print(f"  {prompt_type}: avg latency={avg_latency:.1f}ms, ms/token={avg_ms_per_tok:.2f}")
    
    # Save
    with open(f'block_test_{block_size}.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-size', type=int, default=16)
    args = parser.parse_args()
    run_test(args.block_size)
```

### Step 3: Run the Sweep

```bash
chmod +x block_size_sweep.sh
./block_size_sweep.sh
```

### Step 4: Analyze Results

Create `analyze_block_sizes.py`:

```python
#!/usr/bin/env python3
"""Compare block size experiment results."""
import json
import statistics

def load_and_analyze(block_size):
    with open(f'block_test_{block_size}.json') as f:
        results = json.load(f)
    
    # Calculate fragmentation (theoretical)
    # For sequences of various lengths
    total_tokens = sum(r['tokens'] for r in results)
    total_blocks = sum((r['tokens'] + block_size - 1) // block_size for r in results)
    wasted = total_blocks * block_size - total_tokens
    frag_pct = wasted / (total_blocks * block_size) * 100
    
    avg_ms_per_token = statistics.mean([r['ms_per_token'] for r in results])
    
    return {
        'block_size': block_size,
        'fragmentation_pct': frag_pct,
        'avg_ms_per_token': avg_ms_per_token,
        'total_blocks': total_blocks
    }

print("Block Size Analysis")
print("="*60)
print(f"{'Block Size':>12} {'Fragmentation':>15} {'ms/token':>12} {'Blocks':>10}")
print("-"*60)

for bs in [16, 32, 64]:
    try:
        r = load_and_analyze(bs)
        print(f"{r['block_size']:>12} {r['fragmentation_pct']:>14.1f}% {r['avg_ms_per_token']:>11.2f} {r['total_blocks']:>10}")
    except FileNotFoundError:
        print(f"{bs:>12} (not tested)")
```

Run analysis:
```bash
python3 analyze_block_sizes.py
```

### Expected Artifact: `block_size_analysis.md`

```markdown
# Block Size Tuning Analysis

## Test Configuration
- Model: Phi-3-mini-4k-instruct
- Workloads: short (~10 tok), medium (~50 tok), long (~200 tok)
- Iterations: 5 per workload type

## Results

| Block Size | Fragmentation | ms/token | Total Blocks |
|------------|---------------|----------|--------------|
| 16         | X.X%          | X.XX     | XXX          |
| 32         | X.X%          | X.XX     | XXX          |
| 64         | X.X%          | X.XX     | XXX          |

## Trade-off Analysis

### Block Size 16 (Default)
- ✅ Lowest fragmentation
- ✅ Best for varied sequence lengths
- ❌ Slightly higher per-token overhead

### Block Size 32
- Balanced fragmentation/overhead
- Good for medium-length sequences

### Block Size 64
- ✅ Lowest per-token overhead
- ❌ Highest memory waste
- Best only when memory is plentiful

## Recommendation for [Your GPU]
- **Primary recommendation**: Block size X
- **Rationale**: [why this is optimal for your workload]
```

---

## Logging Template for Tomorrow

```markdown
## Day 09 Results

### Commands Run
- [list key commands executed]

### Files Created
- baseline_memory.md
- prefix_cache_results.md  
- block_size_analysis.md
- [Python scripts]
- [CSV/JSON data files]

### Key Metrics
- Memory per token (measured): X KB
- Blocks for 4k tokens: X
- Prefix cache TTFT improvement: X%
- Optimal block size: X

### Observations / Surprises
- [What matched theory vs surprised you]
- [Any performance anomalies]
- [Memory behavior patterns]

### Next Focus
- [What to explore deeper based on findings]
```
