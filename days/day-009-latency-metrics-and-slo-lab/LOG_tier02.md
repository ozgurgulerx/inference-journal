# Day 009 – KV Cache Lab (Hands-On Microtasks 1-9)

> **This tier is purely procedural**: run the experiments, generate artifacts, and keep the report honest.

---

## Task 1: Environment Setup & Package Install

**Objective**: Provision the RunPod instance with all required tools and repos for Day 009.

**Prerequisites**: Fresh GPU VM (Ubuntu) with internet access.

**Time**: ~10 min

### Commands

In a tmux session:

```bash
# Create day009 folder structure
mkdir -p ~/day009/{scripts,data,logs,reports,archive}
cd ~/day009

# System packages
sudo apt-get update && sudo apt-get install -y git wget tmux htop

# Python packages
pip install --upgrade pip
pip install vllm transformers pycuda pynvml requests pandas

# Set vLLM V1 engine
echo 'export VLLM_USE_V1=1' >> ~/.bashrc
source ~/.bashrc

# Verify GPU
nvidia-smi
```

### Create `scripts/setup_env.sh`

```bash
#!/bin/bash
# Day 009 Environment Setup
set -e

echo "=== Day 009 Environment Setup ==="

# Create folder structure
mkdir -p ~/day009/{scripts,data,logs,reports,archive}

# Install dependencies
sudo apt-get update && sudo apt-get install -y git wget tmux htop
pip install --upgrade pip
pip install vllm transformers pycuda pynvml requests pandas

# Set environment
export VLLM_USE_V1=1
echo 'export VLLM_USE_V1=1' >> ~/.bashrc

# Verify
echo "=== Verification ==="
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Setup Complete ==="
```

**Expected Artifacts**:
- `scripts/setup_env.sh`
- `logs/setup.txt` (console output)

**Success Criteria**:
- `python -c "import vllm"` runs without error
- `nvidia-smi` shows GPU

**Failure Modes**:
- pip install failures → retry with `--no-cache-dir`
- GPU driver issues → ensure NVIDIA driver present on RunPod base image

---

## Task 2: Sanity Check Model Load & Generation

**Objective**: Verify the model loads in vLLM and a simple prompt generates output.

**Prerequisites**: Task 1 done; choose model (e.g., `meta-llama/Llama-2-7b-chat-hf`).

**Time**: ~5 min

### Commands

```bash
cd ~/day009

# Set model (adjust as needed)
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Quick generation test using vLLM CLI
vllm complete --model $MODEL \
  --prompt "Hello, my name is" \
  --max-tokens 10 \
  2>&1 | tee logs/sanity_check.log
```

### Alternative: Python Script

Create `scripts/sanity_check.py`:

```python
#!/usr/bin/env python3
"""Verify model loads and generates output."""
import os
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "meta-llama/Llama-2-7b-chat-hf")

print(f"Loading model: {MODEL}")
llm = LLM(model=MODEL, trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
prompts = ["Hello, my name is"]

print("Generating...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text!r}")
    
print("\n=== Sanity check PASSED ===")
```

Run:
```bash
python scripts/sanity_check.py 2>&1 | tee logs/sanity_check.log
```

**Expected Artifacts**:
- Console print of completion (e.g., "Hello, my name is John...")
- `logs/sanity_check.log`

**Success Criteria**:
- Model loads into memory (~7GB VRAM for 7B model)
- Produces tokens without errors
- Latency is reasonable (few seconds)

**Failure Modes**:
- OOM on model load → use smaller model or lower precision (`--dtype half`)
- Missing model → ensure HuggingFace auth if model is gated
- vLLM errors → check `VLLM_USE_V1=1` is set

---

## Task 3: Baseline Profiling (No Prefix Reuse, Default Block)

**Objective**: Measure baseline latency and throughput with prefix caching disabled and default block size.

**Prerequisites**: Task 2 success.

**Time**: ~15 min

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Run baseline benchmark
vllm bench throughput \
  --model $MODEL \
  --dataset-name random \
  --input-len 50 \
  --output-len 50 \
  --num-prompts 200 \
  --no-enable-prefix-caching \
  --block-size 32 \
  --output-json logs/baseline.json \
  2>&1 | tee logs/baseline_run.log
```

### Create `scripts/run_baseline.sh`

```bash
#!/bin/bash
# Baseline profiling - no prefix caching, default block size
set -e
cd ~/day009

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"

echo "=== Baseline Profiling ==="
echo "Model: $MODEL"
echo "Prefix caching: OFF"
echo "Block size: 32"

vllm bench throughput \
  --model $MODEL \
  --dataset-name random \
  --input-len 50 \
  --output-len 50 \
  --num-prompts 200 \
  --no-enable-prefix-caching \
  --block-size 32 \
  --output-json logs/baseline.json

echo "=== Results saved to logs/baseline.json ==="
```

### Create `reports/baseline_report.md` Template

```markdown
# Baseline Profiling Report

## Configuration
- Model: [MODEL_NAME]
- Input length: 50 tokens
- Output length: 50 tokens
- Num prompts: 200
- Prefix caching: OFF
- Block size: 32

## Results
- Throughput: X tokens/sec
- TTFT p50: X ms
- TTFT p95: X ms
- TTFT p99: X ms
- E2E latency p50: X ms
- E2E latency p95: X ms

## Observations
- [Note any patterns or surprises]
```

**Expected Artifacts**:
- `logs/baseline.json` with throughput, latency distribution
- `reports/baseline_report.md` summarizing findings

**Success Criteria**:
- Run completes without errors (few minutes)
- JSON shows sane numbers (TTFT ~hundreds ms, throughput ~thousands tokens/sec)

**Failure Modes**:
- Benchmark CLI fails → write custom loop with `vllm.LLM`
- OOM with 200 prompts → reduce `--num-prompts` to 100

---

## Task 4: Prefix Caching OFF – Shared Prefix Scenario

**Objective**: Simulate workload where many requests share a common prefix, without caching, to measure redundant computation cost.

**Prerequisites**: Task 3 done; example prefix prepared.

**Time**: ~15 min

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Run shared prefix test WITHOUT caching
vllm bench throughput \
  --model $MODEL \
  --dataset-name sharegpt \
  --input-len 100 \
  --output-len 20 \
  --num-prompts 100 \
  --no-enable-prefix-caching \
  --block-size 32 \
  --output-json logs/prefix_off.json \
  2>&1 | tee logs/prefix_off_run.log
```

### Create `scripts/run_prefix_off.sh`

```bash
#!/bin/bash
# Shared prefix scenario - caching OFF
set -e
cd ~/day009

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"

echo "=== Prefix Caching OFF Test ==="
echo "Model: $MODEL"
echo "Prefix caching: OFF"

vllm bench throughput \
  --model $MODEL \
  --dataset-name sharegpt \
  --input-len 100 \
  --output-len 20 \
  --num-prompts 100 \
  --no-enable-prefix-caching \
  --block-size 32 \
  --output-json logs/prefix_off.json

echo "=== Results saved to logs/prefix_off.json ==="
```

**Expected Artifacts**:
- `logs/prefix_off.json` with metrics
- Note: TTFT expected to be high (each request processes full prefix)

**Success Criteria**:
- Run completes
- Average latency higher than baseline (long prefill per request)

**Failure Modes**:
- Dataset issues → use `--dataset-name random` with `--prefix-len 100` if available
- OOM → reduce `--num-prompts` to 50

---

## Task 5: Prefix Caching ON – Shared Prefix Scenario

**Objective**: Rerun the same shared-prefix workload with prefix caching enabled.

**Prerequisites**: Task 4 completed.

**Time**: ~15 min

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Run shared prefix test WITH caching
vllm bench throughput \
  --model $MODEL \
  --dataset-name sharegpt \
  --input-len 100 \
  --output-len 20 \
  --num-prompts 100 \
  --enable-prefix-caching \
  --block-size 32 \
  --output-json logs/prefix_on.json \
  2>&1 | tee logs/prefix_on_run.log
```

### Create `scripts/run_prefix_on.sh`

```bash
#!/bin/bash
# Shared prefix scenario - caching ON
set -e
cd ~/day009

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"

echo "=== Prefix Caching ON Test ==="
echo "Model: $MODEL"
echo "Prefix caching: ON"

vllm bench throughput \
  --model $MODEL \
  --dataset-name sharegpt \
  --input-len 100 \
  --output-len 20 \
  --num-prompts 100 \
  --enable-prefix-caching \
  --block-size 32 \
  --output-json logs/prefix_on.json

echo "=== Results saved to logs/prefix_on.json ==="
```

### Create `reports/prefix_caching_report.md`

```markdown
# Prefix Caching Comparison Report

## Configuration
- Model: [MODEL_NAME]
- Shared prefix: ~100 tokens
- Output length: 20 tokens
- Num prompts: 100

## Results

| Metric | Caching OFF | Caching ON | Improvement |
|--------|-------------|------------|-------------|
| Throughput (tok/s) | X | X | +X% |
| TTFT p50 (ms) | X | X | -X% |
| TTFT p95 (ms) | X | X | -X% |
| E2E p50 (ms) | X | X | -X% |

## Analysis
- First request: pays full prefix computation cost
- Subsequent requests: skip prefix (KV blocks reused from cache)
- Memory savings: ~90% for prefix portion across all requests

## Key Insight
> "TTFT dropped from Y to Z ms (~N% improvement) when enabling prefix caching, 
> thanks to reusing cached KV for the 100-token prefix."
```

**Expected Artifacts**:
- `logs/prefix_on.json` with metrics
- `reports/prefix_caching_report.md` comparing Task 4 vs Task 5

**Success Criteria**:
- Metrics show notable improvement vs Task 4
- Throughput higher, TTFT/latencies lower

**Failure Modes**:
- Results similar to Task 4 → verify `--enable-prefix-caching` was used
- Cache eviction → unlikely for 100 prompts, but check memory usage

---

## Task 6: Block Size Sweep – 16 vs 32

**Objective**: Evaluate how KV cache block size affects performance and memory.

**Prerequisites**: Baseline done.

**Time**: ~20 min

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Test block_size=16
vllm bench throughput \
  --model $MODEL \
  --dataset-name random \
  --input-len 50 \
  --output-len 50 \
  --num-prompts 200 \
  --no-enable-prefix-caching \
  --block-size 16 \
  --output-json logs/block_sweep_16.json \
  2>&1 | tee logs/block_16_run.log

# Test block_size=32
vllm bench throughput \
  --model $MODEL \
  --dataset-name random \
  --input-len 50 \
  --output-len 50 \
  --num-prompts 200 \
  --no-enable-prefix-caching \
  --block-size 32 \
  --output-json logs/block_sweep_32.json \
  2>&1 | tee logs/block_32_run.log
```

### Create `scripts/run_block_sweep.sh`

```bash
#!/bin/bash
# Block size sweep: 16, 32, 64
set -e
cd ~/day009

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"

for BLOCK_SIZE in 16 32 64; do
    echo "=========================================="
    echo "Testing block_size=$BLOCK_SIZE"
    echo "=========================================="
    
    vllm bench throughput \
      --model $MODEL \
      --dataset-name random \
      --input-len 50 \
      --output-len 50 \
      --num-prompts 200 \
      --no-enable-prefix-caching \
      --block-size $BLOCK_SIZE \
      --output-json logs/block_sweep_${BLOCK_SIZE}.json \
      2>&1 | tee logs/block_${BLOCK_SIZE}_run.log || echo "block_size=$BLOCK_SIZE failed or unsupported"
done

echo "=== Block sweep complete ==="
```

**Expected Artifacts**:
- `logs/block_sweep_16.json`
- `logs/block_sweep_32.json`
- Comparison data for `reports/block_tuning_report.md`

**Success Criteria**:
- Both runs complete
- block_size=16 may show slightly lower throughput than 32 (more overhead)
- Check vLLM logs at startup for "# GPU blocks: N"

**Failure Modes**:
- `--block-size 16` unsupported → note in report (GPU may override to 32)

---

## Task 7: Block Size Sweep – 64 (Exploratory)

**Objective**: Attempt block size = 64 to observe vLLM behavior beyond documented GPU limit.

**Prerequisites**: Task 6 done.

**Time**: ~10 min

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Attempt block_size=64 (may fail or be clamped)
vllm bench throughput \
  --model $MODEL \
  --dataset-name random \
  --input-len 50 \
  --output-len 50 \
  --num-prompts 100 \
  --no-enable-prefix-caching \
  --block-size 64 \
  --output-json logs/block_sweep_64.json \
  2>&1 | tee logs/block_64_run.log
```

**Expected Artifacts**:
- `logs/block_sweep_64.json` OR error log
- Note in `reports/block_tuning_report.md`

**Success Criteria**:
- Confirm system's response to unsupported block size
- If error → that IS the result (learning constraint)
- If runs by using 32 → note results identical to Task 6

**Expected Outcome**: On CUDA devices, block sizes >32 are not supported. The config may be clamped or raise exception.

---

## Task 8: High-Resolution GPU Memory Logging

**Objective**: Collect detailed GPU memory usage over time during generation.

**Prerequisites**: Model loaded, test prompt ready.

**Time**: ~15 min

### Create `scripts/sample_gpu_mem.sh`

```bash
#!/bin/bash
# High-frequency GPU memory logging (50ms intervals)
echo "timestamp,memory_used_mib"
while true; do
    TS=$(date +%s.%3N)
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    echo "$TS,$MEM"
    sleep 0.05
done
```

### Commands

```bash
cd ~/day009

# Terminal 1: Start memory logging
chmod +x scripts/sample_gpu_mem.sh
scripts/sample_gpu_mem.sh > logs/gpu_mem_usage.csv &
GPU_LOG_PID=$!

# Terminal 2: Start vLLM server
export MODEL="meta-llama/Llama-2-7b-chat-hf"
vllm serve --model $MODEL --host 0.0.0.0 --port 8000 &
sleep 30  # Wait for load

# Run long generation
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL'",
    "prompt": "Write a very long detailed essay about the history of computing...",
    "max_tokens": 1000,
    "temperature": 0.7
  }'

# Stop logging
kill $GPU_LOG_PID
pkill -f "vllm serve"
```

### Analyze Memory Trace

```python
#!/usr/bin/env python3
"""Analyze GPU memory trace."""
import pandas as pd

df = pd.read_csv('logs/gpu_mem_usage.csv')
df['timestamp'] = pd.to_numeric(df['timestamp'])
df['elapsed'] = df['timestamp'] - df['timestamp'].iloc[0]

print("GPU Memory Analysis:")
print(f"  Initial: {df['memory_used_mib'].iloc[0]} MiB")
print(f"  Peak:    {df['memory_used_mib'].max()} MiB")
print(f"  Final:   {df['memory_used_mib'].iloc[-1]} MiB")
print(f"  Delta:   {df['memory_used_mib'].max() - df['memory_used_mib'].iloc[0]} MiB")

# Check if memory is stepwise or flat
changes = df['memory_used_mib'].diff().abs()
significant_changes = changes[changes > 10].count()
print(f"  Significant changes (>10 MiB): {significant_changes}")
```

**Expected Artifacts**:
- `logs/gpu_mem_usage.csv` with timestamped memory
- Analysis in `reports/memory_analysis.md`

**Success Criteria**:
- Fine-grained memory log captured
- Likely outcome: Memory mostly flat (vLLM pre-allocates KV cache at startup)

**Failure Modes**:
- nvidia-smi too slow → increase sleep to 0.1
- OOM during generation → reduce `max_tokens` to 500

---

## Task 9: Inspect KV Block Allocation Logs

**Objective**: Use vLLM's internal logging to find how many KV blocks are allocated.

**Prerequisites**: vLLM emits log info at startup.

**Time**: ~10 min

### Commands

```bash
cd ~/day009
export MODEL="meta-llama/Llama-2-7b-chat-hf"

# Start vLLM and capture startup logs
vllm serve --model $MODEL --host 0.0.0.0 --port 8000 \
  2>&1 | tee logs/vllm_startup.log &

sleep 60  # Wait for full startup

# Search for block allocation info
grep -i "block" logs/vllm_startup.log
grep -i "GPU blocks" logs/vllm_startup.log
grep -i "cache" logs/vllm_startup.log

# Capture metrics endpoint
curl -s http://localhost:8000/metrics | grep -i cache > logs/vllm_cache_metrics.txt

pkill -f "vllm serve"
```

### Extract Block Info

Look for lines like:
```
# GPU blocks: 3471, # CPU blocks: 680
```

Calculate:
```python
#!/usr/bin/env python3
"""Calculate token capacity from block allocation."""

GPU_BLOCKS = 3471  # From logs
CPU_BLOCKS = 680   # From logs  
BLOCK_SIZE = 16    # Default

gpu_token_capacity = GPU_BLOCKS * BLOCK_SIZE
cpu_token_capacity = CPU_BLOCKS * BLOCK_SIZE
total_capacity = gpu_token_capacity + cpu_token_capacity

print(f"GPU Token Capacity: {gpu_token_capacity:,} tokens")
print(f"CPU Token Capacity: {cpu_token_capacity:,} tokens")
print(f"Total Capacity: {total_capacity:,} tokens")

# Cross-verify with memory
# Per-token memory for Llama-2-7B: ~16 KB
PER_TOKEN_KB = 16
estimated_kv_memory_gb = (gpu_token_capacity * PER_TOKEN_KB) / 1024 / 1024
print(f"Estimated KV Memory: {estimated_kv_memory_gb:.2f} GB")
```

**Expected Artifacts**:
- Block counts from logs
- Analysis in `reports/memory_analysis.md`

**Success Criteria**:
- Retrieved block counts
- Can compute: total_token_capacity = GPU_blocks × block_size

**Failure Modes**:
- No log appears → search for `_kv_cache` or `BlockTables` in Python debug
- If not found, rely on Task 11 formula instead

---

## Logging Template

```markdown
## Day 09 Progress (Tasks 1-9)

### Environment
- [ ] Task 1: Environment setup complete
- [ ] Task 2: Model sanity check passed

### Core Experiments  
- [ ] Task 3: Baseline profiling done
- [ ] Task 4: Prefix OFF results captured
- [ ] Task 5: Prefix ON results captured
- [ ] Task 6: Block sweep 16/32 done

### Instrumentation (Stretch)
- [ ] Task 7: Block 64 attempted
- [ ] Task 8: GPU memory trace captured
- [ ] Task 9: Block allocation logs extracted

### Key Numbers
- Baseline throughput: _____ tokens/sec
- Baseline TTFT p95: _____ ms
- Prefix ON vs OFF TTFT improvement: _____%
- GPU blocks allocated: _____
- Token capacity: _____ tokens
```
