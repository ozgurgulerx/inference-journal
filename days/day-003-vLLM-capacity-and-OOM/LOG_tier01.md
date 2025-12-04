# Day 003 – vLLM Capacity, OOM Surface & Real Use-Cases
## Tier 1: Must-Do Core Block (~2 hours)

> **Prerequisites**: Complete [Day 002](../day-002-GPU-node-bring-up/)  
> **Goal**: Turn your RunPod node into a measured vLLM server for two real workloads  
> **End State**: Reusable configs, benchmark harness, and capacity grid for chat workload  
> **Time**: ~2 hours

---
## Thoughts
Serving qwen2.5 on vLLM was fun. \
Great that I moved past the psycological barrier of procrastination of not initiating this learning which I think where the most of the tech complexity will be. 
I'm trying to maximise the impact of this learning effort by having chatgpt5.1's guidance to focus and organise it to target more pragmatic / high impact technical learning which any book cannot meet. (It will itself be outdated quiet soon I would assume as the pace of tech increases).

## 🎯 Core Idea

Turn your RunPod node(s) into **measured**, not "hopeful", vLLM servers for two real workloads:

1. **Latency-optimized chat API** – short turns, interactive user
2. **Throughput-optimized batch summarization** – documents, offline jobs

And, if budget allows, you'll get one A100/H100 anchor to see how the same config scales on "real" inference hardware.

---

## 📊 Key Metrics We'll Measure

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **TTFT** | Time to First Token | User perceives "responsiveness" |
| **ITL** | Inter-Token Latency (time between tokens) | Streaming smoothness |
| **TPOT** | Time Per Output Token = `(e2e - ttft) / output_tokens` | Decode efficiency |
| **E2E** | End-to-end latency | Total request time |
| **System TPS** | Total tokens/sec across all requests | Infrastructure capacity |
| **User TPS** | Tokens/sec per user ≈ `1/ITL` | Individual UX |

---

## Tier 1 Tasks (~2 hours)

---

### ✅ Task 1.1: Encode a Reusable vLLM Config for Chat
**Tags**: `[Inference–Runtime]` `[Phase1-vLLM-Baseline]`  
**Time**: 30 min  
**Win**: One clean YAML for "chat on a 16GB-ish GPU" (RTX 2000 / T4 class)

#### 🔧 Lab Instructions

Create the config directory and start vLLM directly (simpler than YAML config):

```bash
mkdir -p ~/configs/vllm

# Create a simple start script (CLI args are more reliable than YAML)
cat > ~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --enable-chunked-prefill
EOF

chmod +x ~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh
```

> **Note**: CLI args are more reliable than YAML config files across vLLM versions.

#### 📁 Artifacts
- `~/configs/vllm/qwen2p5_1p5b_chat_16gb.yaml`
- `~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh`

#### 💡 Why This Matters
You now have a **named recipe** for "chat on small GPU", not a pile of CLI flags.

#### 🏆 Success Criteria
- [ ] YAML config created with documented knobs
- [ ] Wrapper script runs without errors
- [ ] Config values match your Day 002 findings

---

### ✅ Task 1.2: Build an Async Chat-Load Benchmark Harness
**Tags**: `[Inference–Runtime]` `[Phase1-LoadTesting]`  
**Time**: 45 min  
**Win**: Measure p50/p95 TTFT + end-to-end latency + rough tokens/sec

#### 🔧 Lab Instructions

First, install the required dependency:

```bash
pip install aiohttp
```

Create the benchmark script:

```bash
mkdir -p ~/scripts/benchmarks ~/benchmarks

cat > ~/scripts/benchmarks/vllm_chat_bench.py << 'EOF'
#!/usr/bin/env python3
"""Simple async benchmark for vLLM. Measures TTFT, E2E latency, throughput."""

import argparse
import asyncio
import json
import statistics
import time
import aiohttp

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

async def run_single_request(session, url, prompt, max_tokens):
    """Send one request, return (ttft_ms, e2e_ms, output_tokens)."""
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": False}
    
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        t1 = time.perf_counter()  # first byte
        data = await resp.json()
    t2 = time.perf_counter()
    
    # Handle both completion and chat response formats
    if "choices" in data and len(data["choices"]) > 0:
        text = data["choices"][0].get("text", "") or data["choices"][0].get("message", {}).get("content", "")
    else:
        text = ""
    
    ttft_ms = (t1 - t0) * 1000
    e2e_ms = (t2 - t0) * 1000
    out_tokens = len(text.split())  # rough estimate
    
    return ttft_ms, e2e_ms, out_tokens


async def run_benchmark(url, prompt, n_requests, concurrency, max_tokens):
    """Run n_requests with concurrency limit, return stats dict."""
    results = []  # list of (ttft, e2e, tokens)
    sem = asyncio.Semaphore(concurrency)
    
    async def worker():
        async with sem:
            async with aiohttp.ClientSession() as session:
                result = await run_single_request(session, url, prompt, max_tokens)
                results.append(result)
    
    t_start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(n_requests)])
    t_end = time.perf_counter()
    
    wall_clock_s = t_end - t_start
    ttfts = [r[0] for r in results]
    e2es = [r[1] for r in results]
    tokens = [r[2] for r in results]
    
    def percentile(vals, p):
        if not vals: return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p), len(s) - 1)]
    
    total_tokens = sum(tokens)
    
    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "wall_clock_s": round(wall_clock_s, 2),
        "p50_ttft_ms": round(statistics.median(ttfts), 2) if ttfts else 0,
        "p95_ttft_ms": round(percentile(ttfts, 0.95), 2),
        "p50_e2e_ms": round(statistics.median(e2es), 2) if e2es else 0,
        "p95_e2e_ms": round(percentile(e2es, 0.95), 2),
        "throughput_tok_s": round(total_tokens / wall_clock_s, 2) if wall_clock_s > 0 else 0,
        "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Chat Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--n-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    
    prompt = "Explain the trade-offs between max_model_len and max_num_seqs for vLLM serving in 3 sentences."
    
    result = asyncio.run(run_benchmark(
        url=args.url,
        prompt=prompt,
        n_requests=args.n_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    ))
    print(json.dumps(result, indent=2))
EOF

chmod +x ~/scripts/benchmarks/vllm_chat_bench.py
```

#### 🧪 Run the Benchmark

Terminal 1 – Start vLLM:
```bash
~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh
```

Terminal 2 – Run benchmark:
```bash
python ~/scripts/benchmarks/vllm_chat_bench.py \
  --n-requests 32 --concurrency 8 --max-new-tokens 128 \
  > ~/benchmarks/day003_chat_baseline_rtx16gb.json

cat ~/benchmarks/day003_chat_baseline_rtx16gb.json
```

#### 📁 Artifacts
- `~/scripts/benchmarks/vllm_chat_bench.py`
- `~/benchmarks/day003_chat_baseline_rtx16gb.json`

#### 🏆 Success Criteria
- [ ] Benchmark script runs without errors
- [ ] JSON output contains TTFT, TPOT, E2E, and throughput metrics
- [ ] Results saved to benchmarks directory

<details>
<summary><strong>📚 Understanding the Metrics</strong></summary>

| Metric | Formula | Good Value (Chat) |
|--------|---------|-------------------|
| **TTFT** | Time until first token arrives | < 200ms |
| **TPOT** | `(e2e - ttft) / output_tokens` | < 50ms |
| **ITL** | Time between consecutive tokens (streaming) | < 50ms |
| **System TPS** | `total_tokens / wall_clock_time` | Higher = better |
| **User TPS** | `≈ 1000 / TPOT_ms` | > 20 tok/s |

**Why TPOT matters**: TPOT captures decode efficiency. High TPOT = slow token generation = poor streaming UX.

**System vs User TPS**:
- System TPS = total capacity (how many tokens/sec your GPU produces)
- User TPS = what one user experiences (degrades under load)

</details>

---

### ✅ Task 1.3: Map a Chat-Capacity Grid on Your GPU
**Tags**: `[Inference–Runtime]` `[Phase1-LoadTesting]`  
**Time**: 45 min  
**Win**: Find a safe-ish zone for chat on your current RunPod GPU

#### 🔧 Lab Instructions

Create the grid sweep script:

```bash
cat > ~/scripts/benchmarks/run_chat_capacity_grid.sh << 'OUTER'
#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://127.0.0.1:8000/v1/completions}"
OUT_CSV="${OUT_CSV:-$HOME/benchmarks/day003_chat_capacity.csv}"
GPU_NAME="${GPU_NAME:-RTX-16GB}"

mkdir -p ~/benchmarks

echo "gpu,concurrency,max_tokens,p50_ttft_ms,p95_ttft_ms,p50_e2e_ms,p95_e2e_ms,throughput_tok_s" > "$OUT_CSV"

for conc in 1 4 8 16; do
  for maxtok in 64 128 256; do
    echo "[*] Testing: concurrency=$conc, max_tokens=$maxtok"
    
    # Run benchmark and save JSON
    python ~/scripts/benchmarks/vllm_chat_bench.py \
      --url "$URL" \
      --n-requests 16 \
      --concurrency "$conc" \
      --max-tokens "$maxtok" \
      > /tmp/bench_result.json
    
    # Parse JSON and append CSV row
    python3 -c "
import json
with open('/tmp/bench_result.json') as f:
    d = json.load(f)
print(f\"$GPU_NAME,$conc,$maxtok,{d['p50_ttft_ms']},{d['p95_ttft_ms']},{d['p50_e2e_ms']},{d['p95_e2e_ms']},{d['throughput_tok_s']}\")
" >> "$OUT_CSV"
    
  done
done

echo ""
echo "[✓] Results: $OUT_CSV"
cat "$OUT_CSV"
OUTER

chmod +x ~/scripts/benchmarks/run_chat_capacity_grid.sh
```

#### 🧪 Run the Grid

With vLLM serving in another terminal:

```bash
GPU_NAME="RunPod-RTX2000-16GB" ~/scripts/benchmarks/run_chat_capacity_grid.sh
```

#### 📝 Document Findings

```bash
mkdir -p ~/artifacts

cat > ~/artifacts/day003_chat_capacity_notes.md << 'EOF'
# Day 003 – Chat Capacity Notes

## GPU: [YOUR GPU NAME]
## Model: Qwen/Qwen2.5-1.5B-Instruct (BF16)
## Config: gpu_memory_utilization=0.8, max_model_len=4096, max_num_seqs=128

## Best Combinations (high throughput, reasonable p95)

| Concurrency | max_new_tokens | p95 E2E (ms) | Throughput (tok/s) | Notes |
|-------------|----------------|--------------|--------------------| ------|
| [FILL] | [FILL] | [FILL] | [FILL] | Sweet spot |
| [FILL] | [FILL] | [FILL] | [FILL] | Still acceptable |

## Jitter / Unstable Zone

| Concurrency | max_new_tokens | Issue |
|-------------|----------------|-------|
| [FILL] | [FILL] | p95 spiked to X ms |

## Key Observations

1. [YOUR OBSERVATION]
2. [YOUR OBSERVATION]
3. [YOUR OBSERVATION]

EOF

echo "Edit ~/artifacts/day003_chat_capacity_notes.md with your findings"
```

#### 📁 Artifacts
- `~/scripts/benchmarks/run_chat_capacity_grid.sh`
- `~/benchmarks/day003_chat_capacity_rtx16gb.csv`
- `~/artifacts/day003_chat_capacity_notes.md`

#### 💡 Why This Matters
You now have **measured chat capacity** for a given GPU – not vibes. This already puts you closer to the top 1% than most practitioners.

#### 🏆 Success Criteria
- [ ] Grid sweep completed for all concurrency × max_new_tokens combos
- [ ] CSV saved with all results
- [ ] Notes file documents best combos and jitter zones

---

## Tier 1 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **1.1** | Created reusable vLLM YAML config for chat | ⬜ |
| **1.2** | Built async benchmark harness | ⬜ |
| **1.3** | Mapped chat capacity grid | ⬜ |

### Artifacts Created
```
~/configs/vllm/
├── qwen2p5_1p5b_chat_16gb.yaml
└── serve_qwen2p5_1p5b_chat_16gb.sh

~/scripts/benchmarks/
├── vllm_chat_bench.py
└── run_chat_capacity_grid.sh

~/benchmarks/
├── day003_chat_baseline_rtx16gb.json
└── day003_chat_capacity_rtx16gb.csv

~/artifacts/
└── day003_chat_capacity_notes.md
```

---

**→ Continue to [Tier 2](LOG_tier02.md)**: Batch summarization workload + A100/H100 anchor run
