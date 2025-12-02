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

Create the config directory and YAML:

```bash
mkdir -p ~/configs/vllm

cat > ~/configs/vllm/qwen2p5_1p5b_chat_16gb.yaml << 'EOF'
# vLLM config for chat-like workloads on a ~16GB GPU (RunPod RTX 2000 / T4 class)

model: "Qwen/Qwen2.5-1.5B-Instruct"

# Use bf16 to exploit tensor cores while keeping memory reasonable
dtype: "bfloat16"

# Networking
host: "0.0.0.0"
port: 8000

# Memory & capacity knobs
gpu_memory_utilization: 0.8   # leave some headroom to avoid OOM jitter
max_model_len: 4096           # typical chat context window
max_num_seqs: 128             # concurrent sequences vLLM may keep in memory
max_num_batched_tokens: 2048  # prefill chunk size; good balance for small model

# Runtime features
tensor_parallel_size: 1
enable_prefix_caching: true
enable_chunked_prefill: true

# Logging
disable_log_requests: true
log_level: "INFO"
EOF
```

Create a wrapper script:

```bash
cat > ~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Disable HF transfer optimization to keep downloads predictable
export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve --config ~/configs/vllm/qwen2p5_1p5b_chat_16gb.yaml
EOF

chmod +x ~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh
```

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

```bash
mkdir -p ~/scripts/benchmarks ~/benchmarks

cat > ~/scripts/benchmarks/vllm_chat_bench.py << 'EOF'
#!/usr/bin/env python3
"""
Async chat-style benchmark for vLLM with full metrics.

Measures (per vLLM/Anyscale conventions):
  - TTFT: Time to First Token
  - ITL: Inter-Token Latency (streaming mode)
  - TPOT: Time Per Output Token = (e2e - ttft) / output_tokens
  - E2E: End-to-end latency
  - System TPS: Total throughput (tokens/sec)
  - User TPS: Per-user throughput ≈ 1/ITL

Use this for 'chat-like' workloads:
  - short prompt, short-ish answers
  - moderate concurrency
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


async def run_request(session, url: str, prompt: str, max_new_tokens: int):
    """Send a single completion request and measure TTFT + E2E latency."""
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "stream": False,
    }

    t_start = time.time()
    async with session.post(url, json=payload) as resp:
        t_first = time.time()  # first byte back from server
        data = await resp.json()
    t_end = time.time()

    ttft_ms = (t_first - t_start) * 1000.0
    e2e_ms = (t_end - t_start) * 1000.0

    # crude token estimate (word-based)
    text = data["choices"][0]["text"]
    out_tokens = len(text.split())

    return ttft_ms, e2e_ms, out_tokens


async def run_bench(
    url: str,
    n_requests: int,
    concurrency: int,
    max_new_tokens: int,
    prompt: str,
):
    """Run n_requests with given concurrency and return aggregate stats."""
    ttfts, e2es, toks = [], [], []
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(i: int):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                ttft_ms, e2e_ms, out_tokens = await run_request(
                    session, url, prompt, max_new_tokens
                )
                ttfts.append(ttft_ms)
                e2es.append(e2e_ms)
                toks.append(out_tokens)

    await asyncio.gather(*[worker(i) for i in range(n_requests)])

    def p95(values):
        if not values:
            return 0.0
        values_sorted = sorted(values)
        idx = max(0, int(0.95 * len(values_sorted)) - 1)
        return values_sorted[idx]

    total_tokens = sum(toks)
    total_time_s = sum(e2es) / 1000.0 if e2es else 0.0
    throughput_tok_s = total_tokens / total_time_s if total_time_s > 0 else 0.0

    # Calculate TPOT (Time Per Output Token)
    tpots = []
    for ttft, e2e, tok in zip(ttfts, e2es, toks):
        if tok > 0:
            tpot = (e2e - ttft) / tok
            tpots.append(tpot)

    return {
        "p50_ttft_ms": statistics.median(ttfts) if ttfts else 0.0,
        "p95_ttft_ms": p95(ttfts),
        "p50_tpot_ms": statistics.median(tpots) if tpots else 0.0,
        "p95_tpot_ms": p95(tpots),
        "p50_e2e_ms": statistics.median(e2es) if e2es else 0.0,
        "p95_e2e_ms": p95(e2es),
        "system_throughput_tok_s": throughput_tok_s,
        "user_tps_approx": 1000.0 / statistics.median(tpots) if tpots else 0.0,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--n-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    chat_prompt = (
        "You are an LLM inference engineer. In 3–4 concise sentences, "
        "explain the trade-offs between max_model_len, max_num_seqs, and "
        "concurrency for serving Qwen2.5-1.5B on a 16GB GPU."
    )

    res = asyncio.run(
        run_bench(
            url=args.url,
            n_requests=args.n_requests,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            prompt=chat_prompt,
        )
    )
    print(json.dumps(res, indent=2))
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
cat > ~/scripts/benchmarks/run_chat_capacity_grid.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://127.0.0.1:8000/v1/completions}"
OUT_CSV="${OUT_CSV:-$HOME/benchmarks/day003_chat_capacity_rtx16gb.csv}"
GPU_NAME="${GPU_NAME:-RunPod-RTX-small}"

mkdir -p ~/benchmarks

echo "workload,gpu,concurrency,max_new_tokens,p50_ttft_ms,p95_ttft_ms,p50_e2e_ms,p95_e2e_ms,throughput_tok_s" > "$OUT_CSV"

for conc in 1 4 8 16; do
  for mnt in 64 256 512; do
    echo "[*] chat capacity: conc=${conc}, max_new_tokens=${mnt}"
    
    python ~/scripts/benchmarks/vllm_chat_bench.py \
      --url "$URL" \
      --n-requests 32 \
      --concurrency "$conc" \
      --max-new-tokens "$mnt" \
      > /tmp/chat_bench.json

    # Parse and append to CSV
    python3 << PY
import json
data = json.load(open("/tmp/chat_bench.json"))
print(f"chat,${GPU_NAME},${conc},${mnt},{data['p50_ttft_ms']:.2f},{data['p95_ttft_ms']:.2f},{data['p50_e2e_ms']:.2f},{data['p95_e2e_ms']:.2f},{data['throughput_tok_s']:.2f}")
PY
  done
done >> "$OUT_CSV"

echo ""
echo "[✓] Results written to: ${OUT_CSV}"
EOF

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
