# Day 003 – vLLM Capacity, OOM Surface & Real Use-Cases
## Tier 2: Extension (~1–2 hours)

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md)  
> **Goal**: Add batch summarization workload + A100/H100 anchor comparison  
> **End State**: Second workload benchmarked, GPU scaling data collected  
> **Time**: ~1–2 hours

---

## Progression from Tier 1

In Tier 1, you built a **vLLM server config**, an **async benchmark harness** (`vllm_chat_bench.py`), and mapped a **chat capacity grid** on the RTX 2000 16GB. Here we reuse those same tools and extend them in two directions:

1. **New workload**: Throughput-heavy batch document summarization (offline, latency-relaxed)
2. **New hardware** (optional): A100/H100 anchor run for scaling comparison

Same tools, new workload + new GPU.

---

## Tier 2 Tasks

---

### ✅ Task 2.1: Throughput Use-Case – Batch Document Summarization
**Tags**: `[Inference–Runtime]` `[Phase1-LoadTesting]` `[Phase3-Optimization]`  
**Time**: 45–60 min  
**Win**: Second workload – lots of medium-long docs → maximize tokens/sec

#### � Tier 1 Reference

We'll compare batch metrics against the **chat capacity grid** from Tier 1:
- Chat baseline: `~/benchmarks/day003_chat_capacity_rtx16gb.csv`
- Chat observations: see [Tier 1 Task 1.3 Findings](LOG_tier01.md#-findings-runpod-rtx-2000-ada-16gb)

This lets us directly contrast **interactive chat** (low-latency, moderate throughput) vs **offline batch** (latency-relaxed, max throughput).

#### � Lab Instructions

**Step 1: Prepare sample documents**

```bash
mkdir -p ~/data

cat > ~/data/day003_docs_sample.txt << 'EOF'
[DOC]
Large language models for finance: The adoption of LLMs in financial services has accelerated dramatically. Key use cases include risk assessment, fraud detection, customer service automation, and regulatory compliance. However, challenges remain around model interpretability, data privacy, and the need for domain-specific fine-tuning. Financial institutions must balance the potential for efficiency gains against the risks of model hallucination and regulatory scrutiny.

[DOC]
Inference engineering in practice: The gap between training a model and serving it at scale is substantial. Inference engineers focus on latency optimization, memory efficiency, and throughput maximization. Key techniques include quantization, KV cache optimization, continuous batching, and speculative decoding. The choice of serving framework (vLLM, TensorRT-LLM, Triton) significantly impacts achievable performance.

[DOC]
GPU vs TPU trade-offs for LLM serving: NVIDIA GPUs remain the dominant hardware for LLM inference due to mature software ecosystems (CUDA, TensorRT) and widespread availability. Google TPUs offer compelling price-performance for specific workloads but require JAX expertise and have limited availability outside Google Cloud. The emergence of AMD MI300X and Intel Gaudi provides new options but with less mature tooling.

[DOC]
Kubernetes for ML infrastructure: Running LLM inference on Kubernetes requires careful resource management. Key considerations include GPU scheduling, node affinity, resource quotas, and autoscaling policies. Tools like KubeRay, Seldon, and KServe simplify deployment but add operational complexity. Most teams benefit from starting with single-node serving before graduating to K8s-orchestrated inference.

[DOC]
Cost optimization for LLM inference: The cost of running LLMs at scale can be substantial. Strategies for optimization include: right-sizing GPU instances, using spot/preemptible instances for batch workloads, implementing request batching, applying quantization to reduce memory requirements, and caching frequent responses. Understanding the cost-per-token across different configurations is essential for production deployments.
EOF
```

**Step 2: Create batch summarization benchmark**

```bash
cat > ~/scripts/benchmarks/vllm_batch_summarize_bench.py << 'EOF'
#!/usr/bin/env python3
"""
Batch document summarization benchmark for vLLM.

Measures throughput-focused metrics for offline batch processing:
  - Total tokens/sec
  - p50/p95 latency per document
  - Aggregate processing time

Use this for 'batch' workloads:
  - Longer prompts (documents)
  - Higher concurrency
  - Throughput > latency priority
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


def load_documents(filepath: str) -> list[str]:
    """Load documents from a [DOC] delimited file."""
    content = Path(filepath).read_text()
    docs = []
    for chunk in content.split("[DOC]"):
        chunk = chunk.strip()
        if chunk:
            docs.append(chunk)
    return docs


async def summarize_doc(session, url: str, doc: str, max_new_tokens: int):
    """Send a summarization request for a single document."""
    prompt = (
        "Summarize the following technical document in 5 bullet points. "
        "Be concise and focus on key insights.\n\n"
        f"Document:\n{doc}\n\n"
        "Summary:"
    )
    
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "stream": False,
    }

    t_start = time.time()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
    t_end = time.time()

    latency_ms = (t_end - t_start) * 1000.0
    text = data["choices"][0]["text"]
    out_tokens = len(text.split())
    
    return latency_ms, out_tokens


async def run_batch_bench(
    url: str,
    docs: list[str],
    concurrency: int,
    max_new_tokens: int,
    repeat: int = 1,
):
    """Run batch summarization with given concurrency."""
    # Repeat docs to get more samples
    all_docs = docs * repeat
    
    latencies = []
    tokens = []
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(doc: str):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                lat_ms, out_toks = await summarize_doc(
                    session, url, doc, max_new_tokens
                )
                latencies.append(lat_ms)
                tokens.append(out_toks)

    t_start = time.time()
    await asyncio.gather(*[worker(doc) for doc in all_docs])
    total_time_s = time.time() - t_start

    def p95(values):
        if not values:
            return 0.0
        values_sorted = sorted(values)
        idx = max(0, int(0.95 * len(values_sorted)) - 1)
        return values_sorted[idx]

    total_tokens = sum(tokens)
    throughput_tok_s = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "workload": "batch_summarize",
        "n_documents": len(all_docs),
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
        "total_time_s": round(total_time_s, 2),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput_tok_s, 2),
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(p95(latencies), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--docs-file", default="data/day003_docs_sample.txt")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=4, 
                        help="Repeat docs N times for more samples")
    args = parser.parse_args()

    docs = load_documents(args.docs_file)
    print(f"[*] Loaded {len(docs)} documents, will process {len(docs) * args.repeat} total", 
          file=__import__('sys').stderr)

    res = asyncio.run(
        run_batch_bench(
            url=args.url,
            docs=docs,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            repeat=args.repeat,
        )
    )
    print(json.dumps(res, indent=2))
EOF

chmod +x ~/scripts/benchmarks/vllm_batch_summarize_bench.py
```

**Step 3: Run batch benchmark**

```bash
# vLLM should still be serving from Tier 1 (same server, same config)

# Test at different concurrency levels (higher than chat since latency is relaxed)
python ~/scripts/benchmarks/vllm_batch_summarize_bench.py \
  --docs-file ~/data/day003_docs_sample.txt \
  --concurrency 16 --max-new-tokens 256 --repeat 4 \
  > ~/benchmarks/day003_batch_c16_rtx16gb.json

python ~/scripts/benchmarks/vllm_batch_summarize_bench.py \
  --docs-file ~/data/day003_docs_sample.txt \
  --concurrency 32 --max-new-tokens 256 --repeat 4 \
  > ~/benchmarks/day003_batch_c32_rtx16gb.json

# View results
cat ~/benchmarks/day003_batch_c16_rtx16gb.json
cat ~/benchmarks/day003_batch_c32_rtx16gb.json
```

**Step 4: Create batch capacity CSV**

```bash
cat > ~/scripts/benchmarks/run_batch_capacity_grid.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://127.0.0.1:8000/v1/completions}"
OUT_CSV="${OUT_CSV:-$HOME/benchmarks/day003_batch_capacity_rtx16gb.csv}"
GPU_NAME="${GPU_NAME:-RunPod-RTX-small}"
DOCS_FILE="${DOCS_FILE:-$HOME/data/day003_docs_sample.txt}"

mkdir -p ~/benchmarks

echo "workload,gpu,concurrency,max_new_tokens,total_time_s,throughput_tok_s,p50_latency_ms,p95_latency_ms" > "$OUT_CSV"

for conc in 8 16 32; do
  for mnt in 256 512; do
    echo "[*] batch capacity: conc=${conc}, max_new_tokens=${mnt}"
    
    python ~/scripts/benchmarks/vllm_batch_summarize_bench.py \
      --url "$URL" \
      --docs-file "$DOCS_FILE" \
      --concurrency "$conc" \
      --max-new-tokens "$mnt" \
      --repeat 4 \
      > /tmp/batch_bench.json

    python3 << PY
import json
data = json.load(open("/tmp/batch_bench.json"))
print(f"batch,${GPU_NAME},${conc},${mnt},{data['total_time_s']},{data['throughput_tok_s']},{data['p50_latency_ms']},{data['p95_latency_ms']}")
PY
  done
done >> "$OUT_CSV"

echo ""
echo "[✓] Results written to: ${OUT_CSV}"
EOF

chmod +x ~/scripts/benchmarks/run_batch_capacity_grid.sh

# Run it
GPU_NAME="RunPod-RTX2000-16GB" ~/scripts/benchmarks/run_batch_capacity_grid.sh
```

**Step 5: Update notes with batch comparison**

Add to `~/artifacts/day003_chat_capacity_notes.md`:

```bash
cat >> ~/artifacts/day003_chat_capacity_notes.md << 'EOF'

---

## Batch Summarization Results

### Best Batch Configuration

| Concurrency | max_new_tokens | Throughput (tok/s) | p95 Latency (ms) |
|-------------|----------------|--------------------| -----------------|
| [FILL] | [FILL] | [FILL] | [FILL] |

### Chat vs Batch Comparison (RTX 2000 16GB)

*Chat numbers from Tier 1 (`day003_chat_capacity_rtx16gb.csv`), batch numbers from Tier 2 (`day003_batch_capacity_rtx16gb.csv`)*

| Workload | Best Concurrency | Best Throughput | Acceptable p95 |
|----------|------------------|-----------------|----------------|
| Chat | [FILL] | [FILL] tok/s | [FILL] ms |
| Batch | [FILL] | [FILL] tok/s | [FILL] ms |

### Key Differences

1. Batch can tolerate higher p95 latency (no interactive user waiting)
2. Batch achieves higher throughput by pushing concurrency harder
3. Batch uses longer outputs (256+ tokens) where chat was optimized for 128
4. [YOUR OBSERVATION]

EOF
```

---

#### 📊 Findings: Batch Summarization on RTX 2000 Ada 16GB

Using the same vLLM config as Tier 1, but with longer prompts and a summarization instruction, we swept concurrency × max_new_tokens for an offline batch workload.

##### Raw Data

```csv
workload,gpu,concurrency,max_new_tokens,total_time_s,throughput_tok_s,p50_latency_ms,p95_latency_ms
batch,RunPod-RTX2000-16GB,8,256,12.09,243.03,3653.98,4569.61
batch,RunPod-RTX2000-16GB,8,512,12.63,276.87,3947.04,5888.02
batch,RunPod-RTX2000-16GB,16,256,7.2,430.22,4757.83,5184.75
batch,RunPod-RTX2000-16GB,16,512,9.43,355.52,4036.16,9423.26
batch,RunPod-RTX2000-16GB,32,256,5.3,567.9,4657.35,5300.39
batch,RunPod-RTX2000-16GB,32,512,6.54,443.28,4318.5,5970.2
```

##### Best Batch Configurations

| Concurrency | max_new_tokens | Throughput (tok/s) | p95 Latency (ms) | Comment |
|-------------|----------------|--------------------|------------------|---------|
| 32 | 256 | ~568 | ~5300 | Best 256-token setting |
| 32 | 512 | ~443 | ~6000 | Best 512-token setting |

##### Key Patterns Observed

**1. Throughput scales cleanly with concurrency (for 256 tokens)**

| Concurrency | Throughput | p95 Latency | Scaling |
|-------------|------------|-------------|---------|
| 8 | 243 tok/s | 4570ms | baseline |
| 16 | 430 tok/s | 5185ms | 1.8× |
| 32 | 568 tok/s | 5300ms | 2.3× |

Going from 8 → 32 concurrent docs gives **2.3× throughput** with p95 rising only from ~4.6s to ~5.3s.

**2. Jitter ridge at concurrency=16 for 512-token outputs**

| Concurrency | max_tokens=512 | p95 Latency |
|-------------|----------------|-------------|
| 8 | 277 tok/s | 5888ms |
| 16 | 356 tok/s | **9423ms** ⚠️ |
| 32 | 443 tok/s | 5970ms |

Concurrency=16 with 512 tokens is a "bad ridge" — p95 jumps to ~9.4s while both 8 and 32 stay under 6s. Likely an interaction between sequence length, batching, and scheduler behavior.

**3. Longer outputs reduce throughput but batch can absorb it**

At concurrency=32:
- 256 tokens → 568 tok/s, p95 ~5.3s
- 512 tokens → 443 tok/s, p95 ~6.0s

Doubling output length drops throughput ~22% but latency only increases ~13%. Acceptable for offline jobs.

##### Chat vs Batch Comparison (RTX 2000 Ada 16GB)

*Chat numbers from Tier 1 (`day003_chat_capacity_rtx16gb.csv`), batch numbers from this task.*

| Workload | Best Config | Throughput (tok/s) | p95 Latency | Notes |
|----------|-------------|--------------------|-------------|-------|
| **Chat** | conc=16, max_tokens=128 | ~620 | ~2600ms | Interactive SLO |
| **Batch** | conc=32, max_tokens=256 | ~568 | ~5300ms | Offline, relaxed |

**Key insight**: Chat at 16×128 achieves ~620 tok/s with p95 ~2.6s, while batch at 32×256 reaches ~568 tok/s with p95 ~5.3s. Batch trades latency headroom for the ability to push longer outputs and higher concurrency.

##### Operating Recommendations

**For 256-token summaries:**
```yaml
concurrency: 32
max_new_tokens: 256
expected_throughput: ~568 tok/s
expected_p95: ~5.3s
```

**For 512-token summaries:**
```yaml
concurrency: 32  # NOT 16 — jitter ridge
max_new_tokens: 512
expected_throughput: ~443 tok/s
expected_p95: ~6.0s
```

**Avoid:** `concurrency=16, max_tokens=512` — p95 spikes to ~9.4s.

---

#### 📁 Artifacts
- `~/data/day003_docs_sample.txt`
- `~/scripts/benchmarks/vllm_batch_summarize_bench.py`
- `~/scripts/benchmarks/run_batch_capacity_grid.sh`
- `~/benchmarks/day003_batch_capacity_rtx16gb.csv`
- `~/benchmarks/day003_batch_c16_rtx16gb.json`
- `~/benchmarks/day003_batch_c32_rtx16gb.json`

#### 💡 Why This Matters

| Aspect | Chat (Tier 1) | Batch (Tier 2) |
|--------|---------------|----------------|
| **Priority** | Low latency (TTFT, p95 E2E) | High throughput (tok/s) |
| **User** | Human waiting at screen | Background job, no one watching |
| **Concurrency** | Capped to protect latency | Push as high as GPU allows |
| **Output length** | Short (~128 tokens) | Longer (~256+ tokens) |
| **SLO** | p95 E2E < 3s | p95 E2E < 30s is fine |

Knowing both profiles lets you **right-size endpoints**: one pool for chat, one for batch, different configs.

#### 🏆 Success Criteria
- [ ] Batch benchmark script working
- [ ] Batch capacity grid completed
- [ ] Notes compare chat vs batch workloads

---

### ✅ Task 2.2 (Optional): GPU Scaling Anchor – RTX vs A100/H100
**Tags**: `[Inference–Runtime]` `[Phase1-ScalingAcrossGPUs]`  
**Time**: 30–45 min (if you already have access)  
**Win**: Build intuition for when to recommend bigger GPUs

> ⚠️ **This is optional.** Only do this if you already have A100/H100 credits or cheap access. The core learning from Day 003 is on the RTX 2000; this step is just for extra intuition about bigger GPUs.

#### 🔧 If You Have A100/H100 Access

Reuse the **exact same scripts** from Tier 1 and Tier 2.1 — only the GPU changes:

```bash
# Chat capacity grid
GPU_NAME="RunPod-A100-40GB" \
OUT_CSV="$HOME/benchmarks/day003_chat_capacity_a100_40gb.csv" \
~/scripts/benchmarks/run_chat_capacity_grid.sh

# Batch capacity grid
GPU_NAME="RunPod-A100-40GB" \
OUT_CSV="$HOME/benchmarks/day003_batch_capacity_a100_40gb.csv" \
~/scripts/benchmarks/run_batch_capacity_grid.sh
```

Optionally push concurrency higher (A100/H100 can handle 64–128 concurrent).

#### 📝 Paper Exercise Alternative

**Don't want to pay for A100/H100 time?** Use public benchmarks to fill the comparison table:

- vLLM benchmarks show A100 40GB typically achieves **2–4× throughput** vs RTX-class for similar models
- NVIDIA's published numbers for Qwen-class models on A100 show ~1500–2500 tok/s at high concurrency
- Use these to estimate your speedup factor and cost-per-token

This still teaches the **decision logic** (when to recommend RTX vs A100/H100) without spending money.

#### 📊 Comparison Framework

Fill this table with measured or estimated data:

| Workload | Metric | RTX 2000 16GB | A100 40GB (est.) | Speedup |
|----------|--------|---------------|------------------|---------|
| **Chat** | Peak throughput | ~620 tok/s | ~1500–2000 tok/s | ~2.5–3× |
| **Chat** | Max stable conc | 16 | 64–128 | ~4–8× |
| **Batch** | Peak throughput | ~568 tok/s | ~1200–1800 tok/s | ~2–3× |
| **Batch** | p95 @ peak conc | ~5300ms | ~2000–3000ms | ~2× better |

#### 💡 Decision Logic

| Scenario | Recommendation |
|----------|----------------|
| Dev/test, low traffic, cost-sensitive | RTX-class (16–24GB) |
| Production chat, strict latency SLOs | A100 40GB |
| High-throughput batch or long-context | A100 80GB / H100 |

**Key insight**: A100 isn't just "faster" — it lets you push **much higher concurrency** while keeping latency acceptable. That's the real scaling win.

#### 📁 Artifacts (if run)
- `~/benchmarks/day003_chat_capacity_a100_40gb.csv` *(optional)*
- `~/benchmarks/day003_batch_capacity_a100_40gb.csv` *(optional)*
- `~/artifacts/day003_gpu_scaling_notes.md` *(optional)*

#### 🏆 Success Criteria
- [ ] Understand when RTX-class is sufficient vs when A100/H100 is needed
- [ ] (Optional) Run same scripts on A100/H100 and compare

---

## Tier 2 Summary

| Task | What You Did | Required? |
|------|--------------|-----------|
| **2.1** | Batch summarization workload on RTX | ✅ Core |
| **2.2** | GPU scaling anchor (RTX vs A100/H100) | ⚪ Optional |

### Key Comparisons Enabled

| Comparison | Data Source |
|------------|-------------|
| Chat vs Batch (RTX) | `day003_chat_capacity_rtx16gb.csv` + `day003_batch_capacity_rtx16gb.csv` |
| RTX vs A100 | Measured (if run) or estimated from public benchmarks |

### Artifacts
```
~/data/
└── day003_docs_sample.txt

~/scripts/benchmarks/
├── vllm_batch_summarize_bench.py
└── run_batch_capacity_grid.sh

~/benchmarks/
├── day003_batch_capacity_rtx16gb.csv
├── day003_batch_c16_rtx16gb.json
├── day003_batch_c32_rtx16gb.json
├── day003_chat_capacity_a100_40gb.csv   (optional, if run)
└── day003_batch_capacity_a100_40gb.csv  (optional, if run)

~/artifacts/
└── day003_gpu_scaling_notes.md          (optional)
```

---

**→ Continue to [Tier 3](LOG_tier03.md)**: Capacity frontier analysis + "Life of a request" documentation
