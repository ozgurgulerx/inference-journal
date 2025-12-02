# Day 002 – GPU Node Bring-Up on RunPod
## Tier 3: vLLM Runtime Tuning & Serving Parameters

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) and [Tier 2](LOG_tier02.md) first  
> **Goal**: Understand how vLLM serving parameters change behavior on a single 16GB GPU  
> **End State**: A small but real performance profile + one baseline "prod" config you trust  
> **Time**: ~2 hours

---

## 🎯 What You'll Learn

How vLLM partitions VRAM between:
- Model weights
- KV cache
- Compiled kernels / CUDA graphs

How these knobs interact:
- `--gpu-memory-utilization`
- `--max-model-len`
- `--max-num-seqs`
- `--dtype` (implicitly bfloat16)
- `--enforce-eager` (for debugging)

How to interpret vLLM's startup logs:
- KV cache size
- Maximum concurrency
- CUDA graph compile timings

---

## 🔑 Key vLLM Serving Parameters

| Parameter | What It Controls | Default |
|-----------|------------------|---------|
| `--gpu-memory-utilization` | Fraction of VRAM for KV cache (0.0–1.0) | 0.9 |
| `--max-model-len` | Maximum sequence length (tokens) | Model's max |
| `--max-num-seqs` | Maximum concurrent sequences | 256 |
| `--enforce-eager` | Disable CUDA graphs (debug mode) | False |
| `--disable-log-requests` | Suppress per-request logging | False |

---

## 📘 Key Concepts Explained

<details>
<summary><strong>Click to expand deep technical overview</strong></summary>

### 1. KV Cache (Key–Value Cache)

**What it is:**  
During generation, transformer models compute attention over past tokens. To avoid recomputing history every step, vLLM stores:
- **K** (Key tensor)
- **V** (Value tensor)

…for every past token in GPU memory.

**Why it matters:**  
KV cache is the **#1 factor** that determines:
- How many concurrent requests the GPU can handle
- Maximum context length
- Memory pressure
- Throughput

**vLLM's role:**  
vLLM implements **PagedAttention**, which:
- Stores KV cache in a paged memory structure
- Reduces fragmentation
- Allows dynamic batching
- Scales concurrency far better than HuggingFace

**Memory cost:**  
For Qwen2.5-1.5B, a single token KV entry is ~80–120 KB. Thus:
```
KV cache memory ≈ tokens_in_context × KV_per_token
```

---

### 2. `gpu-memory-utilization`

**What it is:**  
Parameter that controls how much VRAM vLLM is allowed to use.

```bash
--gpu-memory-utilization 0.8
```

vLLM will use ~80% of available VRAM for:
- Model weights
- KV cache
- CUDA graphs
- Compilation artifacts

| Value | Use Case |
|-------|----------|
| 0.3–0.5 | Debugging, multi-process |
| 0.6–0.8 | Balanced production |
| 0.9–0.95 | Maximum throughput |

**Higher value = more KV cache = more concurrency.**

---

### 3. `max-model-len`

**What it is:**  
The maximum context length (max tokens per request).

```bash
--max-model-len 4096
```

**Why it matters:**  
Longer max context = more KV memory per request.

| Short max-model-len | Long max-model-len |
|---------------------|-------------------|
| ✅ High concurrency | ❌ Low concurrency |
| ✅ Low memory use | ❌ High memory use |
| ✅ Fast batching | ❌ Harder batching |

**You are trading context length for concurrency.**

---

### 4. `max-num-seqs`

**What it is:**  
The maximum number of sequences vLLM is allowed to handle concurrently.

```bash
--max-num-seqs 16
```

This is an upper bound on concurrency. Actual concurrency is limited by:
```
min(max-num-seqs, KV_capacity)
```

**Increase when:** You want more throughput and have KV cache headroom.  
**Decrease when:** You need low, predictable latency.

---

### 5. Continuous Batching (vLLM Superpower)

**Traditional inference** (HuggingFace):
```
req1 → finish → req2 → finish → req3 → finish
```

**vLLM's continuous batching:**
```
Batch(t) = all requests currently active → process together
```

This allows:
- Higher GPU utilization
- Massive throughput gains
- Lower cost per request

If 8 users request simultaneously, vLLM merges them:
```
[8 prompts] → (one batched prefill)
[8 decodes] → (minimized overhead per token)
```

**This is why your concurrency test shows: latency increases slightly, but throughput increases massively.**

---

### 6. Prefill Phase vs Decode Phase

In autoregressive LLMs, inference has **2 phases**:

**1️⃣ Prefill (Prompt Processing)**
- Model processes the whole prompt (e.g., 2000 tokens)
- Heavy: O(N²) attention if not optimized
- vLLM speeds this up via FlashAttention, chunked prefill, CUDA graphs

**2️⃣ Decode (Token-by-Token Generation)**
- LLM generates one token at a time
- Cheap: dominated by KV lookups + small matmuls
- vLLM optimizes with KV reuse, fused kernels, continuous batching

---

### 7. Chunked Prefill

You saw in logs:
```
Chunked prefill is enabled with max_num_batched_tokens=2048.
```

**What it means:**
- vLLM splits very long prompts into smaller chunks
- Reduces GPU memory spikes
- Allows batching multiple users even if some have long prompts
- Improves stability on small GPUs (16GB, 12GB, 8GB)

Without chunked prefill, a single long prompt can block the entire batch.

---

### 8. FlashAttention

You saw in logs:
```
Using FLASH_ATTN backend.
```

**FlashAttention:**
- Fuses multiple attention ops (Softmax, matmul)
- Computes attention without storing full attention matrix
- Reduces memory bandwidth cost
- Provides large speed-ups for long-context prompts

This is foundational for high throughput and fast prefill.

---

### 9. CUDA Graphs

You saw:
```
Capturing CUDA graphs...
Graph capturing finished in 4 secs
```

**CUDA graphs:**
- Record GPU execution ("trace" the ops once)
- Replay that graph with near-zero CPU overhead
- Remove dynamic Python overhead
- Give predictable, fast, low-latency inference

vLLM uses 3 types of CUDA graphs:
1. Prefill graph
2. Decode graph
3. Mixed prefill-decode graphs

**This is what makes vLLM much faster than direct PyTorch code.**

---

### 10. TTFT (Time to First Token)

For a user, this is the feeling of **"responsiveness"**.

| Mode | TTFT Definition |
|------|-----------------|
| Non-streaming | Time until entire completion ready |
| Streaming | Time until first token arrives (~50–200ms) |

vLLM makes TTFT faster via:
- CUDA graph replay
- Chunked prefill
- Optimized attention kernels
- Continuous batching

---

### 11. Token Throughput (tok/s)

This is the real metric for batch workloads:
```
throughput = tokens_generated ÷ time
```

vLLM can reach **tens of thousands of tokens/second** on large GPUs due to:
- Continuous batching
- Paged KV cache
- CUDA graph replay
- FlashAttention

Your Tier 3 testing shows:
- Single-user throughput (baseline)
- Multi-user throughput (continuous batching gain)

---

### 12. BF16 vs FP16

You saw:
```
dtype=torch.bfloat16
```

vLLM auto-selects dtype based on model & GPU support.

**BF16 advantages:**
- Wider exponent range
- Safer for long-context math
- Same memory footprint as FP16
- Often yields better model quality

If BF16 wasn't supported, vLLM would fall back to FP16.

---

### 13. KV Cache Tokens vs VRAM

When vLLM logs:
```
GPU KV cache size: 188,512 tokens
```

This means your GPU can hold ~188K past tokens in KV cache at once.

This number depends on:
- GPU VRAM
- `gpu-memory-utilization`
- Model hidden sizes
- dtype (BF16 = larger KV entries than INT4)

---

### 14. Maximum Concurrency

Logged as:
```
Maximum concurrency for 32,768 tokens per request: 5.75x
```

**Meaning:**  
If every user sent a full-length 32K context, the GPU can serve ~5–6 users at once.

If you reduce `--max-model-len`, this grows quickly:
- 32K tokens → ~6 concurrent
- 4K tokens → ~46 concurrent

</details>

---

## Tier 3 Tasks (~2 hours)

---

### ✅ Task 3.1: Capture Baseline vLLM Runtime Settings
**Tags**: `[vLLM]` `[Introspection]`  
**Time**: 15 min  
**Win**: You know exactly what your current serving config does

#### 🔧 Lab Instructions

Restart vLLM cleanly to re-capture logs:

```bash
pkill -f "vllm serve" || true
sleep 3

export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.6 2>&1 | tee ~/artifacts/tier03_baseline.log
```

Wait for startup to complete, then extract these values from the log:

```
dtype=torch.bfloat16
max_seq_len=32768
Available KV cache memory: X GiB
GPU KV cache size: Y tokens
Maximum concurrency for 32,768 tokens per request: Zx
```

<details>
<summary><strong>Sample output (16GB GPU, RTX 2000 Ada)</strong></summary>

```
INFO [...] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO [...] vLLM API server version 0.11.2
INFO [...] Resolved architecture: Qwen2ForCausalLM
INFO [...] Using max model len 32768
INFO [...] dtype=torch.bfloat16, max_seq_len=32768
INFO [...] Using FLASH_ATTN backend.
INFO [...] Loading weights took 5.46 seconds
INFO [...] Model loading took 2.8871 GiB memory and 6.44 seconds
INFO [...] torch.compile takes 8.29 s in total
INFO [...] Available KV cache memory: 5.03 GiB
INFO [...] GPU KV cache size: 188,512 tokens
INFO [...] Maximum concurrency for 32,768 tokens per request: 5.75x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 51/51
Capturing CUDA graphs (decode, FULL): 100%|██████████| 35/35
INFO [...] Graph capturing finished in 4 secs, took 0.49 GiB
INFO [...] init engine took 14.77 seconds
INFO [...] Starting vLLM API server on http://0.0.0.0:8000
```

**Key values to note:**
| Metric | Value |
|--------|-------|
| Model weights | 2.89 GiB |
| KV cache memory | 5.03 GiB |
| KV cache capacity | 188,512 tokens |
| Max concurrency @ 32k | 5.75x |
| CUDA graphs | 0.49 GiB |
| Total startup time | ~15 seconds |

</details>

<details>
<summary><strong>🔍 Understanding Each Value (Deep Dive)</strong></summary>

#### 1. `dtype=torch.bfloat16` (BF16 Precision)

**What it means:**  
`dtype` is the numeric precision used for model weights and activations during inference.

**BF16 = Brain Floating Point 16-bit**, invented by Google for TPUs, now standard on NVIDIA GPUs.

| Property | BF16 | FP16 |
|----------|------|------|
| Bits | 16 | 16 |
| Exponent bits | 8 | 5 |
| Memory | Same | Same |
| Numerical stability | ✅ Higher | ⚠️ Lower |

**Why vLLM uses BF16:**
- Same memory footprint as FP16
- Much larger exponent range → more stable
- Required for long-context models (>8k, >32k) to avoid degradation
- Same speed as FP16 on tensor cores

**Mental model:** BF16 = FP16 performance + FP32 stability.

---

#### 2. `max_seq_len=32768` (Max Context Length)

**What it means:**  
Maximum number of tokens any single request can contain (prompt + generated tokens).

**Where it comes from:**  
vLLM auto-infers this from the model's HuggingFace config. Qwen2.5-1.5B supports 32k context.

**Why it matters:**  
KV cache cost scales linearly with `max_seq_len`:
```
KV memory per token ≈ 2 × hidden_size × num_heads × dtype_size
```

| max_seq_len | Effect |
|-------------|--------|
| 32768 | ~6 concurrent full-context users |
| 4096 | ~46 concurrent users |
| 2048 | Ultra-fast, hundreds of req/s |

**Mental model:** `max_seq_len` is the height of each request slot in GPU memory.

---

#### 3. `Available KV cache memory: X GiB`

**What it means:**  
After accounting for model weights, CUDA graphs, and overhead, this is how much VRAM remains for KV cache.

**How it's computed (16GB GPU example):**
```
Total VRAM:                    16.0 GB
- Model weights (BF16):        ~2.9 GB
- CUDA graphs + compile:       ~0.5 GB
- PyTorch temp + reserved:     ~1.4 GB
= Available for vLLM:          ~11.2 GB
× gpu-memory-utilization 0.6:  ~6.7 GB theoretical
= Usable after fragmentation:  ~5.0 GB
```

**Why it's critical:**  
This is your "budget" for concurrent requests. More KV cache = more users.

---

#### 4. `GPU KV cache size: Y tokens`

**What it means:**  
Total number of tokens the GPU can hold in KV cache at once.

**How it's calculated:**
```
KV cache size (tokens) = available_KV_memory_bytes ÷ KV_per_token_bytes
```

For Qwen2.5-1.5B, each token's KV entry is ~80–120 KB.

**Example:**  
5.03 GiB ÷ ~28 bytes per token ≈ 188,512 tokens

**Mental model:** This is the "total seats" on your GPU.

---

#### 5. `Maximum concurrency for 32,768 tokens per request: Zx`

**What it means:**  
If every request used the full 32k context, how many parallel requests can the GPU support?

**Calculation:**
```
max_concurrency = KV_cache_tokens ÷ max_seq_len
188,512 ÷ 32,768 ≈ 5.75x
```

**Why this matters:**  
This is the true upper bound for concurrency.

| max_seq_len | Concurrency |
|-------------|-------------|
| 32768 | ~6 users |
| 4096 | ~46 users |
| 2048 | ~92 users |

**Mental model:** `max_concurrency = KV_capacity ÷ request_KV_usage`

---

#### Summary Table

| Concept | Meaning | Why It Matters |
|---------|---------|----------------|
| `dtype=bfloat16` | Numerical format | Stable long-context, fast tensor cores |
| `max_seq_len=32768` | Max tokens per request | Controls memory per request |
| `Available KV cache` | VRAM for KV cache | Determines parallel capacity |
| `GPU KV cache size` | Token capacity | How many tokens in-flight |
| `Max concurrency` | Full-context requests | Throughput & user capacity |

</details>

Create a notes file:

```bash
cat > ~/artifacts/tier03_notes.md << 'EOF'
# Tier 3 – Baseline vLLM Settings (Qwen2.5-1.5B)

## Baseline Config
- gpu-memory-utilization: 0.6
- dtype: bfloat16
- max_seq_len: 32768

## From Logs
- KV cache memory: [X] GiB
- KV cache capacity: [Y] tokens
- Max concurrency @ 32k tokens: [Z]x

## VRAM Partitioning
GPU VRAM splits roughly into:
1. Model weights (~2.9 GB)
2. KV cache (controlled by gpu-memory-utilization)
3. Compiled kernels / CUDA graphs (~0.5 GB)
4. Overhead / fragmentation
EOF
```

> 💡 **Why this matters:** `gpu-memory-utilization` directly controls how much VRAM goes to KV cache → more KV cache = more concurrent long contexts = more throughput.

#### 🏆 Success Criteria
- [ ] Baseline log captured
- [ ] KV cache size and max concurrency noted
- [ ] Notes file created

---

### ✅ Task 3.2: Systematic `--gpu-memory-utilization` Sweep
**Tags**: `[Memory]` `[Capacity Planning]`  
**Time**: 30 min  
**Win**: Understand how VRAM usage and KV capacity scale

#### 📖 What You're Testing

| Setting | Expected Behavior |
|---------|-------------------|
| 0.3 | Low KV cache, good for debugging |
| 0.6 | Balanced (your baseline) |
| 0.9 | Maximum KV cache, production use |

#### 🔧 Lab Instructions

Create a helper script:

```bash
mkdir -p ~/scripts ~/artifacts/tier03-util

cat > ~/scripts/vllm_with_util.sh << 'EOF'
#!/bin/bash
UTIL="$1"
if [ -z "$UTIL" ]; then
  echo "Usage: vllm_with_util.sh <gpu-memory-utilization>"
  exit 1
fi

pkill -f "vllm serve" || true
sleep 3

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Starting vLLM with gpu-memory-utilization=${UTIL} ==="

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization "${UTIL}" 2>&1 | tee ~/artifacts/tier03-util/vllm_util_${UTIL}.log &
PID=$!

# Wait for engine to initialize
sleep 60

echo "=== nvidia-smi for util=${UTIL} ===" | tee ~/artifacts/tier03-util/nvidia_util_${UTIL}.txt
nvidia-smi | tee -a ~/artifacts/tier03-util/nvidia_util_${UTIL}.txt

echo "=== Sending test request ===" 
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"util test","max_tokens":10}' > /dev/null

echo "=== Done for util=${UTIL} (PID=${PID}) ==="
EOF

chmod +x ~/scripts/vllm_with_util.sh
```

Run the sweep (one at a time):

```bash
~/scripts/vllm_with_util.sh 0.3
# Wait, observe, then Ctrl+C or pkill

~/scripts/vllm_with_util.sh 0.6
# Wait, observe, then Ctrl+C or pkill

~/scripts/vllm_with_util.sh 0.9
# Wait, observe, then Ctrl+C or pkill
```

For each run, extract from the log:
- `Available KV cache memory`
- `GPU KV cache size`
- `Maximum concurrency for 32,768 tokens per request`

Fill in this table:

| gpu-memory-utilization | KV cache GiB | KV tokens | Max concurrency (32k) |
|------------------------|-------------:|----------:|----------------------:|
| 0.3                    |      ? GiB   |     ?     |          ?x           |
| 0.6                    |      ? GiB   |     ?     |          ?x           |
| 0.9                    |      ? GiB   |     ?     |          ?x           |

> 🧠 **Key Insight:** Higher `gpu-memory-utilization` = more VRAM for KV cache = more concurrent long contexts. On a 16GB card, 0.8–0.9 is ideal for production; 0.3–0.5 is for debugging.

#### 🏆 Success Criteria
- [ ] All three configs tested
- [ ] Table filled with actual values
- [ ] Understand the tradeoff

---

### ✅ Task 3.3: Interplay of `--max-model-len` & KV Cache
**Tags**: `[Context Length]` `[KV Cache]`  
**Time**: 25 min  
**Win**: Understand why long context is expensive and how to trade it off

#### 📖 The Math

From your baseline logs:
```
GPU KV cache size: 188,512 tokens
Maximum concurrency for 32,768 tokens per request: 5.75x
```

This means:
- Total KV capacity: ~188K tokens
- If each request uses 32K tokens → ~5–6 concurrent full-length requests
- If each request uses 4K tokens → many more concurrent requests!

#### 🔧 Lab Instructions

Test with reduced `--max-model-len`:

```bash
pkill -f "vllm serve" || true
sleep 3

export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 2>&1 | tee ~/artifacts/tier03-maxlen-4096.log &

sleep 60
```

From the log, grab:
- `GPU KV cache size`
- `Maximum concurrency for 4,096 tokens per request`

Compare:

| Config | max-model-len | Max Concurrency |
|--------|---------------|-----------------|
| Baseline | 32768 | ~5–6x |
| Reduced | 4096 | ??x |

> 🧠 **Key Insight:** KV cache capacity (in tokens) is roughly fixed by VRAM. Shorter max sequence length = more concurrent sequences. For chatbots with typical 2–4K contexts, setting `--max-model-len 4096` significantly increases concurrency. **Context length is a resource.**

#### 🏆 Success Criteria
- [ ] Reduced max-model-len config tested
- [ ] Observed increased concurrency
- [ ] Understand the tradeoff

---

### ✅ Task 3.4: Latency & Streaming Check with Tuned Settings
**Tags**: `[Latency]` `[Streaming]`  
**Time**: 20 min  
**Win**: See how your tuned config feels under load

#### 🔧 Lab Instructions

Start with tuned config:

```bash
pkill -f "vllm serve" || true
sleep 3

export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --max-num-seqs 16 2>&1 | tee ~/artifacts/tier03-tuned.log &

sleep 60
```

**Test 1: Non-streaming latency**

```bash
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-1.5B-Instruct",
    "prompt":"Explain what KV cache is in LLM inference:",
    "max_tokens":128
  }' > /tmp/nonstream.json

cat /tmp/nonstream.json | python3 -m json.tool
```

Record the `real` time from `time`.

**Test 2: Streaming latency**

```bash
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-1.5B-Instruct",
    "prompt":"Explain what KV cache is in LLM inference:",
    "max_tokens":128,
    "stream": true
  }'
```

You'll see tokens arriving chunk-by-chunk.

> 🧠 **Feel the difference:**
> - Non-streaming: One big JSON at the end
> - Streaming: Much better perceived latency (tokens appear immediately)

#### 🏆 Success Criteria
- [ ] Both tests completed
- [ ] Understand streaming vs non-streaming UX difference

---

### ✅ Task 3.5: Micro-Concurrency Test
**Tags**: `[Concurrency]` `[Continuous Batching]`  
**Time**: 20 min  
**Win**: Observe continuous batching behavior with real numbers

#### 🔧 Lab Instructions

Create a concurrency test script:

```bash
cat > ~/scripts/tier03_concurrency_test.py << 'EOF'
import requests
import concurrent.futures
import time
import statistics

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def one_request(i: int) -> float:
    start = time.time()
    r = requests.post(URL, json={
        "model": MODEL,
        "prompt": f"Concurrency test request {i}: explain briefly.",
        "max_tokens": 64
    })
    _ = r.json()
    return time.time() - start

def run_concurrency_test(num_clients: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as ex:
        latencies = list(ex.map(one_request, range(num_clients)))
    return {
        "clients": num_clients,
        "mean_latency": statistics.mean(latencies),
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0],
        "min_latency": min(latencies),
        "max_latency": max(latencies),
    }

if __name__ == "__main__":
    print("=" * 60)
    print("CONCURRENCY TEST")
    print("=" * 60)
    for clients in [1, 4, 8, 16]:
        res = run_concurrency_test(clients)
        print(f"{clients:2d} clients → "
              f"mean={res['mean_latency']:.2f}s, "
              f"p95={res['p95_latency']:.2f}s, "
              f"min={res['min_latency']:.2f}s, "
              f"max={res['max_latency']:.2f}s")
EOF
```

Run it:

```bash
python3 ~/scripts/tier03_concurrency_test.py | tee ~/artifacts/tier03_concurrency.txt
```

> 🧠 **What to look for:**
> - 1 client → baseline single-request latency
> - 4 clients → small latency increase, good throughput
> - 8/16 clients → vLLM batching kicks in; latency increases but total work done is much higher

This gives you your first concrete feeling of "how many users can this GPU handle concurrently?"

#### 🏆 Success Criteria
- [ ] Concurrency test completed
- [ ] Results saved
- [ ] Understand batching behavior

---

### ✅ Task 3.6: Create Baseline Prod Config Script
**Tags**: `[Runtime]` `[Ops]`  
**Time**: 10 min  
**Win**: A single command you trust as your default SLM server

#### 🔧 Lab Instructions

```bash
mkdir -p ~/configs

cat > ~/configs/qwen2p5_slm_baseline.sh << 'EOF'
#!/bin/bash
# Qwen2.5-1.5B Instruct – Baseline Serving Config (16GB GPU)
# Optimized for: Chat/interactive use with moderate concurrency

export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --disable-log-requests
EOF

chmod +x ~/configs/qwen2p5_slm_baseline.sh
```

Now your "prod-ish" SLM server is literally:

```bash
~/configs/qwen2p5_slm_baseline.sh
```

**This is exactly how a real inference engineer works.**

#### 🏆 Success Criteria
- [ ] Config script created
- [ ] Script is executable
- [ ] You understand every flag

---

## Tier 3 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **3.1** | Capture baseline settings | ⬜ |
| **3.2** | gpu-memory-utilization sweep | ⬜ |
| **3.3** | max-model-len vs KV cache | ⬜ |
| **3.4** | Latency & streaming check | ⬜ |
| **3.5** | Concurrency test | ⬜ |
| **3.6** | Create prod config | ⬜ |

### Artifacts Created
```
~/artifacts/
├── tier03_baseline.log
├── tier03_notes.md
├── tier03-util/
│   ├── vllm_util_0.3.log
│   ├── vllm_util_0.6.log
│   └── vllm_util_0.9.log
├── tier03-maxlen-4096.log
├── tier03-tuned.log
└── tier03_concurrency.txt

~/configs/
└── qwen2p5_slm_baseline.sh
```

---

## 🎉 Tier 3 Complete!

By finishing this 2-hour block, you've:

- ✅ Mapped how `gpu-memory-utilization` changes KV cache size & concurrency
- ✅ Seen how `max-model-len` is a capacity lever
- ✅ Felt the difference between streaming vs non-streaming
- ✅ Observed continuous batching with concurrent loads
- ✅ Written a baseline serving config script for Qwen2.5-1.5B on a 16GB GPU

**This is the foundation for:**
- **Tier 4**: Quantization (AWQ / GPTQ) for larger models
- **Larger models**: Qwen2.5-3B, Llama-3.x-8B
- **Multi-GPU / distributed** serving later

---

## 🔜 Next Step

When you're ready, continue to Tier 4:

**→ [LOG_tier04.md](LOG_tier04.md)** – Quantization & larger models
