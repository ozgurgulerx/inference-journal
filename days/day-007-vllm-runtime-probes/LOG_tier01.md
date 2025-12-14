# Day 007 – vLLM SLM: TTFT, Prefix Caching, KV Scaling
## Tier 1 – Baseline vLLM Probe + Cold/Warm TTFT

> **Goal**: Establish a repeatable baseline for **first-token latency (TTFT)** and end-to-end latency using a single SLM on vLLM.
> 
> **Outcome**: A minimal probe script + a short markdown note that captures cold vs warm request timings and your baseline server flags.

---

## Tier 1 – Must Do (Core Block)

**Title** – Baseline vLLM Probe + Cold/Warm TTFT  
**Time Budget** – ~60–90 min

---

### 0) Pick the SLM (freeze it for the whole day)

Choose one:

- `microsoft/Phi-3-mini-4k-instruct`  
- `Qwen/Qwen2.5-1.5B-Instruct`

Record it at the top of your notes (and reuse the same `MODEL` in every script).

---

### 1) Create a “known-good” vLLM launch command

Create a small launcher script:

- `days/day-007-vllm-runtime-probes/serve_slm_vllm.sh`

Command skeleton (keep it minimal; you can refine later):

```bash
#!/usr/bin/env bash
set -e

MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --port $PORT
```

Start the server:

```bash
chmod +x days/day-007-vllm-runtime-probes/serve_slm_vllm.sh
./days/day-007-vllm-runtime-probes/serve_slm_vllm.sh
```

**Log**: paste the exact command/flags into `first_token_latency.md`.

---

### 2) Write a tiny TTFT probe script (OpenAI-compatible request)

Create:

- `days/day-007-vllm-runtime-probes/ttft_probe.py`

Requirements:

- Measures total wall time.
- Parses response to extract tokens (roughly OK if you only get usage tokens).
- Keeps code small (≤ ~25 lines target).

Pseudo-structure:

- `t0 = time.time()`
- POST to `http://localhost:8000/v1/completions`
- `t1 = time.time()`
- Print `wall_s` and (if available) `usage.total_tokens`

Suggested request payload (keep constant):

- `max_tokens`: 64
- `temperature`: 0
- prompt: a short single-turn prompt

Example skeleton:

```python
#!/usr/bin/env python3
import json
import time

import requests


URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "Say hello from a small language model."


def main() -> None:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 64,
        "temperature": 0.0,
    }

    t0 = time.time()
    resp = requests.post(URL, json=payload, timeout=30)
    t1 = time.time()

    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "").strip()
    total_tokens = data.get("usage", {}).get("total_tokens")

    print(
        json.dumps(
            {
                "wall_s": t1 - t0,
                "total_tokens": total_tokens,
                "output_preview": text[:80],
            }
        )
    )


if __name__ == "__main__":
    main()
```

---

### 3) Measure cold vs warm request latency (TTFT proxy)

**Cold request** means: first request after a server restart.

Workflow:

1. Restart the server.
2. Run one probe request and record `cold_wall_s`.
3. Immediately run a second request and record `warm_wall_s`.

Record at least:

- `cold_wall_s`
- `warm_wall_s`
- server flags
- model name
- an approximate split between **TTFT** (time to first token) and **end-to-end latency**

Create:

- `days/day-007-vllm-runtime-probes/first_token_latency.md`

Template:

```markdown
# Day 007 – TTFT Baseline (SLM)

## Server Config
- MODEL=
- vLLM flags:
 - GPU=
 - OS / kernel=
 - vLLM version=

## Single-Run Measurements (Cold vs Warm)
| run   | ttft_s | e2e_s | tokens | notes |
|-------|--------|-------|--------|-------|
| cold_1|        |       |        | first request after server start |
| warm_1|        |       |        | immediate second request |

## Warm-Run Variance (steady state)
Prompt: "<short single-turn prompt>"

| run | ttft_s | e2e_s | tokens | notes |
|-----|--------|-------|--------|-------|
| 1   |        |       |        |       |
| 2   |        |       |        |       |
| 3   |        |       |        |       |
| 4   |        |       |        |       |
| 5   |        |       |        |       |

Summary:
- min_ttft_s =
- median_ttft_s =
- max_ttft_s =

## Sanity Metrics
- GPU memory before server start:
- GPU memory after cold request:
- GPU util during warm probes:
- CPU util during cold vs warm:

Practical way to capture these:

- **GPU memory before server start** – in a clean shell, run:

  ```bash
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  ```

  Record the value (in MiB) before you launch vLLM.

- **GPU memory after cold request** – after the first `cold_1` probe has completed, run the same `nvidia-smi` command and log the new `memory.used` value.

- **GPU util during warm probes** – while you run the 3–5 warm runs, in another shell run either:

  ```bash
  watch -n 1 nvidia-smi
  # or, for a short sample:
  nvidia-smi dmon -s puc -d 1 -c 10
  ```

  and jot down the typical GPU utilization and power usage range during warm requests.

- **CPU util during cold vs warm** – find the vLLM server PID (e.g. `ps aux | grep api_server`) and then:

  ```bash
  top -Hp <PID>
  # or, more structured:
  pidstat -h -p <PID> 1 10
  ```

  Run briefly during cold_1 and during your warm runs, and record approximate CPU utilization ranges.

## Observations
Use this section to **interpret** your numbers, not just restate them. Concretely:

- What dominated cold time? (weights load? graph warmup? KV alloc?)  
  - Based on logs and your sanity metrics, write 2–3 sentences guessing **where the time went**.  
  - Example: “Most of cold_1 seems to be disk + page cache fill (GPU util low, CPU busy, high wall_s), with a smaller bump from initial CUDA graph warmup.”

- Is warm stable across 3–5 runs?  
  - Note whether warm TTFT stays within a small band (e.g. ±10–20%) or if it **drifts**.  
  - Example: “Warm TTFT hovered around 80–90 ms with <10% spread, so I’d treat ~90 ms as my steady‑state TTFT for this SLM.”

- Any outliers or spikes? What coincided? (other jobs, clock changes, etc.)  
  - Call out any runs that were clearly slower/faster and tie them to observations: GPU util dip, other processes, clock changes, etc.  
  - Example: “Run 4 was 2× slower; `nvidia-smi dmon` showed another process briefly hitting the GPU, so I’d ignore that as noise, not a config issue.”

Cold-start stack note (page cache vs HBM):

**Page cache** vs **HBM load**:

Yes — that’s the right mental model, with two tweaks:

1. **Page cache** = *storage → CPU RAM* (kernel keeps file pages cached).
2. **HBM load** = *CPU RAM → GPU HBM* (CUDA/runtime does allocations + copies over PCIe/NVLink).

Two nuances that matter in practice:

- **They overlap sometimes**: frameworks can stream + prefetch, so disk reads and HBM copies can pipeline, but cold start still “feels” like two bumps.
- **Not always a full second bump**: if weights are **mmap’d** and copied lazily, or if you’re using **pinned memory**, or if the runtime does staging/quantization on CPU, the shape changes. But the two-layer idea holds.

A cheap diagnostic you’ll like:

- If during `cold_1` you see **high disk reads**, **Cached/buffers rising**, and **GPU util low early** → page cache fill dominated.
- If you see **GPU memory jump** + **PCIe TX/RX high** (or NVLink) → HBM transfer/alloc dominated.

## How I’d explain TTFT to an SRE
- What TTFT represents in user experience terms.
- Why cold vs warm matter for incident/debugging.
- Which knobs we can safely adjust (model size, `max-model-len`, batching policy) and what they trade off.
```

### Streams, Concurrency, and TTFT (Mental Model)

Short answer: **attention doesn’t mix, but the streams absolutely *do* impact each other.** Just not in the way people first think.

#### Setup (your terms, precise)

- **m streams** = m user prompts currently *active* in vLLM (running)
- **n concurrency** = how many the system *allows in-flight* (client + server admission)
- GPU executes **m sequences in parallel**
- **Attention is per-sequence only** ✅ (no cross-prompt leakage)

So far, all correct.

#### Where the intuition breaks

> “in theory these don’t really impact each other”

They **don’t interact semantically**, but they **absolutely interact physically and temporally**.

They compete on **shared resources**.

##### 1. What does *not* interact

Let’s be explicit:

- ❌ No attention mixing  
- ❌ No KV sharing  
- ❌ No prompt leakage  
- ❌ No cross-token influence  

Each stream has:

- its own KV pages  
- its own attention mask  
- its own token history  

So **model correctness is isolated**.

##### 2. What *does* interact (this is the important part)

**A) Scheduler time-slicing (TTFT impact)**  

- vLLM’s scheduler runs in *ticks*.  
- If `m = 4`, each stream gets serviced frequently; if `m = 64`, each stream waits longer between turns.  
- More streams ⇒ fewer scheduling turns per stream ⇒ **higher TTFT per stream**, even though attention is isolated. **Time is shared.**

**B) GPU compute saturation (throughput vs latency)**  

All streams share:

- SMs  
- memory bandwidth  
- tensor cores  
- kernel launch slots  

As `m` grows:

- tokens/sec ↑ (good)  
- per-request latency ↑ (bad)  

This is the classic **throughput–latency tradeoff**.

**C) KV cache pressure (capacity coupling)**  

Each stream consumes roughly:

```text
KV memory ≈ O(max-model-len × hidden_dim × layers)
```

As `m` increases:

- total KV footprint ↑  
- fewer new streams can be admitted  
- eviction / queuing increases  
- OOM risk rises  

So streams don’t mix — but they **crowd each other out**.

**D) Prefill contention (cold-path interaction)**  

If multiple streams arrive together:

- their **prefills are batched**  
- prefills are heavy (large matmuls)  
- later streams may wait behind earlier prefills  

Result: **TTFT inflation even before decode starts**.

##### 3. The correct mental model

> **Streams are semantically independent but operationally coupled through the scheduler, GPU compute, and KV memory budget.**

Or even sharper:

> **vLLM batches compute, not context — but latency is still a shared resource.**

##### 4. Why this matters operationally

This is why:

- “attention doesn’t mix” ≠ “requests don’t affect each other”  
- p95 TTFT explodes under load even though outputs are correct  
- tuning `max-num-seqs`, `max-model-len`, and client concurrency is mandatory

##### 5. Simple thought experiment

If streams truly didn’t impact each other:

- TTFT would be constant as `m → ∞`  
- GPU utilization wouldn’t matter  
- capacity planning would be trivial  

None of that is true in reality — which proves the coupling is **physical, not logical**.

**Crisp answer you can reuse:**

> They don’t impact each other *semantically*, but they strongly impact each other *through scheduling delay, compute contention, and KV memory pressure*, which directly affects TTFT, throughput, and capacity.

### Batching (What We Mean Precisely)

It is **not** the case that different Streaming Multiprocessors (SMs) are assigned to different user requests or different matrices. GPUs do **not** operate by mapping “one stream → one SM” or “one matrix → one SM.”

Instead, **batching works by *structurally enlarging the dimensions of a single logical computation***.

When multiple inference streams are active, the runtime **stacks token-level work from all streams along the batch dimension**, forming **one larger matrix operation** (for example, a larger GEMM or attention computation). This **single, batched operation** is then launched as **one GPU kernel**.

That kernel:

- is decomposed internally into **many small tiles** (blocks of the matrix),  
- each tile is scheduled dynamically by the GPU across **all available SMs**,  
- and all SMs cooperate to compute different parts of the *same* batched operation.

As a result:

- SMs are not tied to individual streams or matrices,  
- streams do not “own” compute units,  
- and parallelism emerges from **tiling a large batched matrix**, not from running many small matrices independently.

Batching therefore **increases the effective problem size** seen by the GPU, allowing the kernel to:

- launch enough warps to fully occupy SMs,  
- amortize kernel launch overhead,  
- improve memory coalescing,  
- and saturate tensor cores.

This is why batching is essential even on GPUs with many cores: **a single stream typically produces matrices that are too small to efficiently occupy the hardware**, especially during token-by-token decoding. By stacking tokens from multiple streams into one batched computation, the runtime transforms many small, inefficient operations into a single large, hardware-efficient one.

**Ultra-concise version (reuseable definition)**  
> **Batching does not assign streams to SMs; it enlarges the matrix dimensions so a single kernel can be tiled across all SMs, letting the entire GPU cooperatively execute one batched computation.**

---

### 4) Minimal sanity checks (optional but fast)

- Confirm server health:

```bash
curl -s http://localhost:8000/v1/models | head
```

- Confirm GPU memory jump on cold start (just note numbers):

```bash
nvidia-smi
```

---

### 5) Quick TTFT mental model (optional note-to-self)

After you’ve collected numbers, add a short paragraph to `first_token_latency.md` that **turns them into a simple mental model**, not just raw stats. Aim to cover:

- Rough **cold vs warm factor** (e.g. “cold is ~3× warm”) and why that’s acceptable or not.
- Your best guess at how much of cold time is:
  - **weights load / page cache fill** (disk + host RAM work, usually low GPU util),
  - **runtime warmup** (CUDA graphs, kernel caches, compilation/JIT),
  - **first prefill + scheduling** (actual model compute and queuing).
- One sentence you’d use to explain TTFT behavior to an SRE or product owner (“what to expect” after deploy / restart).

You can think of it as a rough equation:

- `TTFT_cold  ≈ load_weights + runtime_warmup + first_prefill + queueing`
- `TTFT_warm ≈ first_prefill + small_scheduling_overhead + queueing`

Use your measurements and sanity metrics to guess the relative sizes of these pieces.

Example note you could write (adapt numbers to your case):

> “On this node with Phi‑3 Mini, cold TTFT is ~2.8× warm (cold ≈ 1.2 s, warm ≈ 430 ms). Most of the extra cold time looks like storage + page cache fill and one‑time CUDA/graph warmup (GPU util is low until the tail of the first request, then normal afterwards). Once warm, TTFT is stable within ~10% at ~400–450 ms, so I’d treat 450 ms as my steady‑state TTFT budget for this SLM. Operationally, this means after a restart we should expect the first 1–2 requests to be slow, but everything after that should sit near the warm number unless the GPU is overloaded or batching is misconfigured.”

---

## Expected Artifact

- `days/day-007-vllm-runtime-probes/serve_slm_vllm.sh`
- `days/day-007-vllm-runtime-probes/ttft_probe.py`
- `days/day-007-vllm-runtime-probes/first_token_latency.md`

---

## What You Should Learn (Mental Models)

- Cold-start cost is a bundle: **weight loading + runtime warmup + initial allocations**.
- Warm-start behavior approximates the steady-state “interactive” experience.
- Your vLLM flags are not just “settings”; they are **capacity commitments** (especially `max-model-len` and `gpu-memory-utilization`).

### Check Your Understanding (Q&A + Pointers)

**Q1. Why is cold-start TTFT usually much larger than warm TTFT? What’s in the “bundle”?**  
**A:** Cold-start TTFT includes **loading model weights from disk into host/GPU memory**, **initializing the runtime** (CUDA graphs, kernels, JIT, allocator state), and **first-time KV / buffer allocations**. Warm TTFT skips most of this one-time work and mainly pays for prefill compute + scheduling. If cold and warm are similar, you likely have a systemic runtime or configuration issue; if cold is 2–5× warm, that’s normal for many setups.  
**Reading:**  
- vLLM blog/paper sections on startup and PagedAttention (KV allocations, runtime initialization).  
- Any “cold vs warm start” LLM serving doc (e.g., vLLM/TGI) that breaks down startup phases.

**Q2. What does “warm-start approximates interactive steady state” actually mean for SLOs?**  
**A:** Once the model is loaded and warmed, most user requests should see **warm TTFT**, not cold. That warm TTFT distribution (mean, p95, p99) is what you treat as your **steady-state latency SLO**. If warm TTFT is tight and stable, users experience a responsive system; cold TTFT mainly matters around deploys/restarts and should be bounded or hidden via pre‑warming.  
**Reading:**  
- SRE/serving guides that distinguish cold vs warm SLOs, and recommend pre‑warmed pools or background probes.

**Q3. How do `max-model-len` and `gpu-memory-utilization` act as capacity commitments, not just “settings”?**  
**A:** `max-model-len` determines how much **KV cache capacity per sequence** vLLM must be prepared to support; increasing it increases the **VRAM reserved per active request**. `gpu-memory-utilization` similarly defines how aggressively vLLM can consume GPU memory. Together, they set how many sequences can be batched concurrently before hitting memory limits. Higher values give more flexibility for long contexts but **reduce safe concurrency headroom** and can push you into OOM territory.  
**Reading:**  
- vLLM paper’s KV cache / PagedAttention evaluation (effect of context length on memory).  
- Day 007 Tier 3 (`LOG_tier03.md`) KV scaling notes and `kv_cache_scaling.csv`.

**Q4. How should you reason about TTFT vs throughput when tuning batching/concurrency?**  
**A:** Bigger batches and higher concurrency improve **tokens/sec** but add **queueing delay** before your request is scheduled, which directly increases TTFT. A good operating point is where warm TTFT still meets the SLO while GPUs are reasonably utilized (e.g., 50–80% under load). If TTFT is too high but GPU util is low, you’re probably batching too aggressively or allowing too much concurrency per GPU; if TTFT is fine but you’re compute‑bound, you can cautiously increase batch size/concurrency until TTFT/p95 starts to degrade.  
**Reading:**  
- vLLM docs/blog on continuous batching.  
- General LLM serving articles that frame batching as a TTFT vs throughput trade‑off.

**Q5. What should an SRE take away from these Tier 1 experiments?**  
**A:** They should leave with a simple mental model:  
- “Cold start is expensive because of weights + runtime warmup; we should hide or bound it.”  
- “Warm TTFT around **X ms** is our steady‑state expectation for this SLM and node.”  
- “Changing `max-model-len`, model size, or batching policy **moves us along a latency–throughput–capacity trade‑off curve**, not just a ‘performance’ dial.”  
This turns Tier 1 from a one‑off experiment into a reusable **playbook** for reasoning about TTFT and capacity on any similar node.  
**Reading:**  
- Day 006 README for OS‑level context (THP, hugepages, page cache, allocator) that feeds into cold vs warm behavior.  
- Day 007 README (this file) as the runtime‑level counterpart.

**Extra note on `max-model-len` vs model context length**  

Yes — with one important constraint:

- **`max-model-len` (vLLM)** = **server cap** on *prompt_tokens + generated_tokens* (total sequence length) for any request. It’s an admission-control + memory-budget knob.  
- **Model context length** = what the model is **actually capable of** (trained/implemented positional encoding limits).

So the “capacity commitment” framing is correct as long as:

- `max-model-len` **≤** the model’s real supported context length → you’re choosing to cap it lower for capacity/latency reasons.  
- If you set `max-model-len` **>** the model’s real limit, you’re not “unlocking more”; you’re in “may break / may need RoPE scaling / may be rejected” territory.

Why this matters: even if the model supports 32K context, setting `max-model-len=32K` forces vLLM to plan KV/cache capacity for that worst case, which can dramatically reduce safe concurrency on a given GPU. That’s exactly why Tier 1 + Tier 3 treat `max-model-len` as both a **functional** and a **capacity** knob.
