# Day 007 – vLLM SLM: TTFT, Prefix Caching, KV Scaling
## Tier 3 – KV Scaling + Micro-Batching Under Load (Capacity Intuition)

> **Goal**:
> - Empirically connect `max-model-len` to VRAM usage (KV cache capacity commitment).
> - Quantify the throughput difference between sequential vs concurrent load.
>
> **Outcome**: A small KV scaling CSV + a batching benchmark note with numbers.

---

## Tier 3 – Stretch (Optional / Ambitious)

**Title** – KV scaling and batching regimes for an SLM vLLM server  
**Time Budget** – ~90–150 min

---

### A) KV cache scaling vs `max-model-len` (~45–75 min)

#### 1) Implement a KV scaling runner

Create:

- `days/day-007-vllm-slm/kv_scaling.sh`

Requirements:

- Loop over a small list of lengths (start conservative): `512 1024 2048 4096`
- For each length:
  - start server
  - wait for ready
  - capture `nvidia-smi --query-gpu=memory.used`
  - write a row to CSV
  - stop server cleanly

CSV:

- `days/day-007-vllm-slm/kv_cache_scaling.csv`

Columns:

- `max_model_len,gpu_mem_used_mb,delta_mb_from_prev,bytes_per_token_est,notes`

Example skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000
LENS=("512" "1024" "2048" "4096")
CSV="kv_cache_scaling.csv"

echo "max_model_len,gpu_mem_used_mb,delta_mb_from_prev,bytes_per_token_est,notes" > "$CSV"

prev_mem=""
prev_len=""

for L in "${LENS[@]}"; do
  echo "[*] Testing max-model-len=$L"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype auto \
    --max-model-len "$L" \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" &

  PID=$!
  sleep 30  # crude wait; replace with readiness check if you have one

  mem_used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)

  delta_mb=""
  bytes_per_token=""
  if [[ -n "$prev_mem" ]]; then
    delta_mb=$((mem_used_mb - prev_mem))
    delta_tokens=$((L - prev_len))
    bytes_per_token=$(python - <<EOF
delta_mb = $delta_mb
delta_tokens = $delta_tokens
print(int(delta_mb * 1024 * 1024 / max(delta_tokens, 1)))
EOF
)
  fi

  echo "$L,$mem_used_mb,$delta_mb,$bytes_per_token," >> "$CSV"

  kill "$PID"
  wait "$PID" || true

  prev_mem=$mem_used_mb
  prev_len=$L
done
```

Tip:

- Include a small baseline row (e.g. with the smallest `max-model-len` you test) so `delta_mb_from_prev` and `bytes_per_token_est` are easy to compute.

#### 2) Interpret the curve

Create:

- `days/day-007-vllm-slm/kv_cache_scaling_notes.md`

Answer:

- Is it roughly linear?
- Are there step changes? (allocation granularity)
- Use the observed slope to back out an approximate **bytes per KV token**:
  - e.g. `bytes_per_token_est ≈ (delta_mb * 1024 * 1024) / delta_max_model_len`.
- Which `max-model-len` feels safe for this GPU *given you still want concurrency headroom*?
- Sketch a “headroom envelope”:
  - For this GPU, what `max-model-len` would you pick if you want ≈N concurrent 4K requests?
  - What trade-offs would you make for a latency-focused vs throughput-focused service?

What’s expected here (and why):

- **Linear vs stepwise behavior**  
  - Plot or mentally compare `gpu_mem_used_mb` vs `max_model_len`. If increases are roughly proportional (similar `delta_mb` per `delta_max_model_len`), the relationship is close to linear, which matches the theory that KV memory is ~O(sequence_length).  
  - If you see **step changes** (e.g. jumps at certain lengths), that hints at allocation granularity: the runtime may reserve KV in chunks (pages/blocks), so memory usage grows in “stairs” rather than a perfect line. Calling this out tells you how “smooth” your capacity trade‑off really is.

- **Bytes per KV token (slope) and why you care**  
  - The slope `bytes_per_token_est` turns your CSV into a usable rule of thumb: “Each extra token of `max-model-len` costs ~X bytes of VRAM per active sequence.”  
  - This matters because it lets you quickly evaluate product asks: if someone wants to double context length, you can estimate how many GiB that will cost across your typical concurrency and whether your GPUs can handle it.

- **Safe `max-model-len` and concurrency headroom**  
  - From the slope and total VRAM, you can budget how much memory to allocate to KV vs weights/runtime and then answer, “Given I want ≈N concurrent 4K‑token sequences, what `max-model-len` keeps us comfortably under VRAM limits?”  
  - The expectation is that you write this down explicitly in `kv_cache_scaling_notes.md`, e.g.: “On a 24 GiB card, with ~X GiB for weights/runtime, `max-model-len=4096` leaves enough KV headroom for ~N concurrent 4K requests; 8192 would squeeze concurrency too much.”

- **Headroom envelope and service trade-offs**  
  - A “headroom envelope” is your personal map of **(context length, concurrency)** pairs that fit within memory and latency budgets. For example:
    - Latency‑focused service: pick a **smaller `max-model-len`** (e.g. 4K), keep concurrency modest, prioritize keeping p95 low and avoiding OOM.  
    - Throughput‑focused service: accept either a larger `max-model-len` with lower concurrency, or keep `max-model-len` smaller but push concurrency higher, as long as p95 is acceptable.  
  - Writing this out forces you to reason like an SRE: not just “what’s technically possible,” but “what’s safe and aligned with the SLO for this GPU + SLM.”

---

### B) Micro-batching vs sequential throughput (~45–75 min)

#### 1) Write a tiny concurrency client

Create:

- `days/day-007-vllm-slm/batch_client.py`

Requirements:

- N requests total (e.g. 32)
- Run sequential and concurrent
- Print:
  - `sequential_s`
  - `concurrent_s`
  - `sequential_qps` / `concurrent_qps`
  - `sequential_tok_s` / `concurrent_tok_s` (if you can get token counts)

If you already have a client from earlier days, reuse it, but keep a Day 007 copy so this day is self-contained.

Example skeleton:

```python
#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def call_once(url: str, prompt: str, max_tokens: int) -> int:
  payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
  resp = requests.post(url, json=payload, timeout=60)
  resp.raise_for_status()
  data = resp.json()
  return data.get("usage", {}).get("total_tokens", 0)


def run_sequential(url: str, prompt: str, max_tokens: int, n: int) -> tuple[float, int]:
  t0 = time.time()
  tokens = 0
  for _ in range(n):
    tokens += call_once(url, prompt, max_tokens)
  t1 = time.time()
  return t1 - t0, tokens


def run_concurrent(url: str, prompt: str, max_tokens: int, n: int, concurrency: int) -> tuple[float, int]:
  t0 = time.time()
  tokens = 0
  with ThreadPoolExecutor(max_workers=concurrency) as ex:
    futures = [ex.submit(call_once, url, prompt, max_tokens) for _ in range(n)]
    for f in futures:
      tokens += f.result()
  t1 = time.time()
  return t1 - t0, tokens


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
  parser.add_argument("--n", type=int, default=32)
  parser.add_argument("--concurrency", type=int, default=8)
  parser.add_argument("--max-tokens", type=int, default=64)
  args = parser.parse_args()

  prompt = "Explain what a GPU is in one sentence."

  seq_s, seq_tokens = run_sequential(args.url, prompt, args.max_tokens, args.n)
  conc_s, conc_tokens = run_concurrent(args.url, prompt, args.max_tokens, args.n, args.concurrency)

  result = {
      "n": args.n,
      "concurrency": args.concurrency,
      "sequential_s": seq_s,
      "concurrent_s": conc_s,
      "sequential_qps": args.n / seq_s,
      "concurrent_qps": args.n / conc_s,
      "sequential_tok_s": seq_tokens / seq_s if seq_s > 0 else 0.0,
      "concurrent_tok_s": conc_tokens / conc_s if conc_s > 0 else 0.0,
  }
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
```

#### 2) Run the benchmark under a fixed server config

- Pick one `max-model-len` (e.g. 4096)
- Keep `max_tokens` constant (e.g. 64)
- Run 3 times and record the best/median (your call; just be consistent)

Create:

- `days/day-007-vllm-slm/batching_benchmark.md`

Include:

- numbers (sequential vs concurrent)
- a small table for a few concurrency levels:

```text
concurrency,mean_e2e_s,p95_e2e_s,qps,tok_s,gpu_util_pct,notes
1,,,,,,
4,,,,,,
8,,,,,,
16,,,,,,
```

- a short note on GPU utilization if you observed it (`nvidia-smi dmon`)
- 3–5 bullets that capture your **“rules of thumb”**:
  - e.g. “On this GPU + SLM, concurrency N–M is the sweet spot for 4K contexts.”
  - “Beyond concurrency K, p95 blows up without meaningful throughput gains.”

What’s expected here (and why):

- **Fixed server config**  
  - You fix `max-model-len`, `max_tokens`, and core vLLM flags so that **concurrency is the only major variable**. This isolates the effect of scheduling/batching, making your conclusions about concurrency regimes clean and reproducible. Changing multiple knobs at once would make it hard to attribute effects.

- **Multiple runs (best or median)**  
  - Running each configuration 3× and taking either the best or median filters out noise from transient system jitter (other processes, clock changes, etc.).  
  - The goal is not lab‑grade statistics, but a stable signal: enough to see how throughput and p95 evolve as you move from concurrency 1 → 4 → 8 → 16.

- **The table columns (theoretical meaning)**  
  - `mean_e2e_s` – average end‑to‑end latency per request at that concurrency. This reflects both compute and queueing; as concurrency increases, you expect it to **flatten or slowly rise** until you hit saturation.  
  - `p95_e2e_s` – tail latency. This is where queuing theory shows up: once the system is near capacity utilization, small fluctuations in arrival rate or service time cause p95 to grow rapidly.  
  - `qps` – queries per second. Roughly, `qps ≈ N / total_time` for N requests. Under ideal scaling, qps increases with concurrency until you reach the GPU’s capacity.  
  - `tok_s` – tokens per second (throughput). This is the main metric vendors quote; it’s driven by effective batch sizes and how busy the GPU is. You should see it ramp up with concurrency, then flatten.  
  - `gpu_util_pct` – rough indicator of how “full” the GPU is. Low concurrency with low util means you’re leaving performance on the table; very high util with exploding p95 means you’re over‑driving the system.  
  - `notes` – qualitative context (e.g., “util ~40–50%,” “p95 jumped 2× vs previous point,” “OOM at concurrency=32”).

- **Rules of thumb (capacity intuition)**  
  - From these numbers, you’re expected to infer:  
    - A **sweet-spot concurrency range** where qps/tok_s are high, GPU util is healthy, and p95 is still acceptable.  
    - A **knee of the curve** where increasing concurrency further mostly increases p95 and TTFT without meaningful throughput gains.  
  - Your 3–5 bullets in `batching_benchmark.md` should crystallize this, e.g.:  
    - “For this SLM on a 24 GiB GPU, concurrency 8–12 gives ~X tok/s at p95 ≤ Y ms; above 16, p95 doubles with only ~10% more throughput.”  
  - The theory behind this is basic queueing: as utilization approaches 100%, queueing delay dominates service time, so latency explodes while throughput saturates. Your benchmark is a concrete way to see that curve for this specific GPU + model.

---

### C) Optional: Concurrency sweep (find “knee of the curve”)

If time permits, extend `batch_client.py` to sweep concurrency levels:

- 1, 2, 4, 8, 16, 32

Write results into:

- `days/day-007-vllm-slm/concurrency_sweep.csv`

Key insight to capture:

- Where does throughput stop scaling?
- Where does latency blow up? (queuing)

What’s expected here (and why):

- **Purpose of the sweep**  
  - The sweep is meant to give you a **fuller picture** of how your system behaves from very low concurrency (1) up to “too high” (16–32). Instead of only seeing a couple of points (e.g. 1 and 8), you see the **shape of the curve** and can identify the knee where behavior changes qualitatively.

- **Throughput curve (tokens/sec vs concurrency)**  
  - In theory, as you increase concurrency from 1 upwards, tokens/sec should rise because continuous batching can form larger, more efficient batches.  
  - However, once the GPU is close to saturation, adding more concurrent requests doesn’t give the GPU more compute to do; it just **queues more work**. At that point, throughput **flattens**: you gain little or nothing in tok/s, even as concurrency increases.

- **Latency curve (p95 vs concurrency)**  
  - From basic queueing theory (e.g., M/M/1 intuition), as server utilization approaches 100%, **queueing delay grows non‑linearly**.  
  - That shows up as p95 (and p99) latency staying reasonable at low/moderate concurrency, then suddenly **bending upward**—the “latency blow‑up.”  
  - The sweep is intended to help you spot that bend: the concurrency range where adding more load mostly hurts latency, not throughput.

- **Using `concurrency_sweep.csv`**  
  - With rows like `concurrency,mean_e2e_s,p95_e2e_s,qps,tok_s,gpu_util_pct`, you can:
    - Plot tok/s vs concurrency and see where it starts to plateau.  
    - Plot p95 vs concurrency and see where it starts to curve sharply upward.  
  - The **knee of the curve** is roughly the region where:  
    - tok/s is near its maximum (within ~5–10% of peak), and  
    - p95 is still within your acceptable SLO.  
  - Your expectation is not to compute an exact mathematical knee, but to be able to say: “On this GPU + SLM, going past concurrency K mostly increases latency without buying us meaningful throughput.”

- **Why this matters operationally**  
  - This sweep turns vague advice like “don’t overload the GPU” into a concrete **capacity guideline**: a concurrency range you can recommend to SREs/product.  
  - It also gives you a basis for alerting: if observed concurrency or p95 drift beyond the sweet spot you measured, that’s a sign the system is operating outside its safe envelope (and either traffic or configuration should be adjusted).

---

## Expected Artifact

- `days/day-007-vllm-slm/kv_scaling.sh`
- `days/day-007-vllm-slm/kv_cache_scaling.csv`
- `days/day-007-vllm-slm/kv_cache_scaling_notes.md`
- `days/day-007-vllm-slm/batch_client.py`
- `days/day-007-vllm-slm/batching_benchmark.md`
- (optional) `days/day-007-vllm-slm/concurrency_sweep.csv`

---

## What You Should Learn (Mental Models)

- `max-model-len` is not “just a limit” — it is a **VRAM reservation decision**.
- Continuous batching is not optional; it’s how the runtime reaches its advertised throughput.
- You can reason about safe operating points by combining:
  - KV memory commitment
  - measured throughput under concurrency
  - acceptable p95 latency for your target workload

### Deeper Explanation

- **`max-model-len` as a VRAM reservation decision**  
  - For a given model, the KV cache footprint per active sequence scales roughly linearly with `max-model-len` (and number of heads/layers). Even with PagedAttention reducing fragmentation, the runtime must be prepared to hold KV for up to `max-model-len` tokens per sequence.  
  - That means choosing `max-model-len` is effectively deciding **how much VRAM you’re willing to commit per sequence**. Doubling `max-model-len` roughly doubles the worst-case KV memory per sequence, shrinking the number of concurrent sequences you can safely support before hitting memory limits.  
  - Practically: if your real workloads rarely exceed 4K tokens, setting `max-model-len=16K` wastes VRAM and hurts capacity; a smaller value gives you more concurrency headroom without affecting most users.

- **Continuous batching and advertised throughput**  
  - The impressive tokens/sec numbers in vLLM benchmarks assume the engine is doing **continuous batching**: new tokens from different requests are batched together at every decode step.  
  - Without enough concurrent work (or with overly strict latency constraints that prevent batching), the GPU runs under‑utilized, and you will not see the “headline” throughput.  
  - Your Tier 3 experiments should make this visible: sequential traffic yields low tok/s; adding concurrency and allowing batches to form increases tok/s dramatically until you hit a knee where queuing starts to dominate latency.

- **Safe operating points from KV + throughput + p95**  
  - KV scaling (`kv_cache_scaling.csv`) tells you how much VRAM is consumed as you increase `max-model-len`—this defines your **memory budget per sequence**.  
  - Throughput and latency from `batching_benchmark.md` show how concurrency affects **tok/s and p95 e2e latency**.  
  - Combining them lets you choose an operating point like: “On this GPU + SLM, with `max-model-len=4K`, concurrency 8–12 gives us ≥X tok/s and p95 ≤ Y ms, with Z GiB of VRAM headroom.” That’s what you’d surface to SREs/product as a recommended configuration.

### Check Your Understanding (Q&A)

**Q1. Why can’t you treat `max-model-len` as a purely “functional” parameter (just controlling how long prompts can be)?**  
**A:** Because it directly controls how much KV memory the runtime must provision per sequence. A higher `max-model-len` increases the worst-case KV footprint, which reduces the number of concurrent sequences you can host before hitting VRAM limits or forcing aggressive eviction. Even if most prompts are shorter, the runtime must plan for the worst case. So `max-model-len` is a capacity decision as much as a functionality decision.

**Q2. How would you use `kv_cache_scaling.csv` to choose a reasonable `max-model-len` for a production service?**  
**A:** From `kv_cache_scaling.csv`, you estimate how many MB of GPU memory each additional 1K tokens of `max-model-len` costs (bytes per KV token). Knowing your GPU VRAM and non‑KV overhead (model weights, runtime buffers), you can budget how much memory to allocate to KV and back out a `max-model-len` that still leaves headroom for your target concurrency. For example: “On a 24 GiB GPU, with ~X GiB used by weights/runtime, we can afford Y GiB for KV, which maps to ~4K tokens at our desired concurrency.”

**Q3. Why does throughput often increase sharply when you move from sequential to moderate concurrency, then flatten or even hurt latency at higher concurrency?**  
**A:** At low concurrency, the GPU is under‑utilized; continuous batching cannot form large batches, so each decode step runs with small effective batch size, limiting tok/s. As you add concurrency, batches become larger, improving GPU utilization and throughput. Beyond a certain point, however, queueing delay and scheduling overhead grow faster than the benefit of larger batches: p95–p99 latency blows up, and tok/s may flatten or even drop due to contention and context switching.

**Q4. How would you define a “safe operating point” for this SLM and GPU based on your Tier 3 data?**  
**A:** A safe operating point is a (server config, concurrency) pair where:  
  - KV memory usage at the chosen `max-model-len` stays within a comfortable VRAM budget (with some headroom).  
  - Throughput (tok/s or QPS) is high enough for your load.  
  - p95 (and ideally p99) latency stays within your SLO.  
  - Concurrency is below the knee where adding more requests mostly increases queuing, not throughput.  
Your Tier 3 tables should let you pick and justify such a point, e.g. “`max-model-len=4096`, concurrency 8–12.”

**Q5. How would you explain to an SRE why “just turning up concurrency” is not a free way to get more throughput?**  
**A:** Increasing concurrency increases the pool of work the scheduler can batch, which helps **until** the GPU is saturated. Past that point, new requests mostly wait in queues, increasing TTFT and p95/p99 without proportionally increasing tok/s. Moreover, higher concurrency with a large `max-model-len` increases KV memory pressure, risking OOMs or forced evictions. So concurrency must be tuned together with `max-model-len` and latency SLOs, not treated as a free dial.

### Advanced Scenarios (Q&A)

**Q6. Product asks to double context length (4K → 8K) but keep the same concurrency and latency SLO. How do you evaluate if this is feasible on your current GPU?**  
**A:** First, use `kv_cache_scaling.csv` to estimate how much extra KV memory 8K will cost. If 4K already consumes most of the KV budget (e.g., KV + weights + runtime ≈ 90% of VRAM), doubling to 8K will likely exceed VRAM or force you to reduce concurrency. Then, use your bytes‑per‑token slope to quantify the extra memory and compute: “At 8K, with concurrency N, total KV ≈ X GiB; is that under the card’s capacity with headroom?” If not, the only options are: (a) reduce concurrency, (b) move to a larger GPU, or (c) accept a smaller effective context in practice. You should explain to product that context length and concurrency trade off through KV memory; you can’t freely double one while holding the others fixed.

**Q7. Your batching benchmark shows that from concurrency 8 → 16 you gain ~20% tok/s but p95 latency doubles. How do you decide if running at 16 is acceptable?**  
**A:** You compare the new p95 to your latency SLO and to user expectations. If p95 at concurrency 16 is still within SLO (and TTFT remains reasonable), a 20% throughput gain might be worth it. If p95 crosses the SLO or makes interactive use cases feel sluggish, you treat 8–12 as the safe range and use 16 only for batch/offline workloads. The key is that the decision isn’t purely “more tok/s is better”; it’s “more tok/s at what cost to tail latency and user experience?” Your benchmark gives you the data to make that trade‑off explicit.

**Q8. You notice that at concurrency 1–2, GPU utilization is ~25–30%, and tok/s is far below vendor benchmarks. What’s the underlying reason, and what do you tell an SRE?**  
**A:** The GPU is under‑fed: with only 1–2 concurrent sequences, continuous batching forms tiny effective batches per decode step, so the large matrix units are idle most of the time. Vendor benchmarks assume a high level of concurrency and continuous batching, which keeps the GPU busy. You’d explain that to reach “headline” tok/s you need to run with enough concurrent sequences (or aggregate requests across users) so that the scheduler can build larger batches; otherwise, under‑utilization is expected, not a bug.

**Q9. In production, p95 latency starts creeping up even though QPS hasn’t changed much. How could your Tier 3 data help you debug this?**  
**A:** You can compare current observed concurrency and p95 to the ranges you profiled. If production is now operating closer to or beyond the concurrency range where your sweep showed the “knee of the curve,” the creeping p95 may simply reflect higher effective concurrency (e.g., more long‑running requests, skewed traffic shape). Alternatively, if concurrency is unchanged but p95 has grown, you might suspect increased average prompt length (effective `max-model-len` usage), causing more KV and compute per request. Tier 3 gives you a baseline: “At concurrency N and typical prompt sizes, p95 used to be Y ms; now we’re outside that envelope, so either concurrency or context length has drifted.”

**Q10. You have two services: one interactive chat app and one offline batch summarization job. How would you configure `max-model-len` and concurrency differently for each, using your Tier 3 insights?**  
**A:** For the interactive chat app, you prioritize **latency and stability**: pick a `max-model-len` that covers typical chat histories (e.g. 4K) without over‑committing KV, and choose a concurrency in the sweet‑spot region where p95 and TTFT stay low (e.g. 8–12), even if that leaves some throughput on the table. For the offline batch job, you can tolerate higher p95 and longer runs, so you might (a) run at higher concurrency closer to the knee to maximize tok/s, and/or (b) accept a larger `max-model-len` if long documents are common, knowing that this will reduce concurrency or push utilization higher. In both cases, your Tier 3 tables guide those choices rather than guessing.
