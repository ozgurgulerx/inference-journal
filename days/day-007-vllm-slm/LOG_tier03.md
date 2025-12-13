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

---

### C) Optional: Concurrency sweep (find “knee of the curve”)

If time permits, extend `batch_client.py` to sweep concurrency levels:

- 1, 2, 4, 8, 16, 32

Write results into:

- `days/day-007-vllm-slm/concurrency_sweep.csv`

Key insight to capture:

- Where does throughput stop scaling?
- Where does latency blow up? (queuing)

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
