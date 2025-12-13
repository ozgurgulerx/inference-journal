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

- `days/day-007-vllm-slm/serve_slm_vllm.sh`

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
chmod +x days/day-007-vllm-slm/serve_slm_vllm.sh
./days/day-007-vllm-slm/serve_slm_vllm.sh
```

**Log**: paste the exact command/flags into `first_token_latency.md`.

---

### 2) Write a tiny TTFT probe script (OpenAI-compatible request)

Create:

- `days/day-007-vllm-slm/ttft_probe.py`

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

Create:

- `days/day-007-vllm-slm/first_token_latency.md`

Template:

```markdown
# Day 007 – TTFT Baseline (SLM)

## Server Config
- MODEL=
- vLLM flags:

## Measurements
| run | wall_s | notes |
|-----|--------|-------|
| cold_1 | | first request after server start |
| warm_1 | | immediate second request |

## Observations
- What dominated cold time? (weights load? graph warmup? KV alloc?)
- Is warm stable across 3–5 runs?
```

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

## Expected Artifact

- `days/day-007-vllm-slm/serve_slm_vllm.sh`
- `days/day-007-vllm-slm/ttft_probe.py`
- `days/day-007-vllm-slm/first_token_latency.md`

---

## What You Should Learn (Mental Models)

- Cold-start cost is a bundle: **weight loading + runtime warmup + initial allocations**.
- Warm-start behavior approximates the steady-state “interactive” experience.
- Your vLLM flags are not just “settings”; they are **capacity commitments** (especially `max-model-len` and `gpu-memory-utilization`).
