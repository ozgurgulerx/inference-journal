# Day 003 — vLLM Capacity, OOM Surface & Real Use-Cases

## Executive Summary

Day 003 transforms a running vLLM server into a **measured, production-ready inference endpoint**. The core thesis: capacity should be empirical, not hopeful. By the end, you have benchmark harnesses, capacity grids, and client-ready playbooks for two distinct workloads—chat and batch.

---

## Goal

Build the instrumentation and intuition to answer: *"For this GPU, this model, this workload—what's safe, what breaks, and why?"*

---

## Core Skills: Benchmarking & Metric Interpretation

Day 003 focuses on two foundational inference engineering skills:

### 1. Writing Your Own Benchmark Harness

For a real inference engineer, being able to write your own benchmark harness is basically non-optional. You don't need to build a full Locust/Gatling clone, but you do need to be comfortable spinning up a small Python (or Go/Rust) tool that can generate realistic load, control concurrency, hit an OpenAI-compatible endpoint, log TTFT/E2E/tokens, and spit out a JSON/CSV like the ones you'll produce today.

Off-the-shelf tools (`genai-perf`, `inference-perf`, `guidellm`, etc.) are great, but in practice you constantly need "one-off" experiments: replay a weird traffic pattern, test just this model, just this GPU, just this scheduler idea. That's impossible if you treat benchmarking as "black-box tool magic."

### 2. Reading Metrics Like an Inference Engineer

The second skill is turning raw numbers into immediate conclusions. When you see:

```json
{"p95_ttft_ms": 2300, "p95_e2e_ms": 2350, "throughput_tok_s": 350, "concurrency": 8}
```

You should instantly infer:
- **No effective streaming** — TTFT ≈ E2E means users wait ~2.3s then get everything at once
- **Capacity estimate** — ~350 tok/s at 8 concurrent users ≈ one RTX 2000 Ada's comfortable zone
- **SLO check** — p95 ~2.3s is borderline for interactive chat, fine for async

And if you crank concurrency to 32 and see throughput plateau while p95 jumps to 9-10s, you immediately recognize queuing hell. If you increase `max_tokens` from 128 to 512 and E2E doubles while TTFT stays flat, you've shifted the bottleneck to decode compute.

This is what Day 003 trains: turning TTFT, E2E, throughput, and concurrency into reflexive judgments about UX, capacity, and the next knob to turn.

---

## Workloads Covered

Different inference workloads optimize for different metrics: real-time chat applications require low TTFT and TPOT, RAG scenarios can be relatively more lenient, while offline batch inference focuses on end-to-end throughput. Day 003 covers both ends of this spectrum:

| Workload | Profile | Priority | Target User |
|----------|---------|----------|-------------|
| **Chat** | Short turns, interactive | Latency (p95 < 1.5s) | End users, support agents |
| **Batch Summarization** | Long docs, offline | Throughput (tok/s) | Backend jobs, analytics |

---

## Tier Breakdown

### Tier 1: Must-Do Core (~2h)
> Encode reusable configs, build benchmark harness, map capacity grid for chat workload.

| Task | Deliverable |
|------|-------------|
| **1.1** | Reusable vLLM config (YAML + shell wrapper) for 16GB GPUs |
| **1.2** | Async benchmark script measuring TTFT, E2E, throughput |
| **1.3** | Chat capacity grid (concurrency × max_tokens → metrics) |

**Key Metrics**: TTFT, TPOT, ITL, System TPS, User TPS

---

### Tier 2: Extension (~1-2h)
> Add batch workload, compare RTX vs A100/H100.

| Task | Deliverable |
|------|-------------|
| **2.1** | Batch summarization benchmark + capacity grid |
| **2.2** | A100/H100 anchor run—same scripts, different GPU class |

**Outcome**: Empirical data on when RTX-class is sufficient vs. when to recommend A100.

---

### Tier 3: Deep Work (~4h)
> Unified capacity analysis, connect numbers to vLLM internals.

| Task | Deliverable |
|------|-------------|
| **3.1** | Capacity frontier report (sweet spots per GPU/workload) |
| **3.2** | "Life of a Request" documentation—scheduler, KV cache, prefill/decode |

**Outcome**: Client-presentable analysis + mental model for debugging OOM and latency spikes.

---

### Tier 4: Consolidation (~45-60m)
> Package everything into reusable playbooks.

| Task | Deliverable |
|------|-------------|
| **4.1** | vLLM Single-GPU Serving Recipes playbook |
| **4.2** | vLLM Best Practices reference (chunked prefill, prefix caching, OOM debugging) |
| **4.3** | Clean git commit |
| **4.4** | Retrieval practice quiz |

---

## Key Artifacts

```
configs/vllm/               → Reusable server configs
scripts/benchmarks/         → Chat + batch benchmark harnesses, capacity grid scripts
benchmarks/                 → Raw JSON/CSV results
reports/                    → Capacity frontier, life-of-request docs
playbooks/                  → Client-ready serving recipes
```

---

## Critical Insights

1. **Chat vs Batch operate differently** — Chat needs low p95, batch tolerates latency for throughput gains
2. **Continuous batching is the unlock** — concurrency of 1 misses the whole point
3. **GPU class determines ceiling** — A100 pushes 4-8x higher concurrency before saturation
4. **OOM has a surface you can map** — `max_model_len × max_num_seqs` vs available KV cache
5. **Preemption warnings = early OOM signal** — monitor logs, not just `nvidia-smi`
6. **Chunked prefill tuning matters** — smaller chunks for ITL, larger for throughput

---

## Key Config Levers

| Parameter | Chat | Batch |
|-----------|------|-------|
| `gpu_memory_utilization` | 0.8 | 0.85-0.9 |
| `max_model_len` | 4096 | Match actual need |
| `max_num_seqs` | 64-128 | 128-256 |
| `max_num_batched_tokens` | 2048 | 8192-16384 |
| `enable_prefix_caching` | true | false |

---

## Success Criteria

By completing Day 003:

- [ ] Two workloads benchmarked with reproducible scripts
- [ ] Capacity grids showing safe operating zones
- [ ] GPU scaling intuition (RTX → A100)
- [ ] Playbook ready to share with clients or teammates
- [ ] Mental model of vLLM request lifecycle

---

## Time Investment

| Tier | Time | Scope |
|------|------|-------|
| Tier 1 | ~2h | Core—minimum viable |
| Tier 2 | ~2h | Extension—batch + GPU comparison |
| Tier 3 | ~4h | Deep—analysis + documentation |
| Tier 4 | ~1h | Consolidation—playbooks + commit |

**Total**: 7-9h for full completion. Tier 1 + 2.1 gets you 80% of the value.

---

## Further Reading

For deeper context on the metrics, tools, and mental models used throughout Day 003:

📚 **[FURTHER_READING.md](FURTHER_READING.md)** — Curated references on LLM inference benchmarking, vLLM internals, and a suggested path for learning benchmarking-by-coding.

---

*Day 003 output: You can now quote capacity numbers, not vibes.*
