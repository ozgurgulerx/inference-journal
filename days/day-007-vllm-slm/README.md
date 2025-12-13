# Day 007 – vLLM SLM: TTFT, Prefix Caching, KV Scaling, and Micro-Batching

**Phase:** 0 – OS & GPU Setup (Days 1–15)  
**Theme:** Turn Day 006’s SLM-as-probe foundation into **repeatable runtime measurements**: first-token path, prefix caching, KV cache scaling, and batching behavior.

**Layers**

- **Runtime (vLLM)** → TTFT (cold vs warm), prefix caching, scheduling/batching behavior
- **Product/SLO** → latency vs throughput trade-offs; minimal metrics you can reuse in “health checks”

---

## Snapshot (Today’s Focus)

You’ve already established an OS-level baseline (THP/hugepages, cold vs warm load, allocator sanity) and confirmed vLLM works on your node.

Day 007 is about owning the **runtime-level behaviors** that actually show up in production SLOs:

- **TTFT vs throughput** under realistic concurrency.
- **Prefix reuse** (prefix caching) as the first “real” cross-request optimization.
- **KV cache capacity commitments** via `max-model-len` and what that does to headroom.

Assumptions:

- You will keep one SLM constant (e.g. `microsoft/Phi-3-mini-4k-instruct` or `Qwen/Qwen2.5-1.5B-Instruct`).
- vLLM server is reachable locally (port 8000) and you can run small Python scripts.

---

## Tier Breakdown

| Tier  | Time        | Scope |
|------:|------------:|------|
| Tier 1| ~60–90 min   | vLLM baseline + **TTFT cold/warm** measurement with a minimal probe |
| Tier 2| ~75–120 min  | **Prefix caching**: measure repeated-prefix wins + record trade-offs |
| Tier 3| ~90–150 min  | **KV scaling + batching under load**: `max-model-len` impact + concurrency regimes |

---

## Navigation

- **[Tier 1 – TTFT Baseline & First-Token Path](./LOG_tier01.md)**
- **[Tier 2 – Prefix Caching & Reuse Measurement](./LOG_tier02.md)**
- **[Tier 3 – KV Scaling + Micro-Batching Under Load](./LOG_tier03.md)**

---

## Cross-Day Context

- **Day 003 – vLLM Capacity & OOM**: introduced benchmarking intuition and capacity surfaces.
- **Day 005 – OS & NUMA Hardening**: keeps CPU-side variance from polluting your runtime results.
- **Day 006 – SLM + OS Memory & First-Token Path**: established SLM probe, cold vs warm, and the mental model for KV scaling.

Day 007 bridges:

- “I can run vLLM” → “I can measure and explain TTFT, batching, and reuse knobs with numbers.”

---

## Logging Template (Day 007)

Use this as your end-of-day write-up.

```markdown
# Day 007 – vLLM SLM: TTFT, Prefix Caching, KV Scaling

## Environment
- GPU:
- OS / kernel:
- vLLM version:
- Model (SLM):
- Key vLLM flags:

## Commands Run
- Server launch commands
- Probe/bench commands

## Key Numbers
- Cold TTFT / Warm TTFT
- Throughput tok/s (1 vs N concurrent)
- Prefix caching impact (TTFT, tok/s, memory)
- KV scaling table (max-model-len vs VRAM used)

## Artifacts Created/Updated
- first_token_latency.md
- prefix_caching_results.md
- kv_cache_scaling.csv
- batching_benchmark.md

## Observations / Surprises
- What caused TTFT spikes?
- Did prefix caching help? When did it not?
- What `max-model-len` feels safe on this GPU?
```

---

[← Day 006](../day-006-slm-memory/README.md) · [Days Index](../README.md)
