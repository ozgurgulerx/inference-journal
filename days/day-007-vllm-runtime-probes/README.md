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

## Status (Completeness / Integrity)

This folder now contains both:

- the **templates / write-up shells** for the day, and
- the **runnable scripts** needed to execute Tier 1–3.

Present in repo:

- `first_token_latency.md`
- `prefix_caching_results.md`
- `kv_cache_scaling_notes.md`
- `batching_benchmark.md`

- Tier 1 scripts: `serve_slm_vllm.sh`, `ttft_probe.py`
- Tier 2 scripts/data: `prefix_prompts.jsonl`, `serve_slm_no_prefix_cache.sh`, `serve_slm_prefix_cache.sh`, `prefix_cache_bench.py`
- Tier 3 scripts: `kv_scaling.sh`, `batch_client.py`

Expected to be created when running the day:

- `kv_cache_scaling.csv` (produced by running `kv_scaling.sh`)
- (optional) `concurrency_sweep.csv` (produced by running `batch_client.py --sweep --out concurrency_sweep.csv`)

If you want Day 007 to be “self-contained” in the repository, the cleanest approach is:

- Run the experiments and commit the resulting CSVs/logs alongside the filled-in markdown.

---

## Required Learning (Must-Learn)

These are the references that make Day 007 “deep” (not just a checklist). They are intentionally mapped to Tier 1/2/3.

### Tier 1 (TTFT / cold vs warm)

- **vLLM paper (PagedAttention)**
  - https://arxiv.org/abs/2309.06180
  - Focus: KV cache as the bottleneck, why batching is hard, and why memory management dominates throughput.
- **Queueing / tail-latency intuition (why p95 blows up near saturation)**
  - The Tail at Scale (Google): https://research.google/pubs/the-tail-at-scale/

### Tier 2 (Prefix caching / prefix reuse)

- **vLLM docs: Automatic Prefix Caching (APC)**
  - https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- **vLLM docs: Prefix caching design**
  - https://docs.vllm.ai/en/latest/design/prefix_caching/
- **YouTube (conceptual, high signal)**
  - vLLM / PagedAttention talk (one good entry): https://www.youtube.com/watch?v=Oq2SN7uutbQ

### Tier 3 (KV scaling + continuous batching)

- **FlashAttention background (why attention is an IO/memory problem)**
  - FlashAttention-2: https://arxiv.org/abs/2307.08691
  - (Optional precursor) FlashAttention: https://arxiv.org/abs/2205.14135
- **YouTube (batching intuition)**
  - Gentle intro to static/dynamic/continuous batching: https://www.youtube.com/watch?v=yjvMtJNecec

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

## Feynman / Consulting Deliverable

At the end of the day, add a short section (2–4 paragraphs) to your notes that answers:

- How you’d explain **TTFT, prefix caching, and KV scaling** to a non-ML infra lead.
- Given your measurements, how you’d configure a **first-cut vLLM SLM service** on this node (model choice, `max-model-len`, concurrency targets).
- What metrics/SLOs you’d track in a simple “health check” dashboard (TTFT, p95 latency, tok/s, GPU util, VRAM headroom).

You can reuse this text directly in consulting or internal docs.

---

## Re-usable Checklist (New Node / New SLM)

When you bring up vLLM on a new node or with a new SLM, reuse this quick checklist:

- [ ] Record environment (GPU, OS/kernel, vLLM version, model).
- [ ] Capture **cold vs warm TTFT** with a tiny probe (`first_token_latency.md`).
- [ ] Run a minimal **prefix caching** experiment with a big shared prefix + variants (`prefix_caching_results.md`).
- [ ] Run **KV scaling** for a few `max-model-len` values and compute an approximate bytes-per-token slope (`kv_cache_scaling.csv` + notes).
- [ ] Run a small **micro-batching / concurrency** sweep and identify a sweet-spot concurrency range (`batching_benchmark.md`).
- [ ] Write down 3–5 rules of thumb you’d reuse for this GPU + SLM in production.

---

## Check Your Learning – Day 007 (20 Questions)

Use these questions to test whether you’ve internalized the runtime‑level behaviors from this day.

1. How do you define **TTFT** in the context of vLLM, and how is it different from full end‑to‑end latency?
2. In your `first_token_latency.md` measurements, what rough factor separated cold TTFT from warm TTFT, and what components do you think contributed most to the gap?
3. If you see unexpectedly high warm TTFT, what are three things you would check first (flags, environment, or workload)?
4. How does `max-model-len` affect TTFT and throughput indirectly, even when your actual prompts are much shorter than the maximum?
5. How would you explain to an SRE why “TTFT vs throughput” is a trade‑off rather than two independent knobs?
6. What is **prefix caching** in vLLM terms, and how is it different from just having a long static system prompt?
7. Describe the repeated‑prefix workload you used in `prefix_prompts.jsonl`. Why did you choose that kind of prefix?
8. For your prefix caching experiments, how did TTFT and tok/s change when you turned prefix caching on for a large prefix at moderate concurrency?
9. What is the concept of **cache hit rate** for prefix caching, and how does mixed traffic (some cached, some not) change your expectations?
10. Which production metrics would you monitor to confirm that prefix caching is actually delivering a benefit in a real system?
11. Under what workload shapes would you **avoid** enabling prefix caching, or at least treat it with caution?
12. How did you estimate **bytes per KV token** from `kv_cache_scaling.csv`, and what rough number did you arrive at for this SLM + GPU?
13. Using that bytes‑per‑token estimate, how would you reason about a safe `max-model-len` if a product team asks for “room for 8 concurrent 4K‑token chats”?
14. In your KV scaling notes, did the memory usage vs `max-model-len` curve look mostly linear, or did you observe step changes? What might cause those steps?
15. How would you explain the phrase “`max-model-len` is a VRAM reservation decision” to a non‑ML infra lead?
16. From your `batching_benchmark.md` results, what concurrency range looked like the **sweet spot** for your SLM and node, and why?
17. At what concurrency did you see throughput flatten or p95/v99 latency blow up, and how would you communicate that “knee of the curve” to a product owner?
18. How does continuous batching allow vLLM to increase tokens/sec without linearly increasing TTFT for each request?
19. If you had to design a minimal “vLLM health check” dashboard based on Day 007, which 4–5 metrics would you include and what thresholds would worry you?
20. Given your Day 007 experiments, what concrete configuration (model, `max-model-len`, concurrency target, prefix caching on/off) would you recommend as a **first-cut vLLM SLM service** for this node?

---

[← Day 006](../day-006-slm-memory/README.md) · [Days Index](../README.md)
