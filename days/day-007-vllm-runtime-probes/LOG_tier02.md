# Day 007 – vLLM Runtime Probes

## Tier 2 – Prefix Caching / Prefix Reuse (Measured)

### Goal

Measure what **prefix caching** buys you in practice:

- **Lower TTFT** (time-to-first-token) / better perceived latency  
- **Higher throughput** (tokens/s, QPS at same p95)  
- Understand **when it helps**, **when it’s noise**, and **what it costs** (VRAM, complexity)

### Outcome

A **small repeated-prefix benchmark** + a write-up that makes the trade-offs legible enough to justify a production decision.

---

## 0) TL;DR Mental Model (anchor your intuition)

### What prefix caching is (in one sentence)

**Cross-request reuse of KV for an identical prompt prefix**, so the engine can skip repeated **prefill** work and start decoding sooner.

### Why it helps TTFT

TTFT is usually dominated by **prefill** for long prompts. If you can reuse the KV for the shared prefix, TTFT for cache hits collapses toward “prefill only the suffix”.

Let:

- `P` = shared prefix tokens  
- `S` = request-unique suffix tokens (question, last turn, small RAG chunk)  
- `Tp(x)` = time to prefill x tokens (roughly linear-ish in x under similar batch regimes)

Then on cache hits (best case):

- **Prefill cost saved ≈ Tp(P)**  
- **New prefill cost ≈ Tp(S)**

A good first-order predictor of savings on hits:

- **prefill_saved_fraction ≈ P / (P + S)**

### Hit rate controls real-world benefit

If `h` is cache hit rate among eligible requests:

- **E[TTFT] ≈ h · TTFT_hit + (1-h) · TTFT_miss**

So: long prefixes *and* high hit rate are necessary. One without the other is mediocre.

### Why it can also boost throughput

Prefill is often the **bottleneck stage** (heavy GEMMs + memory traffic). If you save prefill on hits, the GPU has more headroom for:

- more concurrent sequences  
- higher effective batch  
- higher tokens/sec at similar p95 latency

---

## 1) Workload patterns where prefix caching actually matters

Prefix caching is not “RAG caching”. It’s **prefix reuse caching**. So you want workloads that naturally create **identical leading tokens** across many requests.

### Pattern A — Global system/policy/tool preamble (best ROI)

- Long system prompt (policy + tools + formatting rules) reused for almost every request.  
- Suffix changes per user question.

**Signals it will work**:

- `P ≥ ~512` tokens (often 1K–2K),  
- `S` is small (20–300),  
- hit rate `h ≈ 1.0` on that endpoint.

### Pattern B — Per-tenant / per-workspace static prefix

- Each tenant has a stable “policy + glossary + tone + allowed tools” block.  
- Within a tenant, many requests reuse the same prefix.

**Cache key mental model**: `(model, tenant_id, policy_version)` — production-realistic and gives “hot tenant” wins.

### Pattern C — Session trunk reuse (chat history prefix)

- In chat, history accumulates; if you freeze a “history trunk” and only append the last message(s), you reuse the trunk KV across multiple turns.

**Operational pattern**: maintain a cached trunk until you summarize/trim, then rebuild.

### Pattern D — Tool/schema scaffolding

- Long JSON schema / DSL / tool instructions + examples reused across calls.  
- Only the task spec varies.

### Pattern E — Eval / grader pipelines

- Same rubric / examples reused across many candidates.  
- High concurrency, homogeneous calls → great for prefix caching + batching.

### Anti-patterns (when it won’t help)

1. **Unique long document per request** (classic “RAG dumps 4K doc”): hit rate ~0.  
2. **Short prefixes** (`≤ 256` tokens) with large suffix or long generation: savings in noise.  
3. **Prefix churn** (frequent edits to system prompt): invalidates reuse.  
4. **Many distinct prefixes** with low reuse: wastes VRAM without wins.

---

## 2) Experiment design: what you are actually proving

We want to isolate the effect of prefix caching on:

1. **TTFT proxy** (or TTFT directly if stream timing is captured)  
2. **p95 end-to-end latency** under concurrency  
3. **Throughput** (tokens/sec, requests/sec)  
4. **Memory overhead / stability** (VRAM behavior)

### Key “engineering” rule: keep everything identical except caching

Same:

- model  
- dtype  
- `max-model-len`  
- `gpu-memory-utilization`  
- `max-num-seqs` (if you set it)  
- same prompts (identical distribution)  
- same client load pattern

Only toggle prefix caching.

---

## 3) Workload construction: repeated-prefix prompts

We need prompts that share a large identical prefix and differ only near the end.

### Two prefix regimes (to see scaling)

- **Medium prefix**: ~256–512 tokens  
- **Large prefix**: ~1K–2K tokens

### Prompt structure (JSONL)

Create: `days/day-007-vllm-runtime-probes/prefix_prompts.jsonl`  
One JSON per line:

```json
{"prompt": "<BIG_SHARED_PREFIX>\nQ: <variant>\nA:"}
```

Guidelines:

- 20–50 variants  
- keep suffix small (question is short)  
- ensure prefix is *literally identical* for all prompts in a regime (even tiny differences break reuse)

Optional metadata (helpful for analysis):

```json
{"prompt": "...", "workload": "chat_policy", "prefix_regime": "1024", "id": 17}
```

### Sanity check: is the prefix really identical?

Pragmatic check: compute a hash of the prefix string (or first N chars) before you generate prompts. If it differs, you won’t get hits.

---

## 4) Run two servers: prefix cache OFF vs ON

Create launcher scripts:

### `serve_slm_no_prefix_cache.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT"
```

### `serve_slm_prefix_cache.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT" \
  --enable-prefix-caching
```

> Note: flag name can vary by vLLM version. Always verify via `--help`.

### What to observe in logs (if available)

- any prefix cache initialization line  
- any hit/miss counters (version dependent)  
- any warnings about memory pressure

---

## 5) Benchmark client: measure sequential + concurrent

Create: `days/day-007-vllm-runtime-probes/prefix_cache_bench.py`

### Requirements (what you must capture per request)

- `t_start` → request sent  
- `t_first_token` (or TTFT proxy)  
- `t_done` (end-to-end)  
- response usage: `prompt_tokens`, `completion_tokens` (if returned)  
- status / error

### TTFT measurement options

**Option 1 (best): streaming**  
Use OpenAI-compatible streaming and record the timestamp of the first chunk.

**Option 2 (acceptable proxy): non-streaming**  
Measure wall-time to full response. This can still show improvements, but TTFT is blurred by decode time.

### Modes

- `sequential`: baseline, isolates per-request wins  
- `concurrent`: realistic load, shows scheduler/batching regime changes

### Concurrency sweep (minimal but informative)

- 1  
- 8  
- 16 (or 32 if your GPU can handle without thrash)

### Output artifacts

- per-request CSV/JSON: `runs/<mode>_<cache_on/off>_<prefix_regime>_<conc>.csv`  
- summary JSON: mean, p50, p95 for TTFT and E2E; tokens/sec

---

## 6) Results: one canonical table

Create: `days/day-007-vllm-runtime-probes/prefix_caching_results.md`

Use a single schema so comparisons are trivial:

```csv
mode,prefix_regime,concurrency,mean_ttft_s,p95_ttft_s,mean_e2e_s,p95_e2e_s,req_s,tok_s,notes
no_cache,512,1, , , , , , ,
cache,512,1, , , , , , ,
no_cache,512,16, , , , , , ,
cache,512,16, , , , , , ,
no_cache,1024,16, , , , , , ,
cache,1024,16, , , , , , ,
```

Also capture:

- GPU VRAM (peak and steady state) via `nvidia-smi` snapshots  
- any engine metrics if you can access them (hit rate, cache size)

---

## 7) Interpretation: reason like an inference engineer

You’re not collecting numbers for vanity. You’re testing hypotheses about the first-token path.

### 7.1 Did TTFT drop on cache hits?

Expected:

- stronger effect as prefix grows (512 → 1024 → 2048),  
- stronger effect at concurrency (batching pressure makes prefill dominate).

If TTFT does not improve:

- prefix not identical (no hits),  
- prefix too short (prefill cost small),  
- caching not enabled / flag mismatch,  
- workload not prefix-eligible (suffix dominates).

### 7.2 Throughput at same p95: capacity win or UX-only win?

Two possible outcomes:

- **TTFT ↓ but tok/s flat** → mostly UX improvement (still valuable).  
- **TTFT ↓ and tok/s ↑** → true capacity lever (production gold).

Interpretation trick:

- If p95 E2E stays similar but req/s rises, you improved capacity.  
- If p95 E2E improves at same req/s, you improved SLO headroom.

### 7.3 Concurrency sensitivity

Prefix caching often matters most when the system is under load:

- at concurrency 1, benefits can be visible but smaller;  
- at concurrency 8–16, prefill contention magnifies savings.

### 7.4 Memory overhead and stability

Prefix caching stores KV for prefixes (engine-managed).  
Expected:

- modest, bounded VRAM increase,  
- stable plateau after warm-up.

Red flag:

- VRAM grows with number of distinct prefixes unboundedly,  
- you get closer to OOM,  
- cache thrashes (low hit rate + high churn).

---

## 8) Hit-rate mental model (make it explicit)

### Effective hit rate you tested

Your synthetic benchmark is typically close to:

- **`h ≈ 1.0`** (every request shares prefix).

But real production might be:

- mixed endpoints,  
- multiple tenants,  
- per-session history divergence.

### How mixed traffic dilutes benefit

If only fraction `f` of traffic is prefix-eligible and among those hit rate is `h`, then:

- overall TTFT improvement is limited by `f · h`.

Example:

- 50% eligible traffic, 80% hit rate → only 40% of total requests benefit.

### Practical capacity implication

Prefill saved per request on hits is roughly:

- `saved_prefill ≈ h · P`.

Throughput gains show up only when:

- `P` is large,  
- `h` is high,  
- prefill is your bottleneck (often true).

---

## 9) Consulting-ready conclusion (for non-ML infra / SRE)

Add 3–5 bullets at the end of `prefix_caching_results.md`:

### When I would enable prefix caching

- Enable for endpoints with ≥1K shared prefix tokens and sustained concurrency ≥8.  
- Strong candidates: chat assistants with long policy/tool preambles; per-tenant assistants with stable config.

### How I’d detect it’s working

- TTFT for prefix-eligible traffic drops materially (often 30–60% when `P` dominates).  
- Throughput (tok/s or req/s) increases at similar p95 E2E.  
- VRAM increases modestly and reaches a stable plateau.  
- If available: cache hit rate stays high.

### One failure mode / caveat

- If each request has a unique long prefix (e.g., per-request 4K RAG doc), cache fills with low reuse → VRAM cost without latency/throughput benefit.

### One-sentence explanation for PM/SRE

- “Prefix caching lets us pay the cost of a long shared prompt once and reuse it across many requests, cutting first-token latency and often improving throughput, at the cost of some extra GPU memory.”

---

## 10) Expected artifacts (definition of done)

- `days/day-007-vllm-runtime-probes/prefix_prompts.jsonl`  
- `days/day-007-vllm-runtime-probes/serve_slm_no_prefix_cache.sh`  
- `days/day-007-vllm-runtime-probes/serve_slm_prefix_cache.sh`  
- `days/day-007-vllm-runtime-probes/prefix_cache_bench.py`  
- `days/day-007-vllm-runtime-probes/prefix_caching_results.md`

Done means:

- you can re-run the experiment in \<10 minutes,  
- deltas are visible and interpretable,  
- conclusions are actionable.

