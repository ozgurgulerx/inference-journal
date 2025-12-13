# Day 007 – vLLM SLM: TTFT, Prefix Caching, KV Scaling
## Tier 2 – Prefix Caching / Prefix Reuse (Measured)

> **Goal**: Measure what prefix caching buys you in practice: lower TTFT and/or higher throughput for repeated-prefix workloads.
>
> **Outcome**: A small repeated-prefix benchmark + a write-up that makes the trade-offs legible.

---

## Tier 2 – Deepen (If Time/Energy Allow)

**Title** – Prefix caching impact under repeated-prefix traffic  
**Time Budget** – ~75–120 min

---

### 0) Decide on a repeated-prefix workload

We need prompts that share a large prefix (chat history / system prompt / long policy text) and differ only in the final user question.

Create:

- `days/day-007-vllm-slm/prefix_prompts.jsonl`

Structure (one JSON per line):

- `{"prompt": "<BIG_SHARED_PREFIX>\nQ: <variant>\nA:"}`

Keep:

- Prefix length: target a few hundred to ~1K tokens worth of text (don’t overthink; just make it “obviously big”).
- Variants: 20–50 questions.

---

### 1) Run server with prefix caching OFF vs ON

You want two server configs that differ only in prefix caching.

Create two launcher scripts:

- `days/day-007-vllm-slm/serve_slm_no_prefix_cache.sh`
- `days/day-007-vllm-slm/serve_slm_prefix_cache.sh`

Implementation detail: vLLM has a flag for this; if the exact flag name differs by version, rely on `--help` output for your installed vLLM.

Rules:

- Keep `MODEL`, `gpu-memory-utilization`, `max-model-len`, and `max-num-seqs` identical.
- Only toggle prefix caching.

---

### 2) Write a tiny repeated-prefix bench client

Create:

- `days/day-007-vllm-slm/prefix_cache_bench.py`

Requirements:

- Read prompts from `prefix_prompts.jsonl`
- Run them in two modes:
  - sequential (easy baseline)
  - concurrent (more realistic; reuses Day 003 mental model)
- Output a simple CSV line or JSON summary:
  - mean wall time
  - p95 wall time (rough; ok if approximate)
  - total tokens (if available)

Keep it short.

---

### 3) Record results + interpret like an inference engineer

Create:

- `days/day-007-vllm-slm/prefix_caching_results.md`

Include:

- The exact shared-prefix strategy you used (what kind of prefix? how large?).
- The two server commands (prefix cache off/on).
- A table:

```text
mode,concurrency,mean_wall_s,p95_wall_s,notes
no_prefix_cache,1,...,...,
no_prefix_cache,16,...,...,
with_prefix_cache,1,...,...,
with_prefix_cache,16,...,...,
```

Interpretation prompts:

- Did prefix caching reduce *TTFT proxy* for repeated-prefix prompts?
- Did it increase throughput at the same p95?
- Did you observe any extra memory overhead?
- What workload shapes benefit most? (chat history, RAG with fixed system prompt, tool policies)

---

### 4) One “consulting-ready” conclusion

Add 3–5 bullets at the bottom of `prefix_caching_results.md`:

- When I would enable prefix caching.
- How I would detect if it is working (metrics to watch).
- One failure mode / caveat.

---

## Expected Artifact

- `days/day-007-vllm-slm/prefix_prompts.jsonl`
- `days/day-007-vllm-slm/serve_slm_no_prefix_cache.sh`
- `days/day-007-vllm-slm/serve_slm_prefix_cache.sh`
- `days/day-007-vllm-slm/prefix_cache_bench.py`
- `days/day-007-vllm-slm/prefix_caching_results.md`

---

## What You Should Learn (Mental Models)

- Prefix caching is **cross-request KV reuse**; it pays off when the prefix is large and reused.
- It’s a product lever: lowers perceived latency for chat-like repeated history.
- It’s also a capacity lever: can shift you into a better batching regime by shrinking prefill work.
