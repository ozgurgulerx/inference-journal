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

Define **two prefix regimes** so you can see where prefix caching starts to pay off:

- A “medium” prefix (~256–512 tokens).
- A “large” prefix (~1K+ tokens).

Create:

- `days/day-007-vllm-slm/prefix_prompts.jsonl`

Structure (one JSON per line):

- `{"prompt": "<BIG_SHARED_PREFIX>\nQ: <variant>\nA:"}`

Keep:

- Prefix length: target a few hundred to ~1K tokens worth of text (don’t overthink; just make it “obviously big”).
- Variants: 20–50 questions.

Optional, but useful:

- Tag each JSON line with `prefix_len_tokens` (rough estimate) and a `workload` label (e.g. `"chat_policy"`, `"rag_prefix"`).

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
- For each run, capture:
  - `ttft_proxy_s` (start → first token or response receive)
  - `e2e_s` (start → completion done)
  - `prompt_tokens`, `completion_tokens` (if available)
- Output per-request stats (CSV/JSON), plus a small aggregate summary:
  - mean and p95 wall time (rough is fine)
  - total tokens and tokens/sec

Keep it short.

---

### 3) Record results + interpret like an inference engineer

Create:

- `days/day-007-vllm-slm/prefix_caching_results.md`

Include:

- The exact shared-prefix strategy you used (what kind of prefix? how large?).
- The two server commands (prefix cache off/on).
- A summary table per `(prefix_length, concurrency)` pair:

```text
mode,prefix_len_tokens,concurrency,mean_ttft_s,mean_e2e_s,p95_e2e_s,tok_s,notes
no_prefix_cache,512,1,...,...,...,...,
no_prefix_cache,512,16,...,...,...,...,
with_prefix_cache,512,1,...,...,...,...,
with_prefix_cache,512,16,...,...,...,...,
no_prefix_cache,1024,1,...,...,...,...,
with_prefix_cache,1024,16,...,...,...,...,
```

Interpretation prompts:

- Did prefix caching reduce *TTFT proxy* for repeated-prefix prompts?
- Did it increase throughput at the same p95?
- Did you observe any extra memory overhead?
- What workload shapes benefit most? (chat history, RAG with fixed system prompt, tool policies)
- How does the benefit change as `prefix_len_tokens` grows?

Add a short **“Hit rate mental model”** section:

- What effective cache hit rate did you test? (e.g. 100% reuse vs 50% reuse).
- How would mixed traffic (some cached, some not) change the benefit?

---

### 4) One “consulting-ready” conclusion

Add 3–5 bullets at the bottom of `prefix_caching_results.md`:

- When I would enable prefix caching.
- How I would detect if it is working (metrics to watch).
- One failure mode / caveat.
- How I’d explain prefix caching to a product/SRE audience in 1–2 sentences.

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
