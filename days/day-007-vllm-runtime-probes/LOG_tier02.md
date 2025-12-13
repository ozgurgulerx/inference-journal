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

- `days/day-007-vllm-runtime-probes/prefix_prompts.jsonl`

Structure (one JSON per line):

- `{"prompt": "<BIG_SHARED_PREFIX>\nQ: <variant>\nA:"}`

Example (JSONL):

```json
{"prompt": "<BIG_SHARED_PREFIX>\nQ: How do I reset my password?\nA:"}
{"prompt": "<BIG_SHARED_PREFIX>\nQ: How do I export my account data?\nA:"}
{"prompt": "<BIG_SHARED_PREFIX>\nQ: How can I change my notification settings?\nA:"}
```

Keep:

- Prefix length: target a few hundred to ~1K tokens worth of text (don’t overthink; just make it “obviously big”).
- Variants: 20–50 questions.

Optional, but useful:

- Tag each JSON line with `prefix_len_tokens` (rough estimate) and a `workload` label (e.g. `"chat_policy"`, `"rag_prefix"`).

---

### 1) Run server with prefix caching OFF vs ON

You want two server configs that differ only in prefix caching.

Create two launcher scripts:

- `days/day-007-vllm-runtime-probes/serve_slm_no_prefix_cache.sh`
- `days/day-007-vllm-runtime-probes/serve_slm_prefix_cache.sh`

Example (adapt flags/version as needed):

```bash
#!/usr/bin/env bash
# days/day-007-vllm-runtime-probes/serve_slm_no_prefix_cache.sh
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

```bash
#!/usr/bin/env bash
# days/day-007-vllm-runtime-probes/serve_slm_prefix_cache.sh
set -euo pipefail

MODEL="microsoft/Phi-3-mini-4k-instruct"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT" \
  --enable-prefix-caching  # flag name may differ by vLLM version; check --help
```

Implementation detail: vLLM has a flag for this; if the exact flag name differs by version, rely on `--help` output for your installed vLLM.

Rules:

- Keep `MODEL`, `gpu-memory-utilization`, `max-model-len`, and `max-num-seqs` identical.
- Only toggle prefix caching.

---

### 2) Write a tiny repeated-prefix bench client

Create:

- `days/day-007-vllm-runtime-probes/prefix_cache_bench.py`

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

Example skeleton:

```python
#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import requests

MODEL = "microsoft/Phi-3-mini-4k-instruct"

def load_prompts(path: Path) -> List[str]:
    prompts = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def call_completion(url: str, prompt: str, max_tokens: int = 64) -> float:
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    t1 = time.time()
    resp.raise_for_status()
    _ = resp.json()
    return t1 - t0


def run_sequential(url: str, prompts: List[str]) -> List[float]:
    return [call_completion(url, p) for p in prompts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--prompts", default="prefix_prompts.jsonl")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    times = run_sequential(args.url, prompts)

    mean_wall = statistics.mean(times)
    p95_wall = statistics.quantiles(times, n=20)[-1]
    print(json.dumps({"mode": "sequential", "mean_wall_s": mean_wall, "p95_wall_s": p95_wall}))


if __name__ == "__main__":
    main()
```

---

### 3) Record results + interpret like an inference engineer

Create:

- `days/day-007-vllm-runtime-probes/prefix_caching_results.md`

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
  - **Expected outcome:** For workloads with a large shared prefix, you should see TTFT (or your wall-time proxy) drop meaningfully when prefix caching is on, especially at higher concurrency. If TTFT barely changes, either the prefix is too short, hit rate is low, or caching isn’t configured as expected.

- Did it increase throughput at the same p95?  
  - **Expected outcome:** By skipping repeated prefill work, you often free compute to serve more requests per second at similar p95 e2e latency. If tokens/sec and QPS don’t move while TTFT improves, prefix caching is mainly a UX win; if both TTFT and throughput improve, it’s a strong candidate for production use.

- Did you observe any extra memory overhead?  
  - **Expected outcome:** You should see some additional, *bounded* VRAM usage to hold cached prefixes. If memory footprint grows modestly and is stable, that’s acceptable; if it grows with the number of distinct prefixes in an unbounded way, you need a strategy (limits, eviction, scoping) before enabling it broadly.

- What workload shapes benefit most? (chat history, RAG with fixed system prompt, tool policies)  
  - **Expected outcome:** Workloads with long, static prefixes (system prompts, policies, shared context) and short, varied suffixes should show the biggest gains. Highly unique, document-per-request RAG or very short prefixes should show little to no benefit.

- How does the benefit change as `prefix_len_tokens` grows?  
  - **Expected outcome:** The advantage of prefix caching should grow with `prefix_len_tokens`—little effect at 256 tokens, noticeable at ~512–1K, and very strong when the shared prefix dominates the total sequence length. If you don’t see this trend, revisit your workload construction or cache configuration.

Add a short **“Hit rate mental model”** section:

- What effective cache hit rate did you test? (e.g. 100% reuse vs 50% reuse).
- How would mixed traffic (some cached, some not) change the benefit?

Theoretical expectations to guide your intuition:

- If `L_uncached` is TTFT for a long-prefix request **without** caching and `L_cached` is TTFT **with** caching, then for a cache hit rate `h` (fraction of requests that hit the prefix cache):

  - Expected average TTFT:

    ```text
    E[TTFT] ≈ h * L_cached + (1 - h) * L_uncached
    ```

  - So the **maximum benefit** (best case) is at `h ≈ 1.0` (all repeated-prefix traffic), and benefit shrinks roughly linearly as `h` drops.

- For mixed traffic (some requests have a shared prefix, some don’t):
  - Think of total traffic as a weighted mix of “prefix‑eligible” and “non‑prefix” requests. Only the eligible portion can benefit from caching.  
  - Example: If 50% of your QPS uses a 1K shared prefix and the other 50% is arbitrary RAG with unique 4K docs, and within the prefix‑eligible half your hit rate is 80%, then **global** TTFT improvement is much smaller than in a synthetic 100%‑eligible benchmark.

- Throughput (tokens/sec) follows the same pattern:
  - Prefill work is saved only on hits, so effective prefill cost per request shrinks roughly by a factor of `h * (prefix_len / total_len)`.  
  - If prefixes are long and hit rate is high, you should expect a noticeable throughput bump; if either prefixes are short or hit rate is low, the global throughput gain will be modest.

What “interpret like an inference engineer” means here:

- You are not just reporting numbers; you are **explaining what they imply for capacity and SLOs**.
- Look at **relative deltas**, not absolute values only:
  - How much did mean TTFT and p95 e2e improve with prefix caching **for each prefix length and concurrency**?
  - Is the gain bigger for large prefixes than for medium ones? That tells you when prefix caching really matters.
- Connect improvements to **workload shape**:
  - “For a 1K‑token shared policy + short questions, prefix caching shaved ~40–50% off TTFT at concurrency 16.”
  - “For a 256‑token prefix, the effect was within noise; I wouldn’t enable it just for that.”
- Check for **throughput vs latency trade‑offs**:
  - Did tokens/sec increase for the same p95 e2e latency?
  - If throughput didn’t change much, but TTFT got better, prefix caching is mostly a **user‑experience win**.
- Consider **memory and operational cost**:
  - Did GPU memory usage rise when caching prefixes?
  - Given your expected number of distinct prefixes in production, would that memory cost be acceptable?
- Finish with 3–5 **plain‑language bullets** that a non‑ML infra lead could act on, e.g.:
  - “Enable prefix caching for workloads with ≥1K shared tokens and concurrency ≥8.”
  - “Skip it for ad‑hoc RAG where every query has a unique 4K document context.”

---

### 4) One “consulting-ready” conclusion

Add 3–5 bullets at the bottom of `prefix_caching_results.md`:

- When I would enable prefix caching.
- How I would detect if it is working (metrics to watch).
- One failure mode / caveat.
- How I’d explain prefix caching to a product/SRE audience in 1–2 sentences.

Expectations for this section:

- You are writing for a **non‑ML infra / product audience**, so keep the bullets concrete and actionable, not just technical.

- Example of “When I would enable prefix caching”:
  - “Enable prefix caching for chat‑style workloads with ≥1K shared system/policy tokens and short user messages, especially when concurrency ≥8 and GPUs have at least X GiB of headroom.”

- Example of “How I would detect if it is working (metrics to watch)”:
  - “Monitor TTFT and p95 e2e for the prefix‑eligible endpoints; with caching on, TTFT should drop by ~30–50% for long‑prefix traffic and tokens/sec should rise at similar p95. Watch GPU memory and, if available, a ‘prefix cache hit rate’ metric to ensure hits stay high.”

- Example of “One failure mode / caveat”:
  - “If each tenant/session has its own long, unique prefix (e.g. per‑document RAG with 4K context), the cache fills with many distinct prefixes, chewing up VRAM with low hit rate. In that case, prefix caching can increase memory pressure without meaningful TTFT or throughput gains.”

- Example of “How I’d explain prefix caching to a product/SRE audience”:
  - “Prefix caching lets us pay the cost of a long, shared prompt (system policy, safety text, tools) once and reuse it across many requests. For workloads that reuse the same long intro, it can roughly cut TTFT and improve throughput, at the cost of a bit more GPU memory for cached prefixes.”

---

## Expected Artifact

- `days/day-007-vllm-runtime-probes/prefix_prompts.jsonl`
- `days/day-007-vllm-runtime-probes/serve_slm_no_prefix_cache.sh`
- `days/day-007-vllm-runtime-probes/serve_slm_prefix_cache.sh`
- `days/day-007-vllm-runtime-probes/prefix_cache_bench.py`
- `days/day-007-vllm-runtime-probes/prefix_caching_results.md`

---

## What You Should Learn (Mental Models)

- Prefix caching is **cross-request KV reuse**; it pays off when the prefix is large and reused.
- It’s a product lever: lowers perceived latency for chat-like repeated history.
- It’s also a capacity lever: can shift you into a better batching regime by shrinking prefill work.

### Deeper Explanation

- **Cross-request KV reuse (theory and intuition)**  
  - In a transformer, the KV cache holds per-layer, per-head representations of all processed tokens. Building it for a long prefix of length `P` costs `O(P)` compute and memory traffic.  
  - Without prefix caching, you repeat this prefill cost for every request, even if `P` is identical across them. With prefix caching, you compute KV for the shared prefix **once**, then reuse it across requests, only computing the suffix of length `S`.  
  - The relative prefill cost reduction per request is on the order of:

    ```text
    prefill_saved_fraction ≈ P / (P + S)
    ```

    when the shared prefix dominates total length. This directly shrinks TTFT and frees capacity for more requests.

- **Product lever: perceived latency**  
  - Users feel TTFT, not the internal breakdown of prefill vs decode. For chat-like workloads with a big, static intro and a short reply, **most of TTFT is prefill**.  
  - Caching the prefix effectively turns “long prompt + short reply” into “short prompt + short reply” in terms of TTFT, without changing the actual prompt semantics.  
  - This is why it’s a product lever: you can keep the rich policies and system prompts product wants, while still hitting a tighter TTFT SLO.

- **Capacity lever: batching regime and headroom**  
  - Prefill is less parallelizable and more memory-intensive than decode; reducing the per-request prefill load improves how many active sequences you can handle at a given p95.  
  - With prefix caching, **per-request incremental KV and compute** shrink for the eligible portion of traffic. That can:
    - allow you to run at higher effective concurrency for the same latency; or  
    - keep latency flat while serving more QPS.  
  - In continuous batching systems, this helps the scheduler form healthier batches (more sequences with smaller prefixes), which improves GPU utilization.

### Check Your Understanding (Q&A)

**Q1. Why does prefix caching help more for long prefixes than short ones?**  
**A:** Because the work you skip is proportional to the length of the shared prefix `P`. When `P` is small relative to the suffix `S`, the savings `P / (P + S)` are minimal. When `P` is large (e.g. 1–2K tokens) and `S` is short, most of the TTFT and prefill cost is in the prefix, so caching it produces large relative savings. This is why you should expect much bigger gains at 1K tokens than at 256 tokens.

**Q2. How does hit rate interact with prefix length to determine the real benefit?**  
**A:** The effective gain is the product of two factors: **how much work a cache hit saves** (driven by `P / (P + S)`) and **how often you hit** (`h`). The expected TTFT improvement across traffic is roughly `h * (P / (P + S))`. Long prefixes with high hit rates (e.g. shared policies used across many requests) deliver strong gains; short prefixes or low hit rates dilute the benefit.

**Q3. In what sense is prefix caching a “product lever,” not just a micro‑optimization?**  
**A:** Product teams often insist on long system prompts, policies, and safety/tool descriptions. Those make TTFT worse but are hard to remove. Prefix caching lets you keep those product‑driven requirements while restoring much of the lost TTFT and throughput. You can frame it to stakeholders as: “We can maintain your policy text and still meet latency SLOs, by amortizing that cost across requests.”

**Q4. Why is prefix caching also a “capacity lever”?**  
**A:** From a capacity standpoint, the heavy, repeated prefill work for long prefixes limits how many requests you can serve concurrently at acceptable latency. By eliminating repeated prefill for hits, you reduce per-request compute and memory bandwidth consumption. That effectively expands the **safe operating region**: you can handle more QPS or higher concurrency before hitting latency or OOM limits. In continuous batching, this can be the difference between “GPU is underutilized at lower TTFT” and “GPU is well utilized without blowing up p95.”

**Q5. When would you decide not to use prefix caching, even if your engine supports it?**  
**A:** When you don’t have long, reusable prefixes (e.g. each request has a unique 4K RAG document), or when you can’t bound the number of distinct prefixes and GPU memory is tight. In those cases, prefix caching can inflate VRAM usage for many low‑hit‑rate prefixes while providing little TTFT/throughput benefit. That’s when you either disable it or constrain it (e.g. only for certain endpoints or tenants) and focus on other levers like model size, `max-model-len`, or batching policy instead.
