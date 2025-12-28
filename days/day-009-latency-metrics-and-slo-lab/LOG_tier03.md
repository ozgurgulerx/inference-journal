# LOG (Tier 03) — Turning Metrics Into SLOs (and Control)

This tier is “how you think like a production inference engineer”:

percentiles are not vibes — they are the output of **queueing + workload mix + system policy**.

---

## 1) Convert percentiles into a promise (SLO)

Examples:

- “TTFT p95 < 300ms” for chat UX.
- “TPOT p95 < 80ms” for smooth streaming.
- “E2E p95 < 2.0s at max_tokens=256” for completion endpoints.

Rule: SLOs must specify the **shape envelope**:

- max input tokens, max output tokens, sampling policy
- and the load regime (or the admission policy)

---

## 2) The core move: bound the queue

If you let the queue grow unbounded, **p99 becomes a function of burstiness**, not “hardware speed.”

Minimum policy set:

- concurrency cap (admission control)
- bounded queue (Queue_cap)
- shed policy (reject/degrade when over cap)

---

## 3) How to read your Day 09 results

- TTFT p99 grows with concurrency, TPOT stable:
  - queueing/admission problem (CPU front-end, runtime scheduler, or simply saturation).
- TPOT degrades with prompt mix / context length:
  - decode is memory/KV dominated, or you’re hitting a kernel/memory cliff.
- Errors grow with concurrency:
  - resource exhaustion or timeouts; investigate server logs before trusting percentiles.

---

## 4) The “consulting-quality” latency table (minimal)

Every benchmark table you publish should include:

- model + precision + max_model_len
- max_tokens + prompt mix description
- concurrency and request count
- TTFT/TPOT/E2E p50/p95/p99 + error rate
- (optional) GPU util + mem util snapshot

That is enough to:

- reproduce,
- compare,
- and justify an SLO + admission policy.

