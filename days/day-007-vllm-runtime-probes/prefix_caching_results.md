# Day 007 – Prefix Caching Results

## Workload
- Shared prefix description:
- Workload label(s) (e.g. `chat_policy`, `rag_prefix`):
- Approx prefix lengths tested:
  - medium_prefix_len_tokens ≈
  - large_prefix_len_tokens ≈
- Variants per prefix (number of questions):

## Server Configs
- No prefix cache command:
  - 
- With prefix cache command:
  - 

## Summary Table (by prefix length and concurrency)

```text
mode,prefix_len_tokens,concurrency,mean_ttft_s,mean_e2e_s,p95_e2e_s,tok_s,notes
no_prefix_cache,512,1,,,,,
no_prefix_cache,512,16,,,,,
with_prefix_cache,512,1,,,,,
with_prefix_cache,512,16,,,,,
no_prefix_cache,1024,1,,,,,
with_prefix_cache,1024,16,,,,,
```

## Interpretation
- Did prefix caching reduce TTFT proxy for repeated-prefix prompts?
- Did it increase throughput at the same p95?
- Did you observe any extra memory or CPU overhead?
- How did the benefit change as `prefix_len_tokens` grew?
- Any obvious “diminishing returns”?

## Hit Rate Mental Model
- Effective cache hit rate in this experiment (e.g. ~100% reuse vs 50% reuse):
- How would mixed traffic (some cached, some not) affect realized gains?
- Metrics you’d watch in production to confirm hit rate (and detect regressions):

## Consulting-Ready Conclusions
- When I would enable prefix caching:
- How I would detect that it is working (metrics / dashboards):
- One failure mode or caveat:
- One sentence I’d use to explain prefix caching to a product/SRE audience:

