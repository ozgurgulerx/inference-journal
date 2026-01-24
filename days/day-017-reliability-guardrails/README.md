# Day 017 – Reliability Drills & Guardrails

## Goals
- **Primary Objective:** Observe failure behavior and safety overhead in the serving stack.
- **Focus Theme(s):** Reliability & safety; graceful degradation; guardrail performance impact.

## Setup & Context
- **Hardware/Infra:** Single GPU node; ability to cap GPU memory (e.g., `CUDA_VISIBLE_DEVICES` + `torch.cuda.set_per_process_memory_fraction` or cgroups) and manipulate network/timeouts.
- **Model(s):** Chat model from earlier days (BF16) plus optional quantized/speculative variants to see differing stability.
- **Runtime/Engine:** vLLM (with timeouts and max batch settings); optional second runtime for failover tests.
- **Dataset/Workload:** Mix of short chat and long batch jobs to trigger stress; include prompts that touch safety filters.
- **Key Config Knobs:** Timeout values, retry/backoff, KV eviction thresholds, guardrail hook enable/disable, healthcheck intervals.

## Experiments & Measurements
1. **Induced OOM/pressure**
   - Method: Constrain GPU memory, run high-concurrency job until OOM/eviction.
   - Metrics: Failure mode (crash vs eviction), recovery time, dropped requests, logs emitted.
2. **Timeout/hang simulation**
   - Method: Inject artificial delay or block one request; test timeout + cancellation behavior.
   - Metrics: p99 impact, stuck requests, queue growth, error responses.
3. **Failover path**
   - Method: Run two instances; kill one mid-load; observe routing and cold-start hit.
   - Metrics: Latency spike magnitude/duration, success rate.
4. **Guardrail integration**
   - Method: Add lightweight safety filter (toxicity/PII) post-generation; measure overhead.
   - Metrics: Added latency per request, throughput drop, false-positive/negative notes.
5. **Chaos + logs**
   - Method: Combine guardrails with induced failures; ensure logs/metrics are actionable.
   - Metrics: Alertability, signal quality (did we see the failure quickly?).

## Analysis & Insights
- Document dominant failure modes and workable mitigations (eviction, retries, backpressure, failover).
- Summarize guardrail overhead and recommend when to enable by default.

## Performance & Cost Evaluation
- Quantify latency/throughput hit from guardrails and retries; estimate extra $/1M tokens when safety is on.

## Output Quality & Alignment Check
- Note any behavior changes under stress (truncated outputs, refusals).
- Track guardrail accuracy qualitatively on a small sample.

## Observability & Reliability Notes
- Ensure per-request logging includes error codes, retry counts, and safety decisions.
- Capture actionable runbooks for the failures encountered.

## Next Steps
- Fold mitigations into configs for Day 015 (multi-tenant) and Day 018 (aligned model serving).
