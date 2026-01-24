# Day 016 – Cost & Efficiency Accounting

## Goals
- **Primary Objective:** Build a repeatable cost/efficiency calculator for all experiments.
- **Focus Theme(s):** Observability & cost analytics; per-tenant/per-request accounting.

## Setup & Context
- **Hardware/Infra:** Same GPU(s) used in prior days to reuse benchmarks; cloud hourly pricing noted; ability to log GPU utilization and tokens.
- **Model(s):** Reuse representative setups (BF16 baseline, quantized model, speculative config).
- **Runtime/Engine:** vLLM + at least one alternate runtime from Day 011 for comparison.
- **Dataset/Workload:** Representative chat + batch prompts; reuse Day 003 grids for consistency.
- **Key Config Knobs:** Logging granularity (per request/tenant), sampling interval for GPU/DCGM metrics, cost tables for GPUs.

## Experiments & Measurements
1. **Cost calculator draft**
   - Method: Script to convert tokens/sec + GPU hourly rate → $/1M tokens; include build-time amortization for TensorRT-LLM.
   - Metrics: $/1M tokens, req/sec, wall-clock utilization.
2. **Per-request/tenant attribution**
   - Method: Add logging hooks to benchmark harness to record tokens/request + latency + tenant id.
   - Metrics: $/request, $/tenant/hour, normalized by quality if available.
3. **Config comparison**
   - Method: Compare BF16 vs quantized vs speculative vs alt runtime using the same workload.
   - Metrics: Cost vs p95 latency table; highlight winners.
4. **Dash/snippet**
   - Method: Small dashboard or CSV + plot to visualize cost vs throughput.
   - Metrics: None beyond above; focus on clarity for future days/case studies.

## Analysis & Insights
- Identify the most cost-efficient config per workload; note where latency/SLO trade-offs break.
- Call out any surprising cost regressions (e.g., faster but more expensive due to draft overhead).

## Performance & Cost Evaluation
- Present $/1M tokens and $/request side-by-side with latency/throughput for each config.
- Include GPU utilization to show headroom or fragmentation.

## Output Quality & Alignment Check
- Note any quality drift that would nullify a cost win (e.g., quantization errors, speculative divergences).

## Observability & Reliability Notes
- Ensure logging is cheap; measure overhead of extra metrics.
- Note any missing signals to add later (e.g., cache hit rate, eviction).

## Next Steps
- Bake the calculator into future days; wire cost columns into Day 012/013/015/018 reports.
