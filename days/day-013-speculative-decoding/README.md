# Day 013 – Speculative Decoding Benchmark (Draft + Target)

## Goals
- **Primary Objective:** Enable and measure speculative decoding speedups vs baseline.
- **Focus Theme(s):** Decoding acceleration; scheduler interaction; correctness under acceleration.

## Setup & Context
- **Hardware/Infra:** Single GPU (16–40GB). Ensure draft model fits alongside target or run sequentially.
- **Model(s):** Target: 7B–13B chat model (BF16). Draft: 1B–3B model with compatible tokenizer; optional quantized draft.
- **Runtime/Engine:** vLLM speculative decoding flags; if available, TensorRT-LLM draft/target pair for comparison.
- **Dataset/Workload:** Mixed prompts: short Q&A (<=64 tokens) and long-form generation (~512–1024 tokens).
- **Key Config Knobs:** Draft/token ratio, max draft tokens, acceptance thresholds, `max_num_seqs`, batch/concurrency.

## Experiments & Measurements
1. **Baseline (no speculation)**
   - Method: Run chat + long-form workloads with standard vLLM settings.
   - Metrics: p50/p95 TTFT/E2E, tokens/sec, VRAM, CPU.
2. **Speculation enabled (single draft)**
   - Method: Enable draft model; test both workloads; vary max draft tokens (e.g., 4, 8).
   - Metrics: Speedup factor, acceptance rate/rejection rate, VRAM delta, tokens/sec.
3. **Draft model variants**
   - Method: Swap in smaller vs slightly larger drafts; optionally quantized draft.
   - Metrics: Speedup vs quality drift; draft load overhead.
4. **Load sensitivity**
   - Method: Concurrency sweep to see if speedup holds under load.
   - Metrics: System throughput vs latency; any scheduler stalls.
5. **Correctness/quality check**
   - Method: Diff outputs vs baseline; sanity check 10 examples for divergence.

## Analysis & Insights
- Speedup drivers (prefill reuse, acceptance rate); failure cases (draft mismatch, longer responses).
- Guidance for choosing draft size per workload.

## Performance & Cost Evaluation
- Convert tokens/sec to $/1M tokens baseline vs speculative; note GPU memory headroom.
- Account for cost of running draft + target simultaneously (double load?).

## Output Quality & Alignment Check
- Confirm outputs are identical or note divergence; flag any toxicity/format issues caused by draft mismatch.
- Track acceptance rate as a proxy for risk.

## Observability & Reliability Notes
- Log acceptance/rejection counts, latency per stage, and any timeouts.
- Note recovery if draft or target server dies (fallback path).

## Next Steps
- Carry the best speculative config into multi-tenant (Day 015) and cost modeling (Day 016).
