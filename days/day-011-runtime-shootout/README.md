# Day 011 – Alternate Runtimes & Scheduling Shootout

## Goals
- **Primary Objective:** Compare vLLM vs TensorRT-LLM vs TGI/Triton on the same model/workloads.
- **Focus Theme(s):** Runtime diversity & scheduling; continuous vs static batching; cost-per-token.

## Setup & Context
- **Hardware/Infra:** 1× 24–40GB GPU (A10/T4/A100 class), pinned NUMA; note PCIe vs NVLink if relevant.
- **Model(s):** Llama-3-8B (BF16) + a quantized variant if available to reuse from Day 004.
- **Runtime/Engine:** vLLM (latest), TensorRT-LLM (engine built once), plus TGI or Triton; keep configs side-by-side.
- **Dataset/Workload:** Chat turns (short) and batch summarization (longer docs) reused from Day 003 grids.
- **Key Config Knobs:** `max_num_seqs`, `max_model_len`, batching mode (static vs continuous), scheduler priority/prefill settings, CUDA graphs or not.

## Experiments & Measurements
1. **Baseline – vLLM continuous batching**
   - Method: Run chat + batch harness from Day 003 against vLLM with documented flags.
   - Metrics: p50/p95 TTFT + E2E, tokens/sec, VRAM, CPU, GPU util.
   - Observation: Note scheduler efficiency and any prefix cache wins/losses.
2. **TensorRT-LLM (or TGI/Triton) static batching**
   - Method: Build/load engine, run same prompts & request shapes.
   - Metrics: Same as above + batch padding/waste, compile/build time.
   - Observation: Where static batching helps/hurts vs continuous.
3. **Scheduler knob sweep**
   - Method: Sweep batch size/concurrency knobs per runtime (e.g., max_batch_size, max_queue_delay).
   - Metrics: Throughput vs p95 vs GPU utilization; queue delay effects.
   - Observation: Identify per-runtime sweet spot and failure modes (timeouts, OOM).
4. **Cost table**
   - Method: Convert tokens/sec to $/1M tokens for each runtime using the same GPU hourly price.
   - Metrics: $/1M tokens, req/sec, latency deltas.

## Analysis & Insights
- Compare scheduling behavior (prefill/decoding overlap, padding waste, queueing).
- Call out runtime-specific bottlenecks (engine build time, cold start, memory footprint).

## Performance & Cost Evaluation
- Translate best-run tokens/sec into $/1M tokens; include build-time amortization for TensorRT-LLM.
- Note utilization headroom and whether CPU or PCIe became the bottleneck.

## Output Quality & Alignment Check
- Spot-check 10 chat + 5 summarization outputs across runtimes; note any decoding mismatches or errors.
- If quantized variant is used, note quality vs BF16 for the same runtime.

## Observability & Reliability Notes
- Capture per-request logs (latency, tokens, draft/target mismatch if supported) and GPU util traces.
- Note any instability (engine load failure, timeout behavior) per runtime.

## Next Steps
- Choose the “default” runtime per workload (chat vs batch) based on results.
- Plan follow-up: integrate the best runtime into later multi-tenant and speculative-decoding days.
