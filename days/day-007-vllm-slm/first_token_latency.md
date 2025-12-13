# Day 007 – TTFT Baseline (SLM)

## Server Config
- MODEL=
- vLLM flags:
- GPU=
- OS / kernel=
- vLLM version=

## Single-Run Measurements (Cold vs Warm)
| run   | ttft_s | e2e_s | tokens | notes |
|-------|--------|-------|--------|-------|
| cold_1|        |       |        | first request after server start |
| warm_1|        |       |        | immediate second request |

## Warm-Run Variance (steady state)
Prompt: "<short single-turn prompt>"

| run | ttft_s | e2e_s | tokens | notes |
|-----|--------|-------|--------|-------|
| 1   |        |       |        |       |
| 2   |        |       |        |       |
| 3   |        |       |        |       |
| 4   |        |       |        |       |
| 5   |        |       |        |       |

Summary:
- min_ttft_s =
- median_ttft_s =
- max_ttft_s =

## Sanity Metrics
- GPU memory before server start:
- GPU memory after cold request:
- GPU util during warm probes:
- CPU util during cold vs warm:

## Observations
- What dominated cold time? (weights load? graph warmup? KV alloc?)
- Is warm stable across 3–5 runs?
- Any outliers or spikes? What coincided? (other jobs, clock changes, etc.)

## How I’d explain TTFT to an SRE
- What TTFT represents in user experience terms.
- Why cold vs warm matter for incident/debugging.
- Which knobs we can safely adjust (model size, `max-model-len`, batching policy) and what they trade off.

