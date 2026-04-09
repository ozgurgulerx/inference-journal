# 02 - Transformer Inference Mechanics

## Scope

- Understand prefill vs decode behavior at first-principles level.
- Explain KV-cache lifecycle, reuse, and long-context effects.
- Connect request shape changes to latency, throughput, and memory shifts.

## Notes

Use `notes/` for attention mechanics, KV-cache lifecycle, GQA or MQA, MoE, multimodal serving, and bottleneck explanations.

## Experiments

Use `experiments/` for prompt-length sweeps, output-length sweeps, concurrency studies, and cache reuse benchmarks.

## Exit Criteria

- Every major latency shift can be explained before profiling.
- Cache behavior and memory pressure are measured, not guessed.
