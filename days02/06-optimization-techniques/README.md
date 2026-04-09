# 06 - Optimization Techniques

## Scope

- Study the core performance knobs: quantization, speculative decoding, prefix caching, disaggregation, batching, and parallelism.
- Separate workload-specific wins from cargo-cult tuning.

## Notes

Use `notes/` for tradeoff summaries across FP8, INT4, AWQ, GPTQ, speculation modes, caching, and TP or PP or EP choices.

## Experiments

Use `experiments/` for optimization bakeoffs, quantization comparisons, prefix-cache hit-rate studies, and disaggregation or parallelism ablations.

## Exit Criteria

- Each optimization has a measured benefit and a documented cost.
- Optimization choices are justified by workload behavior, not default habit.
