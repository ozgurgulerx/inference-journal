# 05 - Serving Engines in Depth

## Scope

- Use vLLM as the home base and compare it with SGLang and TensorRT-LLM or Dynamo.
- Understand scheduler design, PagedAttention, prefix caching, disaggregated prefill, and runtime tradeoffs.

## Notes

Use `notes/` for engine architecture summaries, code-path notes, scheduler behavior, cache design, and config surfaces.

## Experiments

Use `experiments/` for same-model cross-engine benchmarks under short, long, concurrent, and quantized workloads.

## Exit Criteria

- Engine differences are measurable and explainable.
- You can justify why engine A wins for workload X.
