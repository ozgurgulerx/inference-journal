# 04 - Compiler and Kernel Construction Path

## Scope

- Understand how high-level PyTorch code becomes efficient execution.
- Study `torch.inference_mode()`, `torch.compile()`, Triton, CUTLASS or CuTe, CUDA graphs, and export/runtime boundaries.

## Notes

Use `notes/` for graph capture, lowering, fusion, graph breaks, kernel construction, and export tradeoffs.

## Experiments

Use `experiments/` for eager vs compiled comparisons, Triton operators, CUTLASS or CuTe reading notes, and kernel design studies.

## Exit Criteria

- One operator is implemented or modified in Triton.
- The compile path is understandable end to end, including when it helps or fails.
