# 05 - Compiler and Kernel Stack

## Scope

- Understand the path from eager PyTorch to optimized kernels.
- Study `torch.compile`, Triton, and lower-level kernel tooling.

## Notes

Use `notes/` for graph capture, lowering, fusion, and kernel construction notes.

## Experiments

Use `experiments/` for eager vs compiled comparisons, Triton operators, and kernel design studies.

## Exit Criteria

- One operator is implemented or modified in Triton.
- The compile path is understandable end to end.
