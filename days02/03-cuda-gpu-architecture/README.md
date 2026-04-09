# 03 - CUDA and GPU Architecture

## Scope

- Build intuition for threads, warps, blocks, grids, and the CUDA memory hierarchy.
- Understand occupancy, coalescing, synchronization, launch overhead, and compute-vs-memory bottlenecks.
- Keep hardware and topology orientation here: HBM, Tensor Cores, MIG, PCIe, NVLink, NVSwitch, and NUMA.

## Notes

Use `notes/` for CUDA execution mechanics, roofline reasoning, hardware sizing, topology maps, and placement rules.

## Experiments

Use `experiments/` for kernel profiling studies, topology-aware benchmarks, and placement comparisons.

## Exit Criteria

- You can diagnose compute-bound vs memory-bound behavior with evidence.
- You can identify when topology or placement is the real bottleneck.
