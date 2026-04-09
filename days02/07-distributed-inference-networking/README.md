# 07 - Distributed Inference and Networking

## Scope

- Understand NCCL, collectives, multi-GPU and multi-node scaling, and network-aware serving behavior.
- Connect TP, PP, EP, and data-parallel choices to communication cost and tail latency.

## Notes

Use `notes/` for collective patterns, topology maps, NUMA alignment, network constraints, and scheduler interactions.

## Experiments

Use `experiments/` for all-reduce and all-gather tests, scaling curves, and within-node vs cross-node comparisons.

## Exit Criteria

- Communication overhead is measured directly.
- You can diagnose why multi-GPU or multi-node serving got slower.
