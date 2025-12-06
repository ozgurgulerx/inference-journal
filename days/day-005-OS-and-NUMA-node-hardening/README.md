# Day 005 – OS & NUMA Node Hardening for Inference

> **Theme**: GPU node topology, NUMA, CPU pinning, and their impact on inference latency/throughput.  
> **Layer focus**: Hardware & OS → Inference Runtime  
> **Goal**: Build a NUMA- and CPU-aware baseline for running an inference server on a single GPU node, and turn it into a reusable “node hardening” pattern.

---

## What You’ll Do

- Inspect CPU, NUMA, and GPU topology on your node.  
- Benchmark a simple inference server **before vs after** NUMA + CPU pinning.  
- (Tier 2) Encode those choices into a **systemd/cgroup-aware service**.  
- (Tier 3) Explore CPU governors, noisy neighbors, and document how these knobs affect p95 and overall capacity in a way that’s useful for future consulting work.

---

## Tiers

- **[Tier 1 – Core: NUMA & CPU-Aware Baseline](LOG_tier01.md)**  
  Inspect topology and run baseline vs pinned inference benchmarks.

- **[Tier 2 – Extension: Systemd & Service-Level Affinity](LOG_tier02.md)**  
  Encode NUMA/CPU policies into a reproducible systemd service template.

- **[Tier 3 – Deep Work: Governors, Noisy Neighbors & Consulting Hook](LOG_tier03.md)**  
  Explore CPU governor + noisy-neighbor impact, and package findings as an “Inference Node OS & NUMA Hardening” module.

---

## Cross-Day References

- **Day 002 – GPU Node Bring-Up**: base OS/GPU setup and first `nvidia-smi`/`lspci` checks.  
- **Day 003 – vLLM Capacity & OOM**: capacity grids that will later run on this hardened node.  
- **Day 004 – Quantization vs BF16**: quantized vs BF16 capacity/quality experiments that will benefit from a stable OS/NUMA baseline.

