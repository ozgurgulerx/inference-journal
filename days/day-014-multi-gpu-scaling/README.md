# Day 014 – Multi-GPU Scaling Trial (Data vs Tensor Parallel)

## Goals
- **Primary Objective:** Measure scaling efficiency when moving from 1 → 2 GPUs for serving.
- **Focus Theme(s):** Heterogeneous/distributed serving; scaling vs complexity; cross-GPU overhead.

## Setup & Context
- **Hardware/Infra:** 2× GPUs on the same host (NVLink vs PCIe noted). If only 1 GPU is available, simulate via smaller models + profiling or use case study data.
- **Model(s):** 13B–70B depending on hardware; pick a size that motivates tensor parallel; also run a 7B for data-parallel test.
- **Runtime/Engine:** vLLM multi-GPU options; TensorRT-LLM pipeline/tensor parallel if available.
- **Dataset/Workload:** Chat (short) + batch summarization (long) with known request shapes.
- **Key Config Knobs:** Parallelism mode (data vs tensor), `tensor-parallel-size`, `pipeline-parallel-size`, scheduler/batching knobs, NCCL env (NVLink vs PCIe).

## Experiments & Measurements
1. **Data-parallel serving (independent replicas)**
   - Method: Run two model replicas on separate GPUs; load-balance requests.
   - Metrics: Throughput vs latency; scaling efficiency vs 1 GPU; CPU/network load.
2. **Tensor-parallel single model**
   - Method: Split a larger model across 2 GPUs; same workloads.
   - Metrics: Latency impact (cross-GPU comm), throughput, VRAM per GPU.
3. **Scale curve**
   - Method: Concurrency sweep (e.g., 1/8/16/32) for each mode.
   - Metrics: Tokens/sec vs p95; efficiency = (2GPU throughput)/(2 × 1GPU throughput).
4. **Failure/coordination check**
   - Method: Kill one process or block NCCL traffic briefly; observe recovery or crash.

## Analysis & Insights
- When scaling helps vs when comms overhead dominates.
- Differences between NVLink vs PCIe scenarios; operator fusion impact if seen.

## Performance & Cost Evaluation
- Translate best setups to $/1M tokens vs single GPU; include overhead of idle time if scaling is poor.
- Note GPU memory headroom and whether batch size can increase on 2 GPUs.

## Output Quality & Alignment Check
- Ensure outputs remain consistent across modes; note any determinism changes.

## Observability & Reliability Notes
- Capture NCCL logs, GPU utilization per device, and latency distribution per mode.
- Note failover behavior and any scheduling imbalances.

## Next Steps
- Feed learnings into Day 015 (multi-tenant) and Day 018 (aligned model serving) for sizing guidance.
