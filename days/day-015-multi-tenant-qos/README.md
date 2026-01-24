# Day 015 – Multi-Tenant Load & QoS Isolation

## Goals
- **Primary Objective:** Stress a single GPU with mixed workloads and test QoS policies.
- **Focus Theme(s):** Multi-tenancy, isolation, fairness; scheduler tuning; MIG/partitioning.

## Setup & Context
- **Hardware/Infra:** 1× GPU with MIG support preferred (A100/H100); otherwise simulate via multiple runtime instances; CPU pinning for noisy-neighbor tests.
- **Model(s):** One interactive chat model + one batch/offline model (could share weights); optional smaller model for low-priority jobs.
- **Runtime/Engine:** vLLM (multi-instance or single with scheduling tweaks); optional second runtime (TGI/TRT-LLM) for workload segregation.
- **Dataset/Workload:** Mix of latency-sensitive chat streams and large batch jobs; include at least one “spiky” tenant.
- **Key Config Knobs:** Concurrency/batch limits per tenant, queue priorities, admission control, MIG partition sizes, rate limits.

## Experiments & Measurements
1. **Baseline: no isolation**
   - Method: Run chat + batch simultaneously with default scheduler.
   - Metrics: p50/p95/p99 latency per workload, throughput, GPU/CPU util.
2. **Soft QoS: priority queues or per-tenant batch caps**
   - Method: Apply queue priorities or per-tenant max batch size.
   - Metrics: Tail latency improvement for chat; throughput change for batch.
3. **Hard isolation: split instances or MIG**
   - Method: Run two runtimes or MIG partitions (if available) per workload.
   - Metrics: Isolation effectiveness vs total throughput loss.
4. **Speculative + QoS combo (optional)**
   - Method: Enable speculative decoding on chat path; observe interactions with batching.
   - Metrics: Latency and acceptance rate changes under contention.
5. **Cost/Fairness accounting**
   - Method: Attribute GPU time/tokens per tenant.
   - Metrics: $/tenant/hour, $/1M tokens per tenant, fairness of SLO attainment.

## Analysis & Insights
- Which policies protect chat p99 with minimal throughput loss.
- Trade-offs between utilization and fairness; when to split workloads physically vs logically.

## Performance & Cost Evaluation
- Show $/1M tokens per tenant before vs after QoS.
- Note utilization headroom after isolation and any CPU bottlenecks from multiple runtimes.

## Output Quality & Alignment Check
- Confirm no unexpected truncations/refusals from aggressive scheduling; note guardrail impact if enabled.

## Observability & Reliability Notes
- Per-tenant logging (latency, tokens, cost) and alerting on SLO breach.
- Behavior under overload (queue growth, drops) and recovery steps.

## Next Steps
- Feed QoS defaults into the playbook and size guidance for client scenarios.
