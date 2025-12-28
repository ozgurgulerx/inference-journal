# Questions For Groq (High-Signal, Non-Fluffy)

Use these to eliminate ambiguity. Anything unanswered becomes an explicit assumption in `risk_register.md`.

---

## A) Compiler + Scheduling

1. What is the **supported shape variability** without recompilation (prompt lengths, output caps, batch size, hidden dims)?
2. What compiler outputs describe **service time** (per token, per request) and the **critical path**?
3. What model/operator patterns trigger **fallback paths** or reduced performance?
4. How stable are compiled artifacts across **compiler versions**? What is the compatibility policy?
5. What is the operational model: compile on customer infra vs Groq-managed compile pipeline?
6. What is the best practice for **artifact provenance** (hash inputs, determinism of compile outputs)?
7. How should we think about **multi-tenancy**: compile-time partitioning vs runtime scheduling?
8. What are the supported/unsupported **precision modes** and their performance implications?
9. What is the recommended workflow for **profiling and schedule inspection** (e.g., GroqView or equivalent), and what artifacts/counters should we export for SRE dashboards?
10. Does the compiler provide an **a priori latency estimator** per artifact/shape, and what is its error envelope?
11. What import formats are officially supported (ONNX, MLIR, framework frontends), and what are the known fidelity/performance pitfalls of each?
12. How are **custom ops** handled (plugins, lowering strategy, fallback behavior), and how does that affect determinism and artifact portability?
13. How do you want us to treat **prefill vs decode**: one artifact per bucket, separate artifacts, or separate “modes” with distinct schedules?
14. What are the compiler-level constraints for attention variants (**MHA vs GQA vs MQA**) and KV-cache layouts (supported `n_heads`, `n_kv_heads`, head dims)?
15. What is the support story for **MoE models** (gating, routing, expert parallelism): supported patterns, compile-time constraints, and expected performance cliffs?

---

## B) Hardware Architecture

1. What are the key on-chip resource limits relevant to inference: **SRAM capacity**, bandwidth, compute throughput?
2. What causes performance cliffs: spills, placement failures, bandwidth saturation, fabric contention?
3. What are the thermal/power behaviors under sustained batch-1 decode at high utilization?
4. What telemetry is available for debugging schedule inefficiency (bubbles/stalls)?
5. What is the practical granularity of “a replica”: full chip vs partitions?
6. Are there named numeric modes (e.g., “TruePoint” as described in Groq materials), and what are the exact accumulator/rounding behaviors we should assume for accuracy validation?
7. Where does the **KV cache** live for representative LLM artifacts (pure on-chip SRAM vs staged vs off-chip memory), and what are the placement constraints as `seq_len` grows?
8. What is the effective **memory hierarchy** we should model (SRAM tiers, any off-chip DRAM), and what telemetry exists to confirm bandwidth/latency behavior under decode?
9. For multi-chip artifacts, what is the **communication model on the critical path** per token (what moves across chips, and when)?
10. What packaging/topology assumptions does the hardware require at scale (board vs chassis vs rack), and what are the lead-time / supply constraints for any required memory adjacency (e.g., if any SKUs rely on HBM-class packaging vs SRAM-first designs)?

---

## C) RealScale Fabric / Networking

1. What is the **determinism boundary**: within chassis, rack, pod?
2. What is the topology (conceptually) and what are the key constraints (cable lengths, port mapping)?
3. What are failure domains: link, node, segment, rack? What are detection + reroute times?
4. How is congestion avoided/managed? Is there admission control at the fabric level?
5. What cross-rack patterns are supported for model partitioning (if any), and with what SLO envelope?
6. What is Ethernet used for and what is explicitly **not** supported on Ethernet critical path?

---

## D) Inference / Runtime Engineering

1. What is the recommended **shape bucketing strategy** for real workloads?
2. What are best practices for **admission control** and queue caps to protect p99?
3. How do you measure and report:
   - decode tok/s,
   - prefill tok/s,
   - service time distribution,
   - queueing delay?
4. What is the recommended strategy for **long-context** requests?
5. What is the maximum safe utilization for stable p99 (the “knee”), and how does it vary by model?
6. Do you recommend any runtime-level batching, or is the intended strategy “batch-1 + concurrency + routing”? If batching exists, is it an explicit compile-time schedule feature?
7. For **low-batch decode**, what is the recommended concurrency model (streams in flight, queue structure), and what are the knobs that actually move TTFT vs steady-state tok/s?
8. Can we disaggregate: **prefill on one pool** (capacity-optimized) and **decode on another** (latency-optimized)? If yes, what is the recommended interface (KV transfer, state format) and overhead?
9. How should we think about **GQA/MQA** in production: are there preferred model families/configs because KV-cache bandwidth dominates decode?
10. What are the operational constraints for **MoE** serving (routing determinism, load balance across experts, hot experts, tail behavior)?

---

## E) Data Center / Facilities

1. What are the rack-level power and cooling requirements per SKU (nominal + worst case)?
2. What are the recommended aisle containment and airflow specs?
3. What are the fabric cabling requirements and installation tolerances?
4. What is the bring-up/burn-in procedure and acceptance testing plan for a new pod?
5. What spares inventory is recommended on-site (cables, nodes, PSUs, fans)?
6. If the deployment is a GroqRack-style system, what are the **pod unit boundaries** and “do not violate” constraints for expansion (cabling, power, airflow)?
