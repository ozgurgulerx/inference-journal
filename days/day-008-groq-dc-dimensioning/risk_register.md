# Risk Register (Groq LPU Inference + DC)

**Scales:** Likelihood (L/M/H), Impact (L/M/H).  
**Labels:** Fact (from Groq reference) / Inference / Assumption to validate.

---

## R1 — Shape Sprawl → Artifact Explosion

- Likelihood: H | Impact: H
- Description: Too many supported prompt/output shapes cause compile+QA+ops overload and unpredictable capacity.
- Triggers: Frequent “just increase max tokens” requests; many customer-specific configs.
- Detection: Artifact count rising; compile queue delays; inconsistent service curves.
- Mitigations:
  - Enforce buckets + caps at gateway.
  - Long-context as separate tier.
  - Artifact registry + CI gating.
- Owner: Inference Lead

## R2 — Compilation Time Blocks Rapid Iteration

- Likelihood: M | Impact: H
- Description: Compile times or artifact validation slows A/B tests and emergency patches.
- Triggers: Model updates demanded daily; frequent prompt/template changes affecting graph.
- Detection: Compile pipeline backlog; missed release windows.
- Mitigations:
  - Precompile known variants.
  - Compile farm capacity planning.
  - Change taxonomy (weights-only vs graph/shape changes).
- Owner: ML Platform

## R3 — Cross-Rack Critical Path Introduces Non-Deterministic Tail

- Likelihood: M | Impact: H
- Description: Model partitioning across racks introduces congestion variance and failure coupling.
- Triggers: Attempts to scale a single model beyond rack/pod deterministic boundary.
- Detection: p99 spikes correlated with DC network events; hard-to-reproduce tail.
- Mitigations:
  - Keep critical-path comm within validated deterministic domain.
  - If cross-rack required, demand Groq reference architecture + SLO envelope.
- Owner: NetEng

## R4 — Queueing Knee Misestimated → p99 Collapse

- Likelihood: H | Impact: H
- Description: Running too “hot” pushes utilization near saturation; queueing dominates.
- Triggers: Cost pressure to increase utilization without revalidation.
- Detection: p99 grows with small load increases; queue depth rising.
- Mitigations:
  - Knee-finding load ramps per bucket.
  - Token-budget admission.
  - Bounded queues + shedding.
- Owner: SRE

## R5 — Long Prompts Cause Tail Externalities

- Likelihood: H | Impact: H
- Description: Small fraction of long prompts consumes capacity and increases queueing for all.
- Triggers: Product enables long context universally.
- Detection: Latency spikes correlate with high prompt lengths; prefill dominates.
- Mitigations:
  - Dedicated long-context pools.
  - Pricing/SLO tier separation.
  - Caps + routing by bucket.
- Owner: Product + Inference

## R6 — Artifact Compatibility Drift (Compiler/Hardware)

- Likelihood: M | Impact: H
- Description: Compiler or hardware version changes alter performance or compatibility.
- Triggers: Upgrades without full perf regression suite.
- Detection: Service time shifts at constant load; unexpected compile failures.
- Mitigations:
  - Version pinning; compatibility matrix.
  - Canary rollout with perf gates.
  - Rollback plan for artifacts and compiler.
- Owner: ML Platform

## R7 — DC Power/Cooling Mismatch

- Likelihood: M | Impact: H
- Description: Sustained power/thermal load exceeds design assumptions causing throttling/trips.
- Triggers: Rack density increased; containment not enforced.
- Detection: Inlet temps rising; fan alarms; PDU trips.
- Mitigations:
  - Worst-case sustained power spec + derating.
  - Burn-in thermal mapping.
  - Strict airflow/cabling standards.
- Owner: Facilities

## R8 — Fabric Cabling Errors Break Topology Assumptions

- Likelihood: M | Impact: M/H
- Description: Wrong mapping or intermittent cables cause instability and hard debugging.
- Triggers: Manual installs without validation; missing labels.
- Detection: Link flaps; intermittent errors.
- Mitigations:
  - Serialized cabling plan and QR labels.
  - Link burn-in and periodic validation.
  - Spare cables on-site.
- Owner: NetEng + Facilities

## R9 — Observability Insufficient To Separate Queue vs Service

- Likelihood: H | Impact: H
- Description: Without instrumentation, you can’t tell if tail is queueing or service time shift.
- Triggers: Only end-to-end latency tracked.
- Detection: Incidents with “unknown root cause.”
- Mitigations:
  - Instrument `t_enqueue`, `t_start`, `t_first_token`, `t_done`.
  - Per-bucket dashboards and alerts.
- Owner: SRE + Inference

## R10 — Overreliance on “Public Positioning” Without Validation

- Likelihood: M | Impact: H
- Description: Assumptions about determinism/fabric without Groq-confirmed envelopes lead to wrong design.
- Triggers: Architectural decisions without Groq reference artifacts.
- Detection: Disagreements in technical review; unexplained deviations.
- Mitigations:
  - Track assumptions explicitly.
  - Convert critical assumptions into questions (see `questions_for_groq.md`).
- Owner: Tech Lead
