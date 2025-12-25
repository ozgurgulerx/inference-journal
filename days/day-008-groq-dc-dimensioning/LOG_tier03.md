# LOG (Tier 03) — Deep Trade-offs, Failure Modes, Expert Reasoning

Tier 03 is the interrogation layer: if a Groq staff engineer tries to break your plan, you survive by being explicit about constraints, failure modes, and what you’ll validate.

---

## 1) Determinism: The Hidden Cost Is “Shape Discipline”

### 1.1 Determinism does not mean “no tail”

- **Inference:** Tail latency shifts from compute variance → queueing + shape outliers + failover events.
- **Failure mode:** A small tail of long prompts (`P99(P)`) consumes disproportionate capacity, causing global queue buildup.
- **Mitigation:** bucket + cap + price (different SLO tiers) + route long-context to a dedicated pool.

### 1.2 Multi-tenancy is compile-time + admission-time

GPU multi-tenancy often relies on runtime isolation (MIG, MPS, cgroups + kernel scheduling).

**Groq-native view (Inference):**

- isolation is achieved by:
  - dedicating compiled artifacts to partitions/pools,
  - strict admission per pool,
  - shaping traffic per artifact (rate limiting).

**Failure mode:** “Noisy neighbor” is now “shape neighbor” (one shape class starving others).

---

## 2) Compiler Failure Modes (What Breaks in Real Life)

### 2.1 Compile-time variability becomes an ops risk

Potential risks (Assumptions to validate with Groq):

- compiler version changes performance materially,
- compilation is sensitive to minor model graph changes,
- compilation time is non-trivial for large models,
- compiled artifacts may be hardware-generation specific.

Mitigations:

- pin compiler versions per release train,
- treat compiled artifacts as immutable build outputs,
- maintain a compile farm and an artifact registry.

### 2.2 “Model parallelism” on Groq ≠ GPU tensor/pipeline parallelism

GPU world:

- tensor parallel: shard matmuls across devices with collective comms,
- pipeline parallel: split layers across devices with micro-batches,
- expert parallel: MoE routing + all-to-all.

Groq-native translation (Inference):

- “parallelism” is **compiler placement across LPUs + deterministic fabric schedule**.
- performance is not “collective bandwidth” alone; it is **end-to-end schedule critical path**.

Decision questions:

- What is the partition strategy (layer-wise, tensor-wise, operator-wise)?
- What communications are on the critical path per token?
- What is the failure domain of each partition?

---

## 3) Fabric / Networking: Where Determinism Dies

### 3.1 Across-rack scaling: what you must not hand-wave

If you split a model across racks:

- **Inference:** tail becomes sensitive to:
  - congestion and routing variability,
  - link failures and re-convergence,
  - clock drift / jitter budgets (if tightly scheduled).

Decision forcing:

- If cross-rack is required, demand a Groq-validated reference architecture and SLO envelope.
- Otherwise, enforce “critical path stays within rack/pod.”

### 3.2 Ethernet: what it is for (be explicit)

Use Ethernet for:

- ingress traffic (client → frontends),
- control plane, metrics, logs, orchestration,
- artifact distribution (compiled schedules),
- non-critical background transfers.

**Assumption to validate:** Whether any Groq deployment uses Ethernet for model-parallel critical path; do not assume.

---

## 4) Performance Reality: “Saturation” and the Queueing Knee

### 4.1 Deterministic service time means the knee is sharper

In an M/D/1 approximation:

`Wq ∝ 1/(1-ρ)` (diverges as ρ→1)

**Inference:** When service time is stable, you see a cleaner knee: queueing grows rapidly near saturation.

Decision:

- choose `ρ_target` and defend it with a burst model (not a single average).

### 4.2 Burstiness and admission: p99 is a control problem

If arrivals have burst factor `B` (peak/mean over window):

- capacity must cover `λ_peak`, not `λ_mean`.
- queue cap must bound time-in-queue.

**Mitigation:** token-budget admission (predict service time), not request-count admission.

---

## 5) Model Shape Failure Modes (Common “Why is this slow?” Root Causes)

1. **Too many shape variants**
   - **Symptom:** artifact sprawl; ops can’t manage; compilation backlog.
   - **Fix:** discretize into buckets; enforce request normalization.
2. **Max sequence creep**
   - **Symptom:** service time rises silently; SLO breaks during peak.
   - **Fix:** enforce caps; create a long-context tier.
3. **Incompatible ops / graph changes**
   - **Symptom:** compiler fallback path or placement inefficiency.
   - **Fix:** keep a “compiler-friendly operator subset” policy; validate changes early.

---

## 6) Data Center Failure Modes (The Physical World Still Wins)

### 6.1 Power

Failure modes:

- PDU trip / breaker derating,
- transient spikes tripping protection,
- uneven phase loading.

Mitigation:

- per-rack power telemetry,
- conservative derating policy,
- staged power-up and burn-in.

### 6.2 Cooling

Failure modes:

- hot aisle recirculation,
- cable bundles blocking airflow,
- fan failures cascading to thermal throttling.

Mitigation:

- enforce airflow/cable standards,
- thermal mapping during burn-in,
- strict hot-aisle containment discipline.

### 6.3 Cabling / Fabric mapping

Failure modes:

- wrong port mapping breaks topology assumptions,
- intermittent connectors cause “ghost” errors.

Mitigation:

- serialized cable plan + QR labeling,
- link burn-in + periodic validation,
- keep spares on-site.

---

## 7) Expert-Level “Why” Answers (Short)

- **Why does Groq p99 look stable at low load?**
  - **Inference:** deterministic service time + low queueing → tight distribution.
- **Why does p99 still explode at high load?**
  - **Inference:** queueing divergence near ρ→1; determinism doesn’t fix queueing theory.
- **Why is GPU-style dynamic batching dangerous here?**
  - **Inference:** it introduces waiting windows and heterogeneity that break deterministic scheduling assumptions.

---

## 8) Tier 03 Acceptance Self-Test

You can defend:

- an explicit list of assumptions about compiler, fabric, and shapes,
- a queueing + admission policy that bounds p99,
- a failure-domain-aware redundancy plan,
- a cross-rack scaling position (allowed vs forbidden, with rationale).

