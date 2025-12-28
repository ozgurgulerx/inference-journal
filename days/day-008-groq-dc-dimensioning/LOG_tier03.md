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

### 2.3 Cross-compiler literacy: how GPU toolchains mislead Groq designs

This is the “translate without importing assumptions” section. You should be fluent in GPU inference compilers (TensorRT/ORT/TVM) to avoid abusing Groq.

| Ecosystem | What it optimizes | Typical mode | Groq-native translation |
|---|---|---|---|
| TensorRT | kernel fusion + tactic selection | runtime/build-time autotuning + kernels | Groq compiler emits a deterministic whole-graph schedule; you don’t “swap tactics at runtime” without changing the artifact |
| ONNX Runtime | graph rewrites + backend EP selection | dynamic selection of providers | Groq import/compile is the backend; treat provider/backend as an explicit deployment choice |
| TVM | schedule search/autotuning | explores many schedules per target | Groq compiler is effectively the schedule author; your lever is shape discipline + artifact selection, not runtime exploration |

**GPU intuition trap:** “We can just port our TensorRT/Triton habits.” On Groq, the runtime lever is admission and routing; the performance lever is compilation inputs and artifact choice.

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

## 5.1 Compression as a DC Design Lever (Quantization, Pruning, Distillation)

This is where general inference engineering skills directly change your facility plan.

- **Inference:** If a model can be distilled or structurally reduced to fit into fewer LPUs (or smaller partitions), your **rack count, cabling, and redundancy math** can change more than any micro-optimization.
- **Assumption to validate:** Exact numeric modes supported by your Groq SKU/compiler (e.g., “TruePoint” claims in Groq materials). Treat numerics as part of the compiled artifact and validate accuracy per mode.

### Quantization (Groq-native stance)

- **Inference:** Quantization is not “turn on INT8 and win.” It is:
  - an accuracy contract (golden sets),
  - a compiler/schedule contract (what ops are supported),
  - an ops contract (separate artifacts per numeric mode).

### Pruning (Groq-native stance)

- **Inference:** Unstructured sparsity rarely helps unless the hardware/compiler exploits it. Structured pruning changes shapes and can change placement and schedule efficiency (good or bad).
- **Decision:** Only pursue pruning if it simplifies shapes or reduces partition count in a way that materially reduces failure-domain coupling or rack count.

### Distillation (Groq-native stance)

- **Inference:** Distillation is often the cleanest way to “fit” within SRAM-driven constraints: a smaller student model can shift you from multi-LPU partitioning to single-replica service units, simplifying fabric and tail behavior.

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

---

## 9) MoE workloads (why they’re trickier for deterministic schedules)

MoE changes each layer’s MLP from “always run the same weights” to “route each token to a subset of experts.”

- **Inference:** The MoE risk isn’t the math; it’s **routing irregularity**:
  - hot experts (load imbalance) can create tails,
  - expert routing can create all-to-all style communication in multi-device settings,
  - the expert set per token is data-dependent (harder to statically schedule).
- **Assumption to validate (Groq specifics):** Whether MoE is supported by constraining routing patterns/top-k and compiling deterministic expert placement + comm.

Meeting questions:

- What is the stable service-curve contract for MoE (worst-case vs typical routing)?
- How do you prevent hot experts from dominating p99?
- What is the multi-chip communication pattern on the critical path per token?

---

## 10) Daily Pulse — Prefill/Decode Disaggregation + Memory Economics + ASIC Strategy

### 10.1 Disaggregation is an “HBM budget optimizer” (deep but simple)

- **Prefill** wants **capacity** (big contexts; lots of KV written).
- **Decode** wants **low latency + bandwidth**; at low batch you can’t hide memory stalls behind batching the way GPUs often do.

**Inference:** Splitting inference into **prefill vs decode** lets you spend expensive bandwidth memory (HBM-like) only where it buys user-perceived latency, and serve decode with a different hardware point optimized for KV-heavy, low-batch behavior.

### 10.2 Groq-style SRAM decode thesis (why it exists)

- **Inference:** A compiler-scheduled, SRAM-first decode engine lets you reserve HBM-heavy GPUs for regimes where they dominate (training, high-batch inference, heavy prefill) and push low-batch, latency-sensitive decode to an SRAM-first deterministic pipeline.
- Even if SRAM is expensive per bit, the bet is that it buys **very high effective bandwidth/low latency** where decode is KV-cache dominated.

### 10.3 “Rubin / Rubin CPX / Rubin SRAM” (third-party thesis; capture without claiming)

Treat the following as **Assumption to validate** (until NVIDIA product docs confirm):

- “Rubin CPX” (GDDR DRAM) as capacity-optimized prefill hardware (massive context windows, lower bandwidth).
- “Rubin” (HBM DRAM) as balanced training + high-density/batched inference.
- A Groq-derived “Rubin SRAM” as ultra-low-latency decode hardware (agentic/reasoning), with prefill likely on CPX or standard Rubin.

### 10.4 Lossless distillation → ultra-dense distillation

**Dense (lossless) paragraph:** Inference is splitting into prefill and decode: prefill/context building wants memory capacity (often cheaper/denser per $), while decode wants low latency + bandwidth and, at low batch, cannot hide memory stalls behind batching like GPUs, so decode becomes KV-cache-latency/bandwidth dominated. Disaggregation therefore acts as an “HBM budget optimizer”: reserve expensive HBM-heavy GPUs for regimes where they dominate and push low-batch decode to SRAM-first deterministic pipelines where a compiler schedules a distributed on-chip SRAM working set, trading capacity for extremely high effective bandwidth and predictable service time.

**Ultra-dense (still lossless) paragraph:** Prefill is capacity-hungry; decode is low-batch KV-cache-latency/bandwidth-hungry; disaggregate to spend HBM only where it wins, and use SRAM-first deterministic pipelines for latency decode because batching can’t hide stalls there.

### 10.5 Open questions (record, don’t guess)

- Who is **Jay Y. Lee** (Samsung leadership context), and what is his education/background? *(Research TODO; cite sources before asserting.)*
- How are recent **DRAM/HBM price cycles** influencing GPU BOM and the incentive to add SRAM-first decode SKUs? *(Hypothesis: HBM price pressure increases value of disaggregation.)*

### 10.6 How to learn to make inferences like this (repeatable method)

- Build a bottleneck model per phase (prefill vs decode): compute vs memory vs comm, especially at low batch.
- Translate claims into primitives: KV bytes/token, bandwidth needs, latency budget, utilization knee, queueing sensitivity.
- Track memory roadmaps (DDR/GDDR/HBM) as first-class constraints (capacity, bandwidth, supply, cost).
- Separate **Fact vs Inference vs Assumption**, and keep a falsification checklist.
