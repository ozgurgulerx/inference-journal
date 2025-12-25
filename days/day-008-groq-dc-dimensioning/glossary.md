# Glossary (Groq-Native)

**Admission control:** A policy that decides whether a request is admitted for service now, queued, or rejected, based on predicted cost and SLO protection.

**Artifact (compiled):** The output of compiling a model graph for a target Groq system under specific shape constraints; treated as a versioned deployment unit.

**Bucket (shape bucket):** A discrete class of requests with bounded prompt/output token sizes (and sometimes batch constraints) mapped to a specific compiled artifact or pool.

**Deterministic service time:** For a fixed compiled artifact and fixed shape bucket, the service time is intended to be stable/predictable (excluding queueing delay).

**Failure domain:** The boundary within which a fault can occur and affect service (e.g., link segment, node, rack, pod).

**Headroom (`ρ_target`):** The target utilization chosen to protect tails and allow failure tolerance; lower headroom means lower utilization.

**Knee (`ρ_knee`):** The utilization region where small increases in load cause large increases in queueing delay (tail latency).

**LPU:** Groq’s Language Processing Unit; treated here as a compiler-scheduled deterministic compute engine optimized for inference.

**GroqWare:** Groq’s software/tooling suite (vendor term); treat the official Groq documentation as the source of truth for components and support boundaries.

**GroqFlow:** Groq’s model import/compile workflow/tooling (vendor term); in this playbook, it is the place you get: compilation, artifact metadata, and (ideally) performance reports.

**GroqView:** Groq’s profiling/visualization tooling (vendor term); use it (or the current equivalent) to extract schedule-level evidence for “where time goes” beyond end-to-end latency.

**GroqRack:** A rack-scale Groq deployment unit (vendor term); treat its topology/power/cooling/cabling constraints as “part of the machine,” not optional infra.

**Pool:** A set of replicas serving the same compiled artifact and shape bucket(s), with its own admission control and queue.

**Prefill:** The phase where the prompt tokens are processed to initialize attention/KV state for generation.

**Decode:** The autoregressive generation phase where tokens are generated one at a time using the existing context/KV state.

**Queueing delay (`T_queue`):** Time spent waiting before service begins; dominant tail driver under load in deterministic systems.

**Service time (`T_service`):** Time spent in computation and deterministic data movement for a request, excluding queueing.

**Shape discipline:** The operational practice of constraining and bucketing request shapes so compiled schedules remain valid and predictable.

**SLO:** Service Level Objective; the target percentile(s) for latency/availability (e.g., p99 < 300ms).

**TTFT:** Time To First Token; must be defined precisely (does it include queueing? prefill? network?).

**TruePoint:** A Groq vendor term for a numeric format/approach described in some Groq materials; do not assume details—confirm the exact precision/accumulation/rounding behavior for your SKU and compiler version before using it in accuracy-critical claims.

---

## Comparative Ecosystem Terms (For Translation, Not As Design Targets)

**TensorRT:** NVIDIA’s GPU inference compiler/runtime. Useful as a contrast case: GPUs often rely on kernel libraries and tactic selection, while Groq relies on whole-graph static scheduling.

**Tensor Cores / matrix engines (GPU):** Dedicated matrix-math units inside a GPU SM that execute matrix-multiply instructions issued by warps (LLM matmuls spend most time here). They accelerate math but do not remove runtime scheduling and memory variability.

**GEMA (generic accelerator term):** “General Matrix Accelerator” (informal/generic label) for matrix engines; on NVIDIA GPUs, the analogous unit is the Tensor Core. Use as a conceptual term only—prefer vendor terms when speaking precisely.

**ONNX / ONNX Runtime:** A portable model format and a runtime with multiple hardware “execution providers.” Useful analogy for “portable graph → hardware-specific backend,” but do not assume Groq behaves like ORT providers.

**TVM:** A compiler stack emphasizing schedule exploration/autotuning. Useful for understanding schedule search, but Groq compilation aims at deterministic schedules rather than runtime exploration.

**Olive (Microsoft):** A hardware-aware optimization toolchain built on ONNX Runtime concepts; useful as a template for building a policy-driven artifact pipeline (optimize → measure → select), including for Groq artifacts.

---

## GPU Intuition Traps (Glossary Entries)

**Dynamic batching (GPU-style):** Opportunistic runtime merging of heterogeneous requests to increase throughput; often increases queueing delay and violates deterministic service assumptions on Groq unless explicitly supported.

**“Caches will save us”:** A mental model where irregular access patterns are forgiven by caching; Groq-native thinking requires explicit shape and residency planning.
