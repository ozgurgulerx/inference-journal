# Glossary (Groq-Native)

**Admission control:** A policy that decides whether a request is admitted for service now, queued, or rejected, based on predicted cost and SLO protection.

**Artifact (compiled):** The output of compiling a model graph for a target Groq system under specific shape constraints; treated as a versioned deployment unit.

**Bucket (shape bucket):** A discrete class of requests with bounded prompt/output token sizes (and sometimes batch constraints) mapped to a specific compiled artifact or pool.

**Deterministic service time:** For a fixed compiled artifact and fixed shape bucket, the service time is intended to be stable/predictable (excluding queueing delay).

**Failure domain:** The boundary within which a fault can occur and affect service (e.g., link segment, node, rack, pod).

**Headroom (`ρ_target`):** The target utilization chosen to protect tails and allow failure tolerance; lower headroom means lower utilization.

**Knee (`ρ_knee`):** The utilization region where small increases in load cause large increases in queueing delay (tail latency).

**LPU:** Groq’s Language Processing Unit; treated here as a compiler-scheduled deterministic compute engine optimized for inference.

**Pool:** A set of replicas serving the same compiled artifact and shape bucket(s), with its own admission control and queue.

**Prefill:** The phase where the prompt tokens are processed to initialize attention/KV state for generation.

**Decode:** The autoregressive generation phase where tokens are generated one at a time using the existing context/KV state.

**Queueing delay (`T_queue`):** Time spent waiting before service begins; dominant tail driver under load in deterministic systems.

**Service time (`T_service`):** Time spent in computation and deterministic data movement for a request, excluding queueing.

**Shape discipline:** The operational practice of constraining and bucketing request shapes so compiled schedules remain valid and predictable.

**SLO:** Service Level Objective; the target percentile(s) for latency/availability (e.g., p99 < 300ms).

**TTFT:** Time To First Token; must be defined precisely (does it include queueing? prefill? network?).

---

## GPU Intuition Traps (Glossary Entries)

**Dynamic batching (GPU-style):** Opportunistic runtime merging of heterogeneous requests to increase throughput; often increases queueing delay and violates deterministic service assumptions on Groq unless explicitly supported.

**“Caches will save us”:** A mental model where irregular access patterns are forgiven by caching; Groq-native thinking requires explicit shape and residency planning.

