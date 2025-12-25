# LOG (Tier 01) — Groq Architecture Mastery (Mental Models + Inference Behavior)

This tier rebuilds the mental model: Groq is not “a faster GPU.” It is a compiler-scheduled, deterministic machine where inference performance is a *static property of the compiled schedule* plus *queueing and admission control*.

---

## 0) Truth Labels (Use These In Meetings)

You must separate confidence levels explicitly:

- **Fact (from Groq reference):** Documented by Groq (official public material or a Groq engineering artifact shared in your engagement). If you can’t point to a specific reference yet, downgrade to **Assumption to validate**.
- **Inference:** Derived from facts + standard systems/queueing/compiler principles.
- **Assumption to validate:** Anything you’re carrying from outside Groq references or not yet confirmed for your specific hardware/software version.

If you cannot cite a Groq reference in your engagement, label it **Assumption to validate** and add it to `questions_for_groq.md`.

---

## 1) Groq LPU Mental Model (Rebuilt From First Principles)

### 1.0 A GPU mental model (for contrast)

If you want a clean, GPU-native hierarchy to anchor the contrast, use `gpu_to_lpu_bridge.md` (includes a local visual: `assets/gpu_arch.png`).

### 1.1 Deterministic single-program execution (vs GPU “many schedulers”)

- **Fact (from Groq reference):** Groq emphasizes deterministic execution; the compiler produces a static schedule. *(Attach the specific Groq reference.)*
- **Inference:** You should model the LPU like a **time-scheduled pipeline**: for a given compiled graph shape, the time per token and time per request are largely predictable.
- **GPU intuition trap:** On GPUs, *runtime scheduling* is a major “hidden system.” On Groq, scheduling is *front-loaded* into compilation.

**Consequence:** Your performance debugging shifts from “kernel-level profiling + occupancy” to:

1. “Did we compile the right shape?”
2. “Is the schedule efficient (placement, tiling, overlap)?”
3. “Are we saturating the pipeline (admission/concurrency)?”
4. “Are we queueing (SLO violation is often queueing, not compute variance)?”

### 1.2 “No warp scheduling, no cache hierarchy” → what replaces it

- **Assumption to validate (wording):** “No warp scheduling” and “no cache hierarchy” are GPU-native terms. What matters is that Groq seeks **predictable control** rather than opportunistic caches and dynamic scheduling.
- **Inference:** Replace these GPU concepts with Groq-native concepts:

| GPU concept | Groq-native replacement question |
|---|---|
| Warp scheduling variance | Is the static schedule balanced, and are there bubbles/stalls? |
| Cache hit/miss | Are tensors staged explicitly as planned? Any spill/oversubscription? |
| HBM contention | What is the deterministic memory bandwidth plan per cycle? |
| Kernel launch overhead | What is the fixed pipeline stage overhead per token/request? |

### 1.3 SRAM-as-primary memory: why it helps and why it constrains

- **Fact (from Groq reference):** Groq emphasizes SRAM-first design and predictable data movement. *(Attach the specific Groq reference.)*
- **Inference:** SRAM-first eliminates cache-miss tail and makes latency predictable, but it creates **hard capacity constraints**:
  - tensor residency windows must fit,
  - shape changes can break placement,
  - “just make it bigger” often becomes “recompile / reshard / change topology.”

**Design rule:** When you ask “can this model run,” you are asking:

- can it be placed given SRAM capacity,
- can the schedule meet timing given bandwidth,
- can the fabric movement be overlapped predictably.

### 1.4 Deterministic pipelines collapse p99 variance — but not queueing

- **Inference:** If service time is stable, **compute variance decreases**, so p99 collapses *for a fixed concurrency regime*.
- **Inference:** Under load, p99 is dominated by **queueing delay**, not compute jitter. That means:
  - admission control is the SLO lever,
  - headroom targets must be explicit,
  - “run hotter and rely on batching” is a GPU carry-over that fails.

### 1.5 When determinism is a constraint (not a benefit)

Determinism constrains you when:

- model shapes vary widely (dynamic shapes),
- you need frequent A/B model swaps without precompilation,
- you rely on opportunistic batching over heterogeneous request lengths,
- you need fine-grained multi-tenancy with unpredictable interference.

**Meeting statement:** “We’re buying determinism; we must also buy *shape discipline* and *compilation operations*.”

---

## 2) Compiler-First Inference (The Compiler *Is* The Runtime)

### 2.1 The deployment artifact

Treat a Groq deployment artifact as:

`(model weights, graph, max seq, batch policy, precision, compiler version, target LPU/fabric generation) → compiled schedule/binary`

**Inference:** Ops workflows must include:

- compile farm / CI for compilation,
- artifact versioning and provenance,
- rollbacks across compiler versions,
- canarying compiled schedules (not just containers).

### 2.2 Compile-time graph placement vs runtime scheduling

- **Inference:** GPUs schedule at runtime; Groq schedules at compile time. So you must ask:
  - what parts are pipelined,
  - what is the resource allocation per stage,
  - what is the “critical path” per token.

### 2.3 What model shapes compile well vs poorly (pattern catalogue)

**Compile-friendly patterns (Inference):**

- mostly static transformer graphs with fixed max sequence/batch,
- stable operator set (matmuls/attention/MLP) with predictable tensor sizes,
- limited conditional control flow in the hot path.

**Compile-hostile patterns (Assumption to validate for your model):**

- highly dynamic control flow,
- heavy ragged batching with many sequence lengths,
- frequent shape changes (dynamic prompt lengths without bucketing),
- unusual ops not covered by compiler optimizations.

### 2.4 Compilation time as an operational variable

If compilation takes minutes/hours (Assumption to validate), then:

- model swaps must be precompiled,
- A/B tests become “A/B compiled artifacts,”
- upgrades become “compiler + schedule compatibility” exercises.

**Decision question:** “What is our acceptable compile latency and artifact count?”

---

## 3) RealScale Fabric (Rack-Scale Intuition Without Overclaiming)

### 3.1 What to assume until confirmed

- **Assumption to validate:** Exact RealScale topology, link speed/latency, routing behavior, oversubscription, and failure handling are product/version dependent.
- **Fact (from Groq reference):** Groq markets a rack-scale system with a purpose-built fabric intended for predictable inference scaling. *(Attach the specific Groq reference.)*

### 3.2 Design intent (what “fabric” means here)

- **Inference:** The critical path is not “Ethernet + ToR oversubscription.” It’s a fabric designed for predictable chip-to-chip movement for model partitioning and/or pipelining.
- **Inference:** Failure domains are likely aligned to “link / node / chassis / rack segment,” not “random ECMP path.”

### 3.3 Why avoid switch-based scale-out inside racks (implications)

- **Assumption to validate:** The exact mechanism (direct chip-to-chip links vs embedded switching) depends on the Groq system SKU; confirm the physical topology and switching points.
- **Inference:** Traditional switch-based scale-out inside a rack introduces:
  - variable queueing in switches,
  - contention under incast/collective-like patterns,
  - complex failure and reconvergence behavior,
  - debugging ambiguity (“is it compute, switch buffer, or ECMP?”).
- **Inference:** A purpose-built rack fabric can instead provide:
  - predictable hop count and latency,
  - constrained routing (fewer states),
  - easier scheduling of communication as part of the compiled execution plan.

**Consequence:** You trade flexible, general-purpose networking for **topology discipline** (cabling rules, bounded expansion patterns, stricter failure domains).

### 3.4 What breaks across racks (the honest answer)

- **Inference:** Once you leave a tightly controlled rack-scale fabric, you re-enter the world of:
  - variable latency and congestion,
  - larger failure domains,
  - reduced determinism guarantees.

**Guiding principle:** Keep the model’s critical-path communication within the fabric domain; use Ethernet for control-plane, ingress/egress, and non-critical flows unless Groq explicitly guarantees otherwise.

### 3.5 When Ethernet becomes unavoidable (and what it’s used for)

- **Inference:** Ethernet is unavoidable for:
  - client ingress/egress,
  - orchestration/control plane,
  - metrics/logs/telemetry,
  - artifact distribution,
  - non-critical background transfers.
- **Assumption to validate:** Whether any deployment uses Ethernet for model-parallel critical path; do not assume it is safe for deterministic latency without Groq-confirmed envelopes.

---

## 4) Performance Reality: Replace GPU Mental Models

### 4.1 Batch-1 dominance

- **Fact (from Groq reference):** Groq targets low-latency inference; batch-1 is a core use case. *(Attach the specific Groq reference.)*
- **Inference:** Because the schedule is fixed, you often get **high, stable batch-1 throughput** and predictable per-token latency.

### 4.2 TTFT: what changes

GPU TTFT is often dominated by:

- dynamic batching windows,
- kernel launch + scheduling jitter,
- cache/HBM effects,
- KV cache paging/pinning behavior.

**Groq-native TTFT framing (Inference):**

- TTFT becomes “time to enter pipeline + prefill schedule time + queueing.”
- Variance is mostly queueing and admission policy, not compute variance.

### 4.3 “Saturation” on Groq

GPU saturation often means “SM occupancy + memory bandwidth + kernel overlap,” with messy interference.

**Groq-native saturation (Inference):**

- the pipeline has a fixed service curve; saturation means you’ve filled the schedule with no slack,
- beyond that, you only add queueing delay (SLO death), not throughput.

**Decision heuristic:** Tune admission to keep utilization below the knee where p99 explodes (see Tier 02).

---

## 5) GPU-Trained Intuition Traps (Call Them Out Explicitly)

1. **“Dynamic batching will save p99.”** On deterministic hardware, batching windows often *increase* p99 unless the schedule is explicitly compiled for it.
2. **“Caches will smooth irregularity.”** Groq’s value proposition is minimizing cache-driven variability; you must manage shapes instead.
3. **“We’ll just scale out over Ethernet.”** That’s where determinism usually dies; keep critical comms within the intended fabric.
4. **“Multi-tenancy is just a scheduler problem.”** On Groq, it’s a *compile-time + admission + isolation* problem.
5. **“Same model, same performance.”** On Groq, the shape constraints and compilation target can change performance materially.

---

## 6) LLM “Reasoning” vs Hardware Determinism (Do Not Confuse These)

This is the key bridge to runtime/capacity planning:

- **Inference:** LLM decoding is autoregressive: the system generates tokens one at a time; output length is variable.
- **Inference:** “Reasoning” behaviors (step-by-step explanations) usually increase output tokens and/or the number of internal steps, which increases compute and cost.
- **Groq-native implication:** Even if the LPU’s execution is deterministic for a given shape, **your request cost is still variable** because users choose prompts and the model chooses output length. Therefore:
  - determinism reduces compute variance *given a shape*,
  - but **shape + length control** is still required for SLOs.

**Decision framing:** “We can make service time predictable per bucket; we cannot make user demand predictable without bucketing, caps, and routing.”

**Complementary resource (not Groq-specific, but directly relevant):** Sebastian Raschka’s *Build a Reasoning Model (From Scratch)* (Manning MEAP, 2025) is a good way to internalize why inference-time “reasoning” increases compute (more tokens and/or more inference calls). Use that intuition when you choose bucket caps, admission policies, and degraded-mode behavior.

---

## 6) Tier 01 Checklist (What You Must Be Able To Do From Memory)

- Explain why determinism collapses compute variance but not queueing variance.
- Define the Groq deployment artifact as a compiled schedule with explicit shapes.
- Describe why “shape discipline” is an operational requirement, not an optimization.
- State what you assume about RealScale, and list exactly what you need Groq to confirm.

---

## Tier 01 Reinforcement Reading (High Signal)

Use these to solidify the mental model beyond slogans:

- **Primary sources (recommended first):**
  - Groq ISCA 2020 & 2022 papers (TSP + scale-out design). *(Assumption to validate links; prefer official PDFs if provided by Groq.)*
  - “Inside the LPU: Deconstructing Groq’s Speed” (Groq blog, 2025). *(Fact once you pin the exact post/version.)*
- **Tooling literacy (required for Tier 02 measurement):**
  - GroqFlow + compiler documentation and any GroqView/profiler docs. *(Fact once pinned.)*
- **Intuition builders (use carefully; third-party):**
  - Abhinav Upadhyay’s LPU architecture deep-dive.
  - Igor Arsovski interview/tutorial video.

**Rule:** Use third-party resources to build intuition; use Groq references to make claims in design reviews.
