# Acceptance Criteria (Day 008 Quality Bar)

Day 008 is “done” only if all criteria below are satisfied.

---

## A) Groq-Native Depth

- Separates **service time determinism** from **queueing tail** and uses this to drive SLO decisions.
- Treats the **compiler as runtime** and defines a deployment artifact and ops lifecycle around it.
- Explicitly explains how SRAM-first / deterministic scheduling changes batching, scaling, and debugging.
- Clearly states where determinism becomes a constraint (shape discipline, compilation ops, multi-tenancy limits).

---

## B) RealScale / Fabric Credibility

- Does not invent topology specifics without labeling as **Assumption to validate**.
- Defines a determinism boundary (rack/pod/cluster) and forces a decision on cross-rack critical path.
- Explains what Ethernet is used for (control/ingress/egress) and why it’s risky for critical path.

---

## C) Capacity & Dimensioning (Non-Negotiable)

- Includes a tokens/s-based capacity model that does **not** rely on dynamic batching assumptions.
- Includes queueing model reasoning (at least first-order) and explicitly chooses `ρ_target` and queue caps.
- Includes redundancy sizing (N+1) tied to failure domains and degraded-mode behavior.
- Includes explicit anti-patterns that GPU-centric engineers will attempt.

---

## D) Data Center Translation

- Includes rack power math, cooling considerations, cabling/fabric constraints, and expansion sequencing.
- Describes bring-up/burn-in acceptance tests as a required step, not optional.

---

## E) Meeting Readiness

- Contains a runbook that can lead a hierarchical meeting and force decisions.
- Contains the dedicated section **“Hierarchical Meeting With Groq Engineers: Top-1% Answers”** with:
  - Exec/PM, Compiler, Hardware, Fabric/Network, DC/Facilities, Inference/Runtime layers
  - 10–20 hard questions per layer
  - 3-level answers, red flags, and decision closures

---

## F) Honesty & Epistemics

- All non-verified specifics are labeled **Assumption to validate**.
- Every “Fact (from Groq reference)” claim is supportable by an artifact you can request from Groq.
- External benchmark claims are treated as hypotheses and are not used as primary dimensioning inputs without validation.
