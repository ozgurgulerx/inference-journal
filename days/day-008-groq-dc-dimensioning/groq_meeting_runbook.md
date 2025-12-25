# Groq Design Review Runbook (Inference + Fabric + DC)

This runbook is designed to **control the narrative** in a hierarchical Groq engineering meeting: you set the frame, force decisions, and surface weak assumptions early.

---

## 0) Pre-Read Package (Send 24–48h Before)

- `README.md` (why Groq changes inference + ops)
- `LOG_tier01.md` (mental model + determinism implications)
- `dimensioning_inputs_template.md` (what data you need from each team)
- A 1-page “current state” summary:
  - model(s) + target shapes (prompt/output caps)
  - target SLOs + traffic envelope
  - current facility constraints (rack power, cooling class, row/room limits)

**Decision rule:** If the pre-read isn’t consumed, you do not “wing it.” You shrink scope to confirming assumptions.

---

## 1) Meeting Objective (State This Out Loud)

By end of meeting we will have:

1. An agreed **shape policy** (bucketing + max tokens + tiers).
2. An agreed **capacity model** (measured rates, headroom, redundancy).
3. An agreed **failure domain** and failover behavior (what degrades, what must hold).
4. A clear **rack/pod design** and an expansion sequence.
5. A list of **open assumptions** with owners and due dates.

---

## 2) Agenda (90 minutes, decision-forcing)

### 0–10m — Frame + Non-Negotiables

- Reframe: “Groq performance is a property of the compiled schedule + queueing.”
- State the GPU traps explicitly (dynamic batching, ‘we’ll fix later’ shapes, Ethernet critical path).
- Confirm: “We will label facts vs assumptions.”

### 10–25m — Workload + SLO Envelope (Product + Inference)

- Confirm traffic: `λ_peak`, prompt/output distributions, burst factor.
- Confirm SLOs: p50/p95/p99, TTFT vs full latency definitions.
- Force a decision: **shape policy** (buckets + caps + long-context tier).

### 25–45m — Compiler + Hardware Constraints (Groq)

- Confirm: supported operator set, compilation constraints, artifact lifecycle.
- Confirm: what is deterministic vs what varies (and why).
- Force a decision: “Which shapes do we support in the first production pod?”

### 45–60m — Fabric / Scaling Domain (Groq + NetEng)

- Confirm: RealScale domain boundaries, failure behavior, cross-rack position.
- Force a decision: “Critical-path comm stays within X (rack/pod).”

### 60–80m — DC Translation (Facilities)

- Confirm: rack power/cooling class, containment, cable plant constraints.
- Force a decision: rack density + row layout + expansion staging.

### 80–90m — Close: Decisions + Owners + Next Steps

- Read back decisions and open assumptions.
- Assign owners + due dates.
- Define the next checkpoint (compile/perf data + facility test plan).

---

## 3) Whiteboard Flow (What You Draw)

### Board A — End-to-End Latency Decomposition

`T_total = T_queue + T_service(shape, compiled)`

Then draw:

- arrival bursts,
- admission gate (token-budget),
- replicas/pools by shape bucket.

### Board B — Shape Buckets and Pools

- buckets by prompt length and output cap,
- mapping of buckets → compiled artifacts → replica pools.

### Board C — Failure Domains

- component → link segment → rack → pod,
- which failures trigger degrade vs shed.

### Board D — Rack/Pod Layout (Physical)

- rack groups as pods,
- cable constraints (fabric),
- power/cooling limits and expansion sequence.

---

## 4) Decision Log Template (Fill Live)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Shape buckets | | | | | |
| Max prompt/output | | | | | |
| Headroom `ρ_target` | | | | | |
| Failure-domain target | | | | | |
| Cross-rack policy | | | | | |
| Rack power class | | | | | |
| Pod unit size | | | | | |

---

## 5) “Decision Forcing” Prompts (Use These Verbatim)

- “If we can’t state the shape policy, we can’t size anything. What are the buckets and caps?”
- “What is the deterministic service-time envelope for this compiled artifact?”
- “Where does determinism end: rack, pod, or cluster? We must draw the boundary.”
- “Under N-1 (rack or link), do we shed load or violate p99? Pick one.”
- “What are the artifacts we will version-control: compiler version, schedule, and target hardware?”

---

## 6) Post-Meeting Actions (Within 72h)

- Fill `capacity_model_template.md` with the agreed inputs and first measurements.
- Update `risk_register.md` with new assumptions and risks.
- Produce a one-page “pod blueprint” diagram with:
  - rack count,
  - power/cooling,
  - fabric cabling,
  - spares strategy.

