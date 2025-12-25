# LOG (Tier 02) — Capacity, Dimensioning, Facility Mapping, Scaling Strategy

This tier converts Groq determinism into: tokens/s math, admission control, headroom, redundancy, and facility-level design.

---

## 0) The Only Two Latency Sources You Own

For a single request:

`Latency = QueueingDelay + ServiceTime(shape, compiled_schedule)`

- **Fact (from Groq reference):** ServiceTime is intended to be predictable for a fixed compiled shape. *(Attach the specific Groq reference.)*
- **Inference:** Your p99 is controlled mostly by **QueueingDelay**, which you control via:
  - admission control / concurrency caps,
  - capacity headroom,
  - shape discipline and bucketing,
  - failure-domain-aware redundancy.

---

## 1) Groq-Specific Dimensioning Model (Tokens/s Without GPU Assumptions)

### 1.1 Define the primitives (you will measure these)

For each compiled artifact `a` (a model + shape profile):

- `R_decode[a]` = sustained **decode** throughput (tokens/s) at batch-1-equivalent, at the chosen precision/shape.
- `R_prefill[a, P]` = effective **prefill** throughput (prompt tokens/s) for prompt length bucket `P`.
- `L_fixed[a]` = fixed overhead per request (enqueue + ingress + pipeline entry), excluding queueing.

**Assumption to validate:** Whether Groq provides these directly from compiler reports or profiling tools; if not, you will measure them with microbenchmarks.

### 1.2 Convert QPS to required tokens/s

Given arrival rate `λ` (requests/s), prompt token distribution `P`, output token distribution `O`:

`Tokens/s_required ≈ λ * ( E[O] + α * E[P] )`

Where:

- `α` is a **prefill penalty factor** mapping prompt tokens to “decode-equivalent tokens.”
- **Inference:** `α` captures attention’s quadratic-ish prefill cost vs decode’s linear-ish growth.
- **Assumption to validate:** Fit `α` empirically for your compiled artifact and typical prompt buckets.

If you have separate measured rates:

`ComputeTime_per_req ≈ (E[P]/R_prefill) + (E[O]/R_decode) + L_fixed`

### 1.3 Replica math (no dynamic batching assumption)

Let each LPU (or LPU partition) provides `μ` requests/s service rate for the given shape:

`μ ≈ 1 / E[ServiceTime_per_req]`

Then required replicas:

`N_min = ceil( λ / (ρ_target * μ) )`

Where `ρ_target` is your utilization target (headroom).

**GPU intuition trap:** Importing a GPU “run at 90% and rely on batching.” For deterministic service times, queueing explodes sharply; pick `ρ_target` based on SLO.

### 1.4 Queueing model: use M/D/1 (first-order)

- Arrivals are often bursty (not Poisson), but M/D/1 is a useful lower bound.
- Service time is closer to deterministic (D) than exponential (M) when determinism holds.

**Inference (M/D/1 mean waiting time):**

`Wq ≈ ρ^2 / (2 * μ * (1 - ρ))`

Total mean latency:

`E[T] ≈ Wq + 1/μ`

**Action:** Pick `ρ_target` such that Wq at p95/p99 is tolerable given your burstiness (Tier 03 expands).

### 1.5 Concurrency handling: admission is a first-class knob

Define:

- `C_max_per_replica` = max concurrent requests admitted per replica for your SLO.
- `Queue_cap` = bounded queue to avoid unbounded tail and to shed load predictably.

**Groq-native strategy (Inference):**

- Use **shape buckets** (prompt length buckets + max output tokens caps).
- Admit based on a **deterministic cost model** (estimated service time) not just “number of inflight requests.”

---

## 2) Practical Sizing Worksheet (Fill-In)

### 2.1 Inputs (collect via `dimensioning_inputs_template.md`)

| Symbol | Meaning | Typical source |
|---|---|---|
| `λ_peak` | peak requests/s | product traffic |
| `P50/P95(P)` | prompt length distribution | logs |
| `P50/P95(O)` | output length distribution | logs |
| `SLO_p99` | p99 latency budget | product |
| `R_decode` | decode tok/s per replica | benchmark |
| `R_prefill(Pbucket)` | prefill tok/s per bucket | benchmark |
| `failover_time` | time to re-route after failure | infra |
| `N+1` policy | redundancy target | SRE |

### 2.2 Derived quantities

Compute per-request service time bound for each bucket:

`S_bucket = Pbucket / R_prefill(Pbucket) + Obucket / R_decode + L_fixed`

Compute service rate:

`μ_bucket = 1 / S_bucket`

Pick utilization target `ρ_target` such that `p99_queue + p99_service ≤ SLO_p99`.

### 2.3 Headroom strategy (explicit)

You must choose and defend:

- `ρ_target_normal` (e.g., 0.50–0.70)
- `ρ_target_degraded` under N-1 or link failure (e.g., ≤0.80)
- `shed_policy` (when to reject)

**Inference:** Determinism makes the “knee” more predictable; that’s a *feature*—use it to engineer stable p99.

---

## 3) Anti-Patterns (Things GPU Engineers Will Try That Break on Groq)

1. **Continuous dynamic batching across heterogeneous lengths**
   - **Why it fails (Inference):** determinism + static schedules dislike ragged shapes; batching windows add queueing; performance becomes “compile mismatch.”
2. **“Just increase max sequence length in prod”**
   - **Why it fails:** shape change → recompile/repipeline; service time changes; SLO math invalid.
3. **Treating compilation as a build step you can do “on deploy”**
   - **Why it fails:** compilation latency and schedule validation become part of release engineering.
4. **Assuming Ethernet is “fine” for critical-path model parallelism**
   - **Why it fails:** unpredictable congestion kills deterministic latency guarantees.
5. **Ignoring shape buckets**
   - **Why it fails:** a small fraction of long prompts can dominate capacity and tail.

---

## 4) RealScale / Rack-Scale Design Translation

### 4.1 Why ~30kW-class racks matter (power math template)

Let:

- `P_node` = node power (kW)
- `N_nodes_rack` = nodes per rack
- `P_rack = P_node * N_nodes_rack + P_overhead` (fans/ToR/etc.)

**Decision:** choose rack density to fit facility constraints (power + cooling + cable plant).

**Assumption to validate:** Actual node/rack power of your Groq system SKU.

### 4.2 Cooling assumptions (air vs liquid)

- **Inference:** If the system is engineered for 30kW-class air, DC bring-up is faster (no CDU loops).
- **Constraint:** Air cooling limits density and requires disciplined hot-aisle containment and airflow management.

**Decision questions for Facilities:**

- Can we sustain `ΔT` and airflow at `P_rack` without hotspots?
- What is the derating at altitude/ambient extremes?

### 4.3 Cabling realities with a rack-scale fabric

- **Inference:** A purpose-built rack fabric often implies:
  - specific cable lengths and routing,
  - strict port mapping,
  - low tolerance for “we’ll tidy later.”

**Action:** Make cabling part of the critical path plan; treat it like a backplane, not “just networking.”

### 4.4 Floor layout and expansion sequencing

Expansion plan should be written as:

1. **Pod unit**: smallest repeatable rack group that contains a failure domain.
2. **Row unit**: pods + shared power distribution.
3. **Room unit**: rows + spares + staging.

**Inference:** Deterministic systems benefit from repeatable pods—don’t “snowflake” racks.

### 4.5 Why Groq can enable faster bring-up (and what still limits it)

Potential accelerators (Inference):

- less tuning around kernel scheduling variance,
- fewer “mysterious” tail spikes,
- simpler runtime stack (compiler + runtime vs large GPU kernel zoo).

Hard limits (still real):

- power delivery lead times,
- cooling capacity,
- cable plant and physical install,
- spares logistics and RMA cycles,
- compilation/validation pipeline maturity.

---

## 5) Redundancy Models (N+1, Spares, Failover)

### 5.1 Choose failure domains explicitly

Define:

- **Component FD:** single LPU / node.
- **Fabric FD:** link group / segment.
- **Rack FD:** whole rack outage (PDU/ToR/thermal trip).
- **Pod FD:** multiple racks + shared infra.

**Decision:** Which FD must not violate SLO? Which FD can shed load?

### 5.2 Capacity under failure

If you require N+1 at rack level:

`N_operational = N_total - 1 rack`

Dimension such that:

`λ_peak ≤ ρ_target_degraded * N_operational * μ`

**Action:** Run a forced-failure game day: verify queueing behavior and routing times.

---

## 6) Tier 02 Outputs (Artifacts You Must Produce)

- A filled `capacity_model_template.md` with measured `R_prefill` and `R_decode` per bucket.
- An admission-control policy: `C_max`, `Queue_cap`, shedding thresholds.
- A rack/pod plan with explicit failure domains and spare strategy.
