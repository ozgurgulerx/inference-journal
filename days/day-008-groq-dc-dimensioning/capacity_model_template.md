# Capacity Model Template (Groq-Native)

Fill this template after you have real Groq measurements for your compiled artifacts.

**Rule:** No GPU-only assumptions (dynamic batching, cache variance smoothing) unless Groq confirms equivalence.

---

## 1) Workload Summary

| Item | Value |
|---|---|
| Model(s) | |
| Endpoints | |
| `λ_mean` (req/s) | |
| `λ_peak` (req/s) | |
| Burst window definition | |
| SLO p99 (ms) | |

Token distributions:

| Metric | P50 | P95 | P99 |
|---|---:|---:|---:|
| Prompt tokens `P` | | | |
| Output tokens `O` | | | |

---

## 2) Shape Policy

Buckets:

| Bucket | `P_max` | `O_max` | Traffic share | Notes |
|---|---:|---:|---:|---|
| S | | | | |
| M | | | | |
| L | | | | |
| XL (long-context) | | | | |

Admission:

| Field | Value |
|---|---|
| Queue cap | |
| Token-budget admission | |
| Shed thresholds | |

---

## 3) Measured Performance (Per Bucket / Artifact)

For each bucket, paste the measured values:

| Bucket | Artifact ID | `R_prefill` (tok/s) | `R_decode` (tok/s) | `L_fixed` (ms) |
|---|---|---:|---:|---:|
| S | | | | |
| M | | | | |
| L | | | | |
| XL | | | | |

---

## 4) Service Time Model (Per Bucket)

For bucket `b`:

`S_b = (P_b / R_prefill_b) + (O_b / R_decode_b) + L_fixed_b`

Choose `P_b` and `O_b` as p95 or p99 depending on your SLO objective.

| Bucket | `P_b` | `O_b` | `S_b` (sec) | `μ_b = 1/S_b` (req/s) |
|---|---:|---:|---:|---:|
| S | | | | |
| M | | | | |
| L | | | | |
| XL | | | | |

---

## 5) Headroom + Queueing Targets

Define utilization targets:

| Mode | `ρ_target` | Rationale |
|---|---:|---|
| Normal | | |
| Degraded (N-1) | | |

**Queueing check (M/D/1 first-order):**

`Wq ≈ ρ^2 / (2 * μ * (1 - ρ))`

Compute for each bucket at `ρ_target` and confirm it fits SLO.

---

## 6) Replica Count (Per Bucket Pool)

For each bucket pool:

`N_b = ceil( λ_peak_b / (ρ_target * μ_b) )`

| Bucket | `λ_peak_b` | `μ_b` | `ρ_target` | `N_b` |
|---|---:|---:|---:|---:|
| S | | | | |
| M | | | | |
| L | | | | |
| XL | | | | |

Total replicas:

`N_total = Σ_b N_b`

---

## 7) Failure / Redundancy Sizing

Pick failure domain:

- Component (node/LPU)
- Link segment
- Rack
- Pod

State redundancy policy:

| Policy | Value |
|---|---|
| N+1 scope | |
| Cold standby | |
| Spare rack | |
| RMA turnaround assumption | |

Degraded capacity check:

`λ_peak_total ≤ ρ_target_degraded * (N_total - N_spares_unavailable) * μ_effective`

---

## 8) DC Translation

Assume:

| Item | Value |
|---|---:|
| Node power `P_node` (kW) | |
| Nodes per rack `N_nodes_rack` | |
| Rack overhead `P_overhead` (kW) | |

Compute:

`P_rack = P_node * N_nodes_rack + P_overhead`

Racks needed:

`R = ceil(N_total / N_nodes_rack)`

Then map to:

- rows,
- pods,
- spares,
- growth plan.

