# capacity_calc_pseudocode.md

This is intentionally pseudocode: use it to implement a real sizing calculator in your preferred language.

---

## Inputs

- Buckets `b ∈ {S, M, L, XL}`
- For each bucket `b`:
  - `lambda_peak_b` (req/s)
  - `P_b` (prompt tokens, choose p95/p99 as policy)
  - `O_b` (output tokens, choose p95/p99 as policy)
  - `R_prefill_b` (prompt tok/s measured)
  - `R_decode_b` (decode tok/s measured)
  - `L_fixed_b` (sec)
- Global:
  - `rho_target_normal`
  - `rho_target_degraded`
  - `redundancy_policy` (e.g., rack N+1)

---

## Core Functions

### Service time per request

```
S_b = (P_b / R_prefill_b) + (O_b / R_decode_b) + L_fixed_b
mu_b = 1 / S_b
```

### Replica count per bucket

```
N_b = ceil(lambda_peak_b / (rho_target_normal * mu_b))
```

### Degraded-mode check (N-1)

Decide what “N-1” means (node loss, rack loss, link segment loss). Convert that to effective capacity loss:

```
N_total = sum_b N_b
N_operational = N_total - N_lost

# Conservative: use worst bucket utilization after loss
assert lambda_peak_total <= rho_target_degraded * N_operational * mu_effective
```

Where `mu_effective` may be a weighted average service rate or computed per bucket with redistributed traffic.

---

## Queueing sanity check (first-order)

For each bucket, approximate mean waiting time using M/D/1:

```
rho = lambda_b / (N_b * mu_b)
Wq_mean = rho^2 / (2 * (N_b * mu_b) * (1 - rho))
T_mean = Wq_mean + (1 / (N_b * mu_b))
```

Then compare to SLO expectations. For p99, you must model burstiness explicitly (Tier 03).

---

## Outputs

- Per bucket: `S_b`, `mu_b`, `N_b`, `rho_b` at peak
- Cluster totals: `N_total`, racks required (given nodes/rack)
- Degraded-mode compliance result + required extra spares

