# qps_to_tokens_model.md

This file defines the minimal math to translate QPS into token demand for deterministic capacity planning.

---

## 1) Definitions

- `λ` = requests per second (QPS)
- `P` = prompt tokens per request
- `O` = output tokens per request
- `E[P]`, `E[O]` = expectations over the request distribution

---

## 2) Naive token demand (useful upper bound)

If you assume each token costs the same:

`Tokens/s_required = λ * ( E[P] + E[O] )`

This is rarely accurate for transformers because prefill and decode have different compute cost profiles, but it’s a useful sanity bound.

---

## 3) Prefill-weighted token demand (better first-order)

Introduce `α >= 1` to convert prompt tokens into decode-equivalent tokens:

`Tokens/s_required ≈ λ * ( α * E[P] + E[O] )`

How to fit `α`:

- Measure service time for different prompt lengths with fixed output cap.
- Fit `α` so the model predicts observed service time when combined with a measured `R_decode`.

---

## 4) Bucketed demand (recommended)

Define buckets `b` with traffic share `w_b`, prompt cap `P_b`, output cap `O_b`:

`Tokens/s_required_total ≈ Σ_b ( λ * w_b * ( α_b * E[P|b] + E[O|b] ) )`

Then allocate pools per bucket.

---

## 5) Why this matters more on Groq

- **Inference:** If service time is stable, you can safely plan from token demand + headroom.
- **GPU intuition trap:** relying on dynamic batching to “absorb” token demand variability hides queueing costs; Groq rewards explicit bucketing and admission.

