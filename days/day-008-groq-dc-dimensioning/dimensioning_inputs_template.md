# Dimensioning Inputs Template (Groq LPU + Fabric + DC)

Use this as the single source of truth for all inputs needed to size Groq for a workload. Keep it versioned.

**Labels:** Mark each field as **Fact (from Groq reference)**, **Inference**, or **Assumption to validate**.

---

## 1) Workload Envelope (Product / Inference)

### 1.1 Traffic

| Field | Value | Label | Notes |
|---|---:|---|---|
| `λ_mean` (req/s) |  |  | |
| `λ_peak` (req/s) |  |  | peak window definition |
| Burst factor `B = λ_peak/λ_mean` |  |  | |
| Request mix (% by endpoint/model) |  |  | |

### 1.2 Token Distributions (Per endpoint/model)

| Field | P50 | P95 | P99 | Label | Notes |
|---|---:|---:|---:|---|---|
| Prompt tokens `P` |  |  |  |  | tokenizer definition |
| Output tokens `O` |  |  |  |  | max cap policy |
| Total tokens `P+O` |  |  |  |  | |

### 1.3 SLO / SLI Definitions

| Metric | Definition | Target | Label | Notes |
|---|---|---:|---|---|
| p50 latency |  |  |  | |
| p95 latency |  |  |  | |
| p99 latency |  |  |  | |
| TTFT |  |  |  | include queueing? |
| Availability |  |  |  | per endpoint |

---

## 2) Shape Policy (Non-Negotiable)

### 2.1 Buckets

Define discrete buckets (example):

| Bucket | Prompt cap `P_max` | Output cap `O_max` | Concurrency cap | Notes |
|---|---:|---:|---:|---|
| S |  |  |  | |
| M |  |  |  | |
| L |  |  |  | |
| XL (long-context tier) |  |  |  | routed separately |

### 2.2 Admission Policy

| Field | Value | Label | Notes |
|---|---:|---|---|
| `Queue_cap` (requests) |  |  | bounded queue |
| Token-budget admission? | yes/no |  | definition |
| Shed policy |  |  | 429 vs degrade |

---

## 3) Groq Compiler / Artifact Inputs

| Field | Value | Label | Notes |
|---|---|---|---|
| Compiler version |  |  | |
| Artifact ID scheme |  |  | hash inputs |
| Supported operator set constraints |  |  | |
| Compile time per artifact |  |  | wall clock |
| Artifact portability |  |  | hw generation coupling |

---

## 4) Performance Measurements (Per artifact + bucket)

For each bucket/artifact:

| Field | Value | Label | Notes |
|---|---:|---|---|
| `R_decode` (tok/s) |  |  | steady-state |
| `R_prefill` (tok/s) |  |  | by prompt bucket |
| `L_fixed` (ms) |  |  | ingress/pipeline entry |
| Max stable utilization `ρ_knee` |  |  | where p99 explodes |

---

## 5) Fabric / Network Inputs

| Field | Value | Label | Notes |
|---|---|---|---|
| Deterministic domain boundary | rack/pod/cluster |  | confirm with Groq |
| Failure domains |  |  | link/rack/pod |
| Cross-rack critical path allowed? | yes/no |  | if yes: SLO envelope |
| Ethernet role |  |  | control/ingress/egress |

---

## 6) Data Center / Facilities Inputs

### 6.1 Rack

| Field | Value | Label | Notes |
|---|---:|---|---|
| Rack power budget (kW) |  |  | derating |
| Rack cooling class | air/liquid |  | |
| Hot-aisle containment available | yes/no |  | |

### 6.2 Power Delivery

| Field | Value | Label | Notes |
|---|---|---|---|
| PDU type / phase |  |  | |
| Breaker limits and derating |  |  | |
| UPS / generator assumptions |  |  | |

### 6.3 Cabling

| Field | Value | Label | Notes |
|---|---|---|---|
| Fabric cable type/length constraints |  |  | |
| Labeling/testing process |  |  | |
| Spare cable stock |  |  | |
