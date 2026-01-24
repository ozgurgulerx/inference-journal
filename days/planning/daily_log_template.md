# Daily Log Template (Measurement-First)

Use this template for every new day to keep the journal metrics-driven, cost-aware, and behavior-conscious.

## Day {DD} – {Descriptive Title}

### Goals
- **Primary Objective:** What you plan to achieve or learn today.
- **Focus Theme(s):** Tie to the top-10 themes (e.g., Long Context & Memory, Decoding Acceleration, Multi-Tenancy & QoS).

### Setup & Context
- **Hardware/Infra:** GPU/CPU, memory, special settings (NUMA, MIG, FP8/FP16/INT4).
- **Model(s):** Names, sizes, precision, and any draft/secondary models.
- **Runtime/Engine:** vLLM/TRT-LLM/TGI/Triton/etc. with key flags.
- **Dataset/Workload:** Prompts or datasets, synthetic vs real, request shapes.
- **Key Config Knobs:** Parameters you will vary (e.g., `max_num_seqs`, `max_model_len`, speculative settings, batch/concurrency).

### Experiments & Measurements
For each experiment:
1. **Name:** Clear label.
2. **Method:** Commands or scripts (high level).
3. **Metrics:** p50/p95 latency, throughput (tok/s or req/s), VRAM, CPU, network, cache hit/reject rate (for speculative), rejection/quality stats.
4. **Results:** Raw numbers or a compact table.
5. **Observation:** 1–2 line interpretation.

### Analysis & Insights
- What the data shows and why (bottlenecks, trade-offs, anomalies).
- Compare variants (e.g., runtime A vs B, quant vs BF16).

### Performance & Cost Evaluation
- Convert throughput to **$ per 1M tokens** using the GPU hourly rate.
- Note utilization headroom, memory pressure, and scheduler efficiency.
- Call out cost/perf winners for the tested scenario.

### Output Quality & Alignment Check
- Qualitative or quantitative check (BLEU/pass@k/accuracy; correctness/regressions).
- Behavior/guardrail notes (toxicity/PII filters, refusals, draft-model mismatch).

### Observability & Reliability Notes
- Logging/metrics added (per-request cost, per-tenant stats).
- Failure drills run (OOM, timeout, node down), recovery behavior, and guardrail overhead.

### Next Steps
- Immediate follow-ups, longer-term ideas, and blockers/risks.
