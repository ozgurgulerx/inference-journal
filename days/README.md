# 100 Days of Inference Engineering

> **Dec 1, 2025 → Mar 10, 2026** | 3-4 hrs/day focused effort

**[📋 Full Roadmap & Checklist →](learning_goals.md)**

---

Throughout this 100-day journey I’m also using **NotebookLM** and **OpenAI DeepResearch** extensively—to synthesize reading, design experiments, and pressure-test my own explanations as I go.

> **Audit update:** Use the measurement-first daily log template in `days/planning/daily_log_template.md`. Days 011–019 have been added to close gaps (alternate runtimes, long-context vs RAG, speculative decoding, multi-GPU, multi-tenant QoS, cost accounting, reliability/guardrails, alignment, and MoE/SSM coverage).

## The 100-Day Plan

| Phase | Days | Dates | Focus |
|-------|------|-------|-------|
| **0** | 1-15 | Dec 1-15 | OS & GPU Setup |
| **1** | 16-35 | Dec 16 → Jan 4 | vLLM Mastery |
| **2** | 36-55 | Jan 5-24 | Quantization |
| **3** | 56-80 | Jan 25 → Feb 18 | Optimization |
| **4** | 81-100 | Feb 19 → Mar 10 | Ship & Share |

---

## Daily Logs

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| [001](day-001-initial-setup/) | Dec 1 | Initial Setup | ✅ |
| [002](day-002-GPU-node-bring-up/) | Dec 2 | GPU Node Bring-Up | 🔄 |
| [003](day-003-vLLM-capacity-and-OOM/) | Dec 3 | vLLM Capacity & OOM | ⏳ |
| [004](day-004-quantization-vs-bf16/) | Dec 4 | Quantization vs BF16 | ⏳ |
| [005](day-005-OS-and-NUMA-node-hardening/) | Dec 5 | OS & NUMA Node Hardening | ⏳ |
| [006](day-006-slm-memory/) | Dec 6–7 | SLM + OS Memory & vLLM | ⏳ |
| [007](day-007-vllm-runtime-probes/) | Dec 8 | vLLM SLM: TTFT, Prefix Caching, KV Scaling | ⏳ |
| [008](day-008-groq-dc-dimensioning/) | Dec 9 | Groq DC Dimensioning | ⏳ |
| [009](day-009-latency-metrics-and-slo-lab/) | Dec 10 | Latency Metrics, KV Deep Dive & SLO Lab | ⏳ |
| [010](day-010-attention-architecture-gqa-mqa/) | Dec 11 | Attention Variants (MHA/GQA/MQA) | ⏳ |
| [011](day-011-runtime-shootout/) | Dec 12 | Alternate Runtimes & Scheduling | ⏳ |
| [012](day-012-long-context-vs-rag/) | Dec 13 | Long-Context vs RAG | ⏳ |
| [013](day-013-speculative-decoding/) | Dec 14 | Speculative Decoding Benchmark | ⏳ |
| [014](day-014-multi-gpu-scaling/) | Dec 15 | Multi-GPU Scaling | ⏳ |
| [015](day-015-multi-tenant-qos/) | Dec 16 | Multi-Tenant Load & QoS | ⏳ |
| [016](day-016-cost-efficiency-accounting/) | Dec 17 | Cost & Efficiency Accounting | ⏳ |
| [017](day-017-reliability-guardrails/) | Dec 18 | Reliability Drills & Guardrails | ⏳ |
| [018](day-018-alignment-e2e-demo/) | Dec 19 | Alignment + Serving E2E | ⏳ |
| [019](day-019-state-space-or-moe/) | Dec 20 | State-Space / MoE Hands-On | ⏳ |

---

## Deliverables Tracker

| Deliverable | Target Day | Status |
|-------------|------------|--------|
| Bootstrap script | 3 | ⏳ |
| Grafana dashboard | 15 | ⏳ |
| HF vs vLLM comparison repo | 18 | ⏳ |
| Load test script | 30 | ⏳ |
| Quantization benchmark | 40 | ⏳ |
| Case study #1 | 72 | ⏳ |
| Case study #2 | 80 | ⏳ |
| Optimization playbook | 95 | ⏳ |
| Blog post published | 100 | ⏳ |

---

## Links

- [📋 Learning Goals](learning_goals.md) – Full 100-day checklist
- [🤖 Daily Coach Prompt](learning_prompt.md) – AI prompt for generating daily plans
- [📚 Inference Engineering Book](../books/inference-engineering/README.md)
