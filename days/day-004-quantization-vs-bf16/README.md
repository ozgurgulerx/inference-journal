# Day 004 – Quantization vs BF16 on RTX2000

> **Theme**: Quantization & Mixed Precision  
> **Hardware**: RunPod RTX 2000 Ada 16GB  
> **Model**: Qwen2.5-1.5B-Instruct (BF16) + a quantized variant (AWQ/GPTQ)  
> **Builds on**: Days 001–003 (vLLM serving, chat/batch harnesses, capacity grids)

---

## Executive Summary

You've measured BF16 capacity. Now measure **quantized capacity** on the same stack—and compare throughput, latency, VRAM, quality, and cost. This is the foundation for every "should we quantize?" consulting conversation.

---

## Situational Recap

| What You Have | From |
|---------------|------|
| vLLM serving Qwen2.5-1.5B-Instruct BF16 | Day 003 |
| `vllm_chat_bench.py` (async chat bench) | Day 003 |
| `run_chat_capacity_grid.sh` (concurrency × max_tokens grid) | Day 003 |
| `vllm_batch_summarize_bench.py` + `run_batch_capacity_grid.sh` | Day 003 |
| Chat + batch capacity CSVs with measured sweet spots | Day 003 |
| Baseline: ~50 tok/s single-stream, ~620 tok/s @ conc=16 | Day 003 |

**Next high-leverage theme**: Quantization vs BF16 on this exact stack, reusing all Day 3 scripts.

---

## Tier Breakdown

| Tier | Time | Scope |
|------|------|-------|
| Tier 1 | ~2h | Core—bring up quant model, chat capacity comparison, quality sanity check |
| Tier 2 | ~1–2h | Extension—batch comparison + cost-per-token analysis |
| Tier 3 | ~1–3h | Deep work—mini case study write-up |

**Total**: 4–6h for full completion. Tier 1 alone gives you the key comparison data.

---

## Key Artifacts

```
configs/vllm/
└── serve_quant_model_rtx16gb.sh        → Quantized model server config

scripts/benchmarks/
└── day004_quant_quality_eval.py        → Side-by-side quality comparison

benchmarks/
├── day004_chat_capacity_rtx16gb_bf16.csv
├── day004_chat_capacity_rtx16gb_quant.csv
├── day004_batch_capacity_rtx16gb_bf16.csv  (Tier 2)
├── day004_batch_capacity_rtx16gb_quant.csv (Tier 2)
└── day004_quant_quality_eval.json

artifacts/
├── day004_quant_vs_bf16_notes.md       → Running notes + comparisons
└── day004_quant_cost_comparison.md     (Tier 2)

reports/
└── day004_quant_vs_bf16_rtx2000.md     (Tier 3 case study)
```

---

## Critical Insights (Preview)

1. **VRAM savings** — Quantization should free up memory, enabling higher concurrency or longer contexts
2. **Throughput vs latency** — Quant may improve throughput but watch for p95 degradation
3. **Quality trade-offs** — Small models quantize differently than large ones; spot-check matters
4. **Business case** — $/1M tokens is the metric that closes deals

---

## Success Criteria

By completing Day 004:

- [ ] Quantized model running alongside BF16 on same GPU (different ports)
- [ ] Chat capacity comparison: BF16 vs quant at conc=1/8/16
- [ ] Quality sanity check: 10+ prompts compared, issues noted
- [ ] (Tier 2) Batch comparison + cost-per-token analysis
- [ ] (Tier 3) Mini case study ready for client adaptation

---

## Navigation

- **[Tier 1 – Core Block](LOG_tier01.md)**: Quant model setup + chat comparison + quality check
- **[Tier 2 – Extension](LOG_tier02.md)**: Batch comparison + cost analysis
- **[Tier 3 – Deep Work](LOG_tier03.md)**: Case study consolidation

---

## Cross-Day References

| Artifact | Day | Path |
|----------|-----|------|
| BF16 server config | 003 | `~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh` |
| Chat benchmark harness | 003 | `~/scripts/benchmarks/vllm_chat_bench.py` |
| Chat capacity grid | 003 | `~/scripts/benchmarks/run_chat_capacity_grid.sh` |
| Batch benchmark | 003 | `~/scripts/benchmarks/vllm_batch_summarize_bench.py` |
| Batch capacity grid | 003 | `~/scripts/benchmarks/run_batch_capacity_grid.sh` |
| Sample docs | 003 | `~/data/day003_docs_sample.txt` |
| Baseline CSVs | 003 | `~/benchmarks/day003_*_capacity_rtx16gb.csv` |

---

<p align="center">
  <a href="../day-003-vLLM-capacity-and-OOM/">← Day 003</a> · 
  <a href="../README.md">Days Index</a>
</p>
