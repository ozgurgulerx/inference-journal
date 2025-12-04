# Day 004 – Quantization vs BF16 Notes

*Running notes and comparisons from Day 004 experiments*

---

## Model & VRAM Baseline (Task 1.1)

### Models Used
- **BF16**: [model name, HF ID]
- **Quantized**: [model name, HF ID, quantization method]

### VRAM Comparison

| State | BF16 (MB) | Quant (MB) | Δ |
|-------|-----------|------------|---|
| Idle (no server) | [X] | [X] | — |
| Server loaded | [X] | [X] | [X]% reduction |
| Under load (conc=16) | [X] | [X] | [X]% reduction |

### Startup Observations
- BF16 startup time: ~[X] seconds
- Quant startup time: ~[X] seconds
- Notes: [any differences in download, warmup, etc.]

### Feynman: What I Learned About Quant Setup
> [5–10 sentences: What quantized model I picked, how vLLM sees it, how VRAM changed, what surprised me]

---

## Chat Capacity: BF16 vs Quant (Task 1.2)

### Results Summary

| Concurrency | max_tokens | BF16 tok/s | Quant tok/s | Δ% | BF16 p95 E2E | Quant p95 E2E | Δ% |
|-------------|------------|------------|-------------|-----|--------------|---------------|-----|
| 1 | 128 | | | | | | |
| 8 | 128 | | | | | | |
| 16 | 128 | | | | | | |

### Key Observations
1. [observation about throughput]
2. [observation about latency]
3. [observation about scaling]

### Feynman: Quant vs BF16 Chat Patterns
> At conc=1/8/16, max_tokens=128, quant vs BF16 →
> - tokens/sec ratios: [your numbers]
> - p95 patterns: [your observations]
> - surprises: [what you didn't expect]

---

## Quality Sanity Check (Task 1.3)

### Prompts Tested

| # | Prompt Summary | BF16 Quality | Quant Quality | Notes |
|---|----------------|--------------|---------------|-------|
| 1 | Factual Q&A | | | |
| 2 | Explanation | | | |
| 3 | Reasoning | | | |
| 4 | Code | | | |
| 5 | Summarization | | | |
| ... | ... | | | |

### Quality Assessment

**Overall**: [OK for X / Not acceptable for Y]

**Specific Issues**:
- [issue 1, if any]
- [issue 2, if any]

### Feynman: Quality Risk When Quantizing
> [paragraph: How I think about quality risk when quantizing; when manual spot-checking is enough, when I'd need proper automatic eval, and what failure modes I watch for]

---

## Batch Capacity: BF16 vs Quant (Task 2.1)

### Results Summary

| Concurrency | max_new_tokens | BF16 tok/s | Quant tok/s | Δ% | BF16 p95 | Quant p95 | Δ% |
|-------------|----------------|------------|-------------|-----|----------|-----------|-----|
| 16 | 256 | | | | | | |
| 32 | 256 | | | | | | |

### Feynman: Batch vs Chat Quant Impact
> Batch vs Chat quantization impact:
> - Chat at conc=16: quant gave [X%] throughput change
> - Batch at conc=32: quant gave [Y%] throughput change
> - Why the difference: [your hypothesis]

---

## Deep-Thinking Prompts: My Predictions vs Reality

### Before I Started

| Question | My Prediction |
|----------|---------------|
| VRAM reduction vs BF16? | |
| TTFT improve/same/worse? | |
| Quant help more at conc=1 or conc=16? | |
| p95 latency similar or degrade? | |
| Quality failures: which task types? | |

### After I Finished

| Question | Reality | Match? |
|----------|---------|--------|
| VRAM reduction | | ✅/❌ |
| TTFT change | | ✅/❌ |
| Conc=1 vs conc=16 benefit | | ✅/❌ |
| p95 latency | | ✅/❌ |
| Quality failures | | ✅/❌ |

### What I Got Wrong and Why
> [reflection on predictions vs reality]

---

## References

- BF16 CSV: `~/benchmarks/day004_chat_capacity_rtx16gb_bf16.csv`
- Quant CSV: `~/benchmarks/day004_chat_capacity_rtx16gb_quant.csv`
- Quality eval: `~/benchmarks/day004_quant_quality_eval.json`
- Batch BF16: `~/benchmarks/day004_batch_capacity_rtx16gb_bf16.csv`
- Batch Quant: `~/benchmarks/day004_batch_capacity_rtx16gb_quant.csv`
