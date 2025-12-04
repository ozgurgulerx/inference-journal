# Day 004 – Quantization vs BF16 on RTX2000
## Tier 3: Deep Work (~1–3 hours)

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) and [Tier 2](LOG_tier02.md)  
> **Goal**: Consolidate findings into a reusable case study  
> **End State**: Client-ready mini case study document  
> **Time**: ~1–3 hours (optional, for 4h+ days)

---

## Tier 3 Tasks

---

### ✅ Task 3.1: Mini Quantization Case Study – "RTX2000 Chat & Batch"

**Tags**: `[Inference–Runtime]` `[Model/Training]` `[Product/Business]`  
**Themes**: `[Quant]` `[Batching]` `[Obs/Cost]`  
**Time**: ~1–2 hours  
**Win**: A reusable document you can adapt for client presentations

#### 🎯 Objective

Turn today's work into a mini case study that demonstrates your quantization analysis methodology.

#### 📋 Spec

Create: `~/reports/day004_quant_vs_bf16_rtx2000.md`

**Structure**:

```markdown
# Quantization vs BF16 on RTX 2000 Ada 16GB
*A practical comparison for small-GPU inference deployments*

---

## 1. Setup

### Hardware
- GPU: [exact model]
- VRAM: 16GB
- Provider: RunPod

### Models
- **Baseline (BF16)**: [model name, HF ID, size]
- **Quantized**: [model name, HF ID, quantization method, size]

### vLLM Configuration
| Parameter | Value |
|-----------|-------|
| max_model_len | [X] |
| max_num_seqs | [X] |
| gpu_memory_utilization | [X] |

---

## 2. VRAM Comparison

| Model | Idle VRAM | % of Total |
|-------|-----------|------------|
| BF16  | [X] GB    | [X]%       |
| Quant | [Y] GB    | [Y]%       |

**Observation**: [1–2 sentences on VRAM savings and what it enables]

---

## 3. Chat Workload Results

### Capacity at Key Operating Points

| Config | BF16 tok/s | Quant tok/s | Δ% | BF16 p95 (ms) | Quant p95 (ms) | Δ% |
|--------|------------|-------------|-----|---------------|----------------|-----|
| conc=1, 128 tok | [X] | [Y] | [Z]% | [X] | [Y] | [Z]% |
| conc=8, 128 tok | [X] | [Y] | [Z]% | [X] | [Y] | [Z]% |
| conc=16, 128 tok | [X] | [Y] | [Z]% | [X] | [Y] | [Z]% |

**Key Finding**: [2–3 sentences summarizing the pattern]

---

## 4. Batch Workload Results

### Capacity at Key Operating Points

| Config | BF16 tok/s | Quant tok/s | Δ% | BF16 p95 (ms) | Quant p95 (ms) | Δ% |
|--------|------------|-------------|-----|---------------|----------------|-----|
| conc=16, 256 tok | [X] | [Y] | [Z]% | [X] | [Y] | [Z]% |
| conc=32, 256 tok | [X] | [Y] | [Z]% | [X] | [Y] | [Z]% |

**Key Finding**: [2–3 sentences]

---

## 5. Quality Sanity Check

### Sample Comparisons

| Prompt Type | BF16 Quality | Quant Quality | Issue? |
|-------------|--------------|---------------|--------|
| Factual Q&A | ✅ Good | ✅ Good | None |
| Reasoning | ✅ Good | ⚠️ Minor | [describe] |
| Code | ✅ Good | ⚠️ Minor | [describe] |
| Summarization | ✅ Good | ✅ Good | None |

**Overall Assessment**: [2–3 sentences on quality acceptability]

---

## 6. Cost Analysis

### Assumptions
- Hourly cost: $[X]/hr (RunPod RTX 2000)
- Utilization: 100%

### Chat Workload (conc=16, 128 tokens)

| Model | Throughput | $/1M Tokens |
|-------|------------|-------------|
| BF16  | [X] tok/s  | $[Y]        |
| Quant | [X] tok/s  | $[Y]        |

**Savings**: [X]% cost reduction

### Batch Workload (conc=32, 256 tokens)

| Model | Throughput | $/1M Tokens |
|-------|------------|-------------|
| BF16  | [X] tok/s  | $[Y]        |
| Quant | [X] tok/s  | $[Y]        |

**Savings**: [X]% cost reduction

---

## 7. Recommendations

### When to Quantize on RTX 2000

✅ **Recommend quantization for**:
- [workload type 1]
- [workload type 2]

⚠️ **Use caution for**:
- [workload type]

❌ **Avoid quantization for**:
- [workload type]

### Configuration Recommendations

| Workload | Model | Concurrency | Expected Throughput | p95 Budget |
|----------|-------|-------------|---------------------|------------|
| Chat | Quant | 16 | [X] tok/s | < 3000ms |
| Batch | Quant | 32 | [X] tok/s | < 6000ms |

---

## 8. Methodology & Reproducibility

### Scripts Used
- Chat benchmark: `~/scripts/benchmarks/vllm_chat_bench.py`
- Batch benchmark: `~/scripts/benchmarks/vllm_batch_summarize_bench.py`
- Quality eval: `~/scripts/benchmarks/day004_quant_quality_eval.py`

### Raw Data
- `~/benchmarks/day004_chat_capacity_rtx16gb_bf16.csv`
- `~/benchmarks/day004_chat_capacity_rtx16gb_quant.csv`
- `~/benchmarks/day004_batch_capacity_rtx16gb_bf16.csv`
- `~/benchmarks/day004_batch_capacity_rtx16gb_quant.csv`
- `~/benchmarks/day004_quant_quality_eval.json`

---

## 9. Next Steps

For deeper analysis:
- [ ] Test on A100/H100 to see if quant benefits scale
- [ ] Try FP8 quantization if hardware supports
- [ ] Run proper eval benchmark (e.g., MMLU subset) for quality
- [ ] Test longer contexts (4k, 8k tokens)

---

*Generated: [date] | Hardware: RTX 2000 Ada 16GB | vLLM [version]*
```

#### ✅ Acceptance Criteria

- [ ] Report is readable and complete
- [ ] References all CSVs and scripts by path
- [ ] Can be adapted into a blog post or client slide deck
- [ ] Includes clear recommendations with rationale

#### 📁 Artifacts

- `~/reports/day004_quant_vs_bf16_rtx2000.md`

#### 📝 Feynman Deliverable (~150–300 words)

Close the report with a section:

> **"What I Learned About Quantization vs BF16 on a Small GPU"**
>
> Write this as if explaining to a customer who's asking: "Should we quantize our model?"
>
> Cover:
> - What actually changes (VRAM, throughput, latency)
> - When it's a clear win vs when to be cautious
> - The quality trade-off reality
> - How to think about the cost/quality balance

#### 🏋️ Hard Mode

If you want to push toward 1% competence:

- [ ] **Add failure injection**: Deliberately stress the quant model until it breaks (high concurrency + long context) and document the failure mode
- [ ] **Add A100 estimation**: Based on your RTX2000 data + Day 003 A100 intuition, estimate what quant would do on A100
- [ ] **Add automatic quality check**: Extend your eval script to compute a simple similarity score between BF16 and quant outputs

---

## Tier 4: Optional Reading & Business Tie-In

### 📚 Reading (Off-Hours Only)

If you want to deepen today's theme after the labs:

1. **AWQ/GPTQ Implementation**
   - A recent explainer on how AWQ/GPTQ actually work at the kernel level
   - Tie to: "Does this explain the speed-up/slow-down I observed?"

2. **vLLM Quantization Docs**
   - Official docs/issues related to quantization support
   - Tie to: "Are there flags I missed that could improve my results?"

3. **FP8/INT8 on H100**
   - A blog/paper on FP8 quantization on newer hardware
   - Tie to: "How would my recommendations change on H100?"

### 💼 Business / Positioning Nudge

Add a bullet to your Business notebook (`business/offers.md`):

```markdown
## Quantization Health Check (1–2 week engagement)

**Scope**:
- Benchmark customer's BF16 model vs curated quantized variants
- Run on their target hardware (or cloud equivalent)
- Produce cost model and quality assessment

**Deliverables**:
- Capacity comparison (throughput, latency, VRAM)
- Quality spot-check report
- Cost-per-token analysis
- Recommendation: quantize or not, with rationale

**Seed artifact**: Day 004 case study
```

Note explicitly:
> "Day 04 artifacts = the seed benchmark methodology for this offer."

---

## Tier 3 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **3.1** | Created mini case study with all Day 004 data | ⬜ |

### Final Artifacts

```
~/reports/
└── day004_quant_vs_bf16_rtx2000.md
```

---

## 🎉 Day 004 Complete!

### What You Achieved

By finishing Day 004, you now have:

- ✅ **Quantized model running** alongside BF16 on the same GPU
- ✅ **Empirical comparison**: chat + batch capacity, BF16 vs quant
- ✅ **Quality sanity check**: manual spot-check with documented findings
- ✅ **Cost analysis**: $/1M tokens comparison with business narrative
- ✅ **Reusable case study**: ready for client adaptation

### Key Takeaways

1. **VRAM savings are real** — quantization frees memory for higher concurrency or longer contexts
2. **Throughput gains vary** — depends on workload, concurrency, and how GPU-bound you already were
3. **Quality risk is real** — never ship quantized without at least basic spot-checking
4. **Business case is nuanced** — cost savings must be weighed against quality risk
5. **Methodology matters** — same harness, same configs, apples-to-apples comparison

### Connection to Learning Goals

| Theme | Day 004 Contribution |
|-------|---------------------|
| Quantization & Mixed Precision | Core focus: AWQ/GPTQ comparison |
| KV Cache Design | Observed VRAM impact on max_num_seqs headroom |
| Continuous Batching | Compared quant impact at different concurrency levels |
| Observability & Cost | Built $/1M tokens analysis |
| Reliability | Quality spot-check methodology |

---

## 🔜 Next Steps

When ready for Day 005+, potential directions:

- **Streaming deep dive**: Enable true streaming, measure TTFT drop
- **Longer context testing**: 4k, 8k, 16k tokens — how does quant scale?
- **Different quantization methods**: Compare AWQ vs GPTQ vs INT8
- **A100/H100 anchor**: See if quant benefits change on bigger GPUs
- **Automatic quality eval**: Integrate a proper eval benchmark (MMLU, etc.)

---

*Day 004 concluded. You now have a quantization analysis methodology and reusable case study.* 🚀
