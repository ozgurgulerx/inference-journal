# Day 004 – Quantization vs BF16 on RTX2000
## Tier 2: Extension Block (~1–2 hours)

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md)  
> **Goal**: Extend comparison to batch workloads + tie to business metrics  
> **End State**: Batch capacity CSVs + cost-per-token analysis  
> **Time**: ~1–2 hours

---

## Tier 2 Tasks

---

### ✅ Task 2.1: Batch Summarization – BF16 vs Quant

**Tags**: `[Inference–Runtime]` `[Quant]` `[Batching]` `[LongContext]`  
**Time**: ~45 min  
**Win**: Empirical data on whether quant helps more in throughput-heavy batch mode

#### 🎯 Objective

See whether quantization buys you more in throughput-heavy batch summarization vs interactive chat.

#### 🧠 Deep-Thinking Prompts (Before)

1. **Does quantization help more when you're fully saturating the GPU?** (batch mode)
2. **Will quant quality issues be more visible in summarization?** (longer outputs)
3. **Do you expect p95 blow-up for quant vs BF16 at high concurrency?**

#### 📋 Spec

Reuse Day 3 batch harness:
- `~/scripts/benchmarks/vllm_batch_summarize_bench.py`
- `~/scripts/benchmarks/run_batch_capacity_grid.sh`
- `~/data/day003_docs_sample.txt`

**Reduced grid** (fast comparison):
- Concurrency: `16, 32`
- max_new_tokens: `256`

#### 🔧 Implementation

**For BF16** (server on :8000):
```bash
GPU_NAME="RTX2000-16GB-BF16" \
OUT_CSV="$HOME/benchmarks/day004_batch_capacity_rtx16gb_bf16.csv" \
URL="http://127.0.0.1:8000/v1/completions" \
  ~/scripts/benchmarks/run_batch_capacity_grid.sh
```

**For Quant** (server on :8001):
```bash
GPU_NAME="RTX2000-16GB-QUANT" \
OUT_CSV="$HOME/benchmarks/day004_batch_capacity_rtx16gb_quant.csv" \
URL="http://127.0.0.1:8001/v1/completions" \
  ~/scripts/benchmarks/run_batch_capacity_grid.sh
```

> **Note**: If your grid script sweeps more configs, either modify it to focus on conc=16/32, max_new_tokens=256, or just run the full grid and extract the relevant rows.

#### ✅ Acceptance Criteria

- [ ] Two CSVs exist:
  - `day004_batch_capacity_rtx16gb_bf16.csv`
  - `day004_batch_capacity_rtx16gb_quant.csv`
- [ ] Short comparison table in `day004_quant_vs_bf16_notes.md`:
  - `throughput_tok_s` and `p95_latency_ms` for conc=16/32, max_new_tokens=256

#### 📁 Artifacts

- `~/benchmarks/day004_batch_capacity_rtx16gb_bf16.csv`
- `~/benchmarks/day004_batch_capacity_rtx16gb_quant.csv`
- Summary table in `~/artifacts/day004_quant_vs_bf16_notes.md` (section: "Batch Capacity: BF16 vs Quant")

#### 🔗 Cross-Day Reuse

| From | Artifact | Usage |
|------|----------|-------|
| Day 003 | `~/scripts/benchmarks/vllm_batch_summarize_bench.py` | Core benchmark |
| Day 003 | `~/scripts/benchmarks/run_batch_capacity_grid.sh` | Grid runner |
| Day 003 | `~/data/day003_docs_sample.txt` | Input documents |
| Day 003 | `~/benchmarks/day003_batch_capacity_rtx16gb.csv` | Baseline |

#### 📝 Feynman Deliverable

In `day004_quant_vs_bf16_notes.md`:

> "Batch vs Chat quantization impact:  
> - Chat at conc=16: quant gave [X%] throughput change  
> - Batch at conc=32: quant gave [Y%] throughput change  
> - Why the difference: [your hypothesis]"

#### 🔄 Reflection Prompts (After)

1. Did quantization help more in batch mode than chat mode?
2. Any concerning p95 blow-up for quant vs BF16 at high concurrency?
3. Would you recommend quant for batch workloads specifically?

#### 🏋️ Practice Reps

- [ ] Run conc=8 for batch to see scaling curve
- [ ] Try max_new_tokens=512 and check quality degradation
- [ ] Spot-check one batch output from quant vs BF16

---

### ✅ Task 2.2: Rough Cost-per-Token Comparison

**Tags**: `[Product/Business]` `[Obs/Cost]` `[Quant]`  
**Time**: ~30–45 min  
**Win**: Tie quant vs BF16 into a business-level story: $/1M tokens

#### 🎯 Objective

Turn your throughput numbers into a cost comparison that speaks to business stakeholders.

#### 🧠 Deep-Thinking Prompts (Before)

1. **Under which assumptions does quantization give a clear business win?**
2. **When would you refuse to quantize?** (e.g., high-risk user-facing use cases)
3. **How does utilization factor into cost-per-token?**

#### 📋 Spec

1. **Assume a rough hourly price for RTX2000 RunPod** (you decide — check current pricing)

2. **Use throughput_tok_s** from:
   - BF16: chat sweet spot (e.g., conc=16, 128 tokens)
   - Quant: same config

3. **Compute**:
   ```
   tokens_per_hour = throughput_tok_s × 3600
   cost_per_1M_tokens = (hourly_price / tokens_per_hour) × 1,000,000
   ```

4. **Document** in `~/artifacts/day004_quant_cost_comparison.md`

#### 🔧 Implementation

Create the cost comparison doc:

```markdown
# Day 004 – Quantization Cost Comparison

## Assumptions
- GPU: RunPod RTX 2000 Ada 16GB
- Hourly price: $[YOUR_ESTIMATE]/hr
- Utilization: 100% (continuous load)

## Chat Workload (conc=16, max_tokens=128)

| Model | Throughput (tok/s) | Tokens/Hour | $/1M Tokens |
|-------|-------------------|-------------|-------------|
| BF16  | [X]               | [X × 3600]  | [compute]   |
| Quant | [Y]               | [Y × 3600]  | [compute]   |

**Savings**: [Z]% cost reduction with quantization

## Batch Workload (conc=32, max_new_tokens=256)

| Model | Throughput (tok/s) | Tokens/Hour | $/1M Tokens |
|-------|-------------------|-------------|-------------|
| BF16  | [X]               | [X × 3600]  | [compute]   |
| Quant | [Y]               | [Y × 3600]  | [compute]   |

**Savings**: [Z]% cost reduction with quantization

## Narrative

2–3 sentences: "On this box, quantization changes economics like this..."
```

#### ✅ Acceptance Criteria

- [ ] Doc exists with your assumptions stated
- [ ] Computed $/1M tokens for BF16 vs quant (chat + batch)
- [ ] 2–3 sentence narrative explaining the business impact

#### 📁 Artifacts

- `~/artifacts/day004_quant_cost_comparison.md`

#### 📝 Feynman Deliverable

In the cost doc:

> "The business case for quantization on RTX2000: [your summary in 3–5 sentences, including caveats about quality trade-offs]"

#### 🔄 Reflection Prompts (After)

1. Is the cost savings worth the quality risk for your typical client?
2. How would this calculation change on A100/H100?
3. What's the "break-even" quality threshold where you'd refuse to quantize?

#### 🌍 Wider View

- **Cloud economics**: At scale, 30% throughput improvement = 30% fewer GPUs = real money
- **Quality cost**: One quality incident can cost more than months of GPU savings
- **Client conversations**: "We can cut your inference bill by X%, but here's the trade-off..."
- **A100/H100 context**: Quantization benefits may differ on larger GPUs with more memory bandwidth

---

## Tier 2 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **2.1** | Batch capacity: BF16 vs quant @ conc=16/32 | ⬜ |
| **2.2** | Cost-per-token comparison with assumptions | ⬜ |

### Additional Artifacts

```
~/benchmarks/
├── day004_batch_capacity_rtx16gb_bf16.csv
└── day004_batch_capacity_rtx16gb_quant.csv

~/artifacts/
└── day004_quant_cost_comparison.md
```

---

## Best Practices Update

After completing Tier 2, append these to `INFERENCE_PERFORMANCE_BEST_PRACTICES.md`:

- [ ] **Batch vs Chat quant impact**: Document which workload benefits more from quantization
- [ ] **Cost modeling**: Always tie throughput numbers to $/1M tokens for business conversations
- [ ] **Assumptions transparency**: State hourly cost and utilization assumptions explicitly
- [ ] **Quality-adjusted cost**: Factor in quality risk when presenting cost savings

---

**→ Continue to [Tier 3](LOG_tier03.md)**: Mini case study consolidation
