# 🔥 **AWQ vs GPTQ — Deep, Practical Explanation**

Quantization in LLM inference = **reduce weight precision (usually from FP16/BF16 → INT4)** without hurting quality too much, so you:

* fit larger models in VRAM
* serve more concurrent users
* increase throughput

Both **AWQ** and **GPTQ** are *weight-only post-training quantization* methods.
They do **not** quantize activations — only weights.

But their philosophy, math, and behavior differ significantly.

---

# 🟩 **1. AWQ — Activation-Aware Weight Quantization (2023)**

### **Core idea:**

**Don't quantize weights blindly.
Quantize them based on how important they are for activation quality.**

AWQ looks at how each weight contributes to activations during inference.
It identifies **“critical” channels / heads / blocks** and selectively *attenuates* quantization error.

### **How AWQ works (mechanics):**

1. **Feed calibration samples (200–500 tokens typical)** through the FP16 model.
2. Measure **activation sensitivity** for each block/layer.
3. Compute scaling factors:

   * Big weights → keep more precision
   * Small/noisy weights → can compress safely
4. Apply per-channel INT4 quantization.
5. Produce a quantized checkpoint + scale metadata.

### **Strengths**

| AWQ Strength                 | Why it matters                               |
| ---------------------------- | -------------------------------------------- |
| ✔ **Excellent stability**    | Less likely to degrade reasoning / coherence |
| ✔ **Very low VRAM usage**    | Often 40–60% reduction                       |
| ✔ **Fast loading & serving** | Simple structure, good for vLLM              |
| ✔ **Good on long context**   | Important for Qwen2.5 models                 |
| ✔ **Production-friendly**    | Deterministic, robust                        |

### **Weaknesses**

* Slightly slower TTFT compared to pure FP16 (because of scaling operations).
* Sometimes slightly lower throughput vs GPTQ at batch 1 (rare).

---

# 🟥 **2. GPTQ — Gradient Post-Training Quantization (2022)**

### **Core idea:**

**Quantize weights by solving a one-step optimization problem that minimizes output error.**

GPTQ is more “mathematical” and uses:

* blockwise reconstruction
* error minimization
* quantization-aware second-order approximations (Hessian-based)

### **How GPTQ works (mechanics):**

1. Run calibration samples through FP16 model.
2. Compute approximate Hessian of the weight blocks.
3. Quantize each block while minimizing:

   ```
   || W_fp16  –  W_int4 * scale ||   under Hessian weighting
   ```
4. Bake quantized weights into a single safetensors file.

### **Strengths**

| GPTQ Strength                               | Why it matters                |
| ------------------------------------------- | ----------------------------- |
| ✔ **Slightly faster throughput at batch=1** | Good for low-concurrency apps |
| ✔ **Often smaller files**                   | More aggressive compression   |
| ✔ **Works very well on many 7B–13B models** | Very widely adopted           |

### **Weaknesses**

* **Less stable** than AWQ on reasoning-heavy workloads
* Can produce **more quality artifacts**
* Some GPTQ models suffer from:

  * repetition loops
  * missing token collapse
  * broken long-context behavior
* More variance depending on quantization config (group size, act order, dampening)

---

# ⚡ Summary Table — AWQ vs GPTQ (Engineer Edition)

| Dimension              | **AWQ**                         | **GPTQ**                                      |
| ---------------------- | ------------------------------- | --------------------------------------------- |
| Method Type            | Activation-aware                | Error-minimization                            |
| Quality Stability      | ⭐⭐⭐⭐                            | ⭐⭐⭐                                           |
| VRAM Reduction         | ⭐⭐⭐⭐                            | ⭐⭐⭐⭐                                          |
| Throughput (batch=1)   | ⭐⭐⭐                             | ⭐⭐⭐⭐                                          |
| Throughput (batch>1)   | ⭐⭐⭐⭐                            | ⭐⭐⭐                                           |
| Tail Latency           | Low                             | Medium                                        |
| Long Context Stability | High                            | Medium–Low                                    |
| Failure Modes          | Mild degradation                | Repetition, collapse, weird loops             |
| vLLM Compatibility     | Excellent                       | Good but config-dependent                     |
| Best For               | Chatbots, reasoning, production | Lightweight apps, small models, offline tools |

---

# 🚀 In Your Case: **Qwen2.5-1.5B on RTX2000 Ada**

For your exact setup:

* small model (1.5B)
* BF16 baseline already running
* RTX 2000 Ada 16GB
* vLLM as runtime
* you care about **quality**, stability, KV behavior

### ⭐ **Recommended: AWQ**

Why?

* Extremely stable for long-context models (Qwen is 32k by default)
* Better behavior under concurrency (vLLM continuous batching)
* More robust to scaling issues when GPU is small (16GB)
* Lower VRAM footprint → more KV cache → more concurrency
* Lower risk of weird output artifacts during your quality tests (Tier04 Task 1.3)

### GPTQ is still usable, but:

* slightly riskier for reasoning tasks
* may degrade at 4-bit
* more variance depending on quant parameters
* some GPTQ quantizations for Qwen2.5 have been reported to be unstable in vLLM

---

# 📌 Recommended Model for Day 004 (again, reinforced)

### ✔ **`bartowski/Qwen2.5-1.5B-Instruct-AWQ`**

This is the best quantized version of your Day 2 baseline model.

Works flawlessly with:

```bash
--quantization awq
```

---

## Advanced Quantization Topics – Concrete Examples

### Quant capacity

**Goal**: Turn “INT4 saves VRAM” into **hard numbers about extra capacity** on RTX 2000.

- Run a reduced chat grid (e.g. `conc=1,8,16`, `max_tokens=128`) for BF16 and AWQ.  
- Store a merged CSV: `~/benchmarks/day004_quant_capacity_rtx16gb.csv` with columns like `precision, conc, tok_s, p95_ms`.  
- Derive:
  - “Max safe concurrency” for BF16 vs AWQ at your target p95.  
  - A one-liner, e.g. *“On RTX 2000, AWQ sustains ~1.7× more users at p95 ≤ 3s.”*  
- Add a short capacity summary to `day004_quant_vs_bf16_notes.md` under “Quant Capacity on RTX2000”.

### Quant compute graphs

**Goal**: See **where** quantization is buying you speed (or not) in the kernel timeline.

- Capture a short Nsight Systems trace for a single 200-token generation in BF16 vs AWQ.  
- For each run, screenshot the kernel timeline and annotate:
  - Which kernels shrink / disappear under quant (e.g. GEMMs, dequant ops).  
  - Whether the decode loop still looks memory-bound (lots of small kernels with gaps).  
- Save annotated images under `~/artifacts/day004_quant_compute_graphs/`.  
- In your notes, write 3–5 bullets answering: *“Did INT4 move me closer to a FLOP ceiling or just reduce bandwidth pressure?”*

### Quant quality failure modes

**Goal**: Build a **small catalog of real failure modes** instead of vague “quality may drop”.

- Construct a 15–20 prompt set mixing: factual QA, “explain like I’m 5”, multi-step reasoning, code, and summarization.  
- Run BF16 and AWQ and log obvious issues into `~/artifacts/day004_quant_quality_failures.md` with a table:
  - `prompt_id`, `category`, `bf16_ok?`, `quant_issue?`, `symptom`, `notes`.  
- Look for patterns:
  - Does AWQ fail more on math? On code? On multi-hop reasoning?  
- Add a “Quant Failure Modes” subsection to your Day 4 notes with 3 concrete examples you’d actually show a client.

### Quant cost models

**Goal**: Turn throughput numbers into **$/1M tokens** that product teams understand.

- Pick 1–2 representative configs (e.g. chat conc=16, batch conc=32).  
- Using your measured tokens/sec and an hourly RTX 2000 price, compute:
  - `cost_per_1M_tokens_bf16` and `cost_per_1M_tokens_awq`.  
- Write a short `~/artifacts/day004_quant_cost_model.md` with:
  - A tiny table summarizing the numbers.  
  - 2 bullets on what this implies for *daily* or *monthly* spend at your expected traffic.  
- Pull a one-sentence takeaway into the case study: *“INT4 AWQ cuts serving cost/1M tokens by ~X% at equal p95 latency.”*

### Quant-under-concurrency

**Goal**: Understand **how quant changes your latency curve**, not just peak throughput.

- Fix `max_tokens` and sweep concurrency upward for BF16 and AWQ until p95 crosses your chat SLO (e.g. 3s).  
- Plot or at least tabulate `conc` vs `p95` for both precisions; store raw data in `~/benchmarks/day004_quant_concurrency_sweep.csv`.  
- Extract:
  - “Knee points” where p95 starts to blow up.  
  - A rule of thumb, e.g. *“On RTX 2000, AWQ keeps p95 < 3s up to conc≈24; BF16 only to conc≈14.”*  
- Add these knee points to your “Quantization Risk Brief” as concrete operating guidance.

### The real reasons enterprises choose INT4

**Goal**: Distill **business, ops, and platform** reasons—not just “it’s faster”.

- Based on your experiments, draft `~/reports/day004_int4_business_case.md` with 5–8 bullets such as:
  - Higher tenant density per GPU (more workspaces / orgs per card).  
  - Meeting latency SLOs on cheaper SKUs (RTX/A-series instead of A100/H100).  
  - Making A/B experiments cheaper by reducing the GPU footprint for each variant.  
  - Keeping a single “INT4-optimized” platform config instead of bespoke BF16 setups per model.  
- Close with 2 bullets on **when INT4 is a bad idea** (e.g. safety-critical QA, strict factual accuracy), tied back to the failure modes you observed.

