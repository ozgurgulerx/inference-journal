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

- **Quant capacity**  
  - Example: reuse your Day 3 chat grid to run BF16 vs AWQ at `conc=1,8,16` with fixed `max_tokens=128`, write results to `~/benchmarks/day004_quant_capacity_rtx16gb.csv`, and add a short note on how many extra concurrent users INT4 buys you at the same p95.

- **Quant compute graphs**  
  - Example: capture a short Nsight Systems trace for a 200-token generation in BF16 vs AWQ, then annotate one screenshot per run highlighting which kernels become memory-bound vs compute-bound after quantization.

- **Quant quality failure modes**  
  - Example: build a 15–20 prompt set mixing factual QA, code, and step-by-step reasoning; run BF16 and AWQ, and log obvious issues (loops, contradictions, dropped constraints) into `~/artifacts/day004_quant_quality_failures.md` with one row per failure.

- **Quant cost models**  
  - Example: using your measured tokens/sec and an hourly GPU price for RTX 2000, compute `$ / 1M tokens` for BF16 vs AWQ in both chat and batch modes, and summarize the deltas in a small table in `~/artifacts/day004_quant_cost_model.md`.

- **Quant-under-concurrency**  
  - Example: hold `max_tokens` fixed and sweep concurrency until p95 latency crosses your SLO for BF16 and AWQ; record the “max safe concurrency” for each and turn that into a simple rule of thumb in your notes (e.g., “on RTX 2000, AWQ safely carries ~1.5–2× BF16 users at p95 ≤ 3s”).

- **The real reasons enterprises choose INT4**  
  - Example: draft a short bullet list for `~/reports/day004_int4_business_case.md` covering concrete drivers like GPU SKU consolidation, higher tenant density, meeting fixed latency SLOs on smaller GPUs, and enabling cheaper A/B capacity experiments—grounded in your Day 004 measurements instead of generic claims.


