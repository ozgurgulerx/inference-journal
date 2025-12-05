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

## Quantization: What Actually Changes (INT4 AWQ/GPTQ)

### 1. VRAM: Weight Compression (The Good Part)

- BF16 weights: **2 bytes**; INT4 weights: **0.5 bytes**.  
- In practice you often see **40–60% VRAM savings**, which converts directly into:
  - More room for **KV cache** → higher safe concurrency.
  - Ability to fit **larger models on smaller GPUs** (e.g., 7B on 8–12GB, 1.5B + generous KV on 16GB).

> Mental model: *“Quant = more VRAM for KV cache → more concurrent users.”*

### 2. HBM Traffic: Bandwidth Cost (The Price You Pay)

- Runtimes store weights as **INT4**, but GEMMs still run in **FP16/BF16**. Each token step does:
  1. Load INT4 weights from HBM.  
  2. Dequantize to FP16/BF16.  
  3. Apply per-channel/group scales.  
  4. Feed into GEMM kernels.  
- This adds:
  - Extra memory reads and dequant kernels.  
  - More L2/cache pressure and HBM bandwidth usage.  
  - Slight increases in **TTFT** and sometimes **tail latency** for single-stream loads.

**Why HBM traffic goes up (even though everything is on-GPU):**

- You load **more than just weights**: INT4 chunks **plus** per-group scales and metadata (group structure, zero-points). Net bytes per GEMM tile often *increase* vs plain BF16.  
- Dequantized FP16 weights **are not cached back to HBM** – they live briefly in registers/shared memory and are discarded, so every forward pass must **reload INT4 weights** from HBM.  
- Quantization tends to **fragment kernels** (extra dequant/scale ops), so more small kernels each fetch their own tiles, reducing reuse and increasing total memory traffic.

> Mental model: *“Quant = fewer FLOPs per token, more memory bandwidth pressure.”*

### 3. CUDA Kernel Mix & Execution Graph (Behavior Shift)

- BF16 execution path (simplified):

  ```text
  GEMM → GEMM → FlashAttention → LayerNorm → next token
  ```

- AWQ/GPTQ execution path (typical):

  ```text
  dequant/scale → GEMM → GEMM → FlashAttention → LayerNorm → next token
  ```

  (or fused `int4_dequantize_and_gemm` kernels if optimized)

- Effects:
  - **More kernels**, often smaller and more memory-bound.  
  - Slightly more **kernel launch overhead** and CUDA Graph warmup work.  
  - Decode-phase throughput often stays similar because **attention remains the bottleneck**, not weight precision.

> One-liner (consulting version):  
> **“Quantization trades a bit of latency for a lot of VRAM, turning a FLOPs bottleneck into a memory-bottleneck and buying you concurrency and lower cost.”**

---

## Why Quantization Wins at the System Level

### 1. VRAM → Capacity (the main win)

- INT4 shrinks weight storage (e.g., 1.5B BF16 weights ≈ 2.9 GB → INT4 ≈ 0.9–1.1 GB).  
- vLLM converts freed VRAM into **KV cache**, so you can hold more active sequences at once.  
- On a 16GB RTX-class GPU, that often means **2×+ more concurrent users at the same p95**, even if each request is slightly slower.

> Enterprises don’t pay for raw TTFT; they pay for **throughput and concurrency capacity**.

### 2. Bigger models on cheaper hardware

- BF16 often forces “big model → big GPU” (e.g., 7B at long context wants A100/H100-class memory).  
- INT4 lets you fit **7B/8B models on mid-tier 16–24GB GPUs** with meaningful concurrency.  
- This unlocks migrations like “A100 → L40S/RTX” with large savings in $/hour.

### 3. Lower $/1M tokens via higher system throughput

- Single-stream decode may be similar or slightly slower (e.g., BF16 ≈ 300 tok/s vs INT4 ≈ 280 tok/s).  
- But if BF16 supports 6 concurrent users and INT4 supports 14 at the same SLO:

  ```text
  BF16: 300 tok/s × 6  = 1,800 tok/s
  INT4: 280 tok/s × 14 = 3,920 tok/s
  ```

- Net effect: **~2.2× throughput**, **~2× better cost/1M tokens**, **fewer GPUs for same traffic**.

### 4. Overhead is small vs economics

- Dequant adds maybe **5–15% TTFT** and **a few percent per-token overhead**, plus some HBM pressure.  
- In exchange you get:
  - Higher concurrency and tenant density.  
  - Ability to run **bigger models** on **cheaper GPUs**.  
  - Lower infra cost at a fixed SLA.

> Quantization optimizes the **real bottleneck** in LLM serving (capacity/VRAM), not the kernel FLOP count.

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
