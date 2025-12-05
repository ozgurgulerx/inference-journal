# Day 004 – Quantization vs BF16 on RTX2000
## Tier 5: GPU Hardware Deep Dive (~1–3 evenings)

> **Prerequisites**: Complete Tiers 1–3 (benchmarks + case study)  
> **Goal**: Build NVIDIA GPU hardware intuition for why quantization behaves the way it does  
> **End State**: Reading notes + a concrete study plan for GPU internals  
> **Time**: ~3–6 hours total (spread over a few nights)

---

## Tier 5 Tasks – NVIDIA Hardware Foundations

---

### ✅ Task 5.1: GPU Architecture – SMs, Tensor Cores, Memory Hierarchy

**Tags**: `[GPU/HW]` `[Architecture]` `[Foundations]`  
**Time**: ~60–90 min  
**Win**: You can sketch a modern NVIDIA GPU from memory (SMs, warps, memory hierarchy) and explain compute- vs memory-bound kernels.

#### 🎯 Objective

Understand how an NVIDIA GPU is physically organized so Day 004’s quantization effects (INT4 vs BF16) make sense at the hardware level.

#### 📘 Reading

- Pick **one** NVIDIA architecture whitepaper (search the title + PDF):  
  - Volta (GV100) Architecture Whitepaper  
  - Turing Architecture Whitepaper  
  - Ampere (A100) Architecture Whitepaper  
  - Ada Lovelace (RTX 40) Architecture Whitepaper  

Focus on:
- SM structure (warp schedulers, tensor cores, register file)  
- Memory hierarchy (HBM → L2 → shared memory → registers)  
- What makes kernels compute-bound vs memory-bound  

#### ✅ Acceptance Criteria

- [ ] 10–15 bullet notes in `~/artifacts/day004_tier05_gpu_arch_notes.md`.  
- [ ] You can verbally answer: *“Why does INT4 make decode more memory-bound on RTX 2000?”*

---

### ✅ Task 5.2: GEMM – Why LLMs Are Mostly Matmul

**Tags**: `[GPU/HW]` `[GEMM]` `[TensorCores]`  
**Time**: ~60–90 min  
**Win**: You understand why “LLMs ≈ GEMM engine + KV cache” and how tensor cores are fed.

#### 🎯 Objective

Learn how NVIDIA thinks about GEMM tiling and tensor core utilization so you can reason about throughput and why quantization shifts bottlenecks.

#### 📘 Reading

- “A Beginner’s Guide to GEMM Optimization” (Colfax Research) – search: `colfax gemm tutorial`.  
- Skim the **CUTLASS documentation** overview (tiling, threadblocks, MMA instructions).

Focus on:
- FLOPs, math throughput, and roofline-style reasoning.  
- How tiles move through HBM → L2 → shared memory → registers.  

#### ✅ Acceptance Criteria

- [ ] Short note in `day004_tier05_gpu_arch_notes.md` explaining:  
  - Why LLMs are 80–95% GEMM.  
  - Why small batch sizes under-utilize tensor cores.  

---

### ✅ Task 5.3: Memory Hierarchy & HBM Bandwidth

**Tags**: `[GPU/HW]` `[HBM]` `[MemoryHierarchy]`  
**Time**: ~60–90 min  
**Win**: You can explain, from first principles, why INT4 AWQ/GPTQ raises HBM pressure and makes decode more bandwidth-bound.

#### 🎯 Objective

Make the link from CUDA memory hierarchy → Day 004 observations (TTFT change, dequant overhead, INT4 becoming memory-bound).

#### 📘 Reading

- CUDA Programming Guide – **Memory Hierarchy** section (Global/HBM, L2, shared, registers, coalescing).  
- “Understanding GPU Memory Hierarchy” by Mark Harris (NVIDIA) – search title.

Focus on:
- What HBM bandwidth means in practice.  
- Coalesced vs non-coalesced access.  
- Why fused kernels and good locality matter.  

#### ✅ Acceptance Criteria

- [ ] A paragraph in `day004_tier05_gpu_arch_notes.md` titled **“Why INT4 increases HBM traffic even on-GPU”** tying directly into your Day 004 Nsight / mental model.  
- [ ] You can articulate: *“BF16 is compute-bound, INT4 often becomes memory-bound because …”*

---

### ✅ Task 5.4: CUDA Execution Model – Warps, Blocks, Occupancy

**Tags**: `[GPU/HW]` `[Kernels]` `[Occupancy]`  
**Time**: ~60–90 min  
**Win**: You can read a kernel configuration and intuit why it under- or over-utilizes the GPU.

#### 🎯 Objective

Connect CUDA’s execution model (warps/blocks/streams) with vLLM’s kernel behavior under quantization and different batch sizes.

#### 📘 Reading

- “CUDA by Example” – chapters on the execution model.  
- Nsight Compute User Guide – **Occupancy** and **Warp State** sections.

Focus on:
- Warps, blocks, SM occupancy.  
- How small GEMMs / small batches hurt utilization.  
- Why attention and tiny dequant kernels can fragment scheduling.

#### ✅ Acceptance Criteria

- [ ] 5–10 bullets in `day004_tier05_gpu_arch_notes.md` under **“Execution model takeaways”**.  
- [ ] You can answer: *“Why do some vLLM kernels get slower at small batch sizes?”*

---

### ✅ Task 5.5: Nsight Systems & Nsight Compute – Your Core Profilers

**Tags**: `[GPU/HW]` `[Profiling]` `[Nsight]`  
**Time**: ~45–60 min (intro pass)  
**Win**: You know what Nsight Systems vs Nsight Compute are for and roughly how to use each.

#### 🎯 Objective

Understand at a high level how to capture traces and kernel stats so you can inspect BF16 vs AWQ/GPTQ behavior at the kernel level.

#### 📘 Viewing / Reading

- NVIDIA YouTube tutorials: “Nsight Systems Tutorial”, “Nsight Compute Tutorial”.  
- Skim Nsight Systems docs for **GPU kernel timelines**; Nsight Compute docs for **kernel metrics (FLOPs, bandwidth, occupancy)**.

#### ✅ Acceptance Criteria

- [ ] 5 bullets in your notes titled **“Nsight Systems vs Nsight Compute (who does what)”**.  
- [ ] A TODO in your Day 004 plan: *“Capture BF16 vs AWQ traces for one 200-token run using Nsight Systems.”*

---

### ✅ Task 5.6: LLM-Specific GPU Optimizations

**Tags**: `[GPU/HW]` `[LLM-Kernels]` `[FlashAttention]` `[vLLM]`  
**Time**: ~60–90 min  
**Win**: You can explain why FlashAttention and PagedAttention exist and how they relate to your Day 004 quantization work.

#### 🎯 Objective

Connect generic GPU knowledge to LLM-serving primitives.

#### 📘 Reading

- FlashAttention paper – focus on IO-awareness and tiling, not proofs.  
- vLLM / PagedAttention paper or blog – how KV cache is laid out, why paging helps.  
- TensorRT-LLM whitepaper – skim sections on kernel fusion, quant pipelines, scheduling.

#### ✅ Acceptance Criteria

- [ ] Short section in `day004_tier05_gpu_arch_notes.md` titled **“Why FlashAttention is special for LLMs”**.  
- [ ] 3 bullets connecting PagedAttention to your Day 003/004 capacity experiments.

---

### ✅ Task 5.7: Implementation Practice (Stretch Goal)

**Tags**: `[GPU/HW]` `[HandsOn]` `[Triton/CUDA]`  
**Time**: ~2–4 hours total (multi-day)  
**Win**: You’ve written at least one tiny kernel and looked at its behavior in a profiler.

#### 🎯 Objective

Turn reading into muscle memory by touching real kernels.

#### 📋 Suggested Mini-Projects

- Write a **simple CUDA vector-add** kernel or follow a minimal tutorial; annotate how warps/blocks map to data.  
- Write a **tiny GEMM in Triton** (or study a minimal example) and note how tiling + memory coalescing show up in performance.  
- Once comfortable, plan to:
  - Capture Nsight Systems traces for BF16 vs AWQ vs GPTQ on a small LLM.  
  - Compare kernel mix, HBM throughput, and tensor core utilization.

#### ✅ Acceptance Criteria

- [ ] At least one hands-on kernel experiment completed (CUDA or Triton), with a few bullets on what you observed.  
- [ ] A list of follow-up ideas for Day 010+ (deeper GPU profiling work).

---

### ✅ Task 5.8: Stay Focused on Inference-Relevant GPU Knowledge

**Tags**: `[Meta]` `[Learning]` `[Focus]`  
**Time**: ~10–15 min (meta reflection)  
**Win**: You have a tight mental filter for what GPU topics are relevant to LLM inference.

#### 🎯 Objective

Avoid getting lost in generic CUDA/HPC rabbit holes; focus on the slice of GPU knowledge that moves the needle for LLM serving.

#### 📋 Guidance

De‑prioritize for now:
- Deep CUDA C++ language mastery.  
- Raytracing tutorials and graphics pipelines.  
- HPC solvers (FEM, CFD) and unrelated CUDA samples.

Keep your study centered on:
- GPU architecture and memory hierarchy.  
- GEMM + tensor core utilization.  
- Kernel scheduling and occupancy.  
- DL inference kernel patterns (GEMM, attention, dequant).  
- Nsight profiling.  
- vLLM internals + quantization internals.

#### ✅ Acceptance Criteria

- [ ] One short paragraph in your notes titled **“My GPU learning filter (LLM inference only)”**.  
- [ ] A trimmed reading list you can realistically finish over the next 4–6 weeks.

