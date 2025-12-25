# Day 008 – Groq LPU Inference + Data Center Dimensioning (Groq-Native Playbook)

**Objective:** Build a Groq-specific inference engineering + facility design playbook that lets you lead (not follow) design reviews with Groq compiler, hardware, fabric, and data center engineers.

**Scope:** Groq LPU behavior under real inference load, deterministic scheduling implications, RealScale-style rack-scale fabric intuition, and an explicit capacity/dimensioning model that does **not** import GPU assumptions.

**How To Use This Day**

- Start with `LOG_tier01.md` to rebuild your mental model around determinism + compiler-first execution.
- Use `LOG_tier02.md` to convert that mental model into sizing math, SLOs, rack design, and expansion sequencing.
- Use `LOG_tier03.md` to pressure-test the design against failure modes, awkward model shapes, and “GPU intuition traps”.
- Run a real meeting using `groq_meeting_runbook.md`, and use `questions_for_groq.md` + `answers_bank_top1percent.md` to control the narrative.

---

## What Changes When You Stop Thinking “GPU”

### Determinism Is The Product

- **Fact (from Groq reference):** Groq’s core promise is deterministic execution with compiler-scheduled data movement and compute. *(Attach the specific Groq reference in your engagement notes.)*
- **Inference:** If service time becomes predictable, **p99 latency stops being dominated by kernel/cache variance** and becomes dominated by **queueing + admission control**.
- **GPU intuition trap:** On GPUs, “variance is normal” (warp scheduling, cache misses, HBM contention, kernel launch jitter). On Groq, *variance is a bug*—you dimension for queueing, not micro-variance.

### The Compiler Is The Runtime

- **Fact (from Groq reference):** Groq compiles the model to a static schedule; runtime variability is intentionally minimized. *(Attach the specific Groq reference in your engagement notes.)*
- **Consequence:** Your “deployment artifact” is no longer “container + model weights”; it’s **(model, graph shape, batch/sequence constraints, compiler version, target hardware)** → compiled binary/schedule.
- **GPU intuition trap:** “We’ll just change the batch size later.” On Groq, changing shape may mean **recompile** and potentially different placement; shape is a first-class ops decision.

### Memory Hierarchy Inversion (SRAM-first Mental Model)

- **Fact (from Groq reference):** Groq emphasizes on-chip SRAM and deterministic access patterns rather than GPU-style cache hierarchies. *(Attach the specific Groq reference in your engagement notes.)*
- **Consequence:** You trade cache-miss chaos for **explicit capacity constraints** and **compile-time shape constraints**. You can’t “just rely on cache to save you.”
- **GPU intuition trap:** “The cache will probably hit.” On Groq, you must ask: **where does every tensor live, when, and for how long?**

---

## Deliverables (This Folder)

- `LOG_tier01.md`: Groq LPU mental model, terminology, deterministic inference intuition.
- `LOG_tier02.md`: Dimensioning model, SLO mapping, rack/facilities translation, scale strategy.
- `LOG_tier03.md`: Failure modes, compiler constraints, cross-rack scaling realities, expert reasoning.
- `groq_meeting_runbook.md`: Agenda + decision-forcing prompts + whiteboard flow.
- `dimensioning_inputs_template.md`: The exact fields you must collect.
- `capacity_model_template.md`: A fill-in-the-blanks capacity model.
- `questions_for_groq.md`: High-signal questions grouped by domain.
- `answers_bank_top1percent.md`: Top-1% answer patterns and explanations.
- `risk_register.md`: Operational + technical risks with mitigations.
- `glossary.md`: Consistent vocabulary to prevent “same word, different meaning” failures.
- `acceptance_criteria.md`: Quality bar and self-checks for credibility.
- `scripts/`: Pseudocode templates for modeling.

---

## Learning Resources (Curated Path To Top-1% Groq Inference + DC Design)

Use this as your background reading to turn the playbook into “can speak with Groq staff engineers on a whiteboard” capability.

**Evidence hygiene:** Benchmark numbers and marketing claims below are **Assumption to validate** unless they come from an official Groq reference you can cite for your engagement. Prefer primary sources (Groq docs/papers) for architectural claims.

### Beginner (Foundations: tooling + basic mental model)

- **Groq Tutorial 2025** (Article; GroqFlow basics): `aimltechnews.com` *(Assumption to validate; third-party)*  
  Focus: end-to-end “import → compile → run inference” and the determinism framing.
- **Getting Started with Groq API** (Tutorial; API usage + streaming): `analyticsvidhya.com` *(Assumption to validate; third-party)*  
  Focus: operational patterns for streaming + integration; useful to understand the *product* surface area even if you deploy on-prem.
- **GroqWare / GroqFlow overview** (Official docs): `groq.sa` *(Fact once pinned to your chosen doc version)*  
  Focus: the actual toolchain contract; treat this as the source of truth for what compilation and profiling support exists.

### Intermediate (Hands-on: compiler + profiling + rack-scale intuition)

- **“The Architecture of Groq’s LPU” (Abhinav Upadhyay)** (Deep-dive article): `blog.codingconfessions.com` / `medium.com` *(Assumption to validate; third-party)*  
  Focus: conceptual explanation of TSP/TSP-slice thinking; helpful to build intuition before reading the ISCA papers.
- **“How Groq LPU Works – Understanding LPUs” (Igor Arsovski interview)** (Video): `classcentral.com` *(Assumption to validate; third-party link to video)*  
  Focus: system-level discussion (hardware + compiler + scaling) and practical trade-offs.
- **GroqFlow + Compiler docs + GroqView profiler references** (Docs + examples): `github.com` / `groq.sa` / `docs.alcf.anl.gov` *(Fact once pinned)*  
  Focus: what the compiler exposes (reports, estimators, profilers) for the measurements used in `LOG_tier02.md`.
- **Groq AI Training Workshop (Argonne, 2024)** (Workshop video/slides): `alcf.anl.gov` *(Assumption to validate; third-party host)*  
  Focus: GroqRack-style cluster workflows and operational bring-up patterns.

### Advanced (Expert: primary sources + cutting-edge compiler/hardware co-design)

- **Groq ISCA 2020 & 2022 papers** (Academic papers; primary sources): `fuse.wikichip.org` / `groq.humain.ai` *(Assumption to validate; mirrors/hosts vary)*  
  Focus: the deterministic, statically scheduled streaming architecture and large-scale interconnect design. Read these to justify “why determinism changes p99” without hand-waving.
- **“Inside the LPU: Deconstructing Groq’s Speed” (Andrew Ling, 2025)** (Groq blog): `groq.com` *(Fact once pinned)*  
  Focus: modern LPU details, numerics, SRAM-centric design, and compiler orchestration. Use this to sharpen Tier 03 discussions on performance cliffs and artifact stability.
- **Benchmark/competitive analyses** (Third-party + vendor): `groq.com`, `aiixx.ai`, `blog.startupstash.com` *(Assumption to validate)*  
  Focus: sanity-check throughput/latency envelopes; do not dimension a DC on these alone—convert them into hypotheses and validate with your own compiled artifacts.
- **Groq community (Discord/forum) + API Cookbook**: `community.groq.com` *(Assumption to validate for access details)*  
  Focus: “tribal knowledge” and fast feedback loops; treat community claims as leads, not proofs.

### Complementary (Reasoning-model cost literacy, directly relevant to capacity)

- **Build a Reasoning Model (From Scratch)** (Sebastian Raschka, Manning MEAP 2025) *(External book; use as conceptual foundation)*  
  Focus: inference-time compute scaling, why chain-of-thought increases token volume/cost, and why “reasoning” changes the workload shape. This directly feeds Groq dimensioning because even with deterministic execution per shape, **request cost varies with prompt/output tokens**.

### Complementary (Inference compilers + hardware co-design literacy)

These are not Groq-specific, but they prevent “GPU kernel culture” mistakes and make you conversant with compiler engineers.

- **How compilers and hardware co-design enable efficient DNN execution** (Paper): `arxiv.org` *(Assumption to validate; locate canonical PDF)*  
  Focus: the general principles behind compiler-first hardware. Use this to articulate “why Groq moves complexity into the compiler” without vendor slogans.
- **NVIDIA TensorRT** (Docs + samples + DLI course): `developer.nvidia.com` *(Fact once pinned)*  
  Focus: GPU inference optimization patterns (tactic selection, fusion, FP16/INT8). Valuable mainly as a contrast: GPUs often rely on runtime-validated tactics and kernel libraries; Groq relies on whole-graph static scheduling.
- **ONNX + ONNX Runtime (ORT)** (Docs): *(Assumption to validate; official sites vary)*  
  Focus: graph-level optimizations and the “execution provider” model. Useful for mapping “portable graph” → “hardware-specific backend,” which is conceptually similar to Groq’s import/compile pipeline.
- **Microsoft Olive** (Toolchain; hardware-aware optimization atop ORT): `opensource.microsoft.com` *(Assumption to validate)*  
  Focus: automated, hardware-aware optimization composition (quantization + backend selection + graph transforms). This is a useful template for what your Groq artifact pipeline should become (policy-driven optimization + measurement).
- **Apache TVM** (Compiler stack): *(Assumption to validate; official site varies)*  
  Focus: schedule search/autotuning and IR lowering. Contrast: TVM explores many schedules; Groq compiler aims to produce a deterministic schedule for the LPU.

### Complementary (Compression: quantization, pruning, distillation)

Compression matters for Groq because SRAM capacity constraints often force explicit choices: fit vs shard vs distill.

- **Quantization references** (NVIDIA blogs/docs, HF Optimum): `developer.nvidia.com`, `huggingface.co` *(Assumption to validate for specific posts)*  
  Focus: when lowering precision helps throughput/latency and when accuracy regressions demand calibration/QAT. On Groq, treat numeric modes as part of the compiled artifact and validate accuracy per mode.
- **Pruning and distillation references** (e.g., NVIDIA blogs; Hinton distillation paper): *(Assumption to validate)*  
  Focus: structured pruning changes shapes (can change compile placement); distillation can create a smaller model that fits in fewer LPUs/SRAM, changing the entire DC design.

### Why These Resources Improve Day 008 Specifically

- Tier 01 depends on you understanding **deterministic execution and static scheduling** (papers + Andrew Ling blog).
- Tier 02 depends on you extracting **service-time and throughput primitives** from Groq tooling (GroqFlow/compiler docs + profiler).
- Tier 03 depends on you knowing where determinism ends (fabric boundary) and how failures/queueing dominate tails (papers + workshop + interviews).
