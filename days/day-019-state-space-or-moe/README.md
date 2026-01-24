# Day 019 – State-Space / MoE Model Hands-On

## Goals
- **Primary Objective:** Run at least one non-transformer or MoE model through the same harness and compare serving behavior vs transformer baselines.
- **Focus Theme(s):** High-efficiency architectures; architecture-driven latency/memory shifts; applicability to workloads.

## Setup & Context
- **Hardware/Infra:** 1× GPU (24–40GB). Note if MoE requires more VRAM; consider smaller expert counts to fit.
- **Model(s):** Option A: Small MoE (2–4 experts) such as Mixtral/MoE-Llama variant. Option B: State-space model (e.g., Mamba/Hyena-style) if available with inference weights. Include a transformer baseline of similar quality.
- **Runtime/Engine:** vLLM or HF Transformers for models supported; note any runtime limitations (e.g., MoE routing not supported in some engines).
- **Dataset/Workload:** Same chat + batch prompts as prior days; include a longer-sequence workload to test SSM efficiency.
- **Key Config Knobs:** Expert parallelism/top-k experts, cache handling (if any), precision (BF16 vs FP16 vs INT8), batch/concurrency settings.

## Experiments & Measurements
1. **Baseline transformer run**
   - Method: Run the baseline model with known configs; capture metrics.
   - Metrics: p50/p95 latency, tokens/sec, VRAM, CPU.
2. **MoE run**
   - Method: Serve MoE model; test expert parallel/top-k settings; note engine support limitations.
   - Metrics: Latency, tokens/sec, VRAM per expert, load balance, routing success.
3. **State-space/SSM run (if available)**
   - Method: Serve SSM model; test long-sequence prompts.
   - Metrics: Latency vs sequence length, VRAM scaling, throughput vs baseline.
4. **Quality/behavior check**
   - Method: Compare outputs on 10–15 prompts; note quality and style differences.
5. **Cost & efficiency comparison**
   - Method: Use Day 016 calculator to compute $/1M tokens for each model.
   - Metrics: Cost vs latency vs quality table.

## Analysis & Insights
- When MoE/SSM is advantageous (throughput per quality, long sequences).
- Serving challenges (routing, caching, engine compatibility) and mitigations.

## Performance & Cost Evaluation
- Present side-by-side $/1M tokens and latency numbers; call out VRAM pressure and scaling behavior.

## Output Quality & Alignment Check
- Note any regressions/improvements in factuality or style; watch for instability in MoE routing.

## Observability & Reliability Notes
- Capture routing/attention stats if exposed; log expert selection distribution.
- Note failure cases (unsupported ops, load failures) and workarounds.

## Next Steps
- Feed architecture learnings into the playbook and into future alignment/optimization choices.
