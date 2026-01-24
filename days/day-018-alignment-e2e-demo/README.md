# Day 018 – Alignment + Serving End-to-End Demo

## Goals
- **Primary Objective:** Fine-tune (LoRA/SFT) a small model on a use case, serve it, and measure behavior + performance deltas; optionally run a small DPO loop.
- **Focus Theme(s):** Training/alignment to production; serving impact of alignment; behavior evaluation.

## Setup & Context
- **Hardware/Infra:** 1× GPU (24–40GB) for LoRA/SFT; reuse serving node from prior days.
- **Model(s):** Base chat model (7B-ish) + LoRA adapters; optional reward/preference model for DPO/ORPO-lite.
- **Runtime/Engine:** Training via PEFT/TRL; serving via vLLM (and/or TensorRT-LLM if exportable).
- **Dataset/Workload:** Domain-specific mini dataset (support Q&A or policy compliance); ~1–3k examples for SFT; paired preferences (real or synthetic) for small DPO loop.
- **Key Config Knobs:** LoRA rank/alpha, training epochs, max_tokens during eval, alignment loss weights, response length controls.

## Experiments & Measurements
1. **Baseline serving + eval**
   - Method: Serve base model; run eval harness (automatic + 10 manual spot checks).
   - Metrics: p50/p95 latency, tokens/sec, $/1M tokens, quality/behavior scores.
2. **LoRA/SFT training**
   - Method: Train adapters; log training time and GPU usage.
   - Metrics: Training cost/time; checkpoint size.
3. **Serve aligned model**
   - Method: Load adapters in runtime; rerun eval harness.
   - Metrics: Latency/throughput vs baseline; behavior deltas (accuracy/compliance/refusals).
4. **(Optional) DPO/ORPO loop**
   - Method: Build tiny preference set; run 1–2 epochs; serve updated model.
   - Metrics: Behavior shift vs SFT-only; latency/throughput impacts.
5. **Quality & safety checks**
   - Method: Compare outputs on 15–20 prompts; note regressions and response length drift.

## Analysis & Insights
- Summarize behavior improvements vs cost/latency impacts; note token-length changes.
- Highlight serving pitfalls (adapter loading, caching behavior) and how to mitigate.

## Performance & Cost Evaluation
- Present $/1M tokens and p95s before vs after alignment; include training cost.

## Output Quality & Alignment Check
- Track compliance/refusal rates; accuracy on task; safety filter interaction.

## Observability & Reliability Notes
- Log which adapter/version served each request; ensure rollbacks are easy.
- Note any stability issues (hot-reload, adapter swap).

## Next Steps
- Promote best configs into the optimization playbook; feed behavior metrics into Day 019 architecture comparisons.
