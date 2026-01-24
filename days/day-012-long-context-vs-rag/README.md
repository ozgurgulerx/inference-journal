# Day 012 – Long-Context vs RAG (16k–32k) Showdown

## Goals
- **Primary Objective:** Measure when to prefer long-context models vs retrieval for the same task.
- **Focus Theme(s):** Long context & memory management; RAG vs context trade-offs; cost/quality awareness.

## Setup & Context
- **Hardware/Infra:** 1× GPU with ≥40GB VRAM if possible; otherwise note max context achievable; fast NVMe for embeddings/vector store.
- **Model(s):** Long-context variant (e.g., Llama-2-Long/32k or similar) + a 4k–8k baseline model for RAG; same instruction tuning level.
- **Runtime/Engine:** vLLM (long-context config) and chosen RAG stack (e.g., FAISS/Chroma + 4k model via vLLM).
- **Dataset/Workload:** A long document or multi-chapter text; tasks: summarization + multi-hop Q&A; contexts at 4k, 16k, 32k tokens.
- **Key Config Knobs:** `max_model_len`, `max_num_seqs`, KV block size, chunk size/overlap for RAG, top-k/top-p retrieval settings.

## Experiments & Measurements
1. **Direct long-context inference**
   - Method: Run 4k/16k/32k contexts through the long-context model.
   - Metrics: p50/p95 TTFT/E2E, tokens/sec, VRAM, KV growth, quality notes.
2. **RAG pipeline with 4k model**
   - Method: Chunk the same doc into a vector store; answer queries via retrieval + 4k model.
   - Metrics: Retrieval latency, generation latency, total E2E, tokens/sec, VRAM.
3. **Hybrid strategy check**
   - Method: Use short context for base + attach top retrieved passages; optional windowed attention if supported.
   - Metrics: Same as above; note any caching wins.
4. **Quality & cost table**
   - Method: Judge correctness on 10–15 queries; optionally auto-metric (ROUGE/BLEU) if scriptable.
   - Metrics: Accuracy/consistency, $/1M tokens, $ per answered question for each strategy.

## Analysis & Insights
- Identify context length where long-context latency explodes vs where RAG dominates.
- Note memory pressure and KV eviction risk; highlight when prefix caching helps/hurts.

## Performance & Cost Evaluation
- Convert throughput to $/1M tokens per strategy; include vector DB cost if relevant.
- Surface GPU vs CPU vs I/O bottlenecks; estimate cost per question end-to-end.

## Output Quality & Alignment Check
- Track hallucinations/omissions per strategy; note any refusal/guardrail impacts.
- Capture qualitative degradation beyond certain context lengths.

## Observability & Reliability Notes
- Log per-request latency split (retrieval vs generation) and KV cache usage.
- Note failure cases (OOM, slow retrieval, index warmup) and mitigations.

## Next Steps
- Decide default strategy for long documents in the playbook.
- Feed results into speculative decoding (Day 013) and multi-tenant scenarios (Day 015).
