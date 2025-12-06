### Technical Focus (Anchored in Today’s Inference Frontier)

Over these 100 days, “inference engineering” is explicitly tied to the current state-of-the-art themes you mapped in your research:

- **Long Context & Memory Management**  
  - Practice: vLLM PagedAttention, FlashAttention, long-context configs; measuring latency/memory vs context length; understanding when to prefer RAG vs brute-force context.  
  - Goal: be able to design and defend a long-context strategy (pure context, RAG, or hybrid) for an enterprise workload.

- **Continuous Batching & Smart Scheduling**  
  - Practice: vLLM continuous batching, concurrency grids, TTFT/E2E under load; simple QoS/backpressure gateway; basic multi-tenant policies.  
  - Goal: choose batch/concurrency configs that maximize tokens/sec per dollar without blowing p95/p99.

- **KV Cache Design, Partitioning & Reuse**  
  - Practice: play with max_num_seqs, gpu_memory_utilization, prefix caching; observe fragmentation vs paging; reason about eviction/isolation.  
  - Goal: avoid OOMs and maximize concurrent sessions per GPU, and be able to explain how KV/memory behavior interacts with product requirements.

- **Quantization & Mixed Precision**  
  - Practice: AWQ/GPTQ 4–8 bit, weight-only vs full; compare BF16 vs quantized on real tasks; integrate with TensorRT-LLM or other compilers.  
  - Goal: confidently propose “here’s how we cut your inference bill 2–3× with quantization and what quality trade-offs to expect.”

- **High-Efficiency & Alternative Architectures (Transformers, MoE, SSM/Hybrid)**  
  - Practice: run at least one long-context or GQA/MQA model and one state-space/hybrid/MoE model through your harness; compare latency/memory vs vanilla LLM.  
  - Goal: know when an architecture choice (e.g. SSM for million-token logs, MoE for parameter-efficient quality) is the right tool and what it does to the serving stack.

- **Decoding Acceleration (Speculative/Parallel)**  
  - Practice: speculative decoding in vLLM; measure TTFT/E2E improvements; understand draft model requirements and failure cases.  
  - Goal: be able to say “for this stack, speculative decoding is worth ~2× and here’s how we’d implement it safely.”

- **Heterogeneous & Distributed Serving (Multi-GPU/Multi-Node)**  
  - Practice: at least one small tensor/pipeline-parallel setup (single-node), plus basic understanding of multi-node trade-offs; tie it back to your capacity & cost modeling.  
  - Goal: know when you *must* go multi-GPU/node, what parallelism strategy to pick, and what it will do to latency and reliability.

- **Multi-Tenancy, Isolation & Fairness**  
  - Practice: simple multi-tenant experiments (chat vs batch on same GPU); naive vs policy-based scheduling; basic per-tenant metrics.  
  - Goal: design a simple but sane QoS story for “we host N customers and don’t want one to ruin everyone’s p99.”

- **Observability & Cost Analytics**  
  - Practice: logging token-level metrics from your harness/runtimes; mapping GPU time, KV usage, and latency to $/token and per-tenant cost; basic dashboards.  
  - Goal: answer “who is costing us what and why is this request slow?” with data, not guesses.

- **Reliability & Safety in Production Inference**  
  - Practice: deliberately break configs to see OOM/timeouts; think through rollback/HA scenarios; integrate at least one basic guardrail/safety check in a pipeline.  
  - Goal: be able to outline a dependable serving story for a customer: availability, degradation modes, and basic safety/guardrail integration.

- **Training, Alignment & Governance for Institutions**  
  - Practice: build at least one practical SFT/LoRA pipeline that you can deploy behind vLLM/TRT-LLM; implement one concrete institutional alignment loop (DPO/GRPO or similar) from logs/preferences back into serving; add simple guardrails and eval hooks (policy checks, basic safety tests) and observe their latency/throughput impact.  
  - Goal: explain to an enterprise stakeholder when to use SFT vs LoRA vs DPO/GRPO, how alignment shifts behavior **and** serving capacity/cost, and how to integrate minimal governance without breaking SLOs.

- **Market & Service Positioning**  
  - Practice: map your benchmarks, case study, and playbook to common consultancy offerings (Inference Health Check, Optimization Sprint, Alignment Sprint) and note where your approach (e.g. quant-under-concurrency + institutional alignment) goes deeper than typical vendor decks.  
  - Goal: speak credibly about where your 100-day stack fits in the current consulting landscape and how it can be packaged as concrete services for clients.

All of this is layered explicitly over the **hardware → kernel/compiler → runtime/engine → model/architecture → workload/product → business** stack you mapped in the research, so each hands-on day connects back to a real bottleneck and a real business lever.
