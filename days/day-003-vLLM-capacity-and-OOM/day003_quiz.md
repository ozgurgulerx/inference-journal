# Day 003 – vLLM Serving Quiz

*Answer these without looking at your code first. Use for retrieval practice.*

---

## Concepts & Configuration (Questions 1–10)

**1.** Explain the roles of `max_model_len` and `max_num_seqs` in vLLM. How do they jointly affect VRAM usage and OOM risk?

**2.** In a chat workload on a 16GB GPU, why might `max_model_len=8192` be a bad default, even if you "want flexibility"?

**3.** What does `gpu_memory_utilization` control in vLLM, and why might 0.8 be a reasonable starting value on a small GPU?

**4.** Define TTFT and e2e latency in the context of your `vllm_chat_bench.py` script. Why do we care about p50 vs p95?

**5.** For a latency-sensitive chat API, what is an acceptable trade-off between p95 latency and throughput on a 16GB GPU? How would that differ for batch summarization?

**6.** When increasing concurrency from 1 → 16, what typical patterns did you observe in:
   - GPU utilization
   - throughput tokens/sec
   - p95 latency

**7.** How would you explain continuous batching to a teammate using only your Day 003 experiments (no theory)?

**8.** Why is it valuable to have a dedicated YAML config file (e.g., `qwen2p5_1p5b_chat_16gb.yaml`) instead of passing flags on the CLI each time?

**9.** Suppose a client reports random CUDA OOMs under traffic spikes. Which 3 vLLM config fields would you inspect first, and why?

**10.** Describe a method to empirically find a "safe capacity zone" for a given GPU and workload using your benchmark scripts.

---

## GPU Scaling & Comparison (Questions 11–15)

**11.** On an A100 40GB, what differences did you observe compared to the RTX-class GPU for:
   - max stable concurrency
   - throughput tokens/sec
   - p95 latency

**12.** How would you decide whether a client should upgrade from an RTX-class GPU to A100/H100 based on your Day 003 findings?

**13.** For batch document summarization, why can a higher p95 latency be acceptable, and what metric becomes more important?

**14.** What are the main differences in config between a "chat" recipe and a "batch summarization" recipe in your playbook?

**15.** Explain how prefix caching can impact performance for chat workloads with repetitive system + history prompts.

---

## Practical Scenarios (Questions 16–20)

**16.** If a client insists on 8192-token contexts for chat but only has 16GB GPUs, how would you approach:
   - capacity estimation
   - risk explanation
   - possible mitigations?

**17.** How would you extend your benchmark harness to log approximate cost-per-1M-tokens for different GPU types on RunPod?

**18.** Describe a realistic scenario where:
   - throughput is high,
   - GPU utilization looks good,
   - but user-perceived latency is still bad.
   What might be going on?

**19.** How would you validate that an observed OOM is due to vLLM config rather than a bug in your code or in RunPod's environment?

**20.** Suppose you double `max_new_tokens` for your chat workload but keep concurrency fixed. What impact do you expect on:
   - TTFT
   - e2e latency
   - throughput tokens/sec
   And why?

---

## Synthesis & Communication (Questions 21–25)

**21.** How would you explain the difference between "RTX small chat recipe" and "A100 chat recipe" to a semi-technical product manager in 2–3 sentences?

**22.** What additional metrics or tools (besides your Python bench + `nvidia-smi`) would you introduce in future days to deepen your understanding of vLLM performance?

**23.** If you had to compress Day 003 into a 5-slide client-facing explanation, what 5 key insights would you highlight?

**24.** How does locking in single-GPU capacity knowledge today help you later when moving to:
   - Kubernetes
   - Ray
   - multi-GPU setups?

**25.** After doing Day 003, what are the top 3 questions you still have about vLLM serving that you'd like to answer in future days?

---

## Latest vLLM Features (Questions 26–30)

**26.** What is the difference between TPOT (Time Per Output Token) and ITL (Inter-Token Latency)? When would you use each metric?

**27.** Explain what `PreemptionMode.RECOMPUTE` means in vLLM logs. What causes it, and what are three ways to reduce preemption frequency?

**28.** How does `max_num_batched_tokens` affect the trade-off between TTFT and ITL? What value would you use for:
   - A chat workload prioritizing streaming smoothness
   - A batch workload prioritizing throughput

**29.** What is prefix caching in vLLM, and why is it particularly beneficial for chat workloads with shared system prompts?

**30.** vLLM's chunked prefill feature was a key 2024 improvement. Explain how it helps balance compute-bound (prefill) and memory-bound (decode) operations.

---

## Scoring Guide

| Score | Level |
|-------|-------|
| 25–30 correct | Expert – ready for production consulting |
| 18–24 correct | Proficient – solid foundation |
| 12–17 correct | Developing – review Tier 1–3 materials |
| < 12 correct | Needs work – redo experiments with focus |

---

*Review these questions 24 hours after completing Day 003 for optimal retention.*
