# Day 003 – vLLM Capacity, OOM Surface & Real Use-Cases
## Tier 3: Consolidation, Story & Quiz (~45–60 min)

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) and [Tier 2](LOG_tier02.md)  
> **Goal**: Create client-ready playbook + commit everything + quiz for retrieval practice  
> **End State**: Publishable recipes, clean git history, learning reinforcement  
> **Time**: 45–60 min

---

## Tier 3 Tasks

---

### ✅ Task 3.1: Create "Single-GPU vLLM Serving Recipes" Playbook
**Tags**: `[Business]` `[Documentation]`  
**Time**: 20–30 min  
**Win**: Something you can show to a client or reuse as a template

#### 🔧 Lab Instructions

```bash
mkdir -p ~/playbooks

cat > ~/playbooks/vllm_single_gpu_recipes.md << 'EOF'
# vLLM Single-GPU Serving Recipes

*Practical configurations derived from Day 003 capacity testing*

---

## Quick Reference (RTX 2000 Ada 16GB + Qwen2.5-1.5B)

| Workload | GPU Class | Best Concurrency | Expected Throughput | p95 Latency |
|----------|-----------|------------------|---------------------|-------------|
| Chat | 16GB (RTX/T4) | 8–16 | ~620 tok/s | ~2600ms |
| Chat | 40GB (A100) | 32–64 | ~1500–2000 tok/s (est.) | ~1000ms |
| Batch | 16GB (RTX/T4) | 32 | ~568 tok/s | ~5300ms |
| Batch | 40GB (A100) | 64–128 | ~1200–1800 tok/s (est.) | ~2500ms |

---

## Recipe 1: Chat on 16GB GPU (RTX/T4 Class)

**Use Case**: Interactive chat, customer support, low-latency Q&A

**Config**: `configs/vllm/qwen2p5_1p5b_chat_16gb.yaml`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| gpu_memory_utilization | 0.8 | Safety margin |
| max_model_len | 4096 | Typical chat context |
| max_num_seqs | 128 | Concurrent sequences |
| Concurrency | 8–16 | Balances throughput vs p95 |
| max_tokens | 128 | Sweet spot for chat |
| Target p95 | < 3000ms | Good UX for chat |

**Measured Performance** (Day 003):
- conc=16, max_tokens=128 → ~620 tok/s, p95 ~2600ms

---

## Recipe 2: Batch Summarization on 16GB GPU

**Use Case**: Document summarization, offline processing, batch analytics

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Concurrency | 32 | Maximize GPU utilization |
| max_new_tokens | 256 | Good summary length |
| Target p95 | < 6000ms | Batch jobs tolerate latency |

**Measured Performance** (Day 003):
- conc=32, max_tokens=256 → ~568 tok/s, p95 ~5300ms

**⚠️ Avoid**: conc=16 + max_tokens=512 → p95 spikes to ~9400ms (jitter ridge)

---

## Recipe 3: Chat on A100/H100

**Use Case**: High-traffic production, strict latency SLAs

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| gpu_memory_utilization | 0.85 | More headroom available |
| max_num_seqs | 256 | A100 handles more |
| Concurrency | 32–64 | Much higher capacity |
| Target p95 | < 1000ms | Tighter SLA possible |

---

## Common Pitfalls

1. **max_model_len too high** → Wastes KV cache, reduces concurrency
2. **Concurrency 1–2** → Misses continuous batching benefits
3. **No benchmark harness** → Cannot make data-driven decisions
4. **gpu_memory_utilization=0.95** → OOM under traffic spikes

EOF

echo "✅ Playbook created: ~/playbooks/vllm_single_gpu_recipes.md"
```

#### 📁 Artifacts
- `~/playbooks/vllm_single_gpu_recipes.md`

#### 🏆 Success Criteria
- [ ] Playbook created with all recipes
- [ ] Your actual numbers filled in
- [ ] Document is client-presentable

---

### ✅ Task 3.2: Add vLLM Best Practices Reference
**Tags**: `[Documentation]` `[Advanced]`  
**Time**: 15 min  
**Win**: Document latest vLLM tuning knobs for future reference

#### 🔧 Lab Instructions

```bash
cat >> ~/playbooks/vllm_single_gpu_recipes.md << 'EOF'

---

## Latest vLLM Best Practices (2024/2025)

### Chunked Prefill Tuning

`max_num_batched_tokens` controls the prefill chunk size:

| Goal | Recommended Value |
|------|-------------------|
| Better ITL (chat) | `2048` (smaller chunks) |
| Better TTFT (batch) | `8192–16384` (larger chunks) |
| Max throughput (small models) | `> 8192` |

### Preemption Monitoring

Watch for this warning in logs:
```
WARNING: Sequence group X is preempted by PreemptionMode.RECOMPUTE
```

**If you see frequent preemptions:**
1. Increase `gpu_memory_utilization`
2. Decrease `max_num_seqs` or `max_num_batched_tokens`
3. Consider tensor parallelism

### Prefix Caching (Chat Optimization)

Enable for chat workloads with shared system prompts:
```yaml
enable_prefix_caching: true
```

**Benefit**: Subsequent requests with same prefix skip KV recomputation.

### Key Config Knobs Summary

| Parameter | Chat Workload | Batch Workload |
|-----------|---------------|----------------|
| `gpu_memory_utilization` | 0.8 | 0.85–0.9 |
| `max_model_len` | 4096 | Match actual need |
| `max_num_seqs` | 64–128 | 128–256 |
| `max_num_batched_tokens` | 2048 | 8192–16384 |
| `enable_prefix_caching` | true | false |
| `enable_chunked_prefill` | true | true |

### OOM Debugging Checklist

1. Check preemption count in logs
2. Lower `gpu_memory_utilization` to 0.7
3. Reduce `max_model_len` to actual need
4. Reduce `max_num_seqs`
5. Monitor with `nvidia-smi -l 1`

EOF
```

#### 📁 Artifacts
- Updated `~/playbooks/vllm_single_gpu_recipes.md`

---

### ✅ Task 3.3: Commit Everything
**Tags**: `[Git]`  
**Time**: 5–10 min  
**Win**: Clean git history, all artifacts tracked

#### 🔧 Lab Instructions

```bash
# Review what you're committing
git status

# Stage all Day 003 artifacts
git add \
  configs/vllm \
  scripts/benchmarks \
  benchmarks \
  artifacts \
  reports \
  playbooks \
  data

# Commit with descriptive message
git commit -m "Day 003: vLLM single-GPU capacity for chat + batch, RTX vs A100 anchor"

# Push if remote configured
# git push origin main
```

#### 🏆 Success Criteria
- [ ] All files staged and committed
- [ ] Commit message is descriptive

---

### ✅ Task 3.4: Complete the Quiz
**Tags**: `[Learning]`  
**Time**: 10–15 min to review, answer later for retrieval practice  
**Win**: Force retrieval of concepts tomorrow, not just today's muscle memory

The quiz is stored in a separate file: **[day003_quiz.md](day003_quiz.md)**

Answer these questions **without looking at your code first** to reinforce learning.

---

## Tier 3 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **3.1** | Created serving recipes playbook | ⬜ |
| **3.2** | Added vLLM best practices reference | ⬜ |
| **3.3** | Committed everything | ⬜ |
| **3.4** | Reviewed quiz questions | ⬜ |

---

## 🎉 Day 003 Complete!

### What You Achieved

By finishing Tier 1 + Tier 2.1, you're already bootstrapping **1%-level intuition**:

> "This GPU, this model, this workload → here's what's safe, here's what breaks, here's why."

### Full Artifact Tree

```
~/configs/vllm/
├── qwen2p5_1p5b_chat_16gb.yaml
└── serve_qwen2p5_1p5b_chat_16gb.sh

~/scripts/benchmarks/
├── vllm_chat_bench.py
├── vllm_batch_summarize_bench.py
├── run_chat_capacity_grid.sh
└── run_batch_capacity_grid.sh

~/benchmarks/
├── day003_chat_baseline_rtx16gb.json
├── day003_chat_capacity_rtx16gb.csv
├── day003_batch_capacity_rtx16gb.csv
├── day003_chat_capacity_a100_40gb.csv (if done)
└── day003_batch_capacity_a100_40gb.csv (if done)

~/data/
└── day003_docs_sample.txt

~/artifacts/
└── day003_chat_capacity_notes.md

~/playbooks/
└── vllm_single_gpu_recipes.md
```

### Key Takeaways

1. **Chat vs Batch** have different optimal operating points
2. **Capacity is measurable**, not a guess
3. **GPU class matters** – A100 can push 4–8x higher concurrency
4. **OOM has a surface** – you can map it empirically
5. **Configs should be versioned** – YAML > CLI flags
6. **Metrics matter** – TTFT, TPOT, ITL tell different stories
7. **Preemption = OOM warning** – monitor logs for early signals
8. **Chunked prefill** – tune `max_num_batched_tokens` per workload

---

## 🔜 Next Steps

When ready for Day 004, potential directions:
- Longer context testing (8k, 16k, 32k)
- Multi-model serving
- OOM stress testing
- Quantization (AWQ/GPTQ)

---

*Day 003 concluded. You now have measured capacity data for real workloads.* 🚀
