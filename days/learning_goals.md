# 100 Days of Inference Engineering

> **Start Date**: December 1, 2025  
> **End Date**: March 10, 2026  
> **Duration**: 100 days  
> **Effort**: 3-4 hrs/day focused (20-25 hrs/week)  
> **Goal**: Production-ready LLM inference skills

---

## The 100-Day Timeline

| Phase | Focus | Days | Dates | Daily Goal |
|-------|-------|------|-------|------------|
| **Phase 0** | OS & GPU Setup | 1-15 | Dec 1-15 | Get a GPU box production-ready |
| **Phase 1** | vLLM Mastery | 16-35 | Dec 16 → Jan 4 | Serve models, benchmark everything |
| **Phase 2** | Quantization | 36-55 | Jan 5-24 | INT8/INT4, measure quality vs speed |
| **Phase 3** | Optimization | 56-80 | Jan 25 → Feb 18 | Real workloads, case studies |
| **Phase 4** | Ship & Share | 81-100 | Feb 19 → Mar 10 | Publish repos, write case studies |

---

## In This Document

- [Success Outcomes](#success-outcomes-day-100)
- [Daily Routine](#daily-routine)
- [Phase 0: OS & GPU Setup](#phase-0--os--gpu-setup-days-115) (Days 1-15)
- [Phase 1: vLLM Mastery](#phase-1--vllm-mastery-days-1635) (Days 16-35)
- [Phase 2: Quantization](#phase-2--quantization-days-3655) (Days 36-55)
- [Phase 3: Optimization](#phase-3--optimization-days-5680) (Days 56-80)
- [Phase 4: Ship & Share](#phase-4--ship--share-days-81100) (Days 81-100)
- [Advanced Tracks](#advanced-mastery-tracks) (Post-100)

---

## Success Outcomes (Day 100)

By March 10, 2026, you will have:

- [ ] A production-ready GPU node with tuned OS settings
- [ ] Fluency with vLLM serving, configuration, and debugging
- [ ] Hands-on experience with INT8/INT4 quantization (AWQ, GPTQ)
- [ ] 3+ benchmark repos with real numbers
- [ ] 2 case studies: before/after optimization results
- [ ] 1 published blog post or talk
- [ ] A repeatable optimization playbook

---

## Daily Routine

| Time Block | Focus | Output |
|------------|-------|--------|
| **First 30 min** | Review yesterday's results, plan today | Clear goal |
| **Next 2-3 hrs** | **Hands-on hacking** – code, run, measure | Working code |
| **Last 30 min** | Log results, commit code, note blockers | Day log entry |

> **Rule**: Every day produces either **code**, **numbers**, or **a config**. No passive reading days.

---

## Phase 0 – OS & GPU Setup (Days 1-15)

> **Dec 1-15** | Get a GPU box production-ready. Every day = a working script or config.

### Days 1-3: GPU Node Bring-Up
- [ ] Install NVIDIA drivers, CUDA, cuDNN, NCCL, and TensorRT
- [ ] Verify with `nvidia-smi` and `nvcc`
- [ ] Prepare a repeatable bootstrap script

### Days 4-5: CPU & NUMA
- [ ] Set CPU governor to `performance`
- [ ] Map NUMA topology with `numactl --hardware`
- [ ] Pin vLLM process to GPU-local cores
- [ ] Benchmark: measure latency before/after pinning

### Days 6-7: Memory & Huge Pages
- [ ] Configure THP to `madvise`
- [ ] Set `vm.swappiness=10`
- [ ] Reserve 1GB hugepages, verify with `cat /proc/meminfo`

### Days 8-9: Storage
- [ ] Time model load from NVMe vs network
- [ ] Set I/O scheduler to `none` for NVMe
- [ ] Script to pre-warm page cache

### Days 10-11: Networking
- [ ] Enable jumbo frames (MTU 9000)
- [ ] Configure IRQ affinity for NIC
- [ ] Set `net.core.somaxconn=65535`
- [ ] Measure HTTP latency before/after

### Days 12-13: Containers
- [ ] Install NVIDIA Container Toolkit
- [ ] Run vLLM in Docker with GPU access
- [ ] Create a working `docker-compose.yml`

### Days 14-15: Observability
- [ ] Deploy `node_exporter` + DCGM exporter
- [ ] Create Grafana dashboard for GPU metrics
- [ ] **Deliverable**: Screenshot of working dashboard

---

## Phase 1 – vLLM Mastery (Days 16-35)

> **Dec 16 → Jan 4** | Serve models, benchmark everything. Ship a comparison repo.

### Days 16-18: HF vs vLLM Baseline
- [ ] Serve Llama-3-8B with HuggingFace `transformers`
- [ ] Measure: latency, throughput, memory
- [ ] Serve same model with vLLM
- [ ] Measure: latency, throughput, memory
- [ ] **Deliverable**: Comparison table in a GitHub repo

### Days 19-22: vLLM Deep Dive
- [ ] Experiment with `--max-model-len`, `--gpu-memory-utilization`
- [ ] Enable and test prefix caching
- [ ] Test streaming responses
- [ ] Run with different `--dtype` (fp16, bf16)
- [ ] Document which flags affect what

### Days 23-26: Profiling
- [ ] Run `nvidia-smi dmon` during inference
- [ ] Identify compute-bound vs memory-bound regimes
- [ ] Install Nsight Systems, capture a trace
- [ ] Compare kernel patterns: HF vs vLLM
- [ ] **Deliverable**: 1-page profiling notes

### Days 27-30: Load Testing
- [ ] Build a simple load generator (Python + `aiohttp`)
- [ ] Test with 1, 10, 50, 100 concurrent requests
- [ ] Plot: throughput vs concurrency
- [ ] Find the "sweet spot" for your GPU
- [ ] **Deliverable**: Load test script + results chart

### Days 31-35: Multi-Model Serving
- [ ] Serve 2-3 different models (7B, 13B sizes)
- [ ] Test Mistral, Llama, Qwen variants
- [ ] Benchmark each, note differences
- [ ] **Deliverable**: Multi-model benchmark repo

---

## Phase 2 – Quantization (Days 36-55)

> **Jan 5-24** | INT8/INT4 experiments. Measure quality vs speed trade-offs.

### Days 36-40: AWQ Quantization
- [ ] Download AWQ-quantized Llama-3-8B
- [ ] Serve with vLLM, benchmark throughput
- [ ] Compare: FP16 vs AWQ-INT4
- [ ] Measure memory savings
- [ ] **Deliverable**: Quantization comparison table

### Days 41-45: GPTQ & Other Formats
- [ ] Try GPTQ-quantized model
- [ ] Compare AWQ vs GPTQ on same base model
- [ ] Test FP8 if hardware supports
- [ ] Measure perplexity on small eval set
- [ ] **Deliverable**: Format comparison chart

### Days 46-50: FlashAttention
- [ ] Verify FlashAttention is enabled in vLLM
- [ ] Benchmark with/without (if toggleable)
- [ ] Profile attention kernel time
- [ ] Test with long sequences (4K, 8K, 16K tokens)
- [ ] **Deliverable**: FlashAttention impact notes

### Days 51-55: Speculative Decoding
- [ ] Set up speculative decoding with draft model
- [ ] Measure latency improvement
- [ ] Test different draft model sizes
- [ ] Document when it helps vs hurts
- [ ] **Deliverable**: Speculative decoding benchmark

---

## Phase 3 – Optimization (Days 56-80)

> **Jan 25 → Feb 18** | Real workloads, real case studies. Optimize end-to-end.

### Days 56-60: KV Cache Tuning
- [ ] Experiment with `--max-num-seqs`
- [ ] Test different `--block-size` values
- [ ] Measure KV cache memory vs sequence length
- [ ] Find optimal settings for your GPU
- [ ] **Deliverable**: KV cache tuning guide

### Days 61-65: Production Config
- [ ] Create optimized vLLM config for:
  - [ ] Latency-optimized (chat)
  - [ ] Throughput-optimized (batch)
- [ ] Test with realistic traffic patterns
- [ ] Document the trade-offs
- [ ] **Deliverable**: Two production configs with benchmarks

### Days 66-72: Case Study #1
- [ ] Define workload: "Customer support chatbot"
- [ ] Baseline: HF Transformers
- [ ] Optimized: vLLM + quantization + tuning
- [ ] Measure: latency, throughput, cost
- [ ] Calculate cost savings
- [ ] **Deliverable**: Case study document with charts

### Days 73-80: Case Study #2
- [ ] Define workload: "Batch document processing"
- [ ] Test with 1000+ requests
- [ ] Optimize for maximum throughput
- [ ] Compare different parallelism strategies (if multi-GPU)
- [ ] **Deliverable**: Second case study with ROI analysis

---

## Phase 4 – Ship & Share (Days 81-100)

> **Feb 19 → Mar 10** | Package and publish everything. Build credibility.

### Days 81-85: Polish Repos
- [ ] Clean up all benchmark repos
- [ ] Add proper READMEs with results
- [ ] Add reproduction scripts
- [ ] Tag releases
- [ ] **Deliverable**: 3 polished GitHub repos

### Days 86-90: Write Case Studies
- [ ] Format case study #1 as blog post
- [ ] Format case study #2 as blog post
- [ ] Add charts and visuals
- [ ] **Deliverable**: 2 publishable case studies

### Days 91-95: Create Playbook
- [ ] Document your optimization workflow
- [ ] Create checklists for common scenarios
- [ ] Package configs and scripts
- [ ] **Deliverable**: "Inference Optimization Playbook" repo

### Days 96-100: Share & Launch
- [ ] Publish blog post(s)
- [ ] Share on Twitter/LinkedIn/HN
- [ ] Create a simple portfolio page
- [ ] **Deliverable**: Public presence established

---

## Post-100: Next Steps

> After Day 100, continue building on your foundation:

### Deepen Expertise
- [ ] Try TensorRT-LLM for comparison
- [ ] Explore multi-GPU tensor parallelism
- [ ] Test on different hardware (A100, H100, L40S)

### Build Visibility
- [ ] Give a talk at a meetup
- [ ] Write a longer technical article
- [ ] Engage with the vLLM community

### Productize
- [ ] Define an "Inference Health Check" service
- [ ] Create a pricing model
- [ ] Find 1-2 pilot clients

---

## Advanced Mastery Tracks

### Advanced OS, Kubernetes & Ray Mastery (Top 1%)

- [ ] Build custom low-latency kernels for Ubuntu and RHEL, patching `preempt_rt` and mitigation toggles; manage DKMS modules for NVIDIA drivers
- [ ] Use eBPF, `perf` and `ftrace` to instrument syscalls, interrupts, and scheduler behavior; develop custom eBPF programs to monitor GPU/CPU contention, NVMe/NIC latency, and memory allocations
- [ ] Harden systems using SELinux (RHEL) and AppArmor (Ubuntu) while tuning cgroups v2; design systemd slices with fine-grained memory and CPU controllers for inference pods
- [ ] Engineer NUMA and PCIe topologies: design multi-socket servers with optimal placement of GPUs, NVMe and NICs; tweak BIOS settings (NPS, IOMMU) and verify using `lstopo` and `nvidia-smi topo`
- [ ] Develop custom Kubernetes device plugins and scheduling policies to handle MIG slices, allocate hugepages, and enforce NUMA-aware placement; experiment with topology-manager policies like `restricted` or `single-numa-node`
- [ ] Configure autoscaling for GPU nodes using Cluster Autoscaler or Karpenter, enabling bin-packing and pre-warmed pools; manage multi-region clusters via federation tools
- [ ] Master Ray Serve by creating placement groups, custom resource schedulers, and dynamic batching; integrate Ray clusters with Kubernetes using the Ray operator and optimize autoscaling
- [ ] Implement chaos engineering experiments to simulate kernel panics, NIC failures or GPU hangs; ensure inference services recover gracefully through pod disruption budgets, Ray actor restarts and node health checks
- [ ] Build unified observability by combining Prometheus, Grafana and OpenTelemetry to correlate p99 latency spikes with kernel events and pod reschedules

### Distributed & Advanced vLLM Mastery

- [ ] Deep dive into vLLM architecture: read vLLM's engine, scheduler, memory management, and PagedAttention; write a summary of how requests flow through the system
- [ ] Implement and measure continuous batching: send multiple concurrent generation requests to vLLM; compare throughput vs sequential requests; note improvements
- [ ] Explore vLLM API: experiment with sampling parameters, streaming outputs, prefix caching, and guided decoding; build a small multi-user chat server using vLLM
- [ ] Study distributed inference strategies: data, pipeline, tensor, and expert parallelism; practice serving a large model using tensor parallelism across multiple GPUs; evaluate trade-offs
- [ ] Benchmark other inference engines: compare vLLM to Hugging Face TGI, TensorRT-LLM, FasterTransformer, and FlexGen for a chosen model; document throughput, latency, memory usage, and strengths of each
- [ ] Experiment with compiled kernel frameworks: convert a model to TensorRT-LLM or FasterTransformer; benchmark vs vLLM; document results and insights
- [ ] Develop a custom high-performance serving stack: integrate vLLM scheduling with TensorRT kernels and speculative decoding; prototype and measure improvements

---

## Daily Log Template

```markdown
# Day XX - [Date]

## Goal
What I planned to accomplish today.

## Done
- [ ] Task 1
- [ ] Task 2

## Results
Numbers, screenshots, or code snippets.

## Blockers
What slowed me down.

## Tomorrow
What's next.
```

---

<p align="center">
  <a href="README.md">← Back to Days Index</a>
</p>
