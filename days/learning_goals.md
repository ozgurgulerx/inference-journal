# LLM Inference Mastery Roadmap

> **Start Date**: December 1, 2025  
> **Target Duration**: 12–18 Months  
> **Study Time**: 16 hrs/week (12 hrs effective)  
> **Goal**: Become a recognized LLM inference optimization specialist

---

## Estimated Timeline (Hands-On Focus)

| Phase | Focus | Dates | Duration | Approach |
|-------|-------|-------|----------|----------|
| **Phase 0** | OS Foundations | Dec 1 → Jan 5, 2026 | 5 weeks | 90% hands-on |
| **Phase 1** | vLLM Hacking | Jan 6 → Feb 9, 2026 | 5 weeks | Build & benchmark |
| **Phase 2** | Quantization Experiments | Feb 10 → Mar 23, 2026 | 6 weeks | Code & profile |
| **Phase 3** | Optimization Sprints | Mar 24 → May 18, 2026 | 8 weeks | Real workloads |
| **Phase 4** | Ship & Productize | May 19 → Jun 29, 2026 | 6 weeks | Case studies |
| **Recognition** | Visibility & Clients | Jun 30, 2026 → Dec 2026 | 6 months | Talks + repos |

---

## In This Document

- [Success Outcomes](#0-outcomes--what-success-looks-like)
- [Meta-Routine](#meta-routine-every-week)
- [Phase 0: OS Foundations](#phase-0--os-foundations-dec-1--jan-5) (Dec 1 → Jan 5)
- [Phase 1: vLLM Hacking](#phase-1-jan-6--feb-9--vllm-hacking--benchmarks) (Jan 6 → Feb 9)
- [Phase 2: Quantization](#phase-2-feb-10--mar-23--quantization--kernel-experiments) (Feb 10 → Mar 23)
- [Phase 3: Optimization](#phase-3-mar-24--may-18--optimization-sprints) (Mar 24 → May 18)
- [Phase 4: Ship & Productize](#phase-4-may-19--jun-29--ship--productize) (May 19 → Jun 29)
- [Recognition Phase](#recognition-phase-jun-30--dec-2026--visibility--clients) (Jun 30 → Dec 2026)
- [Advanced Mastery Tracks](#advanced-mastery-tracks)

---

## 0. Outcomes – What "Success" Looks Like

- [ ] I can profile, optimize, and scale LLM inference on GPUs (H100/A100/L40S or similar)
- [ ] I am fluent with vLLM, TensorRT-LLM, and Triton Inference Server
- [ ] I understand and can apply quantization, PagedAttention, KV cache optimization, speculative decoding
- [ ] I have used at least one compiler stack (TVM / TorchInductor+Triton / IREE) on real models
- [ ] I have 3–5 public benchmarks (GitHub + short writeups)
- [ ] I have 2–3 end-to-end optimization case studies (before/after)
- [ ] I have 1–2 talks/posts explaining my findings
- [ ] I have a clear productized inference optimization service
- [ ] I have 1–3 pilot customers or strong pseudo-client case studies

---

## Meta-Routine (Every Week)

| Day | Focus | Duration |
|-----|-------|----------|
| Mon | Skim docs/code (just enough to unblock) | 1 hr |
| Tue–Thu | **Hands-on hacking** (code, experiments, profiling) | 2 hrs/day |
| Fri | Benchmarking & capture results | 1.5 hrs |
| Weekend | **Ship one artifact** (working code, benchmark, script) | 4–5 hrs |

> **Philosophy**: Learn by doing. Read docs only when stuck. Every session should produce runnable code or measurable results.

---

## Phase 0 – OS Foundations (Dec 1 → Jan 5)

> **Approach**: SSH into a GPU box and configure it. Learn by breaking things. Every task = a working script or config.

### OS-01: GPU Node Bring-Up
- [ ] Install NVIDIA drivers, CUDA, cuDNN, NCCL, and TensorRT
- [ ] Verify with `nvidia-smi` and `nvcc`
- [ ] Prepare a repeatable bootstrap script

### OS-02: CPU & NUMA for LLM Inference
- [ ] Set the CPU governor to `performance`
- [ ] Study NUMA topology
- [ ] Pin processes to GPU-local cores
- [ ] Benchmark throughput and latency improvements

### OS-03: Memory Tuning & Huge Pages
- [ ] Configure Transparent Huge Pages (prefer `madvise`)
- [ ] Disable swap via low swappiness
- [ ] Learn when to reserve 2 MB or 1 GB hugepages

### OS-04: Storage & Model Load Performance
- [ ] Benchmark loading models from NVMe vs network storage
- [ ] Tune the I/O scheduler
- [ ] Understand page-cache effects

### OS-05: Networking for LLM Services
- [ ] Tune NIC MTU (e.g., jumbo frames)
- [ ] Enable RSS and IRQ affinity
- [ ] Set key sysctls (`somaxconn`, socket buffers)
- [ ] Observe impact on latency

### OS-06: Containers & GPU Exposure
- [ ] Master the NVIDIA Container Toolkit
- [ ] Understand host driver vs container runtime
- [ ] Use the GPU Operator in Kubernetes to expose GPUs

### OS-07: Observability for Inference Nodes
- [ ] Set up `node_exporter` and DCGM exporter into Prometheus/Grafana
- [ ] Learn to interpret CPU, GPU, storage, and network metrics

---

## Phase 1 (Jan 6 → Feb 9) – vLLM Hacking & Benchmarks

### 1. GPU Basics (Learn While Doing)
- [ ] Run `nvidia-smi dmon` while serving – observe SM utilization, memory bandwidth
- [ ] Vary batch size and watch GPU metrics change
- [ ] Note: when is GPU compute-bound vs memory-bound? (don't read – measure)

### 2. Inference Runtimes – HF vs vLLM (Baseline)
- [ ] Set up a GPU environment (local or cloud)
- [ ] Run a 7B model with naive Hugging Face Transformers (no vLLM)
- [ ] Measure for HF:
  - [ ] Latency per token (single request)
  - [ ] Throughput (tokens/sec) for a small batch
  - [ ] Peak GPU memory usage
- [ ] Install and run vLLM with the same 7B model
- [ ] Measure for vLLM:
  - [ ] Latency per token
  - [ ] Throughput (tokens/sec)
  - [ ] Peak GPU memory usage
- [ ] Create a small comparison table: HF vs vLLM
- [ ] Create a minimal GitHub repo: `transformers_vs_vllm_baseline`

### 3. Basic Profiling
- [ ] Use `nvidia-smi` / `nvtop` to observe GPU utilization while serving
- [ ] Install and run Nsight Systems (or similar profiler)
- [ ] Capture a trace for:
  - [ ] HF-only server run
  - [ ] vLLM server run
- [ ] Note: number of kernels, idle time, kernel launch patterns
- [ ] Write a 1-page note: "What Nsight shows about naive vs vLLM serving"

### 4. Scaling Basics – Load & Parameters
- [ ] Implement or use a simple load generator (Python script or tool)
- [ ] Run with varying:
  - [ ] batch sizes
  - [ ] max_new_tokens
  - [ ] concurrent requests
- [ ] Record or plot: tokens/sec vs batch size
- [ ] Identify "sweet spots" (good throughput, acceptable latency)
- [ ] Write a short summary: "How batch size & sequence length affect my LLM throughput"

---

## Phase 2 (Feb 10 → Mar 23) – Quantization & Kernel Experiments

### 5. Quantization – Just Do It
- [ ] Skip theory – download an AWQ-quantized model and serve it
- [ ] Compare FP16 vs INT8 vs INT4 on same model – measure, don't read
- [ ] Quantize a 7B model to 8-bit (e.g., AWQ or similar)
- [ ] (Optional) Quantize the same model to 4-bit
- [ ] For each version (FP16 / 8-bit / 4-bit):
  - [ ] Measure GPU memory usage
  - [ ] Measure tokens/sec and latency
  - [ ] Evaluate a simple accuracy metric (perplexity or small eval set)
- [ ] Document trade-offs: size vs speed vs quality
- [ ] Create or update a repo: `7B_quantization_experiments`

### 6. Fused Kernels & FlashAttention
- [ ] Read how vanilla attention works vs FlashAttention
- [ ] Use a model/runtime with FlashAttention support
- [ ] Benchmark with FlashAttention enabled vs disabled (if possible)
- [ ] Profile both runs and compare:
  - [ ] Kernel time distribution
  - [ ] Overall latency
- [ ] Write a short doc: "What FlashAttention changes in transformer performance"
- [ ] (Optional advanced) Implement a tiny Triton kernel (e.g., matmul or layernorm)
- [ ] Verify it runs and compare basic performance vs naive PyTorch

### 7. One Compiler Stack (TVM / TorchInductor / IREE)
- [ ] Choose a compiler stack (TVM, TorchInductor+Triton, or IREE)
- [ ] Run a small model or a single transformer block through it
- [ ] Get it to execute end-to-end on GPU
- [ ] Measure execution time vs naive PyTorch
- [ ] Note key issues (build, op coverage, debugging pain points)
- [ ] Write a short note: "First impressions using [chosen compiler] on a transformer block"

---

## Phase 3 (Mar 24 → May 18) – Optimization Sprints

### 8. High-Throughput Inference & Load Behavior
- [ ] Build or use a load-testing harness for your LLM API
- [ ] Simulate different client loads (e.g., 1, 10, 100, 500 concurrent clients)
- [ ] For each load, record:
  - [ ] Throughput (tokens/sec)
  - [ ] p50 / p95 latency
  - [ ] GPU utilization
- [ ] Tune runtime config:
  - [ ] max batch size
  - [ ] max concurrent requests
- [ ] Identify configs that give high throughput and acceptable p95 latency
- [ ] Write a mini-report: "How vLLM behaves under increasing load on my hardware"

### 9. Advanced Techniques – Speculative Decoding & KV Cache
- [ ] Study speculative decoding (draft + target model flow)
- [ ] Try at least one speculative decoding implementation if accessible
- [ ] Compare latency with and without speculative decoding
- [ ] Study KV cache behavior:
  - [ ] How KV cache scales with sequence length
  - [ ] Impact on memory and throughput
- [ ] Experiment with:
  - [ ] Different max sequence lengths
  - [ ] Different KV cache settings (if configurable)
- [ ] Document impact on memory usage and latency

### 10. Full Optimization Case Study
- [ ] Define a realistic workload (e.g., "512-token customer support chat responses")
- [ ] For that workload, run:
  - [ ] Baseline 1: Naive HF Transformers server
  - [ ] Baseline 2: vLLM default configuration
  - [ ] Optimization 1: vLLM + tuned batching + tuned concurrency
  - [ ] Optimization 2: vLLM + quantized model
  - [ ] (Optional) Optimization 3: add FlashAttention, speculative decoding, or other tricks
- [ ] For each configuration:
  - [ ] Measure tokens/sec
  - [ ] Measure p95 latency
  - [ ] Measure GPU memory
  - [ ] Estimate approximate cost per million tokens
- [ ] Summarize results in a comparison table
- [ ] Create a case study document: "Case Study: Reducing LLM inference cost and latency"

---

## Phase 4 (May 19 → Jun 29) – Ship & Productize

### 11. Define Productized Services

**Inference Health Check (2–3 weeks)**:
- [ ] Define inputs from client (current setup, hardware, models, traffic)
- [ ] Define benchmark & profiling plan
- [ ] Define deliverables (before/after benchmarks, config recommendations, report)

**Optimization Sprint (4–6 weeks)**:
- [ ] Implementation of quantization, runtime swap (e.g., to vLLM/TRT-LLM)
- [ ] Batching & KV cache tuning
- [ ] Monitoring/alerting setup (basic)
- [ ] Final report + handover session

- [ ] Write 1-page descriptions for each service

### 12. Credibility Assets
- [ ] Create a simple one-page website or doc:
  - [ ] Who you are
  - [ ] What you offer (two productized services)
  - [ ] 2–3 anonymized case-study summaries
- [ ] Prepare an 8–12 slide deck:
  - [ ] Problem: LLM inference is expensive and slow
  - [ ] Your approach: profiling → tuning → quantization → runtime choice
  - [ ] Example results: your case studies
  - [ ] Simple pricing/engagement model
- [ ] Write 2–3 high-signal posts or articles:
  - [ ] "What most teams get wrong about LLM inference"
  - [ ] "How vLLM + quantization can cut GPU costs significantly"
  - [ ] "Lessons learned from profiling real LLM workloads"

---

## Recognition Phase (Jun 30 → Dec 2026) – Visibility & Clients

### 13. Choose a Niche to Dominate
- [ ] Choose a niche focus (example options):
  - [ ] H100 / L40S large-scale inference
  - [ ] Ultra-low-cost inference for startups
  - [ ] On-prem / air-gapped inference (banks, gov)
  - [ ] Latency-critical chat and agents
- [ ] Write down a clear positioning statement:
  - [ ] "I help X type of customers achieve Y outcome by Z approach"

### 14. Deeper Benchmarks & OSS Tools
- [ ] Design 1–2 serious benchmark suites:
  - [ ] Different models (e.g. 7B, 13B, 70B)
  - [ ] Different hardware types
  - [ ] Different runtimes (vLLM, TRT-LLM, etc.)
- [ ] Publish at least one substantial benchmark report
- [ ] Open-source a small tool:
  - [ ] Simple CLI for running standardized benchmarks, or
  - [ ] Config generator for vLLM/TRT-LLM clusters, or
  - [ ] Visualization/dashboard templates

### 15. Teaching & Visibility
- [ ] Give at least one talk:
  - [ ] Meetup, online webinar, or internal tech talk
  - [ ] Turn one of your case studies into a conference-style talk
- [ ] Publish at least one longer-form article:
  - [ ] "Modern LLM Inference Stack: 2026 Edition" style deep dive

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

## Feedback / Reflection Loop (Continuous)

- [ ] **Every month**: Review what you've shipped (repos, notes, benchmarks)
- [ ] **Every month**: Update a single "Master Log" document with:
  - [ ] What you learned technically
  - [ ] What you shipped
  - [ ] Where you felt bottlenecked
- [ ] **Every quarter**: Update your positioning and offers based on what you now know
- [ ] **Every quarter**: Identify 1–2 potential clients or partners to approach

---

## Weekly Deliverables Checklist

Every week should produce **at least one** of:
- [ ] A working script or config
- [ ] A benchmark result with numbers
- [ ] A GitHub commit with code
- [ ] A short writeup (< 1 page)

**No week without shipping.**

---

<p align="center">
  <a href="README.md">← Back to Days Index</a>
</p>
