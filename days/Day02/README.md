# Day02 - Inference Engineering Systems Map

> A supplemental roadmap for the major systems layers behind modern inference. This folder is the high-level map; each section can later expand into its own deeper notes, experiments, and benchmarks.

## Core Outcome

The goal of this track is simple: explain every latency, throughput, memory, and cost shift from first principles. That means being able to connect model behavior, kernels, compilers, runtimes, topology, networking, and service policy into one coherent picture.

## Section 1 - Transformer Inference Mechanics

This needs to be explicit, not implied. A strong inference engineer should be able to reason cleanly about how work changes between prefill and decode, why KV-cache behavior dominates so many tradeoffs, and how prompt reuse, long context, and architecture choices reshape runtime behavior.

Cover:
- Prefill vs decode
- KV-cache lifecycle
- Prefix reuse and cache hit behavior
- Long-context scaling behavior
- Speculative decoding
- Quantization effects at inference time
- MoE and multimodal serving differences

Engine surface area to study:
- vLLM: PagedAttention, automatic prefix caching, `torch.compile` integration, disaggregated prefill
- SGLang: RadixAttention, continuous batching, chunked and disaggregated prefill, speculative decoding, quantization, multiple parallelism modes
- TensorRT-LLM / Dynamo: disaggregated serving, KV-aware routing, speculative decoding, multi-node deployment

Hands-on goal:
- Benchmark the same model while varying prompt length, output length, cache reuse, and concurrency.
- Explain every latency shift from first principles.

## Section 2 - CUDA and GPU Architecture

This deserves its own full section. The CUDA Best Practices Guide is still the right backbone: memory hierarchy, parallel execution, instruction efficiency, bottleneck identification, and profiling. This is where you build actual intuition for occupancy, coalescing, shared memory, synchronization, launch overhead, and whether a workload is compute-bound or memory-bound.

Cover:
- GPU memory hierarchy
- Warp and thread-block execution
- Occupancy and utilization
- Memory coalescing
- Shared memory usage
- Synchronization costs
- Launch overhead
- Compute-bound vs memory-bound diagnosis
- Profiling workflow with Nsight

Hands-on goal:
- Write and profile a few kernels yourself.
- Skip trivial vector add examples and use fused softmax, RMSNorm, or a simple attention fragment instead.
- Explain the Nsight timeline and kernel counters in plain language.

## Section 3 - Compiler and Kernel Construction Path

There is an important layer between "PyTorch model" and "fast runtime," and it cannot stay fuzzy. You need to understand how inference overhead is removed, how graphs are captured and lowered, and how custom kernels are actually constructed.

Cover:
- `torch.inference_mode()` for removing inference overhead
- `torch.compile()` for graph capture and lowering
- Triton for productive custom kernel work
- CUTLASS and CuTe for tensor-core-oriented kernel engineering

What each tool teaches:
- Triton gives you JIT compilation, autotuning, and a productive GPU programming model.
- CUTLASS 4.4 and CuTe provide reusable CUDA abstractions plus Python and DSL-level control for lower-level kernel design.

Hands-on goal:
- Implement one operator in Triton.
- Compare eager execution vs `torch.compile`.
- Read a CUTLASS or CuTe example and map the high-level tensor program to concrete thread and data layouts.

## Section 4 - Serving Engines in Depth

This should be a real runtime study, not a checklist of user-facing flags. The important step is understanding the architecture and code paths that explain why each engine behaves the way it does.

Recommended framing:
- vLLM as the home base
- SGLang as the contrasting scheduler and cache design
- TensorRT-LLM / Dynamo as the compiled production path

Read closely:
- PagedAttention
- Automatic prefix caching
- Disaggregated prefill
- Metrics and observability surfaces
- NCCL connector paths
- Scheduler design and cache management

Hands-on goal:
- Run the same model across these engines.
- Compare TTFT, ITL, memory footprint, and scaling behavior.
- Explain the differences, not just measure them.

## Section 5 - Distributed Inference and Networking

This is one of the biggest missing layers in most self-study plans. You need real intuition for how topology and communication shape serving performance once a workload leaves the single-GPU world.

Cover:
- NCCL fundamentals
- PCIe vs NVLink vs NVSwitch
- Collective behavior
- Point-to-point transfers
- Tensor parallelism
- Pipeline parallelism
- Expert parallelism
- Data parallelism
- GPUDirect RDMA
- NUMA and topology alignment
- Kubernetes device plugin resource advertisement and allocation

Hands-on goal:
- Run microbenchmarks for all-reduce and all-gather.
- Inspect how topology changes behavior.
- Deploy a 2-4 GPU serving setup and explain whether the bottleneck is compute, memory, or communication.

## Section 6 - Deployment Substrate: Containers and Kubernetes GPU Stack

Inference engineering does not stop when the server starts on one machine. You need enough operational depth to package, schedule, and observe GPU workloads on a real deployment substrate.

Cover:
- GPU-enabled containers
- NVIDIA Container Toolkit
- GPU Operator
- Device plugin behavior
- Container runtime integration
- Node labeling
- DCGM monitoring
- Kubernetes GPU scheduling and health visibility

Hands-on goal:
- Package an engine in Docker.
- Run it with GPU acceleration.
- Deploy it into a small Kubernetes setup with proper GPU scheduling and cluster health visibility.

## Section 7 - Benchmarking and Observability

This is mandatory, not optional. Without disciplined measurement, all runtime opinions become cargo cults. Every experiment needs a stable harness, visible metrics, and a short diagnosis that ties symptoms back to first principles.

Cover:
- AIPerf for TTFT, ITL, TPS, and RPS against OpenAI-style APIs
- DCGM Exporter for Prometheus-visible GPU metrics
- MLPerf for benchmark discipline and latency or accuracy constraints
- `perf stat` and host-side counters for CPU bottleneck analysis
- Trace collection and experiment hygiene

Hands-on goal:
- Build a repeatable benchmark harness.
- Build a GPU dashboard.
- Build a trace collection workflow.
- Make every experiment produce latency percentiles, throughput, GPU utilization, memory behavior, and a short written diagnosis.

## Section 8 - Quantization and Compression

This area needs much more than a surface-level "use INT4 when memory is tight" understanding. Modern inference engineering requires operational fluency with precision formats, layout tradeoffs, and how quantization interacts with compiler and engine support.

Cover:
- Weight-only quantization
- Dynamic quantization
- Dtype and layout tradeoffs
- Engine support differences
- Compiler interactions
- Quality vs latency vs memory tradeoffs

Current toolchain to study:
- `torchao` inference workflows for dynamic and weight-only quantization
- vLLM quantization surfaces
- SGLang quantization surfaces

Hands-on goal:
- Run a real bakeoff across BF16 vs FP8 vs INT4, AWQ, and GPTQ where supported.
- Measure quality, memory, TTFT, ITL, and tokens per second.

## Section 9 - Host-Side Systems and SRE Economics

This is where inference stops being "ML infra" and becomes a production system. The hard problems are often not inside the model alone, but in the request path around it: batching, queueing, routing, retries, cancellation, fragmentation, warmup, and SLO control.

Cover:
- Admission control
- Batch shaping
- Queueing behavior
- Warmup strategy
- Memory fragmentation
- Routing
- Retries and cancellation
- p95 and p99 control
- Cost per token
- SLA-aware planning

Relevant runtime cues:
- TensorRT-LLM / Dynamo request cancellation and SLA-oriented planning surfaces
- AIPerf and MLPerf for disciplined service-level metrics

Hands-on goal:
- Define an SLO for a serving service.
- Tune the system to hit that SLO under a realistic concurrency distribution.

## Expected Artifacts

Each deep dive that grows out of this map should eventually leave behind:
- A benchmark or profiling harness
- A short write-up with diagnosis and tradeoffs
- A reproducible config or deployment recipe
- A clear statement of what actually moved latency, throughput, memory, or cost
