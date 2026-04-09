# days02 - Full-Stack Inference Systems Map

> Working map for a top-1% inference engineering track inside `days02`.

## Guiding Idea

The earlier version was too serving-engine centric. The canonical stack here runs from GPU execution physics to compiler and kernel construction, then into serving engines, distributed communication, deployment substrate, benchmarking and observability, and finally host-side systems plus SRE economics.

Use the Baseten book as the skeleton map, but do not research every section with the same depth:

- Go deep where the tradeoffs are non-obvious and still evolving.
- Stay at survey level where the real job is orientation, not invention.
- Attach one lab and one artifact to each section so the work stays operational.

## Section Map

| Folder | Focus | Depth |
| --- | --- | --- |
| `01-prereqs-evals-economics` | Workloads, SLOs, evals, unit economics, model fit | Medium |
| `02-transformer-inference-mechanics` | Prefill vs decode, KV cache, long context, bottleneck math | Deep |
| `03-cuda-gpu-architecture` | CUDA execution physics, memory hierarchy, topology orientation | Deep |
| `04-compiler-kernel-construction-path` | `torch.inference_mode`, `torch.compile`, Triton, CUTLASS, export paths | Deep |
| `05-serving-engines-in-depth` | vLLM, SGLang, TensorRT-LLM, scheduler and cache behavior | Deep |
| `06-optimization-techniques` | Quantization, speculation, caching, disaggregation, parallelism | Deep |
| `07-distributed-inference-networking` | NCCL, collectives, topology, multi-GPU and multi-node scaling | Deep |
| `08-deployment-substrate-kubernetes-gpu-stack` | Containers, GPU operator, device plugins, Kubernetes scheduling | Deep |
| `09-benchmarking-observability` | AIPerf, DCGM, traces, dashboards, benchmark discipline | Deep |
| `10-host-side-systems-sre-economics` | Queueing, routing, retries, fragmentation, SLOs, cost models | Deep |
| `11-modalities-survey` | Embeddings, VLMs, ASR, TTS, image, video | Medium |
| `12-capstones-artifacts` | Public artifacts, reusable harnesses, polished outputs | Build |

## Light or Reference Topics

Keep these as notes inside the main sections rather than promoting them to top-level folders yet:

- Local inference
- Vendor landscape and procurement details
- Niche modalities until later

## Macro Milestones

### M1 - Workload Thinker

You can define SLOs, evals, cost targets, and model fit.

### M2 - Single-Node Diagnostician

You can profile a model on one GPU or one node and identify the real bottleneck.

### M3 - Engine Operator

You can tune vLLM, SGLang, and TensorRT-LLM and explain why they differ.

### M4 - Distributed Inference Engineer

You can scale across GPUs and nodes and reason about topology, cache, and scheduler behavior.

### M5 - Production Inference Engineer

You can run the service with autoscaling, observability, reliability, and controlled deploys.

### M6 - Top-1%-Trajectory Builder

You have public artifacts that show original measurement, not just summaries.

## Working Conventions

- `00-daily/` holds dated logs and short progress notes.
- `_templates/` holds reusable note, experiment, and summary formats.
- Each numbered section keeps durable concepts in `notes/` and runnable work in `experiments/`.
- `12-capstones-artifacts/` is where polished, shareable outputs land.

## Expected Artifacts

Each deep dive that grows out of this map should eventually leave behind:
- A benchmark or profiling harness
- A short write-up with diagnosis and tradeoffs
- A reproducible config or deployment recipe
- A clear statement of what actually moved latency, throughput, memory, or cost
