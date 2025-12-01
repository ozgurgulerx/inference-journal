# Summary

[Introduction](README.md)

---

# Part I: Foundations

- [Introduction to Inference Engineering](chapters/01-introduction.md)
  - [What is Inference Engineering?](chapters/01-introduction.md#what-is-inference-engineering)
  - [The Inference Stack](chapters/01-introduction.md#the-inference-stack)
  - [Key Performance Levers](chapters/01-introduction.md#key-performance-levers)

- [OS Essentials](chapters/02-os-essentials.md)
  - [Supported Operating Systems](chapters/02-os-essentials.md#1-supported-operating-systems)
  - [Kernel and System Optimizations](chapters/02-os-essentials.md#2-kernel-and-system-level-optimizations)
  - [GPU Driver and Toolkit Setup](chapters/02-os-essentials.md#3-gpu-driver-and-toolkit-setup)
  - [TPU Setup](chapters/02-os-essentials.md#4-tpu-setup)
  - [LLM Serving Frameworks](chapters/02-os-essentials.md#5-llm-serving-frameworks)
  - [Containerization Best Practices](chapters/02-os-essentials.md#6-containerization-best-practices)
  - [Kubernetes Support](chapters/02-os-essentials.md#7-kubernetes-support)
  - [Ray for Distributed Inference](chapters/02-os-essentials.md#8-ray-for-distributed-inference)

---

# Part II: vLLM Deep Dive

- [vLLM Architecture & Internals](chapters/03-vllm-architecture.md)
  - [Paged Attention](chapters/03-vllm-architecture.md#1-paged-attention)
  - [Continuous Batching](chapters/03-vllm-architecture.md#2-continuous-batching)
  - [Request Scheduling](chapters/03-vllm-architecture.md#3-request-scheduling)
  - [Offline Batch Inferencing](chapters/03-vllm-architecture.md#4-offline-batch-inferencing)
  - [Optimized Compute Path](chapters/03-vllm-architecture.md#5-optimized-compute-path)
  - [Prefix Caching](chapters/03-vllm-architecture.md#6-prefix-caching)
  - [Quantization Support](chapters/03-vllm-architecture.md#7-quantization-support)
  - [API and Engine Separation](chapters/03-vllm-architecture.md#8-separation-of-api-and-engine)
  - [Hardware-Aware Deployment](chapters/03-vllm-architecture.md#hardware-aware-deployment)

- [Setup & Basic Usage](chapters/04-setup-basic-usage.md)
  - [Installation](chapters/04-setup-basic-usage.md#installation)
  - [Serving on Single GPU](chapters/04-setup-basic-usage.md#serve-an-slm-on-a-single-gpu-with-vllm)
  - [Streaming Responses](chapters/04-setup-basic-usage.md#enable-streaming-responses)
  - [OpenAI SDK Integration](chapters/04-setup-basic-usage.md#use-the-openai-python-sdk)

- [Performance Tuning & Memory Trade-offs](chapters/05-performance-tuning.md)
  - [Latency Tuning Techniques](chapters/05-performance-tuning.md#51-latency-tuning-and-acceleration-techniques)
  - [Model Quantization](chapters/05-performance-tuning.md#52-model-quantization-and-compression)
  - [Compiler Toolchains](chapters/05-performance-tuning.md#53-compiler-toolchains-and-kernel-optimization)

---

# Part III: Production Deployment

- [Serving Different Models](chapters/06-serving-models.md)
  - [Single GPU Inference](chapters/06-serving-models.md#61-single-gpu-inference)
  - [Distributed Inference](chapters/06-serving-models.md#62-distributed-and-scalable-inference)
  - [Weight Management](chapters/06-serving-models.md#63-weight-management-methods)
  - [Multi-Tenancy](chapters/06-serving-models.md#64-multi-tenancy-and-isolation)

- [Scaling on Multi-GPU & Multi-Node](chapters/07-scaling.md)
  - [Tensor Parallelism](chapters/07-scaling.md#tensor-parallelism)
  - [Pipeline Parallelism](chapters/07-scaling.md#pipeline-parallelism)
  - [Kubernetes Deployments](chapters/07-scaling.md#kubernetes-deployments)

- [Ecosystem Integration](chapters/08-ecosystem-integration.md)
  - [OpenAI SDK](chapters/08-ecosystem-integration.md#openai-sdk)
  - [Hugging Face Hub](chapters/08-ecosystem-integration.md#hugging-face-hub)
  - [LangChain](chapters/08-ecosystem-integration.md#langchain)

---

# Part IV: Operations & Beyond

- [Observability & Debugging](chapters/09-observability.md)
  - [Logging](chapters/09-observability.md#logging)
  - [Metrics](chapters/09-observability.md#metrics)
  - [Tracing](chapters/09-observability.md#tracing)
  - [Best Practices](chapters/09-observability.md#best-practices)

- [Comparisons & Further Reading](chapters/10-comparisons.md)
  - [vLLM vs DeepSpeed](chapters/10-comparisons.md#vllm-vs-deepspeed)
  - [vLLM vs TensorRT-LLM](chapters/10-comparisons.md#vllm-vs-tensorrt-llm)
  - [Further Reading](chapters/10-comparisons.md#further-reading)

- [Ecosystem Players](chapters/11-ecosystem-players.md)
  - [Cloud Platforms](chapters/11-ecosystem-players.md#cloud-platforms)
  - [Inference Providers](chapters/11-ecosystem-players.md#inference-providers)

---

# Appendices

- [References & Resources](appendices/references.md)
- [Glossary](appendices/glossary.md)
