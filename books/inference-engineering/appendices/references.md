# Appendix A: References & Resources

> A curated collection of resources for inference engineering.

---

## Official Documentation

### Inference Engines

| Engine | Documentation | GitHub |
|--------|---------------|--------|
| vLLM | [docs.vllm.ai](https://docs.vllm.ai) | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| TGI | [HF Docs](https://huggingface.co/docs/text-generation-inference) | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| TensorRT-LLM | [NVIDIA Docs](https://nvidia.github.io/TensorRT-LLM/) | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| DeepSpeed | [deepspeed.ai](https://www.deepspeed.ai/) | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| SGLang | [sgl-project.github.io](https://sgl-project.github.io/) | [sgl-project/sglang](https://github.com/sgl-project/sglang) |

### Hardware & Frameworks

| Resource | Link |
|----------|------|
| CUDA Toolkit | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |
| cuDNN | [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) |
| NCCL | [developer.nvidia.com/nccl](https://developer.nvidia.com/nccl) |
| PyTorch | [pytorch.org/docs](https://pytorch.org/docs) |
| Hugging Face | [huggingface.co/docs](https://huggingface.co/docs) |

---

## Academic Papers

### Core Innovations

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [PagedAttention](https://arxiv.org/abs/2309.06180) | Kwon et al. | 2023 | vLLM's memory management |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Dao et al. | 2022 | IO-aware attention |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Dao | 2023 | Improved algorithm |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Frantar et al. | 2022 | Post-training quantization |
| [AWQ](https://arxiv.org/abs/2306.00978) | Lin et al. | 2023 | Activation-aware quantization |

### Optimization Techniques

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) | Leviathan et al. | 2022 | Draft-verify acceleration |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | Xiao et al. | 2022 | Activation smoothing for INT8 |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | Dettmers et al. | 2022 | 8-bit LLM inference |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Dettmers et al. | 2023 | 4-bit fine-tuning |

### Architecture & Systems

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 2017 | Transformer architecture |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | Shoeybi et al. | 2019 | Model parallelism |
| [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) | Rajbhandari et al. | 2019 | Memory optimization |

---

## Blog Posts & Tutorials

### Performance Optimization

| Title | Source | Topic |
|-------|--------|-------|
| "Mastering LLM Techniques: Inference Optimization" | NVIDIA Developer | Comprehensive optimization |
| "Best practices to accelerate inference" | Together AI | Production patterns |
| "Inside vLLM: How Continuous Batching Works" | vLLM Blog | Architecture deep dive |
| "A Guide to Quantization in LLMs" | Hugging Face | Quantization methods |

### Practical Guides

| Title | Source | Topic |
|-------|--------|-------|
| "Deploying LLMs at Scale" | Anyscale | Ray + vLLM |
| "Optimizing Inference Costs" | Modal | Cost reduction |
| "Production LLM Serving" | Baseten | Enterprise patterns |
| "LLM Inference on Kubernetes" | Google Cloud | GKE deployment |

---

## Tools & Libraries

### Inference

| Tool | Purpose | Link |
|------|---------|------|
| vLLM | High-throughput serving | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| llama.cpp | CPU/edge inference | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| Ollama | Local LLM running | [ollama.com](https://ollama.com) |
| LM Studio | Local LLM GUI | [lmstudio.ai](https://lmstudio.ai) |

### Quantization

| Tool | Purpose | Link |
|------|---------|------|
| bitsandbytes | 8-bit/4-bit quantization | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| AutoGPTQ | GPTQ implementation | [github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) |
| AutoAWQ | AWQ implementation | [github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) |

### Observability

| Tool | Purpose | Link |
|------|---------|------|
| Prometheus | Metrics collection | [prometheus.io](https://prometheus.io) |
| Grafana | Visualization | [grafana.com](https://grafana.com) |
| Jaeger | Distributed tracing | [jaegertracing.io](https://www.jaegertracing.io) |

---

## Communities

### Discord Servers

| Community | Focus |
|-----------|-------|
| vLLM Discord | vLLM development and support |
| Hugging Face Discord | General ML/LLM |
| LocalLLaMA Discord | Local inference |

### Forums & Subreddits

| Platform | Focus |
|----------|-------|
| r/LocalLLaMA | Local LLM running |
| r/MachineLearning | General ML |
| Hugging Face Forums | HF ecosystem |
| NVIDIA Developer Forums | GPU/CUDA |

### Twitter/X Accounts

| Account | Focus |
|---------|-------|
| @vaborhey | vLLM updates |
| @_akhaliq | ML news |
| @huggingface | HF ecosystem |

---

## Video Resources

### Talks & Presentations

| Title | Event | Topic |
|-------|-------|-------|
| "Efficient LLM Inference" | MLSys 2023 | PagedAttention |
| "Scaling LLM Serving" | Ray Summit | Distributed inference |
| "TensorRT-LLM Deep Dive" | GTC 2024 | NVIDIA optimization |

### YouTube Channels

| Channel | Focus |
|---------|-------|
| Hugging Face | ML tutorials |
| NVIDIA Developer | GPU computing |
| Weights & Biases | MLOps |

---

## Books

| Title | Authors | Focus |
|-------|---------|-------|
| "Designing Machine Learning Systems" | Huyen | ML systems design |
| "Efficient Deep Learning" | Choudhary et al. | Model optimization |
| "Natural Language Processing with Transformers" | Tunstall et al. | Transformers |

---

## Model Repositories

| Repository | Models | Focus |
|------------|--------|-------|
| Hugging Face Hub | 400,000+ | General purpose |
| NVIDIA NGC | Optimized | Enterprise |
| TheBloke | Quantized | GPTQ/AWQ/GGUF |
| Ollama Library | Curated | Easy to use |

---

<p align="center">
  <a href="../chapters/11-ecosystem-players.md">← Previous: Ecosystem Players</a> | <a href="../README.md">Table of Contents</a> | <a href="glossary.md">Next: Glossary →</a>
</p>
