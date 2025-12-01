# Inference Engineering

<p align="center">
  <img src="https://img.shields.io/badge/Status-Work%20In%20Progress-yellow?style=for-the-badge" alt="Status: WIP"/>
  <img src="https://img.shields.io/badge/Focus-LLM%20Inference-blue?style=for-the-badge" alt="Focus: LLM Inference"/>
  <img src="https://img.shields.io/badge/Engine-vLLM-green?style=for-the-badge" alt="Engine: vLLM"/>
</p>

---

> **Note**: This book is being generated as part of the **100 Days of Inference Engineering** challenge. It represents a learning journey and is continuously evolving. Content is refined through hands-on experimentation and study. Contributions, corrections, and suggestions are welcome.

---

## About This Book

**Inference engineering** is the discipline of designing and operating the stack that transforms a trained model into a fast, reliable, and cost-effective API. This book covers the essential concepts, tools, and techniques required to serve Large Language Models (LLMs) at scale.

### What You'll Learn

- How to configure operating systems and hardware for optimal LLM serving
- Deep understanding of vLLM architecture and memory optimization
- Performance tuning strategies for throughput vs. latency trade-offs
- Distributed inference across multi-GPU and multi-node deployments
- Integration with the broader ML ecosystem

---

## Table of Contents

### Part I: Foundations

| Chapter | Title | Description |
|---------|-------|-------------|
| [1](chapters/01-introduction.md) | **Introduction to Inference Engineering** | Core concepts, goals, and the inference stack |
| [2](chapters/02-os-essentials.md) | **OS Essentials** | Linux setup, kernel tuning, GPU/TPU configuration |

### Part II: vLLM Deep Dive

| Chapter | Title | Description |
|---------|-------|-------------|
| [3](chapters/03-vllm-architecture.md) | **vLLM Architecture & Internals** | PagedAttention, continuous batching, scheduling |
| [4](chapters/04-setup-basic-usage.md) | **Setup & Basic Usage** | Installation, serving models, OpenAI API integration |
| [5](chapters/05-performance-tuning.md) | **Performance Tuning & Memory Trade-offs** | Latency, throughput, quantization, compilers |

### Part III: Production Deployment

| Chapter | Title | Description |
|---------|-------|-------------|
| [6](chapters/06-serving-models.md) | **Serving Different Models** | Single GPU, distributed, weight management |
| [7](chapters/07-scaling.md) | **Scaling on Multi-GPU & Multi-Node** | Tensor/pipeline parallelism, Kubernetes |
| [8](chapters/08-ecosystem-integration.md) | **Ecosystem Integration** | OpenAI SDK, Hugging Face, LangChain |

### Part IV: Operations & Beyond

| Chapter | Title | Description |
|---------|-------|-------------|
| [9](chapters/09-observability.md) | **Observability & Debugging** | Logging, metrics, tracing, best practices |
| [10](chapters/10-comparisons.md) | **Comparisons & Further Reading** | vLLM vs DeepSpeed vs TensorRT-LLM |
| [11](chapters/11-ecosystem-players.md) | **Ecosystem Players** | RunPod, Replicate, Modal, Together, and more |

### Appendices

| Appendix | Title |
|----------|-------|
| [A](appendices/references.md) | **References & Resources** |
| [B](appendices/glossary.md) | **Glossary** |

---

## Quick Navigation

```
inference-engineering/
├── README.md                    # You are here
├── SUMMARY.md                   # GitBook-style navigation
├── chapters/
│   ├── 01-introduction.md
│   ├── 02-os-essentials.md
│   ├── 03-vllm-architecture.md
│   ├── 04-setup-basic-usage.md
│   ├── 05-performance-tuning.md
│   ├── 06-serving-models.md
│   ├── 07-scaling.md
│   ├── 08-ecosystem-integration.md
│   ├── 09-observability.md
│   ├── 10-comparisons.md
│   └── 11-ecosystem-players.md
└── appendices/
    ├── references.md
    └── glossary.md
```

---

## How to Use This Book

### For Beginners
Start with **Chapter 1** (Introduction) to understand the "why" of inference engineering, then proceed to **Chapter 4** (Setup) to get hands-on experience.

### For Practitioners
Jump directly to the chapter that addresses your current challenge:
- Memory issues? → **Chapter 3** (vLLM Architecture)
- Latency problems? → **Chapter 5** (Performance Tuning)
- Scaling needs? → **Chapter 7** (Multi-GPU/Multi-Node)

### For Operators
Focus on **Part III** (Production) and **Part IV** (Operations) for deployment patterns and observability.

---

## Rendering This Book

This book is structured to work with multiple documentation systems:

### GitHub (Default)
Simply browse the markdown files directly on GitHub.

### mdBook (Recommended for local viewing)
```bash
# Install mdBook
cargo install mdbook

# Serve locally
cd books/inference-engineering
mdbook serve --open
```

### GitBook
The `SUMMARY.md` file provides GitBook-compatible navigation.

---

## Contributing

This is a learning project. If you find errors or have suggestions:
1. Open an issue describing the problem or enhancement
2. Submit a PR with corrections or additions

---

## License

This content is part of an educational project. Please cite appropriately if referencing.

---

<p align="center">
  <i>Part of the <a href="../../README.md">Inference Engineering Journal</a></i>
</p>
