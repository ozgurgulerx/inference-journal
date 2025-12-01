# Appendix B: Glossary

> Key terms and definitions for inference engineering.

---

## A

**Activation**
: The output of a neural network layer after applying an activation function. In inference, managing activation memory is crucial for efficiency.

**AWQ (Activation-Aware Weight Quantization)**
: A quantization method that considers activation patterns when quantizing weights, often producing better quality than naive quantization.

**Autoregressive**
: A generation method where each token is predicted based on all previous tokens. Most LLMs use autoregressive decoding.

---

## B

**Batch Size**
: The number of requests processed together. Larger batches improve throughput but may increase latency.

**BF16 (BFloat16)**
: A 16-bit floating-point format with the same exponent range as FP32 but less precision. Popular for training and inference.

---

## C

**CUDA**
: NVIDIA's parallel computing platform and API for GPU programming.

**CUDA Graph**
: A feature that records and replays a sequence of GPU operations, reducing CPU overhead and kernel launch latency.

**Continuous Batching**
: A serving technique where new requests join the batch as others complete, maximizing GPU utilization.

---

## D

**Decode Phase**
: The autoregressive token generation phase after the initial prompt processing. Memory-bandwidth bound.

**DeepSpeed**
: Microsoft's deep learning optimization library for training and inference.

---

## F

**Flash Attention**
: An optimized attention algorithm that reduces memory usage and increases speed through better memory access patterns.

**FP8**
: An 8-bit floating-point format supported by NVIDIA Hopper GPUs (H100), offering significant speedups.

**FP16**
: 16-bit floating-point format, commonly used for inference to reduce memory and increase speed.

---

## G

**GPTQ**
: A post-training quantization method that compresses weights to 4-bit or lower with minimal quality loss.

**GQA (Grouped Query Attention)**
: An attention variant where multiple query heads share key-value heads, reducing memory usage.

---

## H

**HBM (High Bandwidth Memory)**
: Fast GPU memory (e.g., 80GB on A100/H100) with bandwidth up to 3.35 TB/s.

**Hugging Face**
: A company and platform providing ML tools, model hub, and inference infrastructure.

---

## I

**Inference**
: Running a trained model to generate predictions or outputs, as opposed to training.

**ITL (Inter-Token Latency)**
: Time between generating consecutive tokens. Critical for streaming applications.

---

## K

**KV Cache**
: Cached key and value tensors from previous tokens, avoiding recomputation during autoregressive generation.

---

## L

**LoRA (Low-Rank Adaptation)**
: A parameter-efficient fine-tuning method that adds small trainable matrices to a frozen base model.

**LPU (Language Processing Unit)**
: Groq's specialized chip designed for ultra-fast LLM inference.

---

## M

**MIG (Multi-Instance GPU)**
: NVIDIA feature allowing a single GPU to be partitioned into multiple isolated instances.

**MoE (Mixture of Experts)**
: An architecture where only a subset of model parameters are activated per token, improving efficiency.

---

## N

**NCCL (NVIDIA Collective Communications Library)**
: NVIDIA's library for multi-GPU and multi-node collective operations.

**NVLink**
: High-bandwidth interconnect between GPUs (up to 900 GB/s on H100).

---

## P

**PagedAttention**
: vLLM's memory management technique inspired by OS virtual memory, enabling efficient KV cache allocation.

**Pipeline Parallelism (PP)**
: Distributing model layers across devices, processing requests in a pipeline fashion.

**Prefill Phase**
: Initial processing of the input prompt to build the KV cache. Compute-bound.

**Prefix Caching**
: Reusing KV cache from shared prompt prefixes across requests.

---

## Q

**Quantization**
: Reducing numerical precision (e.g., FP16 → INT8) to decrease memory usage and increase speed.

---

## R

**RoPE (Rotary Position Embedding)**
: A position encoding method used in LLaMA and other models for better length generalization.

---

## S

**SLO (Service Level Objective)**
: Target performance metrics (e.g., p95 latency < 500ms).

**Speculative Decoding**
: Using a small "draft" model to predict multiple tokens, verified by the main model in one step.

**Streaming**
: Returning generated tokens incrementally rather than waiting for complete generation.

---

## T

**Tensor Parallelism (TP)**
: Splitting individual layers across multiple GPUs, each computing a portion.

**TGI (Text Generation Inference)**
: Hugging Face's inference server for text generation models.

**TensorRT**
: NVIDIA's SDK for high-performance deep learning inference.

**TTFT (Time to First Token)**
: Latency from request to first generated token. Critical for user experience.

**Throughput**
: Tokens generated per second across all requests.

---

## V

**vLLM**
: High-throughput LLM serving engine featuring PagedAttention and continuous batching.

---

## X

**XLA (Accelerated Linear Algebra)**
: A compiler for accelerating ML computations, used by JAX and TensorFlow.

---

## Metrics Quick Reference

| Metric | Definition | Good Value |
|--------|------------|------------|
| TTFT | Time to first token | < 500ms |
| ITL | Inter-token latency | < 50ms |
| Throughput | Tokens/second | Model-dependent |
| p95 Latency | 95th percentile latency | < 2s |
| GPU Utilization | Compute usage | > 80% |
| KV Cache Usage | Memory for cache | < 90% |

---

## Memory Formulas

**Model Size (FP16)**:
```
Memory = Parameters × 2 bytes
Example: 7B model = 7 × 10^9 × 2 = 14 GB
```

**KV Cache per Token**:
```
KV_size = 2 × hidden_dim × num_layers × bytes_per_element
Example: Llama-70B = 2 × 8192 × 80 × 2 = 2.6 MB/token
```

**Batch Memory**:
```
Total = Model + (KV_size × seq_len × batch_size)
```

---

<p align="center">
  <a href="references.md">← Previous: References</a> | <a href="../README.md">Table of Contents</a>
</p>
