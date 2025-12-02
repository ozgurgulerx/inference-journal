# Day 002 – GPU Node Bring-Up on RunPod
## Tier 2 (Lite): Serve a Small LLM with vLLM

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) first  
> **Goal**: Run your first vLLM server and generate your first GPU-powered inference  
> **End State**: vLLM serving a lightweight model (SLM), verified with a curl request  
> **Time**: ~45 minutes

---

## 📦 Prerequisites

From Tier 1, you should already have:

- ✅ Ubuntu 24.04 running on RunPod
- ✅ NVIDIA driver working (`nvidia-smi`)
- ✅ CUDA runtime functional (`torch.cuda.is_available()`)

That's all you need.

---

## Tier 2 Tasks (~45 minutes)

---

### ✅ Task 2.1: Install vLLM
**Tags**: `[Inference–Setup]` `[vLLM]`  
**Time**: 5 min  
**Win**: vLLM installed successfully

#### 🔧 Lab Instructions

```bash
pip install vllm
```

Verify installation:

```bash
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```
```bash
root@54e7a4f7bf73:/# python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
vLLM version: 0.11.2
```

#### 🏆 Success Criteria
- [ ] `pip install vllm` completes without errors
- [ ] vLLM imports successfully

---

### ✅ Task 2.2: Choose a Small Model (SLM)
**Tags**: `[Model-Selection]`  
**Time**: 2 min  
**Win**: Pick a model that loads fast and fits easily in VRAM

#### Recommended Models (all 3–5GB VRAM)

| Model | VRAM (FP16) | Notes |
|-------|-------------|-------|
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3GB | Very fast, very capable ⭐ **Default** |
| `google/gemma-2-2b-it` | ~4.5GB | High-quality 2B instruction-tuned |
| `microsoft/phi-2` | ~2.5GB | Stable SLM, great for testing |

**Default for this guide**: `Qwen/Qwen2.5-1.5B-Instruct`

> 💡 These small models let you iterate quickly without waiting for large model downloads or worrying about VRAM limits.

---

### ✅ Task 2.3: Start vLLM Server
**Tags**: `[Inference–Runtime]` `[vLLM]`  
**Time**: 10 min  
**Win**: vLLM serving model on port 8000

#### 🔧 Lab Instructions

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.6
```

> ⚠️ **Why disable HF_TRANSFER?** RunPod's base image enables HuggingFace fast-transfer by default but does not include the Rust `hf_transfer` package. Disabling it prevents a startup crash.

> 🔍 **What's happening:**  
> - `--gpu-memory-utilization 0.6` controls how much VRAM vLLM pre-allocates for KV cache  
> - Model loading takes ~60 seconds (first run includes compilation)  
> - Leave the server running in this terminal

#### 📘 Deep Dive: What Happens When vLLM Loads a Model

When you run the command above, vLLM performs **15 internal stages**. Here's exactly what happens:

<details>
<summary><strong>Click to expand full startup trace</strong></summary>

```
INFO [...] vLLM API server version 0.11.2
INFO [...] non-default args: {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'gpu_memory_utilization': 0.6}
config.json: 100%|██████████| 660/660 [00:00<00:00, 6.54MB/s]
INFO [...] Resolved architecture: Qwen2ForCausalLM
INFO [...] Using max model len 32768
tokenizer_config.json: 7.30kB [00:00, 23.4MB/s]
vocab.json: 2.78MB [00:00, 19.9MB/s]
merges.txt: 1.67MB [00:00, 53.9MB/s]
tokenizer.json: 7.03MB [00:00, 95.7MB/s]
INFO [...] Initializing a V1 LLM engine (v0.11.2) with config:
    dtype=torch.bfloat16, max_seq_len=32768, enable_prefix_caching=True
INFO [...] world_size=1 rank=0 backend=nccl
INFO [...] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
INFO [...] Valid backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
INFO [...] Using FLASH_ATTN backend.
model.safetensors: 100%|██████████| 3.09G/3.09G [00:15<00:00, 203MB/s]
INFO [...] Loading weights took 2.46 seconds
INFO [...] Model loading took 2.8871 GiB memory and 34.43 seconds
INFO [...] Dynamo bytecode transform time: 3.62 s
INFO [...] Compiling a graph for dynamic shape takes 11.47 s
INFO [...] torch.compile takes 15.09 s in total
INFO [...] Available KV cache memory: 5.03 GiB
INFO [...] GPU KV cache size: 188,512 tokens
INFO [...] Maximum concurrency for 32,768 tokens per request: 5.75x
Capturing CUDA graphs (mixed prefill-decode): 100%|██████████| 51/51
Capturing CUDA graphs (decode, FULL): 100%|██████████| 35/35
INFO [...] Graph capturing finished in 4 secs, took 0.49 GiB
INFO [...] init engine took 22.78 seconds
INFO [...] Starting vLLM API server on http://0.0.0.0:8000
INFO:     Application startup complete.
```

</details>

**Stage-by-Stage Breakdown:**

| Stage | What Happens | Time |
|-------|--------------|------|
| **1. Environment** | `HF_HUB_ENABLE_HF_TRANSFER=0` disables Rust downloader (avoids crash) | instant |
| **2. Config Download** | `config.json` (660B) – model architecture, hidden size, num layers | <1s |
| **3. Architecture Resolution** | vLLM identifies `Qwen2ForCausalLM`, sets max_seq_len=32768 | <1s |
| **4. Tokenizer Download** | `vocab.json`, `merges.txt`, `tokenizer.json` – BPE vocabulary | ~2s |
| **5. Engine Init** | Creates scheduler, KV cache manager, attention backend selector | <1s |
| **6. Distributed Init** | Sets DP/TP/PP ranks (single GPU = all rank 0) | <1s |
| **7. Weights Download** | `model.safetensors` (3.09 GB) from HuggingFace | ~15s |
| **8. Load to GPU** | Weights transferred to VRAM, BF16 conversion | ~2.5s |
| **9. Attention Backend** | Tests FlashAttention, Flashinfer, Triton → picks fastest | <1s |
| **10. Torch Compile** | TorchInductor fuses ops, generates optimized CUDA kernels | ~15s |
| **11. KV Cache Alloc** | Reserves 5GB for key-value cache (~188K tokens capacity) | <1s |
| **12. CUDA Graph Capture** | Freezes GPU execution paths into static graphs | ~4s |
| **13. Warmup** | Runs test pass, preloads L2 cache | ~1s |
| **14. API Server Start** | Uvicorn starts, routes registered | <1s |
| **15. Ready** | `Application startup complete.` | — |

**Files Downloaded from HuggingFace:**

| File | Size | Purpose |
|------|------|---------|
| `config.json` | 660B | Model architecture definition |
| `tokenizer_config.json` | 7.3KB | Tokenizer settings |
| `vocab.json` | 2.78MB | BPE vocabulary (token → ID) |
| `merges.txt` | 1.67MB | BPE merge rules |
| `tokenizer.json` | 7.03MB | Fast tokenizer binary |
| `generation_config.json` | 242B | Default sampling params |
| `model.safetensors` | 3.09GB | Model weights |

**VRAM Breakdown:**

| Component | Memory |
|-----------|--------|
| Model weights | ~2.9 GB |
| Torch compile cache | ~0.5 GB |
| KV Cache | ~5.0 GB |
| CUDA graphs | ~0.5 GB |
| **Total** | ~9 GB / 16 GB |

#### 🏆 Success Criteria
- [ ] vLLM server starts without errors
- [ ] Model loads successfully (~60 seconds first run)
- [ ] Server listening on port 8000: `Starting vLLM API server on http://0.0.0.0:8000`
- [ ] `Application startup complete.` printed

---

### ✅ Task 2.4: Send Your First Inference Request
**Tags**: `[Inference–Test]`  
**Time**: 5 min  
**Win**: First successful GPU inference 🎉

#### 🔧 Lab Instructions

Open a **second SSH session** to the same pod and run:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "Explain what a GPU is in one sentence:",
    "max_tokens": 50
  }'
```

**Expected output:**
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "choices": [{
    "index": 0,
    "text": " A GPU, or Graphics Processing Unit, is a specialized chip designed to accelerate the processing of graphics and video data. It can perform multiple calculations simultaneously, making it ideal for tasks such as rendering 3D models, playing games, and performing complex simulations",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

Check GPU utilization in the first terminal:
```bash
nvidia-smi
```

> 🔍 **What's happening inside the GPU:**  
> - The prompt is tokenized and sent to the GPU  
> - vLLM's PagedAttention manages the KV cache efficiently  
> - Tensor Cores compute the forward pass  
> - Tokens are generated autoregressively until `max_tokens`

**🎉 You are now serving LLM inference. This is the milestone.**

#### 🏆 Success Criteria
- [ ] curl returns a valid JSON response
- [ ] Response contains coherent text
- [ ] `nvidia-smi` shows VRAM usage increased

---

### ✅ Task 2.5: (Optional) Enable Streaming
**Tags**: `[Inference–Streaming]`  
**Time**: 3 min  
**Win**: See token-by-token output

#### 🔧 Lab Instructions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "Explain KV cache in one sentence:",
    "max_tokens": 50,
    "stream": true
  }'
```

You'll see chunks arriving progressively – this is how ChatGPT-style streaming works.

> 🔍 **What's happening:**  
> - Instead of waiting for all tokens, vLLM sends each token as it's generated  
> - Server-Sent Events (SSE) stream the response  
> - Lower perceived latency for users

---

### ✅ Task 2.6: Save Artifacts
**Tags**: `[Artifacts]`  
**Time**: 3 min  
**Win**: Traceable proof of the first successful inference

#### 🔧 Lab Instructions

```bash
mkdir -p ~/artifacts/tier02-lite

# Capture GPU state during serving
nvidia-smi | tee ~/artifacts/tier02-lite/nvidia_smi_serving.txt

# Capture running processes
ps aux | grep vllm | tee ~/artifacts/tier02-lite/vllm_process.txt

# Save a sample inference response
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "What is inference optimization?",
    "max_tokens": 50
  }' | python3 -m json.tool | tee ~/artifacts/tier02-lite/sample_response.json
```

---

## Tier 2 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **2.1** | Install vLLM | ⬜ |
| **2.2** | Choose small model | ⬜ |
| **2.3** | Start vLLM server | ⬜ |
| **2.4** | Send first inference | ⬜ |
| **2.5** | (Optional) Streaming | ⬜ |
| **2.6** | Save artifacts | ⬜ |

### Artifacts Created
```
~/artifacts/tier02-lite/
├── nvidia_smi_serving.txt
├── vllm_process.txt
└── sample_response.json
```

---

## 🎉 Tier 2 Complete!

By finishing this, you have:

- ✅ Installed vLLM
- ✅ Served a small LLM locally
- ✅ Sent your first inference request
- ✅ Validated GPU-backed serving works
- ✅ Saved artifacts for reproducibility

**This is the foundation for:**
- **Tier 3**: vLLM tuning (KV cache, block sizes, memory)
- **Tier 4**: Quantization and performance optimization

---

## 🔜 Next Step

When you're ready, continue to Tier 3:

**→ [LOG_tier03.md](LOG_tier03.md)** – vLLM configuration tuning (your first real performance work)

