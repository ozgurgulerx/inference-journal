# Day 002 – GPU Node Bring-Up on RunPod (From Scratch)

> **Goal**: Build a complete LLM inference stack from a bare Ubuntu VM – install drivers, CUDA, and everything yourself.  
> **End State**: Understand every layer: OS → Drivers → CUDA → Python → vLLM → LLM API  
> **GPU**: NVIDIA T4 (16GB) – the workhorse of cloud inference

---

## 📚 Pre-Reading (20 min before you start)

| Resource | Why | Time |
|----------|-----|------|
| [NVIDIA Driver vs CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) | Understand driver/runtime relationship | 5 min |
| [CUDA Installation Guide - Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) | Official NVIDIA install docs | 10 min |
| [Ubuntu NVIDIA Drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation) | Ubuntu-specific driver install | 5 min |

#### 📂 Key Concepts to Understand
- **Kernel**: Linux core that talks to hardware
- **Driver**: Software that lets kernel talk to GPU
- **CUDA Toolkit**: Libraries + compiler for GPU programming
- **cuDNN**: Deep learning primitives (convolutions, etc.)

---

## Tier 1 – Must-Do Core Block (~3 hours)

**Objective**: Start with bare Ubuntu, install everything manually, end with working LLM API.

---

### ✅ Task 1.1: Launch Bare Ubuntu VM with T4 GPU
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 15 min  
**Win**: SSH into a fresh Ubuntu box with GPU attached (but no drivers yet!)

#### 📖 Learn First
- [RunPod GPU Types](https://www.runpod.io/gpu-instance/pricing) – T4 is ~$0.20/hr
- [NVIDIA T4 Specs](https://www.nvidia.com/en-us/data-center/tesla-t4/) – 16GB VRAM, Turing architecture

#### 🔧 Lab Instructions

1. **Create account** at [runpod.io](https://runpod.io) and add credits ($5-10)

2. **Deploy a BARE Ubuntu Pod** (not a pre-configured template):
   - Go to **Pods** → **Deploy**
   - Click **"Customize Deployment"** or select a minimal template
   - Template: **`runpod/ubuntu:22.04`** (bare Ubuntu, no CUDA pre-installed)
   - GPU: **NVIDIA T4** (16GB, ~$0.20/hr) – cheapest option for learning
   - Container Disk: **30 GB** (need space for CUDA + models)
   - Volume Disk: **50 GB** (persistent storage for models)
   - Click **Deploy**

3. **Connect via Web Terminal or SSH**:

```bash
# From RunPod UI: Click "Connect" → "Start Web Terminal"
# Or use SSH command from pod details
```

4. **Verify you're on bare Ubuntu** (GPU not yet accessible):

```bash
# Check OS
cat /etc/os-release
```

Expected:
```
PRETTY_NAME="Ubuntu 22.04.x LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
```

```bash
# Check if nvidia-smi works (it should NOT work yet, or show basic info)
nvidia-smi
```

If this fails or shows "NVIDIA-SMI has failed", **that's expected** – we need to install drivers!

```bash
# Check what GPU hardware is attached
lspci | grep -i nvidia
```

Expected output like:
```
00:05.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
```

#### 🏆 Success Criteria
- [ ] SSH/terminal access to Ubuntu 22.04
- [ ] `lspci` shows NVIDIA T4 attached
- [ ] `nvidia-smi` either fails or shows minimal info (drivers not fully set up)

#### 📁 Artifacts
```bash
mkdir -p ~/artifacts
cat /etc/os-release | tee ~/artifacts/os_info.txt
lspci | grep -i nvidia | tee ~/artifacts/gpu_hardware.txt
```

---

### ✅ Task 1.2: System Update & Essential Packages
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 15 min  
**Win**: System fully updated with build tools ready

#### 📖 Learn First
- [Ubuntu Package Management](https://ubuntu.com/server/docs/package-management)
- Why we need `build-essential`: Compiling CUDA samples, kernel modules

#### 🔧 Lab Instructions

```bash
# Update package lists
sudo apt update

# Upgrade all packages (important for kernel compatibility)
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
  build-essential \
  dkms \
  linux-headers-$(uname -r) \
  git \
  curl \
  wget \
  htop \
  tmux \
  vim

# Check kernel version
uname -r
```

Record system info:
```bash
uname -a | tee ~/artifacts/system_info.txt
echo "Kernel: $(uname -r)" | tee -a ~/artifacts/system_info.txt
```

#### 🏆 Success Criteria
- [ ] `apt update && apt upgrade` completes without errors
- [ ] `build-essential` and `linux-headers` installed
- [ ] Kernel version recorded

---

### ✅ Task 1.3: Install NVIDIA Drivers (The Hard Way)
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 25 min  
**Win**: `nvidia-smi` shows your T4 with driver version

#### 📖 Learn First
- [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
- [Ubuntu NVIDIA Driver Install](https://ubuntu.com/server/docs/nvidia-drivers-installation)

**Key Concept**: The driver is the bridge between Linux kernel and GPU hardware.

#### 🔧 Lab Instructions

**Method A: Ubuntu's driver manager (recommended)**

```bash
# Check available drivers
ubuntu-drivers devices
```

Expected output shows something like:
```
vendor   : NVIDIA Corporation
model    : TU104GL [Tesla T4]
driver   : nvidia-driver-535 - distro non-free recommended
driver   : nvidia-driver-525 - distro non-free
```

```bash
# Install the recommended driver
sudo ubuntu-drivers autoinstall

# OR install specific version
sudo apt install -y nvidia-driver-535
```

**Method B: Manual install from NVIDIA (for learning)**

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install driver only
sudo apt install -y nvidia-driver-535
```

**After installation:**

```bash
# Verify driver module is loaded
lsmod | grep nvidia
```

If empty, the driver isn't loaded yet. May need reboot or:

```bash
# Load the driver module
sudo modprobe nvidia
```

```bash
# THE MOMENT OF TRUTH
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx       Driver Version: 535.xxx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:05.0 Off |                    0 |
| N/A   35C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

#### 🏆 Success Criteria
- [ ] `nvidia-smi` runs successfully
- [ ] Shows Tesla T4 with 16GB (15360MiB) memory
- [ ] Driver version displayed (e.g., 535.xxx)

#### 📁 Artifacts
```bash
nvidia-smi | tee ~/artifacts/nvidia_smi_driver.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee ~/artifacts/gpu_info.csv
```

---

### ✅ Task 1.4: Install CUDA Toolkit
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 20 min  
**Win**: `nvcc --version` works, CUDA samples compile

#### 📖 Learn First
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)

**Key Concepts**:
- **CUDA Driver API**: Comes with nvidia-driver, low-level
- **CUDA Runtime API**: Comes with cuda-toolkit, what most apps use
- **nvcc**: NVIDIA CUDA Compiler

#### 🔧 Lab Instructions

```bash
# Install CUDA toolkit (if you added NVIDIA repo earlier)
sudo apt install -y cuda-toolkit-12-2

# OR install full CUDA (includes drivers, use if drivers not installed)
# sudo apt install -y cuda-12-2
```

```bash
# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA compiler
nvcc --version
```

Expected:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.2, V12.2.xxx
```

#### Compile a CUDA program (optional but educational)

```bash
mkdir -p ~/cuda-tests && cd ~/cuda-tests

cat > hello_cuda.cu << 'EOF'
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d, block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Launching kernel...\n");
    hello<<<2, 4>>>();  // 2 blocks, 4 threads each
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}
EOF

nvcc hello_cuda.cu -o hello_cuda
./hello_cuda
```

Expected output:
```
Launching kernel...
Hello from GPU thread 0, block 0!
Hello from GPU thread 1, block 0!
...
Done!
```

#### 🏆 Success Criteria
- [ ] `nvcc --version` shows CUDA 12.x
- [ ] Hello CUDA program compiles and runs
- [ ] You understand: driver vs toolkit vs runtime

#### 📁 Artifacts
```bash
nvcc --version | tee ~/artifacts/cuda_version.txt
```

---

### ✅ Task 1.5: Install Python & Deep Learning Stack
**Tags**: `[OS–Linux]` `[Inference–Runtime]`  
**Time**: 20 min  
**Win**: PyTorch sees your GPU, matrix multiply works

#### 📖 Learn First
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [Python venv](https://docs.python.org/3/library/venv.html)

#### 🔧 Lab Instructions

```bash
# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Add to bashrc for persistence
echo 'source ~/venv/bin/activate' >> ~/.bashrc

# Upgrade pip
pip install --upgrade pip
```

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
# Verify PyTorch sees GPU
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

```bash
# Quick GPU compute test
python3 << 'EOF'
import torch
import time

device = torch.device('cuda')
size = 8000  # Smaller for T4

a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warmup
torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(10):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

tflops = (10 * 2 * size**3) / elapsed / 1e12
print(f"Matrix multiply ({size}x{size}): {elapsed*1000:.1f}ms total, {tflops:.1f} TFLOPS")
EOF
```

#### 🏆 Success Criteria
- [ ] PyTorch reports `CUDA available: True`
- [ ] Shows Tesla T4 with ~16GB memory
- [ ] Matrix multiply benchmark runs (T4 should show ~8-10 TFLOPS)

#### 📁 Artifacts
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" | tee ~/artifacts/pytorch_info.txt
```

---

### ✅ Task 1.6: Install vLLM from Scratch
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 15 min  
**Win**: vLLM installed and importable

#### 📖 Learn First
- [vLLM Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [vLLM GitHub](https://github.com/vllm-project/vllm)

#### 🔧 Lab Instructions

```bash
# Install vLLM
pip install vllm

# This may take 5-10 minutes as it compiles some components
```

```bash
# Verify installation
python3 << 'EOF'
import vllm
print(f"vLLM version: {vllm.__version__}")

from vllm import LLM
print("vLLM LLM class imported successfully!")
EOF
```

#### 🏆 Success Criteria
- [ ] `pip install vllm` completes
- [ ] vLLM imports without errors

---

### ✅ Task 1.7: Serve Your First LLM API
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 30 min  
**Win**: OpenAI-compatible API serving Llama-2-7B

#### 📖 Learn First
- [vLLM OpenAI Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Llama 2 on HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

**Note**: T4 has 16GB VRAM. Llama-2-7B in FP16 needs ~14GB. It will fit!

#### 🔧 Lab Instructions

```bash
# Install HuggingFace CLI for model access
pip install huggingface_hub

# Login to HuggingFace (needed for Llama models)
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens
# Accept Llama 2 license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

```bash
# Create artifacts directory
mkdir -p ~/artifacts

# Start vLLM server with Llama-2-7B
# Note: First run downloads ~13GB model
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee ~/artifacts/vllm_startup.log &
```

```bash
# Watch the log until you see "Uvicorn running"
tail -f ~/artifacts/vllm_startup.log
# Press Ctrl+C when server is ready
```

```bash
# Check GPU memory usage
nvidia-smi
```

Should show ~13-14GB used.

```bash
# Test the API!
curl http://localhost:8000/v1/models

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain what CUDA is in one sentence."}
    ],
    "max_tokens": 100
  }' | python3 -m json.tool
```

#### 🏆 Success Criteria
- [ ] vLLM server starts without OOM
- [ ] `/v1/models` endpoint works
- [ ] Chat completion returns sensible response
- [ ] GPU shows ~14GB used

#### 📁 Artifacts
```bash
nvidia-smi | tee ~/artifacts/nvidia_smi_serving.txt
curl -s http://localhost:8000/v1/models | python3 -m json.tool | tee ~/artifacts/api_models.json
```

---

### ✅ Task 1.8: First Benchmark
**Tags**: `[Inference–Runtime]` `[Phase1-HF_vs_vLLM]`  
**Time**: 15 min  
**Win**: Baseline performance numbers on T4

#### 🔧 Lab Instructions

```bash
cat > ~/benchmark.py << 'EOF'
import requests
import time
import json
import statistics

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-2-7b-chat-hf"

def benchmark(prompt, max_tokens=100):
    start = time.time()
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    })
    elapsed = time.time() - start
    data = r.json()
    tokens = data["usage"]["completion_tokens"]
    return {"elapsed": elapsed, "tokens": tokens, "tok_per_sec": tokens/elapsed}

prompts = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is GPU computing?",
    "Describe CUDA in one paragraph.",
    "What is inference optimization?"
]

print("Running benchmark on T4...")
results = []
for i, p in enumerate(prompts):
    r = benchmark(p)
    print(f"  {i+1}. {r['tok_per_sec']:.1f} tok/s ({r['elapsed']:.2f}s)")
    results.append(r)

avg = statistics.mean([r['tok_per_sec'] for r in results])
print(f"\n=== T4 BASELINE: {avg:.1f} tokens/sec ===")

with open("/root/artifacts/t4_baseline.json", "w") as f:
    json.dump({"results": results, "avg_tok_per_sec": avg}, f, indent=2)
EOF

python3 ~/benchmark.py
```

#### 🏆 Success Criteria
- [ ] Benchmark completes all 5 prompts
- [ ] Results saved to JSON
- [ ] You know your T4 baseline (expect ~20-40 tok/s)

---

## Tier 1 Summary – What You Built

| Layer | What You Installed | Command to Verify |
|-------|-------------------|-------------------|
| **OS** | Ubuntu 22.04 | `cat /etc/os-release` |
| **Kernel** | Linux headers | `uname -r` |
| **Driver** | nvidia-driver-535 | `nvidia-smi` |
| **CUDA** | cuda-toolkit-12.2 | `nvcc --version` |
| **Python** | Python 3.10 + venv | `python3 --version` |
| **ML** | PyTorch + CUDA | `python3 -c "import torch; print(torch.cuda.is_available())"` |
| **Inference** | vLLM | `python3 -c "import vllm"` |
| **API** | OpenAI-compatible | `curl localhost:8000/v1/models` |

**You built the entire stack from scratch!** This is exactly what an inference engineer needs to understand.

### Artifacts Created
```
~/artifacts/
├── os_info.txt
├── gpu_hardware.txt
├── system_info.txt
├── nvidia_smi_driver.txt
├── gpu_info.csv
├── cuda_version.txt
├── pytorch_info.txt
├── vllm_startup.log
├── nvidia_smi_serving.txt
├── api_models.json
└── t4_baseline.json
```

### Commit
```bash
cd ~/artifacts
git init
git add .
git commit -m "day02-tier1: T4 from scratch - Ubuntu → Drivers → CUDA → vLLM → API"
```

---

**→ Continue to [Tier 2](LOG_tier02.md) for HF vs vLLM comparison**

