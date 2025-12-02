# Day 002 – GPU Node Bring-Up on RunPod
## Tier 1: GPU Stack Verification

> **Goal**: Bring up a GPU node and understand every layer of the stack.  
> **End State**: GPU fully verified and inference-ready (drivers, CUDA, health checks complete)  
> **GPU**: NVIDIA T4 or RTX 2000 Ada (both 16GB) – affordable cloud GPUs for learning  
> **Time**: ~1.5 hours

---

## � SSH Key Setup (One-Time Prerequisite)

Before deploying your first pod, you'll need SSH access configured. While RunPod offers a web-based terminal, SSH provides a more robust experience—enabling local tooling, file transfers via `scp`, and persistent sessions with `tmux`.

### Why a Dedicated Key?

Most developers already have an SSH key (`~/.ssh/id_ed25519` or `~/.ssh/id_rsa`) configured for services like GitHub, GitLab, or Azure. Rather than reusing that key—or worse, overwriting it—best practice is to generate a **dedicated key pair** for cloud GPU providers.

This approach offers several advantages:

- **Isolation**: Compromise of one key doesn't affect others
- **Revocability**: You can revoke RunPod access without disrupting other services
- **Clarity**: Easy to audit which key is used where

### Generating a Dedicated Key

Open your local terminal and run:

```bash
ssh-keygen -t ed25519 -a 100 -C "runpod" -f ~/.ssh/id_runpod
```

This creates two files:

```
~/.ssh/id_runpod        # Private key (never share this)
~/.ssh/id_runpod.pub    # Public key (safe to share)
```

When prompted for a passphrase, you may leave it empty for convenience or add one for extra security.

### Adding the Public Key to RunPod

Copy the public key to your clipboard:

```bash
# macOS
pbcopy < ~/.ssh/id_runpod.pub

# Linux
xclip -sel clipboard < ~/.ssh/id_runpod.pub

# Or just print and copy manually
cat ~/.ssh/id_runpod.pub
```

Then navigate to **RunPod → Settings → SSH Public Keys** and paste the key.

### Connecting to Your Pod

Once your pod is running, connect using:

```bash
ssh -i ~/.ssh/id_runpod root@<pod-ip> -p <port>
```

The exact IP and port are displayed in the RunPod dashboard under your pod's connection details.

> **Tip**: For convenience, you can add a host alias to `~/.ssh/config` so that `ssh runpod1` works without remembering the full command.

---

## � Pre-Reading (20 min before you start)

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

## Tier 1 Tasks (~1.5 hours)

**Objective**: Launch a GPU pod, verify drivers, understand the CUDA stack, and complete GPU health checks.

**What You'll Do:**
1. Launch pod & SSH in
2. Verify OS and GPU accessibility
3. (Optional) Install essential packages
4. Verify NVIDIA drivers work
5. Understand the CUDA stack (Driver vs Runtime vs Toolkit)
6. Complete GPU health checks (6 tests)

---

### ✅ Task 1.1: Launch a GPU-Enabled Ubuntu Container
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 15 min  
**Win**: SSH into a running GPU container and verify GPU is accessible

> **Important**: RunPod pods are **not bare-metal VMs** – they are GPU-enabled container images. Even the minimal "Ubuntu" template includes NVIDIA drivers and CUDA runtime pre-installed. Your job in Tier 1 is to **verify** the GPU stack, not install it.

#### 📖 Learn First
- [RunPod GPU Types](https://www.runpod.io/gpu-instance/pricing) – T4/RTX 2000 Ada are ~$0.15-0.20/hr
- [NVIDIA T4 Specs](https://www.nvidia.com/en-us/data-center/tesla-t4/) – 16GB VRAM, Turing (2018)
- [RTX 2000 Ada](https://www.nvidia.com/en-us/design-visualization/rtx-2000/) – 16GB VRAM, Ada Lovelace (2023), faster

#### 🔧 Lab Instructions

1. **Create account** at [runpod.io](https://runpod.io) and add credits ($5-10)

2. **Deploy a minimal Ubuntu Pod**:
   - Go to **Pods** → **Deploy**
   - Click **"Customize Deployment"** or select a minimal template
   - Template: **`runpod/ubuntu`** (Ubuntu 24.04 with GPU drivers auto-provisioned)
   - GPU: **RTX 2000 Ada** or **T4** (both 16GB, ~$0.15-0.20/hr)
   - Container Disk: **30 GB** (space for models)
   - Volume Disk: **50 GB** (persistent storage)
   - Click **Deploy**

> **Note**: RunPod injects NVIDIA drivers and CUDA runtime at container startup. This is why `nvidia-smi` works immediately – it's expected behavior, not an error!

3. **Connect via Web Terminal or SSH**:

```bash
# Option 1: Web Terminal (easiest)
# From RunPod UI: Click "Connect" → "Start Web Terminal"

# Option 2: SSH (see SSH Key Setup section above)
ssh -i ~/.ssh/id_runpod root@<pod-ip> -p <port>
```

4. **Verify the OS**:

```bash
# Check OS version
cat /etc/os-release
```

Expected:
```
PRETTY_NAME="Ubuntu 24.04.x LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
```

5. **Verify GPU is accessible** (should work immediately):

```bash
nvidia-smi
```

> **What is `nvidia-smi`?**  
> NVIDIA System Management Interface – think of it as `htop` but for your GPU.  
> It shows real-time GPU utilization, VRAM usage, temperature, power, and which processes are using the card.  
> You'll use it constantly to: verify models loaded to GPU, check VRAM footprint (7B model ≈ 14-16GB), diagnose OOM errors, and spot memory leaks.  
> **TL;DR**: `nvidia-smi` = the GPU truth meter. Run it to see "what is my GPU doing right now?"

**Expected outcome**: `nvidia-smi` shows your GPU with driver version and CUDA version. This confirms RunPod's auto-provisioning worked.

```bash
# Check what GPU hardware is attached
lspci | grep -i nvidia
```

Expected output like:
```
# For RTX 2000 Ada:
00:05.0 VGA compatible controller: NVIDIA Corporation AD107GL [RTX 2000 Ada Generation]
# Or for T4:
00:05.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
```

#### 🏆 Success Criteria
- [ ] SSH/terminal access to Ubuntu 24.04
- [ ] `lspci` shows NVIDIA GPU attached (RTX 2000 Ada or T4)
- [ ] `nvidia-smi` shows GPU with driver and CUDA version (auto-provisioned by RunPod)

#### 📁 Artifacts
```bash
mkdir -p ~/artifacts
cat /etc/os-release | tee ~/artifacts/os_info.txt
lspci | grep -i nvidia | tee ~/artifacts/gpu_hardware.txt
```

---

### ✅ Task 1.2: System Update & Essential Packages (Optional)
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 10 min  
**Win**: System updated with build tools ready

> **Note**: RunPod containers typically have Python, pip, and common tools pre-installed. This step is optional but useful if you need additional packages or want to ensure everything is up-to-date.

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

### ✅ Task 1.3: Verify NVIDIA Drivers
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 5 min  
**Win**: `nvidia-smi` shows your GPU with driver version

> **RunPod users**: Drivers are already installed. Just run `nvidia-smi` to verify.

#### 📖 Learn First
- [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx) (reference only)
- **Key Concept**: The driver is the bridge between Linux kernel and GPU hardware.

#### 🔧 Lab Instructions

**Step 1: Verify driver is working**

```bash
# This should work immediately on RunPod
nvidia-smi
```

Expected output (example for RTX 2000 Ada):
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx       Driver Version: 535.xxx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 2000 Ada  Off | 00000000:00:05.0 Off |                  Off |
| N/A   35C    P8    10W /  70W |      0MiB / 16376MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

#### 🏆 Success Criteria
- [ ] `nvidia-smi` runs successfully
- [ ] Shows GPU with 16GB memory (RTX 2000 Ada: 16376MiB, T4: 15360MiB)
- [ ] Driver version displayed (e.g., 535.xxx)

#### 📁 Artifacts
```bash
nvidia-smi | tee ~/artifacts/nvidia_smi_driver.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee ~/artifacts/gpu_info.csv
```

---

### ✅ Task 1.4: Understand the CUDA Stack & Complete GPU Health Checks
**Tags**: `[OS–Linux]` `[OS-01]`  
**Time**: 15 min  
**Win**: Understand driver vs runtime vs toolkit, verify CUDA works

#### 📖 The CUDA Stack Explained

This is one of the most important concepts in GPU engineering, and 99% of developers get it wrong.

**The Big Picture**: There are 3 separate layers in the NVIDIA stack, each solving a DIFFERENT problem:

```
+----------------------+   
|  CUDA Toolkit        |   nvcc (compile CUDA C++ code)
+----------------------+   
|  CUDA Runtime        |   PyTorch, JAX, vLLM call GPU ops
+----------------------+
|  NVIDIA Driver       |   nvidia-smi, kernel module
+----------------------+
|  GPU Hardware        |   RTX 2000 Ada, H100, etc.
+----------------------+
```

---

#### 🔧 Layer 1: NVIDIA Driver

**What it is**: Low-level kernel module + user-space libs that talk directly to the GPU.

**Commands that use it**: `nvidia-smi` (GPU utilization, VRAM, power, clocks, temperature)

**If the driver works**: `nvidia-smi` works, GPU is visible  
**If the driver breaks**: PyTorch cannot see GPU, `torch.cuda.is_available()` → False, vLLM crashes

> **On RunPod**: Driver is ALWAYS preinstalled. Never install drivers in a container.

---

#### 🔧 Layer 2: CUDA Runtime

**What it is**: Libraries that PyTorch, TensorFlow, JAX, vLLM, llama.cpp use to run GPU operations.

**What it includes**:
- **cuBLAS** – matrix multiplications
- **cuDNN** – convolutions (deep learning primitives)
- **NCCL** – multi-GPU communication
- cuRAND, cuSPARSE, cuFFT

**Key insight**: PyTorch wheels bundle the CUDA runtime. You do NOT need to install CUDA separately.

```
torch==2.8.0+cu128  →  Includes CUDA 12.8 runtime inside the wheel
```

**If runtime works**: `torch.cuda.is_available()` → True

---

#### 🔧 Layer 3: CUDA Toolkit (nvcc)

**What it is**: Developer tools, mainly `nvcc` – the CUDA C/C++ compiler.

**You NEED Toolkit if**:
- You compile custom CUDA kernels (.cu files)
- You build PyTorch from source
- You study GPU programming

**You DON'T need Toolkit if**:
- You train/fine-tune LLMs
- You run vLLM inference
- You use any Python ML library

> **Warning**: Toolkit is huge (>3GB) and can BREAK the driver if versions mismatch.

---

#### 🎯 TL;DR

| Layer | What It Does | How to Verify | Required For |
|-------|--------------|---------------|--------------|
| **Driver** | Talks to GPU hardware | `nvidia-smi` | Everything |
| **Runtime** | GPU compute libraries | `torch.cuda.is_available()` | PyTorch, vLLM, JAX |
| **Toolkit** | CUDA compiler (nvcc) | `nvcc --version` | Writing CUDA C++ only |

**Your RunPod pod already has layers 1 & 2. You don't need layer 3 unless writing CUDA kernels.**

#### 🔧 Lab Instructions

**⚠️ WARNING: Do NOT run `apt install cuda-toolkit` on RunPod!**  
RunPod images ship with CUDA 12.8 runtime. Installing a different CUDA version via apt will create version mismatches and break your GPU stack.

**Step 1: Verify what's already installed**

```bash
# Driver (already confirmed working)
nvidia-smi | head -3

# Check CUDA version from driver
nvidia-smi | grep "CUDA Version"

# Check if nvcc exists (often missing in Docker images - that's OK)
nvcc --version 2>/dev/null || echo "nvcc not installed (expected on RunPod)"
```

Expected output:
```
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Cuda compilation tools, release 12.8, V12.8.93
```

**Step 2: Verify CUDA runtime works (the important test)**

```bash
# Check if Python/PyTorch already installed (RunPod often has them)
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# If not installed, run:
# apt update && apt install -y python3 python3-pip
# pip3 install torch
```

Expected output:
```
PyTorch: 2.8.0+cu128
CUDA available: True
CUDA version: 12.8
```

If this prints `CUDA available: True` → **your CUDA stack is fully functional**. No toolkit installation needed.

---

**Step 3: GPU Health Check (Required)**

You've verified:
- ✔️ GPU driver (nvidia-smi)
- ✔️ CUDA runtime (torch.cuda.is_available())
- ✔️ PyTorch CUDA version

Now verify the GPU is **truly inference-ready** by testing deeper capabilities. Run these 4 tests in order:

**1️⃣ Check VRAM Allocation**

```bash
python3 << 'EOF'
import torch
x = torch.rand((1000,1000), device='cuda')
print("OK — Tensor allocated on:", x.device)
EOF
```

Expected:
```
OK — Tensor allocated on: cuda:0
```

> 🔍 **What's happening inside the GPU:**  
> - PyTorch requests a block of GPU memory from the CUDA driver  
> - The kernel allocates memory in VRAM and returns a device pointer  
> - The pointer is mapped to the GPU's global memory  
> 
> **If this fails → NOTHING else works** – not even loading a model.

**2️⃣ Check FP16 Matmul (Tensor Cores)**

This is critical for LLM inference – all modern models use half-precision.

```bash
python3 << 'EOF'
import torch
x = torch.randn((4096,4096), device='cuda', dtype=torch.float16)
y = x @ x
print("OK — FP16 matmul, dtype:", y.dtype)
EOF
```

Expected:
```
OK — FP16 matmul, dtype: torch.float16
```

> 🔍 **What's happening inside the GPU:**  
> - The 4096×4096 matrix multiply triggers **Tensor Core kernels**  
> - These run mixed-precision fused-multiply-add (FMA) operations  
> - Tensor Cores deliver **10×–50× the FLOPS** of standard CUDA cores  
> 
> **Why it matters:** Every modern LLM (Llama, Mistral, GPT) relies on FP16/BF16 matmuls.  
> If this fails → vLLM falls back to slow CUDA cores → awful throughput.

**3️⃣ Check cuDNN (Deep Learning Backend)**

cuDNN provides optimized primitives for convolutions, attention, and more.

```bash
python3 << 'EOF'
import torch.backends.cudnn as cudnn
print("cuDNN enabled:", cudnn.enabled)
print("cuDNN version:", cudnn.version())
EOF
```

Expected:
```
cuDNN enabled: True
cuDNN version: 90100
```

> 🔍 **What's happening inside the GPU:**  
> - cuDNN is NVIDIA's deep-learning library with optimized kernels  
> - Provides high-performance attention, convolution, and normalization ops  
> - PyTorch auto-selects these kernels when cuDNN is enabled  
> 
> **Without cuDNN:** Attention becomes slow, model loading is slower, training can be 3×–10× slower.  
> cuDNN = the "turbocharger" for the deep learning stack.

**4️⃣ GPU Matmul Benchmark**

This proves the GPU is operating at full speed.

```bash
python3 << 'EOF'
import torch, time
x = torch.randn((8000,8000), device='cuda')
torch.cuda.synchronize()
t0 = time.time()
y = x @ x
torch.cuda.synchronize()
ms = (time.time() - t0)*1000
tflops = (2 * 8000**3) / (ms/1000) / 1e12
print(f"Matmul time: {ms:.1f}ms ({tflops:.1f} TFLOPS)")
EOF
```

Expected (RTX 2000 Ada):
```
Matmul time: ~15-25ms (~40-70 TFLOPS)
```

> 🔍 **What's happening inside the GPU:**  
> - A massive 8000×8000 GEMM (general matrix multiplication) is dispatched  
> - Thread blocks are distributed across SMs (Streaming Multiprocessors)  
> - Warp schedulers issue FMA instructions every cycle  
> - Tensor Cores operate at high occupancy (near max throughput)  
> 
> **This is the same operation LLMs rely on for:**  
> - Attention score calculation  
> - Feedforward network layers  
> - KV-cache projection  
> 
> **If fast → GPU is delivering advertised FLOPS.**  
> If slow → something is wrong (clock throttling, driver mismatch, thermal limits).

---

**GPU Health Check Summary:**

| Step | Check | What It Verifies | Status |
|------|-------|------------------|--------|
| 1 | nvidia-smi | Driver works | ⬜ |
| 2 | torch.cuda.is_available() | Runtime works | ⬜ |
| 3 | Tensor on cuda:0 | VRAM allocation | ⬜ |
| 4 | FP16 matmul | Tensor Cores | ⬜ |
| 5 | cuDNN enabled | DL primitives | ⬜ |
| 6 | Matmul benchmark | Compute speed | ⬜ |

**Once all 6 pass → your GPU is 100% ready for LLM inference and fine-tuning.**

---

**Step 4 (Optional): Install nvcc locally without breaking drivers**

Only do this if you want to learn CUDA C++ compilation:

```bash
# Safe local install (doesn't touch drivers)
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --toolkit --silent --override

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

#### 🏆 Success Criteria
- [ ] Understand: Driver (nvidia-smi) vs Runtime (PyTorch CUDA) vs Toolkit (nvcc)
- [ ] `nvidia-smi` shows CUDA Version 12.x
- [ ] PyTorch reports `CUDA available: True`
- [ ] (Optional) `nvcc --version` works if you installed it

#### 📁 Artifacts
```bash
nvidia-smi | head -5 | tee ~/artifacts/cuda_stack.txt
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')" | tee -a ~/artifacts/cuda_stack.txt
```

---

## Tier 1 Summary – What You Verified

| Task | What You Did | How to Verify |
|------|--------------|---------------|
| **1.1** | Launch pod, SSH in, check OS | `cat /etc/os-release` |
| **1.2** | Install essential packages | `uname -r`, `which git` |
| **1.3** | Driver verification | `nvidia-smi` |
| **1.4** | CUDA stack understanding + GPU health checks | See table below |

### GPU Health Checks Completed

| # | Check | What It Verifies | Status |
|---|-------|------------------|--------|
| 1 | `nvidia-smi` | Driver works | ⬜ |
| 2 | `torch.cuda.is_available()` | Runtime works | ⬜ |
| 3 | Tensor on `cuda:0` | VRAM allocation | ⬜ |
| 4 | FP16 matmul | Tensor Cores | ⬜ |
| 5 | `cudnn.enabled` | cuDNN primitives | ⬜ |
| 6 | 8000x8000 matmul | Compute speed | ⬜ |

### Artifacts Created
```
~/artifacts/
├── os_info.txt
├── gpu_hardware.txt
├── system_info.txt
├── nvidia_smi_driver.txt
├── gpu_info.csv
└── cuda_stack.txt
```

---

## 🎯 Tier 1 Complete!

Your GPU is now **fully verified and inference-ready**:
- ✅ Ubuntu 24.04 running
- ✅ NVIDIA driver working (`nvidia-smi`)
- ✅ CUDA runtime functional (PyTorch sees GPU)
- ✅ All 6 GPU health checks passed

**Tier 1 stops here.** You understand the full GPU stack from OS to CUDA.

**Next:** Move to [Tier 2](LOG_tier02.md) to install vLLM and serve your first LLM API.

