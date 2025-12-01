# Chapter 2: OS Essentials

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Content covers OS-level configurations for high-performance LLM serving.

---

## In This Chapter

- [Overview](#overview)
- [1. Supported Operating Systems](#1-supported-operating-systems)
- [2. Kernel and System-Level Optimizations](#2-kernel-and-system-level-optimizations)
- [3. GPU Driver and Toolkit Setup](#3-gpu-driver-and-toolkit-setup)
- [4. TPU Setup](#4-tpu-setup)
- [5. LLM Serving Frameworks](#5-llm-serving-frameworks)
- [6. Containerization Best Practices](#6-containerization-best-practices)
- [7. Kubernetes Support](#7-kubernetes-support)
- [8. Ray for Distributed Inference](#8-ray-for-distributed-inference)
- [Quick Reference: Performance Checklist](#quick-reference-performance-checklist)

---

## Overview

Deploying large language models for high-throughput, low-latency inference requires careful tuning of the operating system, hardware drivers, and serving frameworks. This chapter details the OS-level tweaks, installations, and configurations needed to serve LLMs efficiently on cutting-edge GPUs and TPUs.

---

## 1. Supported Operating Systems

### Recommended Distributions

High-performance LLM serving is best supported on **64-bit Linux distributions**:

| Distribution | Use Case | Notes |
|-------------|----------|-------|
| **Ubuntu LTS** (20.04, 22.04, 24.04) | General purpose, DGX | Strong NVIDIA support, frequent updates |
| **RHEL / Rocky Linux / AlmaLinux** | Enterprise/HPC | Stability and long-term support |
| **SLES** | Some HPC contexts | Supported in specific environments |

### Why Linux?

- CUDA, PyTorch, and ML ecosystem are most mature on Linux
- Most optimization tools assume Linux environments
- NVIDIA's Data Center drivers are primarily developed for Linux

### Windows Considerations

While Windows Server supports NVIDIA GPUs, production LLM serving on Windows is uncommon:
- Consider **WSL2** for compatibility with Linux-based ML libraries
- Ensure NVIDIA Data Center driver for Windows is installed
- Performance may be suboptimal compared to native Linux

---

## 2. Kernel and System-Level Optimizations

Default kernel settings are often not optimal for LLM inference. Key optimizations include:

### CPU Frequency Governor

Set CPUs to performance mode to avoid frequency scaling latency:

```bash
# Set performance governor
sudo cpupower frequency-set -g performance

# Or for all cores
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" | sudo tee $cpu
done
```

This prevents the OS from throttling CPU speed during inference preprocessing or I/O tasks.

### CPU Pinning and Isolation

For latency-sensitive workloads:

```bash
# Pin process to specific cores
taskset -c 0-7 python serve.py

# Or use numactl
numactl --cpunodebind=0 --membind=0 python serve.py
```

**In Kubernetes**, set `cpuManagerPolicy: static` to achieve core pinning.

### NUMA Awareness

On multi-socket servers, GPUs and NICs are attached to specific NUMA nodes:

```bash
# Disable automatic NUMA balancing
sudo sysctl -w kernel.numa_balancing=0

# Make persistent
echo "kernel.numa_balancing=0" | sudo tee -a /etc/sysctl.conf
```

Ensure CPU threads feeding a GPU run on the same NUMA node as that GPU:

```bash
# Check GPU NUMA affinity
nvidia-smi topo -m

# Bind to matching NUMA node
numactl --cpunodebind=0 python serve.py
```

### Memory & HugePages

LLMs are memory-hungry. Optimize TLB usage:

```bash
# Check current hugepages
cat /proc/meminfo | grep Huge

# Enable Transparent Huge Pages (THP)
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Or allocate static hugepages at boot
# Add to /etc/default/grub: GRUB_CMDLINE_LINUX="default_hugepagesz=1G hugepages=16"
```

**In Kubernetes**, request hugepages for pods:

```yaml
resources:
  limits:
    hugepages-2Mi: 1Gi
```

### I/O Scheduler

For NVMe SSDs storing model data:

```bash
# Use none/noop scheduler for NVMe
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
```

---

## 3. GPU Driver and Toolkit Setup

### NVIDIA Driver Installation

```bash
# Ubuntu - recommended method
sudo apt update
sudo apt install nvidia-driver-535

# Verify installation
nvidia-smi
```

**Critical settings**:

```bash
# Enable persistence mode (reduces first-use latency)
sudo nvidia-smi -pm 1

# Set power mode for maximum performance
sudo nvidia-smi -pl 400  # Set power limit (adjust for your GPU)
```

### CUDA Toolkit

```bash
# Install CUDA toolkit (example for CUDA 12.2)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2

# Set environment variables
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Essential Libraries

| Library | Purpose | Installation |
|---------|---------|--------------|
| **cuDNN** | Optimized tensor operations | Download from NVIDIA Developer |
| **cuBLAS** | Dense linear algebra | Included with CUDA |
| **TensorRT** | Optimized inference | `pip install tensorrt` or NGC container |
| **NCCL** | Multi-GPU communication | `apt install libnccl2` |

### Multi-Instance GPU (MIG)

For A100/H100 GPUs, partition into smaller instances:

```bash
# Enable MIG mode
sudo nvidia-smi -mig 1

# Create MIG instances (example: 7 x 1g.5gb on A100)
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C

# List instances
nvidia-smi mig -lgi
```

**Use MIG when**:
- Serving multiple smaller models
- Need isolation between workloads
- Model doesn't require full GPU

---

## 4. TPU Setup

### Cloud TPU VM (v4/v5)

TPU v4+ are cloud-only — no on-premises deployment available.

```bash
# Create TPU VM (example)
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b
```

### Framework Installation for TPU

```bash
# Install JAX for TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Or PyTorch XLA
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

### TPU Optimization Tips

1. **Pin to NUMA node 0** for CPU-intensive preprocessing
2. **Use large batch sizes** to fully utilize matrix units
3. **Leverage XLA compilation** for optimized graphs
4. **Use BF16** (TPUs are optimized for it)

---

## 5. LLM Serving Frameworks

### Framework Comparison

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| **vLLM** | PagedAttention, high throughput | Many concurrent users |
| **TGI** | Production-ready, robust | Enterprise deployments |
| **TensorRT-LLM** | Maximum NVIDIA performance | H100/A100 optimization |
| **DeepSpeed** | Large models, multi-GPU | 70B+ parameter models |

### vLLM Installation

```bash
pip install vllm

# Serve a model
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

### Text Generation Inference (TGI)

```bash
# Docker method (recommended)
docker run --gpus all \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --num-shard 1
```

### DeepSpeed Inference

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16
)

# Initialize DeepSpeed inference
ds_engine = deepspeed.init_inference(
    model,
    mp_size=1,  # Number of GPUs
    dtype=torch.float16,
    replace_method='auto'
)
```

---

## 6. Containerization Best Practices

### Use GPU-Optimized Base Images

```dockerfile
# Recommended base images
FROM nvcr.io/nvidia/pytorch:24.01-py3
# or
FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04
```

### NVIDIA Container Toolkit

```bash
# Install on host
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run with GPU access
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Optimized Dockerfile Example

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install vLLM
RUN pip install --no-cache-dir vllm

# Set environment for performance
ENV OMP_NUM_THREADS=1
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# Run as non-root for security
RUN useradd -m -u 1000 llmuser
USER llmuser

EXPOSE 8000
CMD ["vllm", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Singularity/Apptainer for HPC

```bash
# Convert Docker image to Singularity
singularity build vllm.sif docker://vllm/vllm-openai:latest

# Run with GPU
singularity run --nv vllm.sif vllm serve model-name
```

---

## 7. Kubernetes Support

### GPU Scheduling

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: 1
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
```

### NVIDIA GPU Operator

```bash
# Install GPU Operator via Helm
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace
```

### Performance-Aware Scheduling

Enable Topology Manager for NUMA-aware scheduling:

```yaml
# kubelet config
cpuManagerPolicy: static
topologyManagerPolicy: best-effort
```

### MIG in Kubernetes

```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1  # Request specific MIG profile
```

---

## 8. Ray for Distributed Inference

Ray is a distributed computing framework often used with vLLM for multi-node deployments.

### Ray Installation

```bash
pip install ray[default]
```

### Starting a Ray Cluster

```bash
# Head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Worker nodes
ray start --address=<head-ip>:6379
```

### Ray with vLLM

```bash
# Serve model across Ray cluster
vllm serve large-model \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray
```

### Ray Serve for LLM Endpoints

```python
from ray import serve
from vllm import LLM

@serve.deployment(ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self):
        self.llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    
    async def __call__(self, request):
        prompt = await request.json()
        return self.llm.generate(prompt["text"])

app = LLMDeployment.bind()
serve.run(app)
```

### KubeRay for Kubernetes

```yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: vllm-cluster
spec:
  headGroupSpec:
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:latest
          resources:
            limits:
              nvidia.com/gpu: 1
  workerGroupSpecs:
  - replicas: 2
    template:
      spec:
        containers:
        - name: ray-worker
          resources:
            limits:
              nvidia.com/gpu: 1
```

---

## Quick Reference: Performance Checklist

| Setting | Command/Config | Impact |
|---------|----------------|--------|
| CPU Governor | `cpupower frequency-set -g performance` | Consistent CPU speed |
| NUMA Balancing | `sysctl -w kernel.numa_balancing=0` | Reduce page migrations |
| GPU Persistence | `nvidia-smi -pm 1` | Faster first inference |
| Hugepages | `/etc/default/grub` modification | Better TLB efficiency |
| I/O Scheduler | `echo none > /sys/.../scheduler` | Faster model loading |

---

<p align="center">
  <a href="01-introduction.md">← Previous: Introduction</a> | <a href="../README.md">Table of Contents</a> | <a href="03-vllm-architecture.md">Next: vLLM Architecture →</a>
</p>
