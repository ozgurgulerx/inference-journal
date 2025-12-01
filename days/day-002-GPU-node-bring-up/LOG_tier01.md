# Day 002 – Initial Setup and Structure

## Goal
Tier 1 – Must-Do Core Block (~2 hours)
Goal: Turn a fresh Ubuntu GPU VM into a usable inference node + run one small model end-to-end, and commit the trace.

### Task 1: Bring up a GPU node and verify basics
Tags: [OS–Linux] [OS-01]
Time: 45m

19:22 Created a spot VM e2-mini, 20 GB Disk with a T4 card on us-central-01.

```bash
# Install gcloud (if not already)
brew install --cask google-cloud-sdk   # or follow Google’s installer
gcloud init                            # login, pick project, default region/zone
gcloud auth login
```

```bash
gcloud config set project vaulted-blend-456507-a2
```

```bash
gcloud org-policies describe constraints/compute.vmExternalIpAccess \
  --project=vaulted-blend-456507-a2
```

```bash
gcloud compute instances create oz-t4 \
  --zone=us-central1-b \
  --machine-type=n1-standard-2 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  --no-restart-on-failure
```

```bash
# Spot instance version
gcloud compute instances create oz-t4 \
  --zone=us-central1-b \
  --machine-type=n1-standard-1 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  --no-restart-on-failure \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP
```

```bash
gcloud compute instances list
gcloud compute ssh instance-20251118-160537 --zone=us-central1-b
```

22:00 I am in the VM 

```bash
cat /etc/os-release
```

```
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
VERSION="12 (bookworm)"
VERSION_CODENAME=bookworm
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"
```




Lab instructions (Ubuntu 22.04 GPU VM, e.g. GCP/Lambda):
SSH into your GPU VM.

**System prep:**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms git curl wget
uname -a | tee system_info.txt
lspci | grep -i nvidia | tee -a system_info.txt
```

**Install NVIDIA driver** (if not preinstalled; if cloud image already has it, just validate):

```bash
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

**After reboot, verify:**

```bash
nvidia-smi | tee nvidia_smi_initial.txt
```
Create a bootstrap notes file: notes/day02_os01_gpu_node_bringup.md and record:
VM type, GPU type, OS version, driver version.
Any issues/errors and how you fixed them.

**Artifact (GitHub):**
- `notes/day02_os01_gpu_node_bringup.md`
- `system_info.txt`
- `nvidia_smi_initial.txt`

**Commit message:**
```
day02: baseline GPU node bring-up (OS-01)
```

Task 2: Install CUDA Toolkit + basic verification
Tags: [OS–Linux] [OS-01]

Time: 35m

**Lab instructions:**

Install CUDA toolkit via apt (if not already tied to image):

```bash
sudo apt install -y nvidia-cuda-toolkit
nvcc --version | tee nvcc_version.txt
```

Create a tiny CUDA test:

```bash
mkdir -p cuda-tests && cd cuda-tests
cat > vec_add.cu << 'EOF'
#include <stdio.h>
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) y[index] = x[index] + y[index];
}
int main(void) {
    int N = 1 << 20;
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    for (int i = 0; i < N; i++) { x[i] = 1.0f; y[i] = 2.0f; }
    add<<<(N+255)/256, 256>>>(N, x, y);
    cudaDeviceSynchronize();
    printf("y[0] = %f\n", y[0]);
    cudaFree(x); cudaFree(y);
    return 0;
}
EOF
```

```bash
nvcc vec_add.cu -o vec_add
./vec_add | tee cuda_vec_add_output.txt
```

**Log results in** `notes/day02_os01_gpu_node_bringup.md`:
- CUDA version
- Whether test succeeded
- Any errors you hit

**Artifact (GitHub):**
- `cuda-tests/vec_add.cu`
- `cuda-tests/cuda_vec_add_output.txt`
- `nvcc_version.txt`

**Commit:**
```
day02: install CUDA toolkit and verify GPU compute (OS-01)
```

Task 3: Minimal vLLM/HF environment & a single prompt run

Tags: [Inference–Runtime] [Phase1-2 HF_vs_vLLM]

Time: 40m

**Lab instructions:**

Back at home directory:

```bash
mkdir -p inference-tests && cd inference-tests
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch" "transformers" "accelerate" "vllm"
```

Create a simple HF vs vLLM sanity script:

```bash
cat > day02_hf_vllm_sanity.py << 'EOF'
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"  # keep tiny for now
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name} on {device}...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print(f"Model loaded in {time.time() - t0:.2f}s")

prompt = "You are an LLM inference engineer. In one sentence, explain why batching helps."
inputs = tok(prompt, return_tensors="pt").to(device)

t1 = time.time()
out = model.generate(**inputs, max_new_tokens=32)
t2 = time.time()

print(tok.decode(out[0], skip_special_tokens=True))
print(f"Generation time: {t2 - t1:.2f}s")
EOF
```

```bash
python day02_hf_vllm_sanity.py | tee day02_hf_vllm_sanity.log
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader \
   | tee day02_hf_gpu_usage.csv
```

**Append to** `notes/day02_os01_gpu_node_bringup.md`:
- Model name
- Generation time
- GPU memory/usage snapshot

**Artifact (GitHub):**
- `inference-tests/day02_hf_vllm_sanity.py`
- `inference-tests/day02_hf_vllm_sanity.log`
- `inference-tests/day02_hf_gpu_usage.csv`

**Commit:**
```
day02: first HF inference on self-managed GPU node (Phase1 baseline)
```



## Setup
- Hardware: TODO
- Model(s): TODO
- Framework(s): TODO
- Precision: TODO
- Max tokens: TODO

## Experiments
- TODO: Add experiments run today with bullet points and commands.

## Key results (summary)
- TODO: Short bullet list of headline findings.

## Insights
- TODO: What surprised you? What broke? Any quick rules of thumb?

## Next steps
- TODO: Add the next concrete actions for tomorrow.

