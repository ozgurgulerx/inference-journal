# Day 006 – SLM + OS Memory & First-Token Path
## Tier 2 – Allocator & vLLM First-Token Latency (SLM as Probe)

> **Goal**:  
> 1. See how `glibc` vs `jemalloc` affects SLM generation latency.  
> 2. Bridge into vLLM: measure first-token latency + warm-start behavior for the **same SLM**.

---

## Tier 2 – Deepen (If Time/Energy Allow)

**Title** – Allocator & vLLM First-Token Latency (SLM as probe)  
**Time Budget** – ~90–120 min

---

### A. Allocator Impact on Direct HF SLM Generation (~45–60 min)

#### 1. Install jemalloc (if not already)

```bash
sudo apt-get update
sudo apt-get install -y libjemalloc2
```

#### 2. Create a Tiny SLM Generation Script

`days/day-006-slm-memory/slm_gen_latency.py`:

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/Phi-3-mini-4k-instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
)

prompt = "Explain NUMA-aware inference in 2 short bullet points."
inputs = tok(prompt, return_tensors="pt").to(model.device)

torch.cuda.synchronize()
t0 = time.time()
out = model.generate(**inputs, max_new_tokens=64)
torch.cuda.synchronize()
print("gen_latency_s", time.time() - t0)
print(tok.decode(out[0], skip_special_tokens=True))
```

#### 3. Run with Default Allocator vs jemalloc

From `days/day-006-slm-memory`:

```bash
/usr/bin/time -f "glibc_real_s=%E" python slm_gen_latency.py

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
/usr/bin/time -f "jemalloc_real_s=%E" python slm_gen_latency.py
```

Capture from output:

- `gen_latency_s` (printed).  
- `glibc_real_s` / `jemalloc_real_s`.  
- Optionally RSS via:

  ```bash
  ps -C python -o pid,rss,vsz,cmd | head -n 5
  ```

#### 4. Write a Quick CSV

`days/day-006-slm-memory/allocator_latency_comparison.csv`:

```text
allocator,gen_latency_s,wall_real_s,rss_mb
glibc, ... , ... , ...
jemalloc, ... , ... , ...
```

Add 3–5 bullets in `README.md` summarizing whether jemalloc helped (and by how much).

---

### B. vLLM First-Token Latency & Warm Start for SLM (~45–60 min)

> These steps conceptually belong to “Day 7 core” but are executed here so that the **same SLM** probes both HF and vLLM.

#### 1. Serve the Same SLM via vLLM

From repo root or a new folder:

```bash
mkdir -p days/day-007-vllm-slm
cd days/day-007-vllm-slm

python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92
```

Keep this server running in one terminal.

#### 2. Measure Cold vs Warm Request

In another terminal:

```bash
# cold
/usr/bin/time -f "cold_req_real_s=%E" curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "test cold",
    "max_tokens": 16
  }' > /tmp/vllm_cold.json

# warm (immediately after)
/usr/bin/time -f "warm_req_real_s=%E" curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "test warm",
    "max_tokens": 16
  }' > /tmp/vllm_warm.json
```

Record both wall times and, if possible, approximate TTFT from logs.

#### 3. Track GPU Memory Before/After

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

- Start this before the cold request.  
- Let it run through the warm request.  
- Note min/max memory and any jumps between cold and warm.

#### 4. Document First-Token Behavior

Create `days/day-007-vllm-slm/first_token_latency.md` with:

- Cold vs warm wall times (`cold_req_real_s`, `warm_req_real_s`).  
- Any clear drop from first to second request.  
- Rough memory-used timeline from `nvidia-smi`.  
- A paragraph on how you expect this to matter for real SLOs (e.g., first-request penalty, warm pool behavior).

---

### Tier 2 Artifacts

- `days/day-006-slm-memory/slm_gen_latency.py`  
- `days/day-006-slm-memory/allocator_latency_comparison.csv`  
- `days/day-007-vllm-slm/first_token_latency.md`

