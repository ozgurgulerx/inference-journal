# Day 005 – OS & NUMA Node Hardening for Inference
## Tier 1: Core – NUMA & CPU-Aware Baseline (~2 hours)

> **Goal**: Quantify how NUMA + CPU pinning affect latency/throughput for a simple inference server on your GPU node.  
> **End State**: Topology captured, baseline vs pinned benchmarks recorded, with clear hypotheses for later vLLM tuning.

---

## Tier 1 Tasks

---

### ✅ Task 1.1: Capture CPU/NUMA/GPU Topology

**Tags**: `[OS–Linux]` `[NUMA]` `[GPU/HW]`  
**Time**: ~20–30 min  
**Win**: You know which NUMA node and cores are “closest” to your GPU and have artifacts to refer back to later.

#### 🎯 Objective

Record the hardware topology so all future inference experiments can be interpreted in context.

#### 🔧 Lab Instructions

1. Create a small topo folder in your home or repo:

   ```bash
   mkdir -p ~/topo
   ```

2. Capture CPU and NUMA layout:

   ```bash
   lscpu -e > ~/topo/lscpu_day05.txt
   numactl --hardware > ~/topo/numa_day05.txt
   ```

3. (Optional but nice) If `lstopo` is available:

   ```bash
   lstopo-no-graphics > ~/topo/lstopo_day05.txt
   ```

4. Inspect the output and note:
   - Which NUMA node the main GPU is attached to.  
   - Which cores belong to that NUMA node.

#### ✅ Acceptance Criteria

- [ ] `~/topo/lscpu_day05.txt` and `~/topo/numa_day05.txt` committed or tracked.  
- [ ] You can say “GPU is on NUMA node X, nearest cores are Y–Z.”

---

### ✅ Task 1.2: Define a Simple Inference Benchmark Target

**Tags**: `[Inference–Runtime]` `[Benchmarks]`  
**Time**: ~20–30 min  
**Win**: One small, repeatable benchmark (server + client) you can reuse today and in future days.

#### 🎯 Objective

Set up a minimal inference server and client pair so you can measure the effect of OS tuning.

#### 🔧 Lab Instructions

1. Pick a runtime for today:
   - Preferred: existing **vLLM** server with a small model.  
   - Alternative: HF `transformers` model served via a tiny FastAPI/uvicorn or Flask app.

2. Ensure you have:
   - A **server** script (e.g. `serve.py` or a vLLM command) that you can run once and keep alive.  
   - A **client** script (e.g. `bench_client.py`) that:
     - sends N requests (e.g. 50),
     - optionally with concurrency (e.g. 4),
     - prints avg and p95 latency and a throughput estimate (req/s or tokens/s).

3. Place these under a consistent path, e.g.:
   - `code/day005/serve_baseline.py`  
   - `code/day005/bench_client.py`

#### ✅ Acceptance Criteria

- [ ] You can run a command like `python code/day005/bench_client.py --n-requests 50 --concurrency 4` and get latency/throughput metrics.  
- [ ] Scripts are small, readable, and clearly tied to Day 005.

---

### ✅ Task 1.3: Baseline vs Pinned Inference Benchmarks

**Tags**: `[OS–Linux]` `[NUMA]` `[Inference–Runtime]` `[Benchmarks]`  
**Time**: ~60 min  
**Win**: A direct comparison of baseline vs NUMA+CPU-pinned inference behavior on your node.

#### 🎯 Objective

Run the same workload twice—once with default scheduling, once with explicit NUMA/CPU pinning—and compare metrics.

#### 🔧 Lab Instructions

1. **Baseline run (no pinning)**

   - Start the server normally:

     ```bash
     python code/day005/serve_baseline.py  # or your vLLM command
     ```

   - In another shell, start GPU monitoring:

     ```bash
     nvidia-smi dmon -s pucvmt -d 1 -o TD > ~/topo/nvidia_dmon_baseline_day05.log
     ```

   - Run the benchmark:

     ```bash
     python code/day005/bench_client.py --n-requests 50 --concurrency 4
     ```

   - Capture:
     - avg latency, p95 latency  
     - throughput (req/s or tokens/s)

2. **Pinned run (NUMA + CPU affinity)**

   - Stop the baseline server.  
   - Choose a NUMA node and CPU range (e.g. node 0, cores 0–15).  
   - Start the server pinned:

     ```bash
     numactl --cpunodebind=<GPU_NODE> --membind=<GPU_NODE> \
       taskset -c <CPU_RANGE> \
       python code/day005/serve_baseline.py
     ```

   - Start GPU monitoring again:

     ```bash
     nvidia-smi dmon -s pucvmt -d 1 -o TD > ~/topo/nvidia_dmon_pinned_day05.log
     ```

   - Re-run the same benchmark:

     ```bash
     python code/day005/bench_client.py --n-requests 50 --concurrency 4
     ```

3. **Summarize in a simple markdown table**

   - Create `benchmarks/day05_numa_baseline.md` with at least:

     | Mode      | Concurrency | Avg Latency (ms) | p95 Latency (ms) | Throughput | Notes |
     |-----------|-------------|------------------|------------------|-----------|-------|
     | Baseline  | 4           |                  |                  |           |       |
     | Pinned    | 4           |                  |                  |           |       |

   - Add 3–5 bullets under “Observations”:
     - Did p95 improve or worsen?  
     - Any change in throughput?  
     - Any surprises in GPU utilization?

#### ✅ Acceptance Criteria

- [ ] Two successful benchmark runs (baseline + pinned) with recorded metrics.  
- [ ] `benchmarks/day05_numa_baseline.md` exists with a comparison table and short observations.  
- [ ] `~/topo/nvidia_dmon_baseline_day05.log` and `~/topo/nvidia_dmon_pinned_day05.log` captured.

---

### 📝 Feynman Deliverable (Tier 1)

At the bottom of `benchmarks/day05_numa_baseline.md`, add a short section:

> **Feynman: What I Learned About NUMA & Inference Today**  
> 5–10 sentences on:
> - how your GPU, NUMA node, and cores relate,  
> - what pinning did to latency/throughput,  
> - and how you expect this to matter when you start tuning vLLM continuous batching and KV cache later.

