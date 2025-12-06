# Day 005 – NUMA- & CPU-Aware Inference Node Hardening

*Layer focus*: **Hardware & OS → Inference Runtime**  
*Themes*: **Heterogeneous serving (node/topology), Reliability & safety**, early **multi-tenancy/QoS** instincts  

---

## 1. Goals for Day 5

- Build a **NUMA- and CPU-aware baseline** for running an inference server on a single GPU node.  
- Quantify how **CPU/NUMA pinning** and **CPU governor choices** influence latency, p95, and throughput.  
- Take the first step toward a **consultancy-grade “Inference Node OS & NUMA Hardening” module**.

By the end of the day you should be able to:

- Explain which NUMA node and cores are “closest” to your GPU.  
- Show baseline vs pinned latency/throughput numbers.  
- Describe, in business language, how these OS-level changes could save GPUs or meet SLAs.

---

## 2. Hands-On Labs

### Lab 5.1 – NUMA Topology & Baseline vs Pinned Server

**Goal**: Measure the impact of NUMA + CPU pinning on a simple inference workload.

1. **Inspect topology**
   - Run and save:
     - `lscpu -e > topo/lscpu_day05.txt`
     - `numactl --hardware > topo/numa_day05.txt`
     - Optional: `lstopo-no-graphics > topo/lstopo_day05.txt`
   - Identify:
     - GPU’s NUMA node.
     - Core IDs on that NUMA node.

2. **Pick a benchmark target**
   - Either:
     - vLLM server (preferred), or
     - HF `transformers` model + simple HTTP API.
   - Ensure you have:
     - One long-lived **server** process.
     - One **client** script (e.g. `bench_client.py`) that sends N requests and reports avg + p95 latency and throughput.

3. **Baseline run (no pinning)**
   - Start server without pinning.  
   - Run: `python bench_client.py --n-requests 50 --concurrency 4`.  
   - Capture:
     - Latency/throughput summary.
     - `nvidia-smi dmon ... > topo/nvidia_dmon_baseline_day05.log`.  
   - Log numbers into `benchmarks/day05_numa_baseline.md`.

4. **Pinned run (NUMA + CPU affinity)**
   - Start server with:
     ```bash
     numactl --cpunodebind=<GPU_NODE> --membind=<GPU_NODE> \
       taskset -c <CPU_RANGE> \
       python -m vllm.entrypoints.openai.api_server ...
     ```
   - Re-run the same benchmark.  
   - Capture metrics + `nvidia-smi dmon` as `topo/nvidia_dmon_pinned_day05.log`.  
   - Extend `day05_numa_baseline.md` with a table: baseline vs pinned.

5. **Quick interpretation**
   - 3–5 bullets:
     - Which metrics improved/degraded and by how much.  
     - Hypotheses (e.g. CPU-bound vs GPU-bound behavior).  
     - How you expect this to interact with **vLLM continuous batching** later.

**Artifacts**:
- `topo/lscpu_day05.txt`, `topo/numa_day05.txt`, `topo/lstopo_day05.txt`  
- `topo/nvidia_dmon_baseline_day05.log`, `topo/nvidia_dmon_pinned_day05.log`  
- `benchmarks/day05_numa_baseline.md`

---

### Lab 5.2 – Systemd / cgroup-Aware Inference Service (Optional)

**Goal**: Encode your NUMA/CPU choices into a **reproducible service definition**.

1. Create a template service: `systemd/inference-vllm.service` with:
   - `ExecStart` using `numactl` + `taskset`.  
   - `CPUAffinity=` for your chosen core range.  
   - `Restart=always` basics.

2. (Optional) Add a dedicated slice/cgroup for inference and pin CPU/memory there.

3. Start the service via systemd and re-run a short benchmark to verify behavior matches the pinned run.

4. Document a short “How to deploy pinned vLLM via systemd” checklist.

**Artifacts**:
- `systemd/inference-vllm.service`  
- Optional `docs/day05_numa_and_systemd.md`

---

### Lab 5.3 – CPU Governor & Noisy Neighbor Micro-Experiments (Stretch)

**Goal**: Understand how CPU governors and noisy neighbors affect p95, and how NUMA pinning mitigates it.

1. Create a short helper script: `scripts/set_governor.sh` to switch CPU governors.  
2. Benchmark inference under `performance` vs `schedutil` (and `powersave` if available).  
3. Introduce a noisy neighbor using `stress-ng` or a CPU-bound script on:
   - same NUMA node, and  
   - other NUMA node.  
4. Measure p50/p95 with/without noise, pinned vs unpinned.  
5. Summarize impact and choose a recommended governor + pinning policy for “inference nodes”.

**Artifacts**:
- `scripts/set_governor.sh`  
- Extended sections in `benchmarks/day05_numa_baseline.md` (“CPU governor impact”, “noisy neighbor impact”)

---

## 3. Expert & Consultancy Angle (How Day 5 Turns into a Service)

Use today’s work to seed a future **“Inference Node OS & NUMA Hardening”** module:

- In `docs/os_node_os_tuning_summary_day05.md`, capture:
  - Engineering view: what changed technically (topology, pinning, governors, noisy neighbors).  
  - Consultant view:
    - Current node risk: Low / Medium / High.  
    - Key issues (e.g. “inference not pinned to GPU-local cores; p95 inflates under CPU noise”).  
    - Recommendations and **expected SLA / cost impact** (e.g. fewer GPUs needed to hit p95).

- Extract a client checklist:
  - Are inference workers on GPU-local cores/NUMA?  
  - Is the CPU governor appropriate for latency-sensitive workloads?  
  - How bad is noisy-neighbor impact today?  

This makes Day 5 not just a lab, but a building block for an **Inference Health Check** and **Optimization Sprint** offering.

---

## 4. Off-Hours Reading (Optional)

- `man numactl`, `man taskset` – formalize what you used in the labs.  
- Linux perf & scheduler docs (`man perf-stat`, `man perf-sched`) and a short blog/guide on using `perf stat` for latency analysis – to connect CPU migrations/context switches to p95 behavior.  
- A “NUMA and performance tuning” write-up from a reputable source (e.g., Red Hat / SUSE / Intel) – to see how production teams think about local vs remote memory and noisy neighbors.  
- An overview of cgroups v2 and systemd resource control (e.g., `systemd.resource-control` man page) – to prepare for expressing your pinning policies as slices/units instead of ad-hoc commands.  
- vLLM deployment docs, especially flags that interact with memory/batching – to line up your OS-level tuning with future capacity/quantization experiments.
