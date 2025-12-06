# Day 005 – OS & NUMA Node Hardening for Inference
## Tier 3: Deep Work – Governors, Noisy Neighbors & Consulting Hook (~2–3 hours)

> **Prerequisites**: Tiers 1–2 completed  
> **Goal**: Understand how CPU governors and noisy neighbors affect p95, and package your findings as an “Inference Node OS & NUMA Hardening” module for future clients.

---

## Tier 3 Tasks

---

### ✅ Task 3.1: CPU Governor Micro-Benchmark

**Tags**: `[OS–Linux]` `[CPU/Governor]` `[Benchmarks]`  
**Time**: ~45–60 min  
**Win**: Data showing how `performance` vs `schedutil` (and optionally `powersave`) change latency/throughput for your workload.

#### 🎯 Objective

Benchmark your pinned inference server under different CPU governors.

#### 🔧 Lab Instructions

1. Create a small helper script:

   ```bash
   mkdir -p scripts
   cat > scripts/set_governor.sh << 'EOF'
   #!/usr/bin/env bash
   set -euo pipefail
   GOV=${1:-performance}
   for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
     echo "$GOV" | sudo tee "$cpu/cpufreq/scaling_governor" >/dev/null
   done
   echo "Set governor to $GOV"
   EOF
   chmod +x scripts/set_governor.sh
   ```

2. For each governor (`performance`, `schedutil`, `powersave` if available):
   - Switch governor:

     ```bash
     sudo ./scripts/set_governor.sh performance
     ```

   - Ensure your systemd inference service is running.  
   - Run:

     ```bash
     python code/day005/bench_client.py --n-requests 50 --concurrency 4
     ```

   - Record avg and p95 latency + throughput in `benchmarks/day05_numa_baseline.md` under a “CPU governor impact” section.

3. Note any trends:
   - Does `performance` clearly tighten p95?  
   - How much do you lose with `schedutil` or gain in power savings?

#### ✅ Acceptance Criteria

- [ ] `scripts/set_governor.sh` exists and works.  
- [ ] CPU governor experiment results appended to `benchmarks/day05_numa_baseline.md`.  
- [ ] One sentence recommendation: which governor you’d choose for an inference node and why.

---

### ✅ Task 3.2: Noisy Neighbor Experiment

**Tags**: `[OS–Linux]` `[Multi-Tenancy]` `[QoS]`  
**Time**: ~45–60 min  
**Win**: Evidence for how co-located CPU load affects your inference p95, and how much NUMA/CPU pinning helps.

#### 🎯 Objective

Simulate noisy neighbors and measure their impact on your pinned vs unpinned inference server.

#### 🔧 Lab Instructions

1. Install `stress-ng` (or use a small CPU-bound Python loop):

   ```bash
   sudo apt-get update && sudo apt-get install -y stress-ng
   ```

2. Choose two scenarios:
   - **Same NUMA node** as GPU-local cores.  
   - **Other NUMA node** (if you have >1 NUMA).

3. For each scenario and governor (focus on your chosen one, e.g. `performance`):
   - Start noisy neighbor:

     ```bash
     stress-ng --cpu 8 --cpu-method matrixprod --timeout 60s
     ```

   - Run your benchmark during the noise:

     ```bash
     python code/day005/bench_client.py --n-requests 50 --concurrency 4
     ```

   - Compare:
     - p50, p95 vs no-noise case.  
     - Pinned vs unpinned behavior (if you have time to test both).

4. Add a “Noisy neighbor impact” section to `benchmarks/day05_numa_baseline.md`:
   - Small table summarizing p95 with/without noise, pinned vs unpinned.  
   - 3 bullets on what this suggests for multi-tenant deployments.

#### ✅ Acceptance Criteria

- [ ] At least one noisy-neighbor scenario benchmarked.  
- [ ] Qualitative understanding of how much pinning protects you against CPU noise.

---

### ✅ Task 3.3: perf & Scheduler-Level Profiling (Advanced)

**Tags**: `[OS–Linux]` `[perf]` `[Scheduler]` `[Benchmarks]`  
**Time**: ~45–60 min  
**Win**: Evidence for *why* pinning changes behavior (context switches, migrations, cache misses), not just that it does.

#### 🎯 Objective

Use `perf` to compare baseline vs pinned inference in terms of scheduler and micro-architectural signals.

#### 🔧 Lab Instructions

1. Install perf tools if needed:

   ```bash
   sudo apt-get update && sudo apt-get install -y linux-tools-common linux-tools-$(uname -r)
   ```

2. With your inference server running (baseline vs pinned, one at a time), grab a short `perf stat` snapshot:

   ```bash
   # Replace <PID> with your server PID
   sudo perf stat -d -d -d -p <PID> -- sleep 30
   ```

3. Capture and compare metrics for both modes:
   - `context-switches`, `cpu-migrations`  
   - `LLC-load-misses`, `branches`, `branch-misses`  
   - Any changes in instructions per cycle (IPC).

4. Optionally, run:

   ```bash
   sudo perf sched record -- sleep 30
   sudo perf sched latency
   ```

   to see if long run-queue latency correlates with p95 spikes.

5. Add a “perf / scheduler signals” section to `benchmarks/day05_numa_baseline.md` summarizing:
   - “Pinned reduced context switches from X → Y, cpu-migrations from A → B”  
   - Any noticeable change in LLC miss rate.

#### ✅ Acceptance Criteria

- [ ] At least one `perf stat` run captured for baseline and pinned modes.  
- [ ] Short written interpretation of what changed and how it explains your latency results.

---

### ✅ Task 3.4: NUMA-Local vs Remote Memory Access (numastat)

**Tags**: `[OS–Linux]` `[NUMA]` `[Memory]`  
**Time**: ~30–45 min  
**Win**: Validate that your pinned server is actually using local memory, not bouncing across NUMA nodes.

#### 🎯 Objective

Use `numastat` (and optionally `perf mem`) to see local vs remote memory accesses for your inference process.

#### 🔧 Lab Instructions

1. Install numastat if needed:

   ```bash
   sudo apt-get install -y numactl
   ```

2. For baseline and pinned runs, capture:

   ```bash
   numastat -p <PID> > ~/topo/numastat_baseline_day05.txt
   # And for pinned:
   numastat -p <PID> > ~/topo/numastat_pinned_day05.txt
   ```

3. Compare:
   - Local vs remote page counts for each NUMA node.  
   - Whether pinned mode reduces remote allocations.

4. Add a small “NUMA locality” subsection to `benchmarks/day05_numa_baseline.md` linking these observations to your latency differences.

#### ✅ Acceptance Criteria

- [ ] `numastat` snapshots exist for baseline and pinned.  
- [ ] You can describe, in one paragraph, whether your hardened config keeps memory local to the GPU’s NUMA node.

---

### ✅ Task 3.5: IRQ/NIC Affinity for Remote Clients (If Applicable)

**Tags**: `[OS–Linux]` `[Networking]` `[IRQ]` `[QoS]`  
**Time**: ~45–60 min  
**Win**: Initial sense of how NIC interrupt placement can impact inference p95 in networked setups.

#### 🎯 Objective

If your benchmark client hits the server over the network, explore IRQ affinity to decouple NIC interrupts from inference cores.

#### 🔧 Lab Instructions

1. Inspect NIC-related interrupts:

   ```bash
   cat /proc/interrupts | grep -Ei 'eth|enp|eno|mlx5'
   ```

2. Note which CPU cores are currently handling NIC IRQs.  

3. (Advanced / optional) Use `irqbalance` and manual `echo <cpu-list> > /proc/irq/<IRQ>/smp_affinity_list` adjustments on a test box to:
   - Move NIC IRQs off your inference core range.  
   - Re-run a short benchmark and see if p95 tail improves under network load.

4. Document any changes as bullets in `benchmarks/day05_numa_baseline.md` (even “no noticeable change” is useful).

#### ✅ Acceptance Criteria

- [ ] You know which cores handle NIC interrupts on your node.  
- [ ] You have at least a qualitative note on whether moving IRQs helped p95 in your setup.

---

### ✅ Task 3.6: Consulting Packaging – OS & NUMA Hardening Module

**Tags**: `[Product/Business]` `[Consulting]` `[Docs]`  
**Time**: ~45–60 min  
**Win**: A small, reusable doc that turns today’s work into a “module” you can plug into an Inference Health Check or Optimization Sprint.

#### 🎯 Objective

Document Day 5’s work in a format that’s directly usable for client engagements.

#### 🔧 Lab Instructions

1. Create `docs/os_node_os_tuning_summary_day05.md` with two sections:

   **Engineering View**
   - Brief summary of:
     - Node topology (GPU, NUMA, cores).  
     - Baseline vs pinned results.  
     - CPU governor + noisy neighbor findings.

   **Consultant View**
   - “Risk level” for the node: Low / Medium / High.  
   - 3–5 bullet **findings**, phrased in business language  
     (e.g., “Inference processes are not pinned to GPU-local cores, leading to 20–30% p95 inflation under CPU noise”).  
   - 3–5 bullet **recommendations** with expected impact  
     (e.g., “Pinning inference to NUMA 0 cores 0–15 and using `performance` governor is expected to reduce p95 by ~20% at typical conc=32 workloads”).

2. Add a short checklist at the bottom:

   - “Inference Node OS & NUMA Health Check v0.1”
     - [ ] GPU-local NUMA and core mapping documented.  
     - [ ] Inference workers pinned to GPU-local cores.  
     - [ ] CPU governor chosen and justified.  
     - [ ] Noisy-neighbor impact characterized.  
     - [ ] Systemd/cgroup policy defined for inference.

3. (Optional) Add a one-paragraph note on how this module would fit into:
   - an “Inference Health Check” 2–3 week engagement, or  
   - a broader “GPU Consolidation / Cost Optimization” sprint.

#### ✅ Acceptance Criteria

- [ ] `docs/os_node_os_tuning_summary_day05.md` exists with engineering + consultant views.  
- [ ] There is a short, concrete checklist you can reuse for future nodes or clients.

---

### 📝 Feynman Deliverable (Tier 3)

At the end of `docs/os_node_os_tuning_summary_day05.md`, add:

> **Feynman: Why OS & NUMA Tuning Matters for Inference Economics**  
> 1–2 paragraphs explaining, in your own words, how OS/NUMA choices can mean the difference between needing N vs N+1 GPUs, and why that matters more to a client than a 5–10% change in average latency.
