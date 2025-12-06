# Day 005 – OS & NUMA Node Hardening for Inference
## Tier 2: Extension – Systemd & Service-Level Affinity (~60–75 minutes)

> **Prerequisites**: Tier 1 (topology captured + baseline vs pinned benchmark)  
> **Goal**: Encode NUMA/CPU pinning into a reproducible service template so any inference node can be hardened in minutes.

---

## Tier 2 Tasks

---

### ✅ Task 2.1: Create a Systemd Service Template for Inference

**Tags**: `[OS–Linux]` `[systemd]` `[NUMA]`  
**Time**: ~45 min  
**Win**: A reusable `inference-vllm.service` (or equivalent) that bakes in NUMA/CPU affinity.

#### 🎯 Objective

Move from ad-hoc `numactl` + `taskset` commands to a clean systemd unit you can drop onto any node.

#### 🔧 Lab Instructions

1. Create a directory in your repo:

   ```bash
   mkdir -p systemd
   ```

2. Add `systemd/inference-vllm.service` with content along these lines (adjust paths/flags as needed):

   ```ini
   [Unit]
   Description=LLM Inference Server (vLLM)
   After=network.target

   [Service]
   # Adjust ExecStart to your env (venv / conda / path)
   ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 \
             /usr/bin/taskset -c 0-15 \
             /usr/bin/python -m vllm.entrypoints.openai.api_server ...
   WorkingDirectory=/home/ozgur/llm-node

   # Reinforce CPU affinity at the cgroup level
   CPUAffinity=0-15

   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

3. Keep this as a **template**:
   - Add comments indicating:
     - how to adjust NUMA node / CPU ranges,  
     - how to point to different models, ports, or configs.

4. If you prefer not to install system-wide yet:
   - Treat this unit as “source of truth” and keep it repo-local.  
   - When ready, copy it into `/etc/systemd/system/` on a node.

#### ✅ Acceptance Criteria

- [ ] `systemd/inference-vllm.service` exists and reflects your preferred NUMA/CPU policy.  
- [ ] You can explain each line at a high level (ExecStart, WorkingDirectory, CPUAffinity, Restart).

---

### ✅ Task 2.2: Test Systemd Unit and Re-Run a Short Benchmark

**Tags**: `[OS–Linux]` `[systemd]` `[Benchmarks]`  
**Time**: ~30 min  
**Win**: Confirm your systemd-managed server behaves like your Tier 1 pinned mode.

#### 🎯 Objective

Verify that running under systemd with your unit file reproduces the pinned performance profile.

#### 🔧 Lab Instructions

1. On a test node (or your local lab node), copy the service file:

   ```bash
   sudo cp systemd/inference-vllm.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. Start the service:

   ```bash
   sudo systemctl enable --now inference-vllm
   sudo systemctl status inference-vllm
   ```

3. Confirm CPU/NUMA affinity:
   - `ps -o pid,psr,cmd -C python | head`  
   - (Optional) `numactl -s -p <PID>` to inspect NUMA policy.

4. Re-run a short version of your benchmark:

   ```bash
   python code/day005/bench_client.py --n-requests 30 --concurrency 4
   ```

5. Append to `benchmarks/day05_numa_baseline.md`:
   - A short “systemd run” row in your table (same columns as before).  
   - 2–3 bullets on whether performance matches your manual pinned run.

#### ✅ Acceptance Criteria

- [ ] Inference server runs via systemd and shows expected CPU/NUMA placement.  
- [ ] A quick benchmark confirms similar performance to your Tier 1 pinned run.  
- [ ] `benchmarks/day05_numa_baseline.md` updated with systemd-based run notes.

---

### 📝 Feynman Deliverable (Tier 2)

Add a small section to `benchmarks/day05_numa_baseline.md`:

> **Feynman: Encoding NUMA Policy in Services**  
> Briefly explain why expressing pinning at the systemd/cgroup level is more robust than ad-hoc CLI commands, and how you’d standardize this across a fleet of inference nodes.

