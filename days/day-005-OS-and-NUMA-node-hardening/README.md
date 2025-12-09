# Day 005 – OS & NUMA Node Hardening for Inference

> **Theme**: GPU node topology, NUMA, CPU pinning, and their impact on inference latency/throughput.  
> **Layer focus**: Hardware & OS → Inference Runtime  
> **Goal**: Build a NUMA- and CPU-aware baseline for running an inference server on a single GPU node, and turn it into a reusable “node hardening” pattern.

---

## What You’ll Do

- Inspect CPU, NUMA, and GPU topology on your node.  
- Benchmark a simple inference server **before vs after** NUMA + CPU pinning.  
- (Tier 2) Encode those choices into a **systemd/cgroup-aware service**.  
- (Tier 3) Explore CPU governors, noisy neighbors, and document how these knobs affect p95 and overall capacity in a way that’s useful for future consulting work.

---

## Tiers

- **[Tier 1 – Core: NUMA & CPU-Aware Baseline](LOG_tier01.md)**  
  Inspect topology and run baseline vs pinned inference benchmarks.

- **[Tier 2 – Extension: Systemd & Service-Level Affinity](LOG_tier02.md)**  
  Encode NUMA/CPU policies into a reproducible systemd service template.

- **[Tier 3 – Deep Work: Governors, Noisy Neighbors & Consulting Hook](LOG_tier03.md)**  
  Explore CPU governor + noisy-neighbor impact, and package findings as an “Inference Node OS & NUMA Hardening” module.

---

## Cross-Day References

- **Day 002 – GPU Node Bring-Up**: base OS/GPU setup and first `nvidia-smi`/`lspci` checks.  
- **Day 003 – vLLM Capacity & OOM**: capacity grids that will later run on this hardened node.  
- **Day 004 – Quantization vs BF16**: quantized vs BF16 capacity/quality experiments that will benefit from a stable OS/NUMA baseline.

---

## Conceptual Overview – Inference as a DAG & Topological Hardening

*(This post is part of 100 Days of Inference Engineering being documented on GitHub. Let’s learn together…)*  

Inference is a DAG executed across CPU, GPU, memory, and I/O. The model’s forward pass happens on the GPU—matmuls, attention, and KV-cache operations—but everything that feeds the GPU lives on the CPU. Tokenizers, schedulers, batchers, samplers, KV-cache metadata logic, and all request parsing and networking are CPU-bound. A GPU is only fast if the CPU delivers work quickly, accesses local memory, and avoids cross-NUMA chatter.

![Inference DAG – simple view](../assets/dag_simple.png)

Hardware topology determines how well this pipeline flows: which CPU cores and DRAM are local to a GPU, which PCIe root or switch the GPU hangs off of, how GPUs are interconnected via NVLink, and which NIC injects traffic into which NUMA domain. Mapping this topology correctly is the foundation of predictable, low-latency inference.

This diagram exposes the physical cost of each DAG edge:

- CPU-to-GPU edges → PCIe latency  
- CPU-to-memory edges → NUMA-local vs remote latency  
- CPU thread scheduling edges → cross-socket penalties  
- GPU kernel edges → kernel launch overhead + scheduling jitter  

![Inference DAG – annotated topology](../assets/dag_colored.png)

By pinning nodes of the DAG to hardware, you collapse latency along these edges. This is the whole principle behind OS/NUMA hardening.

![CPU-bound segments](../assets/cpu_bound.png)

Topological optimisation techniques such as NUMA pinning and CPU pinning force the OS to respect the DAG’s physical structure, eliminating cross-node penalties and making latency predictable. Here is the full set of “topological hardening” methods for this node.

### Topological Hardening #1 — CPU ↔ NUMA Affinity Enforcement

As core counts increased, a single shared memory controller became a fundamental bottleneck because too many CPU cores were competing for the same memory channels. Modern CPUs solve this by splitting the processor into multiple core groups—often called chiplets—where each group has its own memory controller and directly attached DRAM. This creates physically distinct memory regions: DRAM that is “local” to one chiplet is “remote” to others, accessed through a slower interconnect. The OS reflects this hardware reality by grouping each chiplet (or socket) together with its local DRAM and memory controller into a NUMA node. In an ideal case, cores should access only their local NUMA memory to avoid the extra latency and bandwidth penalties of remote memory access.

By default, the OS does not statically bind CPU cores to their local NUMA memory; the scheduler is free to migrate threads across cores and NUMA nodes, and the kernel may even move memory pages between nodes via automatic NUMA balancing. This “smart” behavior is good for generic workloads but terrible for low-latency inference, because it silently introduces remote memory accesses and cross-node hops. The first layer of hardening is to override this default: pin hot-path threads to a specific set of cores on the NUMA node local to the GPU, and bind their memory allocations to that same node.

![Topology Hardening #1 – CPU↔NUMA](../assets/topo_1.png)

#### Hardening #1 – Execution (Tier 1 / Tier 2)

**Goal**  
Ensure all CPU-bound DAG nodes (tokenizer, scheduler, sampler, runtime threads) run only on the NUMA node local to the GPU, and all their memory allocations come from that NUMA node.

**How to apply it mechanically**

1. Inspect topology:

   ```bash
   lscpu
   numactl --hardware
   nvidia-smi topo -m
   ```

2. Identify the NUMA node local to `GPU0`:
   - Look for GPU0’s **CPU Affinity** in `nvidia-smi topo -m`, e.g. `0-15` ⇒ NUMA node 0.

3. Launch the inference runtime pinned to those cores and DRAM:

   ```bash
   numactl --cpunodebind=0 --membind=0 \
     taskset -c 0-11 \
     python server.py
   ```

4. Disable automatic NUMA balancing (optional, but recommended on dedicated inference nodes):

   ```bash
   echo 0 | sudo tee /proc/sys/kernel/numa_balancing
   ```

5. Reinforce affinity at the service level (Tier 2) using systemd:

   ```ini
   [Service]
   CPUAffinity=0-11
   ```

Relevant tiers: **Tier 1** (manual `numactl`/`taskset` pinning for experiments), **Tier 2** (encoding the same policy in systemd units).

After CPU–NUMA affinity (Hardening #1), the second topological hardening is PCIe topology enforcement: understanding and aligning with the PCIe root complex, PCIe switch hierarchy (PIX/PXB/PHB), and GPU–GPU PCIe connectivity so that GPU workloads remain on their fastest possible PCIe paths.

### Topological Hardening #2 — PCIe Topology Awareness (Root Complex / Switch Affinity)

In a multi-socket, multi-GPU server, each CPU socket exposes its own PCIe root complexes, and GPUs attach underneath them. This creates non-uniform I/O paths: a GPU sitting under PCIe Root 0 (owned by CPU Socket 0 → NUMA 0) is topologically close to cores 0–15 and far from cores 16–31.

By default, inference runtimes ignore this structure. They may place a model shard on GPU0 and GPU2 even though those GPUs live on opposite PCIe roots and must communicate through a slow cross-socket path. They may also schedule GPU0’s workers on CPU cores belonging to the wrong NUMA domain. These mismatches introduce hidden latency, reduce PCIe bandwidth, and destabilize p95/p99 inference behavior.

The second layer of hardening is to respect the PCIe topology exposed in the diagram: each GPU must be paired with the CPU cores and memory on its own PCIe root, and multi-GPU workloads should be restricted to GPUs that share a root complex or switch. Communication-heavy tasks must avoid PHB (cross-root) paths, and no GPU should be driven by a CPU socket that does not own its PCIe parent. This alignment ensures that GPU-CPU and GPU-GPU communication stays on the shortest, highest-bandwidth I/O paths in the system.

![Topology Hardening #2 – PCIe](../assets/topo_2.png)

#### Hardening #2 – Execution (Tier 1 / Tier 2)

**Goal**  
Ensure each GPU is driven by CPU cores within the same PCIe root complex and avoid PHB (cross-root) traffic for inference-critical paths.

**How to apply it mechanically**

1. Map PCIe roots:

   ```bash
   lspci -tv
   ```

2. Check GPU-to-GPU and GPU-to-CPU distances:

   ```bash
   nvidia-smi topo -m
   ```

   - `PIX`, `PXB` → same root/switch (preferred)  
   - `PHB` → cross-root (slow, avoid for tight SLOs)

3. Restrict model shard / tensor-parallel placement:
   - Use only GPUs under the **same PCIe root** for collective-heavy work (tensor parallel, all-reduce, all-gather).

4. Pin workers to the correct CPU domains (Tier 2 via systemd):

   ```ini
   # GPU0 under Root 0 / NUMA 0
   CPUAffinity=0-11

   # GPU2 under Root 2 / NUMA 1
   CPUAffinity=16-23
   ```

5. Ensure the runtime respects these affinities. Example with vLLM:

   ```bash
   CUDA_VISIBLE_DEVICES=0 VLLM_CPU_AFFINITY=0-11 \
     python -m vllm.entrypoints.api_server
   ```

Relevant tiers: **Tier 1** (reading topology, making informed GPU choices), **Tier 2** (baking PCIe-aware CPU affinity into long-lived services).

### Topological Hardening #3 — NVLink / NVSwitch Topology Awareness

Multi-GPU inference is only fast if your GPUs are talking over the right wires. NVLink and NVSwitch create high-bandwidth GPU islands: some GPUs can exchange activations and KV cache at huge speed; others can only reach each other through slow PCIe and host bridges. If you ignore this, your “multi-GPU model” quietly runs over the worst possible paths.

Hardening #3 is simple but brutal in its implications:

- Only group GPUs that are NVLink/NVSwitch-connected into the same model instance.  
- Keep tensor parallel, pipeline stages, and sharded KV cache inside those NVLink islands.  
- Never stretch one logical model across GPUs that only see each other via PHB.  

Done right, GPU-to-GPU traffic stays on the fastest fabric, and you stop burning your throughput and p95/p99 on topology mistakes instead of model limits.

![Topology Hardening #3 – NVLink/NVSwitch](../assets/topo_3.png)

#### Hardening #3 – Execution (Tier 3 / Multi-GPU Work)

**Goal**  
Group GPUs along NVLink/NVSwitch islands—never stretch one model instance across GPUs that only see each other via PCIe/PHB.

**How to apply it mechanically**

1. Inspect NVLink connectivity:

   ```bash
   nvidia-smi topo -m
   ```

   - `NV1/NV2/NV4…` = NVLink paths  
   - `PIX/PXB/PHB` = fallback PCIe/host paths

2. Choose GPU groups that form an NVLink island. Example:

   - `GPU0 <-> GPU1 <-> GPU2` all NVLink-connected  
   - `GPU3` with no NVLink → isolate or use for a separate model.

3. Configure tensor parallel or sharding to stay within that island. Example:

   ```bash
   # 3-way tensor parallel confined to GPUs 0,1,2
   --tensor-parallel-size 3 \
   --tensor-parallel-gpus 0,1,2
   ```

4. Avoid pipeline-parallel stages that cross NVLink boundaries; keep hot-path GPU↔GPU traffic on NVLink/NVSwitch fabric where possible.

Relevant tiers: **Tier 3** (advanced multi-GPU topologies, later scaling work building on Day 005).

### Topological Hardening #4 — NIC ↔ NUMA ↔ GPU Triangulation

For online inference, every request physically follows the same path: NIC → CPU cores → GPU → CPU → NIC. If the NIC, the CPU cores handling the request, and the GPU doing the work sit on different NUMA sides of the machine, each hop pays an extra cross-socket and remote-DRAM penalty. You effectively turn every token into a mini distributed system call inside a single server.

NIC ↔ NUMA ↔ GPU triangulation is about collapsing that path into a local island. The NIC that receives traffic for a given GPU should terminate interrupts on the same NUMA node as that GPU; RSS queues and IRQs are pinned to those local cores, the inference server’s hot-path threads are pinned to those same cores, and their memory allocations are bound to that NUMA node. Traffic for GPU0/1 consistently flows through NIC0 + NUMA0, and traffic for GPU2/3 flows through NIC1 + NUMA1, instead of bouncing across sockets.

Once this triangle is closed—NIC, NUMA, GPU all aligned—network jitter stops turning into mysterious p95/p99 spikes, and your end-to-end latency is limited by the model and hardware, not by accidental tours of the server’s interconnect.

![Topology Hardening #4 – NIC↔NUMA↔GPU](../assets/topo_4.png)

#### Hardening #4 – Execution (Tier 3 / Networked Inference)

**Goal**  
Align NIC interrupts, CPU threads, and GPU work on the same NUMA island so the ingress path doesn’t zig-zag across sockets.

**How to apply it mechanically**

1. Find the NIC’s NUMA node:

   ```bash
   cat /sys/class/net/<iface>/device/numa_node
   ```

2. Pin RSS queues and IRQs to local cores. Example:

   ```bash
   ethtool -X eth0 equal 8
   ```

   IRQ pinning:

   ```bash
   echo 2 > /proc/irq/<irq_number>/smp_affinity_list
   ```

3. Ensure the inference runtime uses the same core set (Tier 3 + systemd):

   ```ini
   [Service]
   CPUAffinity=0-11
   ```

4. Bind server memory allocations to that NUMA node:

   ```bash
   numactl --membind=0 python server.py
   ```

5. Confirm the GPU is under this NUMA node using:

   ```bash
   nvidia-smi topo -m
   ```

Relevant tiers: **Tier 3** (noisy-neighbor / IRQ work, remote clients, production-grade network tuning).

Together, these four topological hardenings turn a generic GPU server into a deterministic inference machine. NUMA affinity ensures CPU locality, PCIe awareness keeps GPU-CPU paths short, NVLink topology shapes efficient multi-GPU execution, and NIC–NUMA–GPU triangulation locks down the network ingress path. Once the hardware is mapped correctly and each component is placed on its optimal island, the entire inference DAG behaves predictably. Latency stops drifting, throughput stabilizes, and the model’s performance reflects fundamental limits—not avoidable topology mistakes. This is the foundation on which every higher-level optimization in LLM serving must be built.

![Node Hardening Playbook](../assets/node_hardenning_playbook.png)

#### Ultra-Terse Execution Checklist (Across Tiers)

```text
#1 CPU–NUMA Affinity (Tier 1/2):
  Pin hot-path CPU threads + memory to NUMA node local to GPU.
  (numactl, CPUAffinity, disable NUMA balancing)

#2 PCIe Topology Awareness (Tier 1/2):
  Keep GPU workloads inside their PCIe root complex.
  Avoid PHB. Map topology (nvidia-smi topo -m, lspci).

#3 NVLink Topology Awareness (Tier 3):
  Only shard / parallelize across NVLink-connected GPUs.
  Never stretch a model across PCIe-only pairs.

#4 NIC–NUMA–GPU Triangulation (Tier 3):
  Align NIC IRQs, CPU cores, and GPU affinity on same NUMA node.
  RSS/IRQ pinning + CPU affinity + membind.
```

Istanbul, 9th December 2025
