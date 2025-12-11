# Day 006 – Caching, Cold vs Warm Loads & vLLM Tunables

This note turns the “cold vs warm loads” concept into concrete knobs you can adjust for vLLM and other inference servers.

---

## Tunable Knobs for vLLM: Making Cold Less Painful, Warm More Reliable

Think in three layers: **storage**, **OS memory**, and **serving patterns**.

---

### 1. Storage & File Placement

These decide how bad a *cold* load is.

- **Keep model weights on local NVMe, not network storage**
  - Local NVMe → lowest disk latency, highest throughput.  
  - Avoid NFS / SMB / object storage mounts for the actual `.safetensors` / shards if you care about cold TTFT.

- **Avoid unnecessary image layers for model files**
  - If your container image includes the model, prefer:
    - Thin base images.  
    - Model files in a single layer (not spread across many).  
  - Cold start = pulling layers + faulting model files; fewer layers → faster cold.

- **Optionally, RAM-disk for ultra-low-latency setups**
  - For extreme setups, place model files on `tmpfs` or a RAM-backed FS.  
  - Trade: more RAM usage vs minimal cold-load penalty.

---

### 2. OS Memory & Page Cache Behavior

These decide *how much* the page cache can help and how stable it is.

- **Don’t starve the page cache with process RSS**
  - Use jemalloc (as in Day 006 allocator notes) to keep RSS ≈ working set.  
  - Lower RSS → more DRAM available for page cache → more robust warm loads.

- **Avoid heavy memory pressure**
  - If the node is running:
    - Multiple big services.  
    - Aggressive THP (`always`).  
    - Large explicit hugepage pools.  
  - The kernel will evict page cache more aggressively.  
  - For inference nodes, aim for:
    - Few tenants.  
    - Predictable memory reservations.  
    - Reasonable headroom (you want spare RAM for cache).

- **Swappiness & cache pressure**
  - Keep `vm.swappiness` low (e.g. 1–10) on inference nodes to avoid swapping out anonymous memory and hammering disks.  
  - `vm.vfs_cache_pressure`:
    - Too high → kernel drops page cache eagerly (bad for warm loads).  
    - Too low → holds cache too aggressively (can starve other uses).  
    - Keep near default unless you have a strong reason and data.

- **Readahead on the model device**
  - For very large sequential model files, increasing block device readahead (`blockdev --setra`) can improve cold load throughput slightly.  
  - Small win compared to “local NVMe vs network,” but still a lever.

- **Use `drop_caches` only for experiments**
  - `echo 3 > /proc/sys/vm/drop_caches` is great for Day 006 SLM experiments.  
  - In prod, avoid it: you’re intentionally throwing away warm cache.

---

### 3. Warm-Start & Pre-Warming Patterns

These don’t change the kernel; they change **how you use it**.

- **Preload the model before admitting traffic**
  - Start vLLM, then immediately send:
    - One or more “dummy” requests that:
      - Use a reasonably long prompt.  
      - Exercise typical context length.  
  - This:
    - Faults model pages into page cache.  
    - Populates GPU HBM.  
    - Triggers JIT/kernel warmup.  
    - Stabilizes first real user TTFT.

- **Keep servers warm; avoid churn**
  - Don’t constantly:
    - Restart pods/containers.  
    - Autoscale down to zero every hour.  
    - Reload models every few minutes.  
  - Every restart = cold cache + cold weights.  
  - Prefer:
    - A minimum replica count.  
    - Graceful draining (preStop hooks) instead of kill/restart.

- **Hide cold loads behind warm-up windows**
  - For clusters:
    - Bring new vLLM replicas up.  
    - Run warm-up traffic.  
    - Only then add them to the load balancer.  
  - Cold TTFT is then never exposed to real users.

- **One-node, many models pattern**
  - If multiple models share the same filesystem and node:
    - A warm load of model A leaves its pages hot.  
    - If you frequently swap A/B models:
      - Choose model sizes & concurrencies so both can coexist in DRAM without constantly evicting each other’s cache.

---

### 4. vLLM-Specific Knobs That Indirectly Help Page Cache

These don’t touch the page cache directly, but they keep the **node’s memory state sane**, which protects page cache.

- **`--gpu-memory-utilization`**
  - If you run too close to the GPU limit, you may restart/evict more often → more cold loads.  
  - Leave a bit of headroom → fewer crashes → fewer cold reboots → page cache stays hot longer.

- **Avoid overcommitting CPU RAM with too many replicas per node**
  - If each vLLM process is 20–30 GB host RSS + page cache, packing too many per node:
    - Forces the kernel to evict page cache.  
    - Increases risk of swapping / OOM.

---

### 5. Simple Rule-of-Thumb Mental Model

You can summarize the page cache knobs like this:

> Put your models on the fastest local storage you can.  
> Keep your process RSS honest (jemalloc, no wild leaks).  
> Give the kernel enough free RAM to cache your model once.  
> Don’t restart containers unnecessarily.  
> Run a small warm-up phase so real traffic never sees a genuinely cold node.

