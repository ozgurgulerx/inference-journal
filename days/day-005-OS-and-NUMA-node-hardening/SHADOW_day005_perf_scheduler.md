# Day 005 – Shadow Subject: Perf & Scheduler X-Ray on NUMA/Pinning

> **Shadow subject (deep tech)**: Use `perf` + basic scheduler tools to explain *why* NUMA/CPU pinning changes p95 latency (or doesn’t), instead of just observing that it does.

This is an **optional, technical side-track** that builds directly on the Day 005 main subject:

- Main: NUMA topology, pinning inference to GPU-local cores, baseline vs pinned benchmarks, optional CPU governor comparison.  
- Shadow: Micro-architectural + scheduler instrumentation of the **same** experiment.

---

## 1. Goals

- Attach `perf` to your inference server in **baseline vs pinned** modes.  
- Capture a **small set of hardware + scheduler counters** and compare.  
- Add a short “Perf & Scheduler Notes” subsection to your Day 005 benchmark doc.

This trains the muscle:

> “I don’t just see p95 move; I can reason from scheduler + micro-architecture counters *why*.”

It also aligns tightly with your 100-day goals:

- Phase 0 isn’t just “we set some sysctls” — it becomes **“we empirically validated that OS/NUMA policy changed scheduling behavior and correlates with p95/p99 changes.”**  
- It feeds directly into future **Inference Health Check / Optimization Sprint** offers where you must distinguish:
  - vLLM config issues vs CPU scheduler/NUMA issues vs GPU sizing issues.  
- It gives you hard evidence for case studies, not just latency charts: *“context switches down X%, cross-NUMA traffic down Y%, p95 latency down Z%.”*

---

## 2. perf stat – Baseline vs Pinned

For each scenario (baseline, pinned):

1. Start your inference server as usual.  
2. Begin your benchmark (e.g. `python code/day005/bench_client.py --n-requests 50 --concurrency 4`).  
3. While the benchmark is running, attach `perf` to the server PID:

   ```bash
   mkdir -p perf

   # Example: baseline
   sudo perf stat -d -d -d -p <PID> -- sleep 30 \
     > perf/perf_stat_baseline_day05.txt 2>&1

   # Example: pinned
   sudo perf stat -d -d -d -p <PID> -- sleep 30 \
     > perf/perf_stat_pinned_day05.txt 2>&1
   ```

4. Focus on:
   - `task-clock`, `context-switches`, `cpu-migrations`  
   - `cache-misses`, `LLC-load-misses` (if present)  
   - `cycles`, `instructions`, and IPC (instructions per cycle)

Look for patterns such as:

- Fewer context switches / migrations in pinned mode.  
- Slightly better IPC or similar IPC but fewer stalls.  
- Or **no significant change** → CPU likely not the bottleneck here.

---

## 3. Quick Scheduler View

Run a short scheduler profile for at least one scenario (baseline or pinned):

```bash
sudo perf sched record -p <PID> -- sleep 20
sudo perf sched latency > perf/perf_sched_latency_day05.txt
```

You’re checking:

- Are there long runqueue latencies?  
- Do they correlate (roughly) with your p95 spikes under load?

Even a coarse observation is enough:

> “Baseline mode shows more scheduler latency and higher context-switch counts; pinned mode has lower variance, matching the tighter p95 we observed.”

---

## 4. Add “Perf & Scheduler Notes” to Day 005 Benchmarks

Extend `benchmarks/day05_numa_baseline.md` with a short subsection, for example:

```markdown
## Perf & Scheduler Notes (Shadow Subject)

- Baseline vs pinned:
  - context-switches: [baseline] → [pinned]
  - cpu-migrations: [baseline] → [pinned]
  - IPC (instructions per cycle): [baseline] → [pinned]
- Observed patterns:
  - [...]
  - [...]

**Interpretation (1 paragraph):**
- For this workload, pinning [reduced/increased] scheduler noise (fewer context switches/migrations) and [slightly improved/did not change] IPC.
- This [matches/does not fully explain] the p95 improvement we saw, suggesting that [CPU scheduling / memory locality / GPU-side bottleneck] is the dominant factor.
```

You don’t need a full perf deep dive—just enough to connect **scheduler/micro-arch signals** to your latency results.

---

## 5. Off-Hours Reading Suggestions (Perf/Scheduler)

- `man perf-stat`, `man perf-sched` – understand the counters and scheduler views you just used.  
- A short “intro to perf for latency analysis” blog or doc (e.g., from Brendan Gregg / Red Hat) – to see how experts interpret similar profiles.  
- Any concise article on Linux scheduler basics (CFS, runqueues) – to give names to the behavior you’re seeing in `perf sched latency`.
