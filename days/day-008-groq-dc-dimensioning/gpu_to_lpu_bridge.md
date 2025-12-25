# GPU → LPU Bridge (Use GPU Intuition Without Importing GPU Assumptions)

This page exists for one reason: **use the GPU mental model as a contrast** so Groq’s “compiler is the runtime” framing clicks, without accidentally carrying GPU-native assumptions into Groq capacity planning.

![GPU architecture and scheduling overview](assets/gpu_arch.png)

---

## The corrected, precise GPU statement

**GPUs run threads. Threads are grouped into warps. Warps run on SMs. Schedulers decide which warps run, when.**

### What each word means (GPU context)

- **Kernel**: the program you launch (the “job description”).
- **Thread**: one logical instance of that program, operating on different data.
- **Warp**: a fixed group of threads (typically 32) that execute in lockstep (SIMT).
- **SM (Streaming Multiprocessor)**: a hardware block that hosts many resident warps and issues their instructions.
- **Warp scheduler (on each SM)**: chooses which ready warp issues an instruction each cycle (helps hide memory latency).

### Why GPUs feel “dynamic” at runtime

Even if *your code* is deterministic, **the exact execution interleaving is not**:

- kernels queue and contend for SM time,
- warps stall on memory and get swapped out,
- caches change what’s fast vs slow from moment to moment.

This is why GPU inference engineering leans heavily on runtime tactics (batching windows, heuristics, warmup, profiling).

---

## The LPU contrast (the only part you should import)

Groq’s promise isn’t “a faster GPU.” It’s closer to:

- **GPU**: “smart workers decide what to do next on the fly.”
- **Groq**: “a perfect timetable; nobody improvises.”

### Translate GPU concepts into Groq-native questions

- Instead of “warp scheduling / occupancy”: **is the compiled schedule balanced, or does it have bubbles/stalls?**
- Instead of “cache hits/misses”: **is data movement explicit and capacity-safe (no spill/oversubscription)?**
- Instead of “kernel-by-kernel tuning”: **is the whole-graph schedule correct for this shape bucket?**
- Instead of “micro-variance dominates p99”: **queueing + admission control dominates p99.**

If you keep this translation discipline, you get the benefit of GPU intuition (hierarchy + scheduling) without importing the wrong levers into Groq dimensioning.

