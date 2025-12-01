Tier 3 – Top-Tier Deep Work (4+ hours)
If you somehow have a big 4–6 hour block, stack these on top.

Task 7: First mini “consulting-style” HF benchmark case
Tags: [Inference–Runtime] [Phase1-4 Scaling Basics] [Business]


Time: 2–3h


Lab instructions:
Define a toy “client workload”: e.g.


 “Generate ~256-token answers to chat-style prompts with GPT-2 on a single GPU node.”



Extend scaling script into a simple benchmark harness:


N prompts (e.g. 10–20)


Vary max_new_tokens and maybe batch size (if simple to add).


Capture metrics: tokens/sec, generation time, GPU memory usage (snapshot via `nvidia-smi --loop-ms=500` redirected to log).


Produce:


`benchmarks/day02_hf_toy_client_workload.csv`

`notes/day02_toy_client_case_study.md` with:


“Before” config (plain HF, default settings)


Observed throughput/latency


Quick thoughts on what could be optimized later (e.g. vLLM, quantization, batching).


**Artifact (GitHub):**
- `benchmarks/day02_hf_toy_client_workload.csv`
- `notes/day02_toy_client_case_study.md`

**Commit:**
```
day02: toy client HF benchmark – baseline case study
```




Task 8: Seed a K8s/Ray future – repo structure + TODOs
Tags: [Platform–Kubernetes] [Platform–Ray] [Business]


Time: 1–1.5h


Lab instructions:
In your learning-trace repo, add directories:
- `k8s/` (future manifests)
- `ray/` (future Ray cluster configs)
- `services/` (consulting service definitions)

Create `services/inference-health-check.md` and write a 1-page skeleton:


Section headers only today:


“01 – GPU Node Baseline (OS-01 checks)”


“02 – Inference Runtime Baseline”


“03 – Load & Latency Snapshot”


“04 – Quick Wins & Recommendations”


Under “01 – GPU Node Baseline”, list today’s OS-01 checks as bullet points.


Add a short `README.md` at repo root explaining what this repo is:


“100-day LLM Inference Engineering Learning Trace”


Brief description of what Day 2 did.


**Artifact (GitHub):**
- `services/inference-health-check.md`
- `README.md`
- `k8s/` and `ray/` folders (even empty with `.gitkeep`)

**Commit:**
```
day02: seed consulting service skeleton + k8s/ray folders
```




3. Reading (Off-Hours) – Optional Theory
Do these outside the 2–4h core block if you want:
CUDA / NVIDIA docs: short section on driver vs runtime vs toolkit.


vLLM quickstart docs (just skim to map HF → vLLM in your head for tomorrow).


A short blog or doc on NVIDIA Container Toolkit – just enough to explain why containers see GPUs.


Don’t go deep; just enough context to name things you already touched in labs.

4. Business / Positioning Nudge
Today’s work directly underpins an “Inference Health Check” service: you’re encoding OS-01/OS-06 into repeatable scripts and checklists that a client would pay for.


Your GitHub learning-trace is the start of real proof: you can point to concrete scripts and benchmarks instead of generic talk. This becomes evidence in sales calls and talks.


The tiny “toy client” HF benchmark is your first proto-case-study: later, you’ll redo it with vLLM / quantization and show “before/after” like a proper consultant.


If you tell me tomorrow “what actually got done from this list”, I’ll treat that as input and design Day 3 to build directly on it (e.g. moving HF baseline → vLLM baseline on the same tuned node).
