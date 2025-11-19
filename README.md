# Inference Engineering Journal

My name is Ozgur Guler.

This repo is my journey into **inference engineering**: building, scaling, and optimizing LLM inference stacks in the real world.

> A story of revival and redemption — moving from “just using APIs” to truly understanding how tokens move through silicon.

## What this repo is

This is a **working lab notebook**, not a polished course. You’ll find:

- Experiments with **vLLM, TensorRT-LLM, Triton**, and friends  
- Benchmarks across **batch sizes, sequence lengths, quantization levels, and hardware**  
- Notes on **latency, throughput, memory, and cost trade-offs**  
- “Postmortems” of things I broke and how I fixed them  

If you’re trying to go from *LLM user* to *inference engineer*, you may find this mess useful.

## How this journal is structured

- `days/` — raw daily logs (one folder per day with `LOG.md`, `metrics.json`, and any scratch code)  
- `topics/` — curated, refactored knowledge pulled from the day logs  
- `benchmarks/` — cleaned results and plots that act as the single source of truth  
- `scripts/` — runners and utilities to produce metrics and plots  
- `JOURNAL_INDEX.md` — quick table that links days and their themes  

I’ll keep refactoring as I learn more. The goal is simple:

> Understand inference deeply enough to design, debug, and optimize **any** LLM deployment, on **any** hardware.

### Current layout

```text
inference-engineering-journal/
├─ JOURNAL_INDEX.md
├─ README.md
├─ days/
│  └─ day-001-initial-setup/
│     ├─ LOG.md
│     ├─ metrics.json
│     ├─ TODO.md
│     └─ code/
├─ topics/
│  ├─ vllm/
│  │  ├─ example_server.py
│  │  ├─ gotchas.md
│  │  └─ overview.md
│  ├─ quantization/
│  ├─ serving-patterns/
│  └─ tensorrt-llm/
├─ benchmarks/
└─ scripts/
```

### Adding a new day

1. Create `days/day-XYZ-topic/` (zero-padded index helps sorting).
2. Copy the `LOG.md` template from `day-001-initial-setup/` and fill it in.
3. Drop raw numbers into `metrics.json` and any helper code into `code/`.
4. Summarize the day in `JOURNAL_INDEX.md`.
5. Every 5–7 days, promote the reusable lessons into `topics/` and clean benchmarks into `benchmarks/`.
