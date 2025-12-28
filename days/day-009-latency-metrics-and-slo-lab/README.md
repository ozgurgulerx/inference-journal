# Day 009 — Latency Metrics Lab (TTFT/TPOT/E2E) + SLO Thinking

**Why this day exists:** If you can’t measure **TTFT/TPOT/E2E** correctly and report **p50/p95/p99 with load context**, you can’t tune inference systems or sell credible “optimization sprints.”

**Theme coverage:** Observability + cost/latency literacy (Product/SLO) + runtime-facing measurement (vLLM/OpenAI API) + queueing intuition.

---

## Snapshot (Today’s Focus)

- Recent artifacts: Day 07 probes (TTFT, prefix caching, KV scaling) + Day 08 (percentiles + prefill/decode + deterministic scheduling mental models).
- Today’s focus: **measure what users feel** (TTFT/TPOT) and what the system can promise (**p95/p99 SLOs**) under load.
- Layers: Runtime + Product/SLO (primary), Platform (load generation) with optional Hardware/OS signals (`nvidia-smi dmon`).
- Assumption: you can run a local OpenAI-compatible endpoint (vLLM preferred). If not, you can still do Tier 1 using `simulate_queue.py`.

---

## 3-Tier Lab Plan (Hands-On Only)

### Tier 1 — Must Do: “One Truthy Latency Report”

**Goal:** Produce one markdown report with TTFT/TPOT/E2E **p50/p95/p99**, plus the **load context** used.

**Time Budget:** 60–90 min

**Steps**

1. Start a local OpenAI-compatible server (vLLM template):
   - `./serve_vllm_openai.sh` (edit `MODEL`, `PORT` via env)
2. Verify streaming works and capture a single request metric:
   - `python3 openai_stream_probe.py --base-url http://localhost:8000/v1 --model "$MODEL"`
3. Run a tiny load test with mixed prompts (induces tails):
   - `python3 openai_loadgen.py --base-url http://localhost:8000/v1 --model "$MODEL" --concurrency 8 --requests 80 --mix mixed --out results.jsonl`
4. Summarize into a markdown table:
   - `python3 summarize_metrics.py results.jsonl --out results.md --title "Day09: $MODEL c=8 r=80 mixed"`
5. Add one paragraph: what drives p99 in your run (queueing vs long prompts vs cold start).

**Expected Artifact**

- `days/day-009-latency-metrics-and-slo-lab/results.md` (generated summary) + optionally `days/day-009-latency-metrics-and-slo-lab/results_template.md` filled with interpretation.

---

### Tier 2 — Deepen: “Tail Anatomy (Prompt Mix + Concurrency Sweep)”

**Goal:** Show how p95/p99 changes with prompt mix and concurrency; identify the “knee.”

**Time Budget:** 60–120 min

**Steps**

1. Run a 2×3 matrix (quick):
   - Concurrency: 1, 8, 32
   - Mix: short vs mixed
2. For each run: save `results_*.jsonl` and summarize.
3. Create a tiny combined table in `results.md`:
   - rows = runs, columns = TTFT p95, TTFT p99, TPOT p95, E2E p95, error rate
4. Optional (hardware signal): run `nvidia-smi dmon -s pucvmt -d 1` during the c=32 run and paste 5–10 lines into the report.

**Expected Artifact**

- `days/day-009-latency-metrics-and-slo-lab/results.md` updated with a small matrix table.

---

### Tier 3 — Stretch: “Queueing Simulator → Admission Control Intuition”

**Goal:** Build intuition for why mean lies and why “p99 is a control problem.”

**Time Budget:** 45–90 min

**Steps**

1. Run the queue simulator with deterministic service time and bursty arrivals:
   - `python3 simulate_queue.py --service-ms 120 --mean-qps 20 --burst-qps 80 --burst-every-s 10 --duration-s 120`
2. Compare p50/p95/p99 from the simulation with your real run from Tier 2.
3. Write 5 bullets mapping sim parameters → real knobs:
   - `service_ms` → model + max tokens + attention/KV behavior
   - `mean_qps/burst_qps` → traffic envelope
   - admission caps → queue bound
4. (Optional) Add a “candidate SLO” line:
   - “TTFT p95 < ___ms at concurrency __ for mix ___”

**Expected Artifact**

- `days/day-009-latency-metrics-and-slo-lab/results.md` with the simulation output pasted and interpreted.

---

## Reading (Off-Hours Only)

- vLLM OpenAI server docs / flags you actually use — to ensure you’re measuring the right endpoint and not fighting defaults.
- Queueing knee intuition (M/D/1 basics) — to explain why tails explode near saturation without hand-waving.
- OpenAI streaming event format (SSE) — to understand why TTFT/TPOT instrumentation is easy to get subtly wrong.

---

## Logging Template for Tomorrow

```markdown
## RECENT_LOG (Day 09)

### Commands run
- …

### Files changed/created
- days/day-009-latency-metrics-and-slo-lab/results.md

### Key numbers
- TTFT p50/p95/p99:
- TPOT p50/p95/p99:
- E2E p50/p95/p99:
- Load context: model=…, concurrency=…, requests=…, mix=…

### Observations / surprises
- What dominated p99? (queueing vs prompt mix vs cold start vs errors)
- Where did throughput plateau?

### Next
- One knob to change next (max_model_len, max_tokens, concurrency cap, prefix cache, etc.)
```
