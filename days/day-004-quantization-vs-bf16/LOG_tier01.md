# Day 004 – Quantization vs BF16 on RTX2000
## Tier 1: Must-Do Core Block (~2 hours)

> **Prerequisites**: Complete [Day 003](../day-003-vLLM-capacity-and-OOM/)  
> **Goal**: Compare BF16 vs quantized model on chat workload + quality sanity check  
> **End State**: Two capacity CSVs, VRAM comparison, quality eval JSON  
> **Time**: ~2 hours

---

## Tier 1 Tasks

---

### ✅ Task 1.1: Bring Up a Quantized Model Under vLLM

**Tags**: `[Inference–Runtime]` `[Quant]` `[KVCache]`  
**Time**: ~45 min  
**Win**: Quantized model running on RTX2000 alongside BF16

#### 🎯 Objective

Serve a quantized variant of a small instruct model side-by-side with your current BF16 Qwen2.5-1.5B on the RTX2000, and confirm it runs and fits in VRAM.

#### 📋 Spec

1. **Pick a vLLM-supported quantized model** that fits in 16GB:
   - Ideally a quantized version of a model you already know (Qwen, Llama, Mistral)
   - Example: an AWQ/GPTQ 4-bit checkpoint from HuggingFace
   - **You choose the exact model ID** — this is your decision to make

2. **Create a new serve script**:
   - Path: `~/configs/vllm/serve_quant_model_rtx16gb.sh`
   - Port: `:8001` (to avoid clashing with BF16 on `:8000`)
   - Same `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs` as Day 3
   - Add required quantization flags (e.g., `--quantization awq` or auto-detect)

3. **Verify**:
   - Run `nvidia-smi` before and after starting quant server
   - Send a single test prompt via curl or Python snippet
   - Record VRAM usage

#### 🧠 Deep-Thinking Prompts (Before)

Before you start, write down your predictions:

1. **What do you expect VRAM reduction to be vs BF16?** (ballpark %)
2. **Do you expect TTFT for a single request to improve, stay same, or get worse?**
3. **What startup differences might you see?** (download time, warmup, etc.)

#### 🔧 Implementation Notes

Create the script yourself. Here's the spec:

```
Script: ~/configs/vllm/serve_quant_model_rtx16gb.sh

Requirements:
- Same structure as serve_qwen2p5_1p5b_chat_16gb.sh
- Point to your chosen quantized model (you pick the HF model ID)
- Port 8001
- Same memory/sequence limits as Day 3 config
- Add --quantization flag if needed (depends on model)
```

**Test command** (after server is up):
```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "YOUR_MODEL_ID", "prompt": "Hello, how are you?", "max_tokens": 50}'
```

#### ✅ Acceptance Criteria

- [ ] Quantized vLLM server starts cleanly (no OOM) on RTX2000
- [ ] Test completion request returns a reasonable answer
- [ ] Recorded: BF16 idle VRAM vs quant idle VRAM (`nvidia-smi` snapshots)
- [ ] Short note on startup time differences

#### 📁 Artifacts

- `~/configs/vllm/serve_quant_model_rtx16gb.sh`
- `~/artifacts/day004_quant_vs_bf16_notes.md` (section: "Model & VRAM Baseline")

#### 🔗 Cross-Day Reuse

| From | Artifact | Usage |
|------|----------|-------|
| Day 003 | `~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh` | Copy structure, change model + port |

#### 📝 Feynman Deliverable

Add a section to `~/artifacts/day004_quant_vs_bf16_notes.md`:

> **5–10 sentences**: "What quantized model I picked, how vLLM sees it, how VRAM changed, what surprised me."

#### 🔄 Reflection Prompts (After)

1. Did VRAM savings match your expectations?
2. Did anything about startup/config for quantization surprise you?
3. What would you tell a client about "getting started with quantization"?

---

### ✅ Task 1.2: Chat Capacity – BF16 vs Quant (Core Slice)

**Tags**: `[Inference–Runtime]` `[Quant]` `[Batching]` `[KVCache]`  
**Time**: ~45 min  
**Win**: Empirical throughput + latency comparison at key operating points

#### 🎯 Objective

Compare BF16 vs quantized model for chat at your most relevant operating points on RTX2000.

#### 📋 Spec

Reuse Day 3 chat harness and grid:
- `~/scripts/benchmarks/vllm_chat_bench.py`
- `~/scripts/benchmarks/run_chat_capacity_grid.sh`

**Reduced grid** (focus on sweet spot):
- Concurrency: `1, 8, 16`
- max_tokens: `128`

#### 🧠 Deep-Thinking Prompts (Before)

1. **Will quantization help more at conc=1 or conc=16?** Why?
2. **Do you expect p95 latency to stay similar or degrade?**
3. **Which metric will show the biggest change?** (throughput, TTFT, or E2E)

#### 🔧 Implementation

**For BF16** (server on :8000):
```bash
# Start BF16 server first
~/configs/vllm/serve_qwen2p5_1p5b_chat_16gb.sh

# In another terminal, run grid
GPU_NAME="RTX2000-16GB-BF16" \
OUT_CSV="$HOME/benchmarks/day004_chat_capacity_rtx16gb_bf16.csv" \
URL="http://127.0.0.1:8000/v1/completions" \
  ~/scripts/benchmarks/run_chat_capacity_grid.sh
```

**For Quant** (server on :8001):
```bash
# Start quant server
~/configs/vllm/serve_quant_model_rtx16gb.sh

# In another terminal, run same grid
GPU_NAME="RTX2000-16GB-QUANT" \
OUT_CSV="$HOME/benchmarks/day004_chat_capacity_rtx16gb_quant.csv" \
URL="http://127.0.0.1:8001/v1/completions" \
  ~/scripts/benchmarks/run_chat_capacity_grid.sh
```

> **Note**: You may need to modify `run_chat_capacity_grid.sh` to honor `URL` and `OUT_CSV` environment variables if it doesn't already. This is a good exercise in script reusability.

#### 📊 Focus Columns

| Metric | What It Tells You |
|--------|-------------------|
| `p50_ttft_ms` | Median time to first token |
| `p95_ttft_ms` | Tail latency for first token |
| `p50_e2e_ms` | Median end-to-end latency |
| `p95_e2e_ms` | Tail end-to-end latency |
| `throughput_tok_s` | System capacity |

#### ✅ Acceptance Criteria

- [ ] Two CSVs exist:
  - `day004_chat_capacity_rtx16gb_bf16.csv`
  - `day004_chat_capacity_rtx16gb_quant.csv`
- [ ] Each has entries for conc=1/8/16, max_tokens=128
- [ ] You can state: "Tokens/sec improved by X% (or not) at conc=8/16"
- [ ] You can state: "p95 TTFT/E2E changed by Y%"

#### 📁 Artifacts

- `~/benchmarks/day004_chat_capacity_rtx16gb_bf16.csv`
- `~/benchmarks/day004_chat_capacity_rtx16gb_quant.csv`
- Summary table in `~/artifacts/day004_quant_vs_bf16_notes.md` (section: "Chat Capacity: BF16 vs Quant")

#### 🔗 Cross-Day Reuse

| From | Artifact | Usage |
|------|----------|-------|
| Day 003 | `~/benchmarks/day003_chat_capacity_rtx16gb.csv` | Baseline comparison |
| Day 003 | `~/scripts/benchmarks/run_chat_capacity_grid.sh` | Run as-is or adapt |
| Day 003 | `~/scripts/benchmarks/vllm_chat_bench.py` | Core benchmark engine |

#### 📝 Feynman Deliverable

In `day004_quant_vs_bf16_notes.md`, write:

> "At conc=1/8/16, max_tokens=128, quant vs BF16 →  
> - tokens/sec ratios: [your numbers]  
> - p95 patterns: [your observations]  
> - surprises: [what you didn't expect]"

#### 🔄 Reflection Prompts (After)

1. Did quantization primarily improve throughput, latency, or neither?
2. Did anything in p95_ttft_ms vs p95_e2e_ms suggest extra overhead for quant?
3. At what concurrency level does the difference matter most?

#### 🏋️ Practice Reps

Run additional variations to build intuition:
- [ ] conc=4, max_tokens=128 (between your tested points)
- [ ] conc=16, max_tokens=64 (shorter outputs)
- [ ] conc=16, max_tokens=256 (longer outputs — watch for quant quality)

---

### ✅ Task 1.3: Tiny Quality Sanity Check (Chat)

**Tags**: `[Model/Training]` `[Quant]` `[Reliability]`  
**Time**: ~30–45 min  
**Win**: Empirical confidence (or caution) about quant quality for chat

#### 🎯 Objective

Manually check whether the quantized model is "good enough" for a simple chat workload vs BF16.

#### 📋 Spec

Create a micro eval script:

```
Script: ~/scripts/benchmarks/day004_quant_quality_eval.py

Behavior:
1. Take a list of 10–15 realistic prompts
2. For each prompt:
   - Call BF16 model (:8000) → save output
   - Call quant model (:8001) → save output
3. Write results to: ~/benchmarks/day004_quant_quality_eval.json
   Format: [{"prompt": "...", "bf16_output": "...", "quant_output": "..."}, ...]
```

#### 🧠 Deep-Thinking Prompts (Before)

1. **Which task types break first under quantization?** (reasoning, code, precise facts, summarization?)
2. **For consulting, would you ever ship a quantized model without manual spot-checking?**
3. **What quality failure modes are you specifically looking for?**

#### 🔧 Implementation Notes

**You implement** the script. Here's a scaffold spec:

```python
# Scaffold: day004_quant_quality_eval.py
# 
# 1. Define PROMPTS list (10-15 diverse prompts)
# 2. For each prompt:
#    - POST to BF16 endpoint, extract response
#    - POST to QUANT endpoint, extract response
# 3. Save as JSON
#
# Prompt ideas:
# - Simple factual Q&A
# - "Explain X in simple terms"
# - Basic reasoning/math
# - Short summarization
# - Code snippet request
# - Edge cases you care about for your use cases
```

**Scripts** (reference only after attempting your own implementation):
- Scaffold: [`scripts/solutions/day004/quant_quality_eval_scaffold.py`](../../../scripts/solutions/day004/quant_quality_eval_scaffold.py)
- Solution: [`scripts/solutions/day004/quant_quality_eval_solution.py`](../../../scripts/solutions/day004/quant_quality_eval_solution.py)

**Example prompts to include**:
- "What is the capital of France?"
- "Explain how a transformer attention mechanism works in 3 sentences."
- "Write a Python function to reverse a string."
- "Summarize: [short paragraph]"
- "If I have 3 apples and give away 2, how many do I have left?"

#### ✅ Acceptance Criteria

- [ ] JSON file exists with at least 10 prompts
- [ ] You've manually inspected all outputs
- [ ] Annotated in `day004_quant_vs_bf16_notes.md`:
  - Any obvious degradations (nonsense, loss of nuance, repeated patterns)
  - Whether quant is "OK for internal tools / batch jobs" vs "not acceptable for X"

#### 📁 Artifacts

- `~/scripts/benchmarks/day004_quant_quality_eval.py` (you implement)
- `~/benchmarks/day004_quant_quality_eval.json`
- Summary in `~/artifacts/day004_quant_vs_bf16_notes.md` (section: "Quality Sanity Check")

#### 🔗 Cross-Day Reuse

| From | Source | Usage |
|------|--------|-------|
| Day 003 | Chat prompts used in benchmarks | Reuse similar prompt patterns |
| Your work | Consulting demo prompts | Real-world relevance |

#### 📝 Feynman Deliverable

Add a paragraph:

> "How I think about quality risk when quantizing: when manual spot-checking is enough, when I'd need proper automatic eval, and what failure modes I watch for."

#### 🔄 Reflection Prompts (After)

1. Did the quant model fail where you expected?
2. Do you now trust it for any specific workload class?
3. What would you tell a client about "quality risk from quantization"?

#### 🌍 Wider View

- **Production reality**: Most teams skip quality checks and regret it later
- **Enterprise**: Regulated industries need documented quality validation
- **Cost of failure**: One bad quant output in a customer-facing system = trust damage
- **Scale**: At 1M+ requests/day, even 0.1% quality regression = thousands of bad responses

---

## Tier 1 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **1.1** | Brought up quantized model, measured VRAM | ⬜ |
| **1.2** | Chat capacity: BF16 vs quant @ conc=1/8/16 | ⬜ |
| **1.3** | Quality sanity check: 10+ prompts compared | ⬜ |

### Artifacts Created

```
~/configs/vllm/
└── serve_quant_model_rtx16gb.sh

~/scripts/benchmarks/
└── day004_quant_quality_eval.py

~/benchmarks/
├── day004_chat_capacity_rtx16gb_bf16.csv
├── day004_chat_capacity_rtx16gb_quant.csv
└── day004_quant_quality_eval.json

~/artifacts/
└── day004_quant_vs_bf16_notes.md
```

---

## Best Practices Update

After completing Tier 1, append these to `INFERENCE_PERFORMANCE_BEST_PRACTICES.md`:

- [ ] **Quantization VRAM savings**: Document actual % reduction on your GPU class
- [ ] **Quality gating**: Never ship quantized model without at least 10-prompt spot check
- [ ] **Side-by-side comparison**: Always run BF16 baseline alongside quant for apples-to-apples
- [ ] **Port separation**: Use different ports for A/B model comparison to avoid confusion

---

**→ Continue to [Tier 2](LOG_tier02.md)**: Batch comparison + cost-per-token analysis
