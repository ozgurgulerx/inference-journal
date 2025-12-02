# Day 003 – vLLM Capacity, OOM Surface & Real Use-Cases
## Tier 3: Deep Work (~4+ hours)

> **Prerequisites**: Complete [Tier 1](LOG_tier01.md) and [Tier 2](LOG_tier02.md)  
> **Goal**: Build capacity frontier analysis + document "Life of a request"  
> **End State**: Unified capacity analysis, client-ready documentation  
> **Time**: ~4 hours

---

## Tier 3 Tasks

---

### ✅ Task 3.1: Build a Capacity Frontier Across GPUs & Workloads
**Tags**: `[Inference–Runtime]` `[Phase3-Optimization]`  
**Time**: 2–3 hours  
**Win**: Unified view – "for each GPU & workload, what's the sweet spot?"

#### 🔧 Lab Instructions

**Step 1: Create the analysis script**

```bash
cat > ~/scripts/benchmarks/analyze_capacity.py << 'EOF'
#!/usr/bin/env python3
"""
Analyze capacity CSVs and identify optimal configurations.

Outputs:
  - Markdown report with recommended settings
  - Identifies "sweet spots" for each GPU + workload combo
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def load_csv(filepath: str) -> list[dict]:
    """Load a benchmark CSV into a list of dicts."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key not in ('workload', 'gpu'):
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def find_sweet_spots(
    rows: list[dict],
    p95_threshold_chat: float = 2000.0,
    p95_threshold_batch: float = 5000.0,
) -> list[dict]:
    """
    Find optimal configurations where:
    - throughput is high
    - p95 is under threshold (varies by workload)
    """
    sweet_spots = []
    
    for row in rows:
        workload = row.get('workload', 'unknown')
        
        # Get p95 field (different CSVs may have different column names)
        p95 = row.get('p95_e2e_ms') or row.get('p95_latency_ms') or 0
        throughput = row.get('throughput_tok_s', 0)
        
        threshold = p95_threshold_chat if workload == 'chat' else p95_threshold_batch
        
        if p95 <= threshold and throughput > 0:
            sweet_spots.append({
                **row,
                'p95': p95,
                'threshold': threshold,
                'headroom_pct': round((1 - p95/threshold) * 100, 1)
            })
    
    # Sort by throughput descending
    sweet_spots.sort(key=lambda x: x.get('throughput_tok_s', 0), reverse=True)
    
    return sweet_spots


def generate_report(all_data: dict[str, list[dict]], output_path: str):
    """Generate a markdown report from analyzed data."""
    
    lines = [
        "# Day 003 – Capacity Frontier Report",
        "",
        "*Auto-generated from benchmark CSVs*",
        "",
        "---",
        "",
    ]
    
    # Group by GPU
    by_gpu_workload = defaultdict(list)
    
    for filepath, rows in all_data.items():
        sweet_spots = find_sweet_spots(rows)
        for spot in sweet_spots[:3]:  # Top 3 per file
            key = (spot.get('gpu', 'unknown'), spot.get('workload', 'unknown'))
            by_gpu_workload[key].append(spot)
    
    # Generate tables
    for (gpu, workload), spots in sorted(by_gpu_workload.items()):
        lines.append(f"## {gpu} – {workload.title()} Workload")
        lines.append("")
        lines.append("| Concurrency | max_tokens | Throughput (tok/s) | p95 (ms) | Headroom |")
        lines.append("|-------------|------------|--------------------| ---------|----------|")
        
        # Dedupe and take top 3
        seen = set()
        count = 0
        for spot in spots:
            key = (spot.get('concurrency'), spot.get('max_new_tokens'))
            if key in seen:
                continue
            seen.add(key)
            
            lines.append(
                f"| {int(spot.get('concurrency', 0))} "
                f"| {int(spot.get('max_new_tokens', 0))} "
                f"| {spot.get('throughput_tok_s', 0):.1f} "
                f"| {spot.get('p95', 0):.0f} "
                f"| {spot.get('headroom_pct', 0):.0f}% |"
            )
            count += 1
            if count >= 3:
                break
        
        lines.append("")
        lines.append(f"**Recommendation**: Use concurrency={int(spots[0].get('concurrency', 8))} "
                    f"with max_tokens={int(spots[0].get('max_new_tokens', 128))} for best throughput "
                    f"within latency budget.")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text('\n'.join(lines))
    print(f"[✓] Report written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="benchmarks",
                        help="Directory containing benchmark CSVs")
    parser.add_argument("--output", default="reports/day003_capacity_frontier.md",
                        help="Output markdown report path")
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir).expanduser()
    
    # Find all day003 capacity CSVs
    csv_files = list(csv_dir.glob("day003_*_capacity_*.csv"))
    
    if not csv_files:
        print(f"[!] No CSV files found in {csv_dir}")
        print("    Looking for: day003_*_capacity_*.csv")
        exit(1)
    
    print(f"[*] Found {len(csv_files)} capacity CSVs:")
    for f in csv_files:
        print(f"    - {f.name}")
    
    # Load all CSVs
    all_data = {}
    for csv_file in csv_files:
        all_data[str(csv_file)] = load_csv(csv_file)
    
    # Generate report
    generate_report(all_data, args.output)
EOF

chmod +x ~/scripts/benchmarks/analyze_capacity.py
```

**Step 2: Run analysis**

```bash
mkdir -p ~/reports

python ~/scripts/benchmarks/analyze_capacity.py \
  --csv-dir ~/benchmarks \
  --output ~/reports/day003_capacity_frontier.md

cat ~/reports/day003_capacity_frontier.md
```

**Step 3: (Optional) Add visualization**

```bash
cat > ~/scripts/benchmarks/plot_capacity.py << 'EOF'
#!/usr/bin/env python3
"""
Simple capacity visualization using matplotlib.
"""

import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[!] matplotlib not installed, skipping plots")


def load_csv(filepath: str) -> list[dict]:
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key not in ('workload', 'gpu'):
                    try:
                        row[key] = float(row[key])
                    except:
                        pass
            rows.append(row)
    return rows


def plot_throughput_vs_concurrency(csv_path: str, output_path: str):
    if not HAS_MATPLOTLIB:
        return
    
    rows = load_csv(csv_path)
    
    # Group by max_new_tokens
    by_mnt = {}
    for row in rows:
        mnt = int(row.get('max_new_tokens', 0))
        if mnt not in by_mnt:
            by_mnt[mnt] = {'conc': [], 'throughput': [], 'p95': []}
        by_mnt[mnt]['conc'].append(row.get('concurrency', 0))
        by_mnt[mnt]['throughput'].append(row.get('throughput_tok_s', 0))
        by_mnt[mnt]['p95'].append(row.get('p95_e2e_ms') or row.get('p95_latency_ms') or 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for mnt, data in sorted(by_mnt.items()):
        ax1.plot(data['conc'], data['throughput'], 'o-', label=f'max_tokens={mnt}')
        ax2.plot(data['conc'], data['p95'], 'o-', label=f'max_tokens={mnt}')
    
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_title('Throughput vs Concurrency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('p95 Latency (ms)')
    ax2.set_title('p95 Latency vs Concurrency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Plot saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    csv_files = list(Path("~/benchmarks").expanduser().glob("day003_*_capacity_*.csv"))
    
    for csv_file in csv_files:
        output = csv_file.with_suffix('.png')
        plot_throughput_vs_concurrency(str(csv_file), str(output))
EOF

# Install matplotlib if needed and run
pip install matplotlib -q
python ~/scripts/benchmarks/plot_capacity.py
```

#### 📁 Artifacts
- `~/scripts/benchmarks/analyze_capacity.py`
- `~/scripts/benchmarks/plot_capacity.py`
- `~/reports/day003_capacity_frontier.md`
- `~/benchmarks/*.png` (capacity plots)

#### 💡 Why This Matters
This is exactly the kind of document you'd show a client to justify your config choices.

#### 🏆 Success Criteria
- [ ] Analysis script processes all benchmark CSVs
- [ ] Markdown report identifies sweet spots per GPU/workload
- [ ] (Optional) Visualization plots generated

---

### ✅ Task 3.2: "Life of a Request" Documentation
**Tags**: `[Documentation]` `[Inference–Runtime]` `[Advanced-vLLM-Internals]`  
**Time**: 1–1.5 hours  
**Win**: Connect the numbers to a mental model of vLLM internals

#### 🔧 Lab Instructions

```bash
mkdir -p ~/reports

cat > ~/reports/day003_life_of_request_vllm_single_gpu.md << 'EOF'
# Day 003 – Life of a vLLM Request on Single-GPU (RunPod)

*A practical guide grounded in Day 003 experiments*

---

## 1. Workloads and Configs Tested

### Chat Workload
- **Model**: Qwen2.5-1.5B-Instruct (BF16)
- **Context**: max_model_len=4096
- **Concurrency tested**: 1, 4, 8, 16
- **Output length**: 64–512 tokens
- **Priority**: Low latency (p95 < 2s)

### Batch Summarization Workload
- **Model**: Same
- **Context**: Same
- **Concurrency tested**: 8, 16, 32
- **Output length**: 256–512 tokens
- **Priority**: High throughput (tok/s)

---

## 2. Request Lifecycle in vLLM

### Step 1: HTTP Request Arrives
```
Client → POST /v1/completions → vLLM API Server
```
- Request is validated and queued
- Tokenizer encodes prompt

### Step 2: Scheduler Enqueues Request
```
Scheduler → Batch Formation → GPU Dispatch
```
- vLLM's continuous batching scheduler merges the request with others
- **Observation from Day 003**: At concurrency=16, we saw batch sizes of 8–16 form naturally
- Higher concurrency → larger batches → better GPU utilization → higher throughput

### Step 3: KV Cache Allocation (PagedAttention)
```
PagedAttention → Allocate KV Pages → Memory Management
```
- vLLM allocates KV cache pages based on:
  - `max_model_len` (4096 in our tests)
  - `max_num_seqs` (128 in our config)
- **Key finding**: With gpu_memory_utilization=0.8, we have ~8 GiB for KV cache
- Each request at 4096 context consumes significant KV budget

### Step 4: Prefill Phase
```
Prompt Tokens → Parallel Processing → Initial KV Cache Population
```
- All prompt tokens processed in parallel
- This is where chunked_prefill helps (max_num_batched_tokens=2048)
- **Observation**: Longer prompts (batch summarization) have higher TTFT

### Step 5: Decode Phase
```
Autoregressive Generation → One Token at a Time → Streaming Output
```
- Each new token requires:
  - KV cache lookup for all previous tokens
  - Single forward pass
  - Token sampling
- **Observation**: Decode is memory-bandwidth bound, not compute bound

### Step 6: Response Completion
```
EOS Token or max_tokens → Cleanup → HTTP Response
```
- KV cache pages freed for reuse
- Response returned to client

---

## 3. Observed Behavior from Experiments

### p95 TTFT vs Concurrency

| Concurrency | p95 TTFT (ms) | Notes |
|-------------|---------------|-------|
| 1 | [YOUR VALUE] | Baseline, no batching |
| 4 | [YOUR VALUE] | Some batching benefit |
| 8 | [YOUR VALUE] | Good utilization |
| 16 | [YOUR VALUE] | Near saturation |

**Pattern**: TTFT increases with concurrency due to scheduler queuing and larger batch prefill.

### Throughput Saturation Point

| GPU | Chat Saturation | Batch Saturation |
|-----|-----------------|------------------|
| RTX 16GB | ~[X] conc | ~[X] conc |
| A100 40GB | ~[X] conc | ~[X] conc |

**Pattern**: Throughput plateaus when GPU compute becomes the bottleneck.

### OOM / Instability Triggers

| Scenario | Result |
|----------|--------|
| max_num_seqs=256 with max_model_len=8192 | [YOUR OBSERVATION] |
| gpu_memory_utilization=0.95 | [YOUR OBSERVATION] |
| concurrency=64 on RTX 16GB | [YOUR OBSERVATION] |

---

## 4. Mental Model

### For Single-GPU vLLM, Capacity is Governed By:

1. **VRAM Budget**
   ```
   VRAM = Model Weights + KV Cache + CUDA Graphs + Overhead
   ```

2. **KV Cache Sizing**
   ```
   KV Tokens = (VRAM - Weights - Overhead) × gpu_mem_util / bytes_per_token
   ```

3. **Concurrency Limit**
   ```
   Max Concurrent Requests ≈ KV Tokens / avg_sequence_length
   ```

4. **Scheduler Pressure**
   - More concurrent requests → larger batches
   - Larger batches → higher throughput but higher latency
   - Beyond saturation → queuing delays dominate

### PagedAttention's Role

- **Reduces fragmentation** but does not make OOM impossible
- Allows dynamic allocation/deallocation as requests start/finish
- Critical for handling variable-length sequences efficiently

---

## 5. Implications for Practice

### Latency-Sensitive Chat
- **Concurrency**: Keep at 8–16 for RTX-class GPUs
- **max_model_len**: 4096 is usually sufficient
- **gpu_memory_utilization**: 0.8 for safety margin
- **Target**: p95 < 1000–2000ms

### Throughput-Sensitive Batch
- **Concurrency**: Push to 16–32+ 
- **max_model_len**: Match your actual document lengths
- **gpu_memory_utilization**: Can go to 0.85–0.9
- **Accept**: Higher p95 (3000–5000ms) for better tok/s

### When to Upgrade GPU

| Symptom | Solution |
|---------|----------|
| p95 exceeds SLA at low concurrency | Faster GPU (A100/H100) |
| OOM at required context length | More VRAM (40GB+) |
| Throughput insufficient | Larger GPU or tensor parallelism |

---

## 6. Key Numbers from Day 003

*Fill in from your experiments:*

| Metric | RTX 16GB | A100 40GB |
|--------|----------|-----------|
| Max stable chat concurrency | [X] | [X] |
| Max chat throughput (tok/s) | [X] | [X] |
| Max batch concurrency | [X] | [X] |
| Max batch throughput (tok/s) | [X] | [X] |
| p95 at optimal config | [X] ms | [X] ms |

---

*This document connects theory to practice based on hands-on experiments.*
EOF

echo "Created ~/reports/day003_life_of_request_vllm_single_gpu.md"
echo "Fill in [YOUR VALUE] placeholders with your actual measurements!"
```

#### 📁 Artifacts
- `~/reports/day003_life_of_request_vllm_single_gpu.md`

#### 🏆 Success Criteria
- [ ] Document created with all sections
- [ ] Placeholders filled with actual measurements
- [ ] Mental model connects to observed behavior

---

## Tier 3 Summary

| Task | What You Did | Status |
|------|--------------|--------|
| **3.1** | Capacity frontier analysis | ⬜ |
| **3.2** | Life of a request documentation | ⬜ |

### Artifacts Created
```
~/scripts/benchmarks/
├── analyze_capacity.py
└── plot_capacity.py

~/reports/
├── day003_capacity_frontier.md
└── day003_life_of_request_vllm_single_gpu.md

~/benchmarks/
└── *.png (capacity plots)
```

---

**→ Continue to [Tier 4](LOG_tier04.md)**: Playbook, commit, and quiz
