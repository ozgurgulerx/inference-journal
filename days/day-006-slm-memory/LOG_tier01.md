# Day 006 – SLM + OS Memory & First-Token Path
## Tier 1 – SLM + OS Memory Baseline (THP, Hugepages, Cold/Warm Load)

> **Goal**: Use a small language model (SLM) as a **fast probe** to measure how THP/hugepages affect load time and memory, creating a baseline before vLLM experiments.

---

**Related theory**:

- `theory/slms_as_probes.md` – why SLMs are ideal OS/runtime probes.  
- `theory/day06_theory_huge_pages.md` – pages, THP vs explicit hugepages, DMA/IOMMU.  
- `theory/caching_cold_warm_loads.md` – page cache, cold vs warm loads, and tunable knobs.

## Tier 1 – Must Do (Core Block)

**Title** – SLM + OS Memory Baseline (THP, hugepages, cold/warm load)  
**Time Budget** – ~75–90 min  
**Outcome** – One SLM, cold vs warm load numbers, and memory snapshots under a known THP/hugepage configuration.

---

### 1. Pick the SLM and Set Up the Day Folder

From repo root:

```bash
mkdir -p days/day-006-slm-memory
cd days/day-006-slm-memory
```

Decide on one SLM and keep it consistent all day. Example:

```python
# MODEL = "microsoft/Phi-3-mini-4k-instruct"  # or "Qwen2.5-1.5B-Instruct"
```

Document the chosen model at the top of `slm_load.py` and in `README.md`.

---

### 2. Inspect Current Memory / THP State (Baseline)

Capture the current hugepage/THP configuration:

```bash
grep -i huge /proc/meminfo | head
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag || true
```

Save these snippets into `README.md` or a small `thp_state_before.txt` if desired.

Optionally, treat this as **THP Mode A** (current defaults) for a before/after comparison.

---

### 3. THP/Hugepages Before/After Experiment

First, run a **baseline load test** under the current THP settings:

```bash
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
/usr/bin/time -f "thp_mode=A (current) cold_load_real_s=%E" python slm_load.py

/usr/bin/time -f "thp_mode=A (current) warm_load_real_s=%E" python slm_load.py
```

> **Note**: `drop_caches` is safe for experiments on a lab node but should never be used on production machines; it intentionally discards the page cache. See `theory/caching_cold_warm_loads.md` for guidance.

Capture these values in a CSV like `thp_load_comparison.csv`:

```text
thp_mode,cold_load_real_s,warm_load_real_s,rss_mb
A-current,....,....,...
```

Then configure THP + hugepages for inference, treating this as **THP Mode B**:

Tweak towards “explicit big pages, no surprise THP everywhere”:

```bash
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never   | sudo tee /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

# 2 GB via 2 MB pages
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
```

Note the exact values you set in `README.md` (THP mode, `nr_hugepages`).

Run the load test again under Mode B:

```bash
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
/usr/bin/time -f "thp_mode=B (madvise+hugepages) cold_load_real_s=%E" python slm_load.py

/usr/bin/time -f "thp_mode=B (madvise+hugepages) warm_load_real_s=%E" python slm_load.py
```

Append these values to `thp_load_comparison.csv`:

```text
thp_mode,cold_load_real_s,warm_load_real_s,rss_mb
A-current,....,....,...
B-madvise,....,....,...
```

---

### 4. Mount a Hugepage Filesystem (Optional but Useful Later)

```bash
sudo mkdir -p /mnt/huge
sudo mount -t hugetlbfs none /mnt/huge
```

You’ll likely use this later for pinned allocations / experiments; for now, just note that it exists.

#### Reverting THP/Hugepages (Safety)

If you need to revert to more typical defaults after experiments:

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo 0      | sudo tee /proc/sys/vm/nr_hugepages
```

Adjust back to your environment’s standard policy as needed.

---

### 5. Create Minimal SLM Loader Script (HF Only, No vLLM Yet)

Create `days/day-006-slm-memory/slm_load.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/Phi-3-mini-4k-instruct"  # keep consistent all day

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
)
print("loaded", MODEL, "on", model.device)
```

Ensure it runs once end-to-end before measuring.

---

### 6. Measure Cold vs Warm Load Times

From `days/day-006-slm-memory`:

```bash
# Cold: clear page cache
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
/usr/bin/time -f "cold_load_real_s=%E" python slm_load.py

# Warm: second run
/usr/bin/time -f "warm_load_real_s=%E" python slm_load.py
```

Record the `cold_load_real_s` and `warm_load_real_s` values (and any notable console logs) into `README.md`.

If you maintained `thp_load_comparison.csv`, ensure both Mode A and Mode B rows include an approximate `rss_mb` column derived from the next step.

---

### 7. Capture Host Memory Footprint After Load

In another terminal while the model is resident in memory:

```bash
ps -C python -o pid,rss,vsz,cmd | head -n 5
```

Optionally convert RSS/VSZ to MB/GB in `README.md` for clarity.

You can reuse these RSS numbers to populate the `rss_mb` column in `thp_load_comparison.csv` for both THP modes.

---

### 8. Document Results

Create or update `days/day-006-slm-memory/README.md` with:

- Chosen `MODEL` string.  
- THP & hugepages configuration lines you set.  
- Cold vs warm `real` times (from `/usr/bin/time`).  
- RSS / VSZ snapshot and any qualitative notes (“warm load almost instant; cold dominated by disk I/O,” etc.).

---

### Tier 1 Artifacts

- `days/day-006-slm-memory/slm_load.py`  
- `days/day-006-slm-memory/README.md` (short, with config + numbers)
