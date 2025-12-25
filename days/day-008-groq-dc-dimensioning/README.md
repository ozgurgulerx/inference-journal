# Day 008 – Groq LPU Inference + Data Center Dimensioning (Groq-Native Playbook)

**Objective:** Build a Groq-specific inference engineering + facility design playbook that lets you lead (not follow) design reviews with Groq compiler, hardware, fabric, and data center engineers.

**Scope:** Groq LPU behavior under real inference load, deterministic scheduling implications, RealScale-style rack-scale fabric intuition, and an explicit capacity/dimensioning model that does **not** import GPU assumptions.

**How To Use This Day**

- Start with `LOG_tier01.md` to rebuild your mental model around determinism + compiler-first execution.
- Use `LOG_tier02.md` to convert that mental model into sizing math, SLOs, rack design, and expansion sequencing.
- Use `LOG_tier03.md` to pressure-test the design against failure modes, awkward model shapes, and “GPU intuition traps”.
- Run a real meeting using `groq_meeting_runbook.md`, and use `questions_for_groq.md` + `answers_bank_top1percent.md` to control the narrative.

---

## What Changes When You Stop Thinking “GPU”

### Determinism Is The Product

- **Fact (from Groq reference):** Groq’s core promise is deterministic execution with compiler-scheduled data movement and compute. *(Attach the specific Groq reference in your engagement notes.)*
- **Inference:** If service time becomes predictable, **p99 latency stops being dominated by kernel/cache variance** and becomes dominated by **queueing + admission control**.
- **GPU intuition trap:** On GPUs, “variance is normal” (warp scheduling, cache misses, HBM contention, kernel launch jitter). On Groq, *variance is a bug*—you dimension for queueing, not micro-variance.

### The Compiler Is The Runtime

- **Fact (from Groq reference):** Groq compiles the model to a static schedule; runtime variability is intentionally minimized. *(Attach the specific Groq reference in your engagement notes.)*
- **Consequence:** Your “deployment artifact” is no longer “container + model weights”; it’s **(model, graph shape, batch/sequence constraints, compiler version, target hardware)** → compiled binary/schedule.
- **GPU intuition trap:** “We’ll just change the batch size later.” On Groq, changing shape may mean **recompile** and potentially different placement; shape is a first-class ops decision.

### Memory Hierarchy Inversion (SRAM-first Mental Model)

- **Fact (from Groq reference):** Groq emphasizes on-chip SRAM and deterministic access patterns rather than GPU-style cache hierarchies. *(Attach the specific Groq reference in your engagement notes.)*
- **Consequence:** You trade cache-miss chaos for **explicit capacity constraints** and **compile-time shape constraints**. You can’t “just rely on cache to save you.”
- **GPU intuition trap:** “The cache will probably hit.” On Groq, you must ask: **where does every tensor live, when, and for how long?**

---

## Deliverables (This Folder)

- `LOG_tier01.md`: Groq LPU mental model, terminology, deterministic inference intuition.
- `LOG_tier02.md`: Dimensioning model, SLO mapping, rack/facilities translation, scale strategy.
- `LOG_tier03.md`: Failure modes, compiler constraints, cross-rack scaling realities, expert reasoning.
- `groq_meeting_runbook.md`: Agenda + decision-forcing prompts + whiteboard flow.
- `dimensioning_inputs_template.md`: The exact fields you must collect.
- `capacity_model_template.md`: A fill-in-the-blanks capacity model.
- `questions_for_groq.md`: High-signal questions grouped by domain.
- `answers_bank_top1percent.md`: Top-1% answer patterns and explanations.
- `risk_register.md`: Operational + technical risks with mitigations.
- `glossary.md`: Consistent vocabulary to prevent “same word, different meaning” failures.
- `acceptance_criteria.md`: Quality bar and self-checks for credibility.
- `scripts/`: Pseudocode templates for modeling.
