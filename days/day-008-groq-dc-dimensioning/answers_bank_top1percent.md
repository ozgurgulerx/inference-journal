# Answers Bank (Top 1%) — Groq-Native Responses Under Whiteboard Pressure

This file is designed for live meetings. Each question includes:

- **One-liner**
- **90-second confident explanation**
- **Deep technical follow-up**
- **Red flags**
- **Close with a decision**

**Constraint:** Avoid GPU analogies unless the point is to explicitly identify a GPU intuition trap.

**Epistemic rule:** When you make a claim about Groq specifics (compiler behavior, fabric, topology, telemetry), either (1) cite it as **Fact (from Groq reference)** in your engagement notes, or (2) explicitly say **Assumption to validate** and ask for confirmation. Use **Inference** only when you can derive it from confirmed premises.

---

## Hierarchical Meeting With Groq Engineers: Top-1% Answers

### Exec / PM (12)

1) **“Why Groq vs GPUs for this product?”**

- One-liner: Deterministic batch-1 performance lets us engineer predictable p99 without hiding behind batching.
- 90s: With GPUs, p99 often swings due to runtime scheduling and memory effects; Groq front-loads scheduling into compilation, making service time predictable for fixed shapes. That shifts the system problem from “fight variance” to “manage queueing,” which is a tractable product/SLO lever.
- Deep: Determinism gives a stable service curve; once we lock shape buckets, we can treat capacity as near-linear scaling, and use admission control to bound tails. The engineering risk becomes shape discipline and artifact ops, not micro-variance firefighting.
- Red flags: “We’ll support arbitrary shapes” / “we’ll just turn on dynamic batching” / “we can always scale over Ethernet.”
- Close: Decide the shape policy and the determinism boundary (rack vs pod) for the first production launch.

2) **“What’s the single biggest operational risk?”**

- One-liner: Shape sprawl—too many variants forces recompiles and breaks predictable capacity planning.
- 90s: Groq’s benefits require stable compilation targets; if product allows unbounded context/output, we’ll either overprovision massively or violate SLO via queueing. We must bucket and cap shapes and treat long-context as a separate tier.
- Deep: Artifact count grows combinatorially with shape dimensions; ops and QA become the bottleneck. The fix is product-level contract: buckets + pricing/SLO tiers + explicit rejection behavior.
- Red flags: “Let users pick any max tokens” / “we’ll compile on deploy.”
- Close: Approve a bucketed contract and a long-context tier.

3) **“How fast do we scale capacity?”**

- One-liner: More replicas buys near-linear throughput, but p99 is bounded by admission and headroom.
- 90s: Because service time is stable, we can size by tokens/s and request service time per bucket. Scaling is adding pods with the same artifact set. The hard part is ensuring failure-domain redundancy and keeping the critical path within the deterministic fabric domain.
- Deep: Build a pod as a repeatable unit with measured service curves; capacity planning becomes adding pods, not tuning. The risk is cross-rack comm and compilation constraints, not kernel tuning.
- Red flags: “We’ll run at 95% utilization” / “we don’t need queue caps.”
- Close: Approve `ρ_target` and the N+1 scope (node vs rack).

4) **“What is the SLO lever unique to Groq?”**

- One-liner: Admission control based on deterministic cost, not heuristic batching.
- 90s: We can compute an estimated service time from (prompt bucket, output cap, artifact) and admit only when the deterministic capacity is available. That makes p99 a policy choice rather than a surprise.
- Deep: Use token-budget admission: reserve prefill and decode budgets; bound queue length/time; separate pools by bucket. This turns tail latency into a system control problem.
- Red flags: “just queue everything” / “let the runtime sort it out.”
- Close: Decide the admission controller model (token-budget vs request-count) and queue caps.

5) **“What does a ‘pod’ mean?”**

- One-liner: A repeatable rack group that preserves determinism and isolates failures.
- 90s: A pod is the smallest unit we can replicate that contains a failure domain, has a known deterministic boundary, and has a stable cable/power design. We scale by adding pods, not by ad hoc racks.
- Deep: Pods enforce repeatable topology and make burn-in and SLO validation deterministic; they also simplify spares and incident response.
- Red flags: “We’ll just add a rack wherever” / “we’ll cable later.”
- Close: Decide the pod unit (racks per pod) and its deterministic boundary.

6) **“Can we do A/B tests?”**

- One-liner: Yes, but we A/B compiled artifacts, not raw weights; precompile is mandatory.
- 90s: Each variant is a compiled artifact with a known service curve. We precompile and canary them and route traffic via L7 policies.
- Deep: We need artifact registry, compile CI, and deterministic performance tests per build. A/B becomes a supply chain problem: artifact builds, signature, rollbacks.
- Red flags: “Compile in prod on request.”
- Close: Decide artifact lifecycle: precompile + registry + canary gates.

7) **“What happens under failures?”**

- One-liner: Determinism makes failures more visible; you either shed load or you violate p99 due to queueing.
- 90s: Losing capacity shifts utilization upward; queueing explodes near saturation. We must choose: degrade SLO, shed traffic, or hold headroom for N-1.
- Deep: Model N-1 at rack or link segment; pre-plan reroute time; implement bounded queues; have spare capacity online.
- Red flags: “We’ll just fail over and keep p99.”
- Close: Decide N+1 scope and degrade vs shed policy.

8) **“Where does Ethernet fit?”**

- One-liner: Ethernet is for ingress/control/egress unless Groq guarantees it for critical-path partitioning.
- 90s: Determinism depends on a controlled fabric domain. Ethernet introduces congestion variability and unpredictable routing; we use it for non-critical flows unless Groq provides an explicit SLO envelope.
- Deep: Separate planes: data-plane inference within deterministic domain; control-plane over Ethernet; isolate and monitor both.
- Red flags: “Let’s just use the existing DC network for model-parallel traffic.”
- Close: Decide the determinism boundary and enforce it in architecture.

9) **“What’s the bring-up timeline driver?”**

- One-liner: Facilities lead times (power/cooling/cabling) and artifact readiness, not kernel tuning.
- 90s: If the pod design is repeatable and air cooling fits, bring-up is mostly physical install + burn-in + telemetry and compile artifact readiness.
- Deep: Build a burn-in plan: thermal, power, link validation, perf acceptance tests.
- Red flags: “We’ll validate after go-live.”
- Close: Approve bring-up acceptance criteria and burn-in plan.

10) **“What does success look like?”**

- One-liner: Predictable p99 under load with explicit headroom and clear failure behavior.
- 90s: We should be able to predict p99 from (traffic envelope, headroom, queue caps) and see it match. Incidents should be capacity/failure-domain events, not latency variance mysteries.
- Deep: Success is an engineered service curve + bounded queueing + stable artifacts + repeatable pods.
- Red flags: “We can’t explain p99 spikes.”
- Close: Decide SLO targets and publish the capacity model.

11) **“What’s the cost model?”**

- One-liner: Cost is tokens/s capacity + headroom + redundancy; determinism reduces overprovision for variance but not for burst.
- 90s: We price capacity by token demand per bucket. Headroom is a deliberate cost to protect p99 and failures. Determinism reduces waste caused by variance-driven overprovisioning, but burst and long prompts still dominate.
- Deep: Use bucketed pricing/SLO tiers; isolate long-context to avoid tail externalities.
- Red flags: “One price, one SLO, any shape.”
- Close: Decide tiering and bucket policy tied to cost.

12) **“What’s the main organizational change?”**

- One-liner: Compilation becomes part of release engineering; shape becomes a product contract.
- 90s: The release unit is compiled artifacts; product must accept shape caps and tiers; infra must support artifact registry and deterministic performance QA.
- Deep: Build a pipeline: model changes → compile → perf tests → canary → rollout; instrument service time and queueing separately.
- Red flags: “Treat Groq like a drop-in GPU.”
- Close: Assign ownership of artifact ops and shape policy.

---

### Compiler Engineers (12)

1) **“What shape variability can we support without recompilation?”**

- One-liner: We assume shape buckets; we won’t promise arbitrary lengths without a validated schedule strategy.
- 90s: Our ops model is discrete buckets with caps. If you can support limited variability inside a bucket (e.g., padding/packing), we’ll use it; otherwise we treat shape as compile-time.
- Deep: We’ll define acceptable shape degrees of freedom: prompt length range, output cap, batch size, and any padding strategy. We’ll version artifacts accordingly and route requests by bucket.
- Red flags: “Just compile one artifact for everything.”
- Close: Decide the supported degrees of freedom per artifact and how padding affects determinism.

2) **“How do we predict service time from compiler outputs?”**

- One-liner: We want compiler-reported critical path and per-token schedule timing to drive admission control.
- 90s: Determinism is only useful if we can extract a predictable service curve. We need either a compiler report or a stable profiling method to compute `S(shape)`.
- Deep: We’ll build a regression model from compiler schedule metrics to measured latencies per bucket; if compiler can emit exact cycle counts per token/request, even better.
- Red flags: “We can’t estimate service time; we just benchmark.”
- Close: Decide the canonical performance report and how it maps to admission policy.

3) **“What are the known compiler cliffs?”**

- One-liner: We need a documented set of patterns that degrade placement or introduce bubbles.
- 90s: We’ll adopt a “compiler-friendly operator subset” policy. If certain ops/graph patterns cause fallback or slow schedules, we’ll detect them in CI.
- Deep: We’ll add compile-time checks: operator whitelist, shape guardrails, and performance gates per artifact.
- Red flags: “Performance changes are unpredictable across minor graph edits.”
- Close: Agree on a list of cliffs + how we detect them automatically.

4) **“How stable are artifacts across compiler versions?”**

- One-liner: We assume artifacts are compiler-version-bound unless you guarantee ABI stability.
- 90s: Release engineering requires pinning compiler versions to avoid unplanned service-time shifts. If there is forward compatibility, we’ll exploit it; otherwise we treat it as a strict dependency.
- Deep: We’ll maintain a matrix: (compiler version × hardware generation × model) with perf baselines and rollback paths.
- Red flags: “Silent performance regressions across updates.”
- Close: Decide the compatibility policy and required validation gates.

5) **“Can we do multi-tenant scheduling at runtime?”**

- One-liner: We prefer pool-level isolation; runtime multi-tenancy must preserve determinism guarantees.
- 90s: If runtime can multiplex artifacts, we need isolation guarantees. Otherwise we’ll allocate pools per artifact/bucket and route at L7.
- Deep: Our approach: admission control and strict queue caps per pool; any runtime multiplexing must be proven not to introduce interference.
- Red flags: “Multi-tenancy without deterministic isolation.”
- Close: Decide tenancy model: per-pool or shared, with explicit isolation guarantees.

6) **“What is the compile/build pipeline we should adopt?”**

- One-liner: Treat compile as CI with artifact registry and performance gates.
- 90s: Every model change triggers compilation for all supported buckets; artifacts are immutable, signed, and stored; we run deterministic perf tests per artifact.
- Deep: Define reproducibility: inputs hashed (weights, graph, compiler flags); compile outputs deterministic; performance regression thresholds enforced.
- Red flags: “We compile manually and hope it works.”
- Close: Decide artifact provenance and required CI gates.

7) **“How do we handle long-context?”**

- One-liner: Long-context is a distinct tier with distinct artifacts and capacity.
- 90s: We separate long-context requests to prevent tail externalities. We compile and benchmark long-context shapes separately and allocate dedicated pools.
- Deep: The service curve is different; prefill dominates; queueing impacts are amplified. We enforce stricter admission and different pricing/SLO.
- Red flags: “One pool serves all context lengths.”
- Close: Approve the long-context tier boundary.

8) **“What is the correct mental model for ‘batch’ on Groq?”**

- One-liner: Batch is a compile-time schedule choice, not a runtime opportunistic merge.
- 90s: If batching exists, it should be explicit: we compile for a chosen batch and keep shapes disciplined. We avoid GPU-style dynamic batching windows.
- Deep: For heterogeneous lengths, we bucket/pad. Any runtime batching must not violate deterministic service-time predictability.
- Red flags: “We’ll just implement vLLM-like continuous batching.”
- Close: Decide: fixed batch schedules vs batch-1 with concurrency.

9) **“How do we expose bubbles/stalls?”**

- One-liner: We need a schedule utilization metric to know if we’re compute- or bandwidth-bound.
- 90s: Deterministic pipelines can have bubbles; we need telemetry that maps directly to schedule inefficiency to guide model/shape changes.
- Deep: Require per-stage utilization, memory bandwidth occupancy, and critical-path attribution.
- Red flags: “Only end-to-end latency is visible.”
- Close: Decide minimum telemetry and how it’s consumed in ops.

10) **“What model changes are low risk?”**

- One-liner: Changes that preserve operator set and shape buckets; anything else requires a new artifact validation cycle.
- 90s: We’ll classify changes: weights-only vs graph changes vs operator changes vs shape changes. Each class has a different compile/test pipeline.
- Deep: Implement artifact promotion gates keyed by change class and measured service curve deltas.
- Red flags: “We can’t predict operational impact of changes.”
- Close: Approve the change taxonomy and gates.

11) **“How do you want us to define success for compilation?”**

- One-liner: Predictable performance with bounded artifact count and stable build times.
- 90s: We want a small set of artifacts with predictable service curves and no surprise cliffs; compilation times must fit release cadence.
- Deep: SLA the compile pipeline: time-to-compile, determinism, perf variance across builds, and failure rates.
- Red flags: “Compilation is a black box.”
- Close: Decide compile SLA and reporting.

12) **“What do you need from us?”**

- One-liner: A stable contract: supported shapes, expected service curves, and version compatibility policy.
- 90s: The infra story depends on those contracts to build admission control, redundancy, and facility sizing.
- Deep: Provide formal interfaces: artifact metadata schema, performance report schema, and compatibility matrix.
- Red flags: “We can’t commit to any envelope.”
- Close: Agree on deliverables and due dates.

---

### Hardware Architects (12)

1) **“Where are the real bottlenecks: compute, SRAM, bandwidth?”**

- One-liner: We want a deterministic attribution: which resource is critical-path for our artifact.
- 90s: For a fixed schedule, bottlenecks are stable. We need a per-artifact breakdown so we can reason about scaling and headroom.
- Deep: Provide saturation curves and counters for compute utilization, SRAM occupancy, and bandwidth; identify bubbles.
- Red flags: “It depends” without counters.
- Close: Decide the canonical bottleneck attribution method.

2) **“What causes performance cliffs?”**

- One-liner: We treat cliffs as design constraints; document them so we can avoid them by policy.
- 90s: Cliffs could be placement failures, spills, bandwidth oversubscription, or schedule imbalance. We need them enumerated.
- Deep: Provide thresholds and symptoms; we’ll encode them into CI and runtime guardrails.
- Red flags: “You’ll notice when it’s slow.”
- Close: Decide the cliff list and how to detect them.

3) **“How does power/thermal behave at sustained decode?”**

- One-liner: We need the sustained power envelope for worst-case load to size racks safely.
- 90s: Deterministic throughput encourages running hot; facilities need sustained draw and transient behavior to avoid trips and hotspots.
- Deep: Provide steady-state kW, transient spikes, derating guidance, and thermal throttling behavior.
- Red flags: “Power is nominal only.”
- Close: Decide the facility design power spec (nominal vs worst-case).

4) **“What is the unit of redundancy?”**

- One-liner: We size redundancy by the true failure granularity: LPU, node, link segment, rack.
- 90s: If one LPU failure disables a partition, that dictates N+1 design. We need to understand what a single failure does to capacity and whether it degrades performance or hard-fails.
- Deep: Provide failure impact by component and how the system reconfigures.
- Red flags: “We don’t know failure impact until it happens.”
- Close: Decide redundancy scope and required spares.

5) **“What’s the right abstraction for ‘replica’?”**

- One-liner: Replica must map to an independently schedulable deterministic service unit.
- 90s: In capacity planning, a replica must have a stable service curve. If the system partitions a chip, we need to know the partition’s service curve.
- Deep: Provide whether partitions share resources and whether interference can occur.
- Red flags: “Partitions are not isolated.”
- Close: Decide the replica definition and pool design.

6) **“How do we measure saturation safely?”**

- One-liner: We find the queueing knee by controlled load ramps with stable shapes.
- 90s: We ramp utilization and measure p99 vs throughput to find `ρ_knee`. This becomes our admission limit.
- Deep: Provide recommended tools and counters to confirm we’re saturating the right resource and not triggering pathological states.
- Red flags: “We just run a big load test.”
- Close: Decide the standard saturation test and acceptance thresholds.

7) **“What is the shape policy from a hardware point of view?”**

- One-liner: Hardware wants bounded tensors and predictable residency; buckets are the natural fit.
- 90s: Buckets prevent worst-case residency from dominating all capacity. Hardware constraints should translate directly into max prompt/output caps.
- Deep: Identify the thresholds where SRAM/bandwidth become limiting; we map those to product tiers.
- Red flags: “No constraints; accept all shapes.”
- Close: Decide the max caps per tier.

8) **“How do failures manifest?”**

- One-liner: We need detection time, isolation, and whether performance degrades before hard failure.
- 90s: Facilities and SRE need to know if failures are fail-stop or partial degradation. Determinism helps detect deviation; we must instrument for it.
- Deep: Provide error modes, telemetry, and recommended remediations.
- Red flags: “Silent data corruption risk” without mitigation.
- Close: Decide failure handling policy and monitoring requirements.

9) **“How do we think about precision and accuracy?”**

- One-liner: Precision choice is a schedule and capacity decision with correctness gates.
- 90s: Lower precision may improve throughput but changes accuracy; we decide per endpoint and validate via golden sets.
- Deep: Provide supported modes and determinism guarantees across modes.
- Red flags: “Precision changes are free.”
- Close: Decide the production precision mode and validation plan.

10) **“What’s the practical max utilization?”**

- One-liner: The answer is “below the queueing knee,” not “as high as possible.”
- 90s: Even if hardware can run at 95% compute utilization, p99 may collapse due to queueing. We choose utilization target from SLO tests.
- Deep: Provide the empirical knee for representative artifacts.
- Red flags: “Run at 90%+ to maximize ROI.”
- Close: Decide `ρ_target_normal` and `ρ_target_degraded`.

11) **“How does the chip behave with heterogeneous workloads?”**

- One-liner: We avoid heterogeneity per pool unless isolation is proven.
- 90s: Determinism can be broken by shared resources across incompatible artifacts/shapes. We prefer pool separation and explicit routing.
- Deep: Provide interference constraints; if none, provide proof/telemetry.
- Red flags: “It should be fine” without evidence.
- Close: Decide pool isolation boundaries.

12) **“What should customers not do?”**

- One-liner: Do not treat Groq like a GPU cluster; do not scale critical path over uncontrolled networks; do not allow unbounded shapes.
- 90s: The platform rewards disciplined shapes and precompiled artifacts. Anti-patterns lead to performance cliffs and tail failures.
- Deep: Provide a “do not do” list tied to failure modes.
- Red flags: Missing anti-pattern guidance.
- Close: Agree on guardrails and enforcement.

---

### Fabric / Network Engineers (12)

1) **“What’s the deterministic boundary?”**

- One-liner: We will not assume determinism beyond the validated fabric domain.
- 90s: We need the official boundary (rack/pod) and the SLO envelope inside it. Beyond that, we architect for variability.
- Deep: Define latency bounds, congestion assumptions, and failure behaviors inside the boundary.
- Red flags: “It’s deterministic everywhere.”
- Close: Decide the boundary and encode it in architecture.

2) **“What’s the topology and cabling constraint?”**

- One-liner: Treat cabling as part of the compute system, not optional networking.
- 90s: We need explicit port mapping, allowed lengths, and routing constraints. This affects floor layout and expansion.
- Deep: Provide topology diagram, port mapping rules, and a test plan for link validation.
- Red flags: “Cabling can be arbitrary.”
- Close: Decide the pod layout that satisfies cabling constraints.

3) **“What’s the failure domain and reroute time?”**

- One-liner: Failures must map to capacity math; reroute time must be in SLO planning.
- 90s: If link failure takes X ms to detect/reroute, and it triggers degraded throughput, we must either shed load or hold headroom.
- Deep: Provide detection mechanisms, reroute behavior, and whether performance changes.
- Red flags: “Reroute is best-effort.”
- Close: Decide N+1 scope and failover behavior.

4) **“How do you avoid congestion?”**

- One-liner: We assume deterministic scheduling requires congestion avoidance by design and admission.
- 90s: If the fabric is part of the compiled schedule, then congestion breaks determinism. We need the system’s congestion model and what traffic is permitted.
- Deep: Provide whether traffic is static, credit-based, or otherwise controlled, and what happens under overload.
- Red flags: “Congestion is handled like Ethernet.”
- Close: Decide allowed traffic patterns and overload behavior.

5) **“Can we run across racks?”**

- One-liner: Only with an explicit Groq reference architecture and SLO envelope; otherwise, no.
- 90s: Cross-rack introduces variability; we restrict critical-path partitioning to within the deterministic boundary unless Groq guarantees otherwise.
- Deep: If yes, define topology, latency bounds, oversubscription, and failure handling; if no, document the constraint.
- Red flags: “We’ll just use the DC spine.”
- Close: Decide cross-rack policy.

6) **“What does Ethernet do in this design?”**

- One-liner: Control plane, ingress/egress, metrics, artifact distribution—keep it off the critical path.
- 90s: We separate planes to preserve deterministic behavior and simplify debugging.
- Deep: Define VLAN/VRF separation, QoS, and telemetry requirements.
- Red flags: “One network for everything.”
- Close: Decide network plane separation and requirements.

7) **“How do you test the fabric?”**

- One-liner: Burn-in with link integrity + latency checks before accepting a pod.
- 90s: Fabric issues look like compute issues unless you validate links explicitly. We need standardized tests and thresholds.
- Deep: Define PRBS/bit error tests, latency checks, and continuous monitoring.
- Red flags: “We rely on app-level errors.”
- Close: Decide acceptance tests and monitoring.

8) **“How do we model capacity with fabric failures?”**

- One-liner: A fabric failure is a capacity reduction event; it pushes utilization upward and increases queueing.
- 90s: We include fabric failures in N+1 planning. Either we hold headroom or we shed load.
- Deep: Define which failures partition the system and how much capacity is lost.
- Red flags: “Fabric failures are rare; ignore them.”
- Close: Decide N+1 assumptions for fabric.

9) **“What’s the observability boundary?”**

- One-liner: We need correlation between inference latency spikes and fabric counters.
- 90s: Determinism helps debugging if we can correlate deviations to fabric events. We need per-link counters and error reporting integrated into SRE tools.
- Deep: Define telemetry APIs and sampling rates.
- Red flags: “No fabric telemetry.”
- Close: Decide minimum fabric observability.

10) **“How do we expand without downtime?”**

- One-liner: Add pods as isolated units; avoid re-cabling existing pods.
- 90s: Expansion should not disturb deterministic domains. We stage, validate, and then add capacity.
- Deep: Define the expansion choreography and required spare ports/cables.
- Red flags: “We’ll hot-swap cables in production.”
- Close: Decide expansion sequencing and isolation rules.

11) **“How do you route traffic to the right pool?”**

- One-liner: L7 routing by bucket/artifact; failover-aware.
- 90s: The router must understand shape buckets to preserve service curves; it must also reroute around failures without violating caps.
- Deep: Define routing policies and health signals.
- Red flags: “Random load balancing across heterogeneous pools.”
- Close: Decide routing policy and health checks.

12) **“What’s your stance on oversubscription?”**

- One-liner: Oversubscription is acceptable only off the critical path.
- 90s: Deterministic service-time assumptions require predictable comm; oversubscription introduces queueing and jitter.
- Deep: Define where oversubscription is allowed and how it is monitored.
- Red flags: “We’ll oversubscribe and rely on average.”
- Close: Decide oversubscription policy.

---

### Data Center / Facilities (12)

1) **“What rack power do you need?”**

- One-liner: We will size to worst-case sustained draw plus derating, not nameplate averages.
- 90s: Deterministic throughput encourages steady high load; facilities must provision sustained kW with safe margins.
- Deep: We need per-node and per-rack worst-case profiles and recommended derating.
- Red flags: “We’ll run at nameplate.”
- Close: Decide rack power budget and derating.

2) **“Air or liquid?”**

- One-liner: We choose the cooling class that matches density and deployment speed; air is faster bring-up if it fits.
- 90s: Liquid increases density but adds infrastructure complexity; air requires strict containment and airflow discipline.
- Deep: Evaluate TCO: CDU loops, maintenance, failure modes, and expansion complexity.
- Red flags: “Cooling doesn’t matter; we’ll figure it out.”
- Close: Decide cooling approach per pod.

3) **“How do you prevent hotspots?”**

- One-liner: Containment + airflow compliance + burn-in thermal mapping.
- 90s: Cable bundles and layout errors create recirculation. We enforce standards and validate under load.
- Deep: Use thermal sensors, IR mapping, and enforce max inlet temps.
- Red flags: “No containment plan.”
- Close: Decide containment standard and validation plan.

4) **“What’s the cabling plan?”**

- One-liner: The fabric cable plant is part of the compute system; it must be installed to spec.
- 90s: Cable constraints affect row layout and expansion. We need a labeled, tested, documented plan.
- Deep: QR labeling, test results stored, spares on-site.
- Red flags: “We’ll cable after racks arrive.”
- Close: Decide cabling process and acceptance tests.

5) **“What’s the bring-up procedure?”**

- One-liner: Stage → power-up → burn-in → fabric validation → perf acceptance.
- 90s: We treat a pod like a product: no production until it passes acceptance criteria.
- Deep: Include thermal, power, link integrity, and service-curve validation.
- Red flags: “Ship to prod without burn-in.”
- Close: Approve bring-up acceptance criteria.

6) **“What spares do you need on-site?”**

- One-liner: Spare capacity is an SLO control; physical spares reduce MTTR.
- 90s: N+1 requires spares in-place. Cable and PSU spares prevent long outages.
- Deep: Define RMA timelines, spare ratios, and hot/cold spares.
- Red flags: “We’ll overnight parts.”
- Close: Decide spare inventory and storage.

7) **“How do you handle maintenance windows?”**

- One-liner: Pods must tolerate planned maintenance via capacity headroom and traffic shifting.
- 90s: With deterministic service curves, we can predict degraded capacity during maintenance and decide whether to shed or throttle.
- Deep: Define maintenance playbooks and traffic migration.
- Red flags: “No maintenance model.”
- Close: Decide maintenance policy and capacity buffer.

8) **“What’s the floor layout constraint?”**

- One-liner: Layout is constrained by power, airflow, and fabric cabling; we design pods as repeatable blocks.
- 90s: Avoid snowflakes; align pods to containment and cable routes.
- Deep: Provide a pod blueprint and expansion map.
- Red flags: “We’ll put racks where there’s space.”
- Close: Approve pod blueprint and expansion sequence.

9) **“What telemetry do you need?”**

- One-liner: Power and thermal telemetry per rack is mandatory to prevent silent derating.
- 90s: Deterministic systems make deviations meaningful; we monitor inlet temps, fan speeds, and power draw.
- Deep: Integrate into DCIM and SRE alerting.
- Red flags: “No rack-level telemetry.”
- Close: Decide telemetry requirements and owners.

10) **“What’s the failure scenario we plan for?”**

- One-liner: Rack loss and airflow failures; these drive redundancy and containment decisions.
- 90s: We plan for worst plausible failure, not best case.
- Deep: Game-day: shut a rack, trip a PDU, simulate fan failure, verify system behavior.
- Red flags: “We don’t test failures.”
- Close: Approve facilities game-day plan.

11) **“How do you expand?”**

- One-liner: Add pods with preplanned power/cooling and pre-staged cabling paths.
- 90s: Expansion should not disturb running pods; staging areas and burn-in are needed.
- Deep: Provide timelines, dependencies, and acceptance gates.
- Red flags: “Expansion is ad hoc.”
- Close: Decide expansion staging and gates.

12) **“What still limits faster bring-up?”**

- One-liner: Power and cooling lead times and cable plant—physics, not software.
- 90s: Even with a simpler runtime story, facilities constraints dominate schedule.
- Deep: Identify long poles early and lock procurement.
- Red flags: Late facilities engagement.
- Close: Assign facilities milestones and procurement owners.

---

### Inference / Runtime Engineers (12)

1) **“How do you define TTFT on Groq?”**

- One-liner: TTFT is queueing + pipeline entry + prefill time for the chosen shape bucket.
- 90s: Determinism makes service time stable; TTFT variance is mostly queueing and bucket selection. We measure them separately.
- Deep: Instrument: `t_enqueue`, `t_start_service`, `t_first_token`; compute queueing vs service components.
- Red flags: “TTFT is just compute.”
- Close: Decide TTFT definition and instrumentation.

2) **“How do you keep p99 stable?”**

- One-liner: Bound the queue and admit by predicted deterministic cost.
- 90s: With a stable service curve, p99 stability is an admission-control problem. We enforce caps and shed early rather than grow queues.
- Deep: Token-budget admission per bucket; separate pools; strict queue caps and timeouts.
- Red flags: “Let the queue absorb spikes.”
- Close: Decide queue caps and admission policy.

3) **“Do we batch?”**

- One-liner: Only if it’s an explicit compiled strategy; otherwise we use concurrency, not dynamic batching windows.
- 90s: Dynamic batching is a common GPU trick but increases waiting and shape heterogeneity. We avoid it unless Groq validates a deterministic batching mode.
- Deep: If batching exists, keep it bucketed and static; otherwise prioritize batch-1 throughput and stable concurrency.
- Red flags: “We’ll implement continuous batching.”
- Close: Decide batching policy and artifact strategy.

4) **“How do you do long-context?”**

- One-liner: Separate tier, separate pools, stricter admission.
- 90s: Long prompts dominate prefill cost and can collapse global p99 if mixed. We isolate and price accordingly.
- Deep: Separate artifacts and capacity; possibly different SLOs.
- Red flags: “Same pool serves all contexts.”
- Close: Approve long-context tier routing.

5) **“What’s the saturation signal?”**

- One-liner: p99 queueing knee, not GPU utilization counters alone.
- 90s: We ramp load with fixed shapes and find the point where p99 increases sharply; that’s our operational max.
- Deep: Combine service-time counters (from Groq tools) with queue metrics to locate the knee.
- Red flags: “We run until it breaks.”
- Close: Decide the knee-based utilization target.

6) **“How do you isolate tenants?”**

- One-liner: Pool-level isolation by artifact and bucket; strict admission.
- 90s: Determinism plus multi-tenancy requires avoiding interference; we route tenants to pools and enforce quotas.
- Deep: Separate queues per tenant; separate compiled artifacts if needed.
- Red flags: “Everything shares one queue.”
- Close: Decide tenancy isolation model.

7) **“How do you roll out new models?”**

- One-liner: Precompile, benchmark service curve, canary by bucket, then ramp.
- 90s: The release unit is a compiled artifact with a measured service curve. We canary in production with strict caps.
- Deep: Artifact provenance, performance regression tests, rollback plan.
- Red flags: “Deploy weights and hope.”
- Close: Approve rollout pipeline.

8) **“What metrics matter most?”**

- One-liner: Queue time vs service time, tokens/s, and per-bucket utilization.
- 90s: Determinism only helps if you separate queueing from compute. We monitor both and alert on deviations.
- Deep: Define dashboards: per-bucket QPS, tokens/s, queue depth/time, error rates, and failure-domain health.
- Red flags: “Only end-to-end latency.”
- Close: Decide metric schema and SLO dashboards.

9) **“What are the ‘hard limits’?”**

- One-liner: Shape caps and artifact set; violating them breaks predictability.
- 90s: We enforce caps at the API gateway. If clients violate, we reject or route to a special tier.
- Deep: Contracts in API and SDK; logs for violations.
- Red flags: “We’ll accept whatever clients send.”
- Close: Approve enforcement points.

10) **“How do you do load tests?”**

- One-liner: Fixed buckets, controlled ramps, and failure injection.
- 90s: We test service curves per bucket and then mixed workloads; we inject failures (N-1) and verify p99 protection.
- Deep: Game-days: link down, rack down, reroute; validate admission behavior.
- Red flags: “One big random load test.”
- Close: Decide test plan and acceptance criteria.

11) **“How do you deal with burstiness?”**

- One-liner: Bound queues and use token-budget admission; burst absorbs into headroom or sheds.
- 90s: Burstiness is a traffic property; we don’t pretend average is enough. We size for `λ_peak` and keep headroom.
- Deep: Multi-timescale rate limiting; early reject; backpressure clients.
- Red flags: “We’ll queue through burst.”
- Close: Decide burst-handling policy.

12) **“What’s your incident response posture?”**

- One-liner: Determinism makes anomalies meaningful; incidents should map to capacity or failure-domain events.
- 90s: We monitor deviations from expected service curves. If service time shifts, it’s likely artifact/hardware; if queue time shifts, it’s load/failure.
- Deep: Runbooks per failure domain; rapid traffic shedding; artifact rollback.
- Red flags: “We can’t attribute latency spikes.”
- Close: Approve incident taxonomy and runbooks.
