# Phase 2 effectome: design proposal

**Status:** proposal, awaiting the maintainer's approval. Nothing here is registered or measured. When the design is approved, the draft preregistration in section 9 becomes `docs/PREREG_PHASE2.md` in its own commit, before any Phase 2 run.

## Goal

Perturb every neuron of the FlyWire v783 connectome-constrained model and record its effect on every other neuron. This produces an in-silico **effectome**, which must recover at least 3 known pathways.

Phase 2 builds on the Phase 1 engine. It does not depend on the Phase 1 real-time claim, but it does need the engine's fidelity tiers (E, F, G) to hold for the perturbed runs too.

## Prior art

- **Pospisil et al. 2024**, *The fly connectome reveals a path to the effectome*, Nature 634. They introduced the term and estimated an effectome from the connectome with a linear dynamical model.
- **Shiu et al. 2024** defined the two perturbations this design uses, activation (Poisson input) and silencing (outgoing weights set to 0), and used them to trace sugar → MN9 and other pathways. Their `model.py`, vendored here, implements both directly: `poi()` with a second rate class `exc2`, and `silence()`.
- **von Reyn et al. 2014** and **Ache et al. 2019** characterised the looming → Giant Fiber (DNp01) escape pathway through LC4 and LPLC2.
- **FlyWire** (Dorkenwald et al. 2024; Schlegel et al. 2024) supplies the connectome and the cell-type annotations used to name stimulus and readout neurons.

The difference here: the effectome is measured by perturbing the full spiking model neuron by neuron, rather than inferred from a linear model. A linear connectome baseline is computed alongside as a comparison (section 6).

## 1. What an effect is

All perturbations use the reference model's own definitions, so each one can be checked exactly against Brian2:

- **Activation** of neuron *i*: Poisson kicks at 150 Hz, with refractory period 0, as in `poi()`.
- **Silencing** of neuron *i*: every outgoing weight of *i* is set to 0, as in `silence()`. The engine implements this as a per-replica mask on spike delivery, which is equivalent.

**The effect of perturbing *i* on neuron *j*** is the paired difference in firing rate, Δr<sub>ij</sub> = r<sub>j</sub>(perturbed) − r<sub>j</sub>(control). It is taken over the same 1.0 s window, with identical stimulus kicks, and averaged over seeds.

Pairing is free. Kicks come from the counter-based hash of (seed, step, neuron), so a perturbed run and its control receive exactly the same input. Every difference between them is caused by the perturbation. This is the method of common random numbers, and it removes most of the sampling noise from each effect.

**An exact shortcut.** If neuron *i* never spikes in the control run, silencing it changes nothing. The two runs are identical spike for spike, so its effect is exactly zero and needs no simulation. Under a stimulus condition only about 76–483 of the 138,639 neurons spike (Phase 1 E tier), so the silencing map needs only a few hundred runs per condition, not 138,639.

## 2. Two maps

### Map A: activation from rest (every neuron)

There is no background stimulus. Each of the 138,639 neurons is activated alone, and the response of every neuron is recorded. The model emits no spikes at rest (tier F1), so Δr equals the response rate.

- Seeds 0–2, 1.0 s each.
- Output: the sparse matrix of all nonzero responses.

### Map B: conditional effectome (silencing under a stimulus)

Under each stimulus condition, every neuron that spikes in that condition's control run is silenced in turn.

- Seeds 0–4, 1.0 s each.
- Conditions: Shiu's sets at 150 Hz (sugar, bitter, water, ir94e, jon_all) plus one looming set, `looming_right`: all LC4 and LPLC2 neurons annotated `side = right` in FlyWire v783 (50 + 102 neurons).
- **B+:** a short list of targeted activations on a background, needed for validation V2. Each bitter GRN is activated individually on the sugar background.

Map A says what each neuron can drive by itself. Map B says what each pathway actually depends on when it is in use. Silencing effects only show up under drive, which is why Map A alone isn't enough.

## 3. Engine work

Phase 2 needs a **batched engine**: R independent replicas of the network advanced together.

- The CSR connectivity is shared. Each replica gets its own state (v, g, refractory counter), stimulus mask, silence mask and spike record.
- Kicks are keyed by (seed, step, neuron), independent of the replica. Replicas on the same seed therefore get identical input, which gives the pairing above at no extra cost.
- The delay ring is sized to observed activity (a cap per slot, with an overflow flag that aborts the run) rather than the worst case of N. This keeps a replica at about 2–3 MB, so 256 replicas fit easily in 8 GB.
- The same kernels run in the C++ emulator, so CI tests the batched engine on the CPU.

**Exactness tests (CI, synthetic networks):**
- A batched run of R replicas matches R separate runs of the existing engine spike for spike.
- Silencing by mask matches silencing by zeroed weights spike for spike.
- Silencing a neuron that never spikes changes nothing.

## 4. Fidelity of the effectome

The effectome is only as good as the perturbed runs, so the Phase 1 tiers are repeated on perturbed networks:

- **E-P (build VM):** 20 perturbations sampled with a fixed seed (10 from Map A, 10 from Map B), run on the CPU float64 engine and on Brian2 via upstream `silence()` and `poi()` with replayed kicks. Pass: identical spikes in all 20.
- **G-P (GPU host):** 50 sampled perturbations, batched GPU float32 against the CPU float32 engine with the same kicks. Pass: Jaccard ≥ 0.99 in every one.

## 5. Validation pathways (draft criteria)

Four pathways are proposed. Each test is fixed before any Phase 2 run, and each comes with a negative control. Neuron IDs are drawn from `data/stimuli_783.json` and the FlyWire v783 annotations.

**Definitions used by the tests:**
- A neuron is **necessary** for a target if silencing it lowers the target's rate in all 5 seeds and by at least 20% on average.
- MN9 means the summed rate of `mn9_left` and `mn9_right`.

| # | Pathway | Literature | Test (draft) |
|---|---|---|---|
| V1 | Sugar → MN9 (feeding initiation) | Shiu et al. 2024 | In B[sugar], at least 14 of the 21 sugar GRNs are necessary for MN9, and at least one non-sensory neuron is necessary. |
| V2 | Bitter suppresses sugar-evoked MN9 | Shiu et al. 2024 | In B+[sugar], activating single bitter GRNs gives a negative summed effect on MN9, and at least 11 of 21 individual effects are negative. |
| V3 | Looming → escape (Giant Fiber) | von Reyn et al. 2014; Ache et al. 2019 | In B[looming_right], DNp01 (right) is among the 5 highest-rate descending neurons, and at least 50% of the stimulated LC4/LPLC2 neurons that are active are necessary for it. |
| V4 | Johnston's organ → descending neurons | Shiu et al. 2024 (their JON readouts) | In B[jon_all], DNg62 and DNge078 (Shiu's `jon_dn1`, `jon_dn2`) each have at least one necessary JON and at least one necessary non-sensory neuron. |

**Pass rule:** at least 3 of the 4 pass, and every result is reported. V3 is the riskiest. Shiu et al. did not test looming, so the model may not drive the Giant Fiber from LC4 and LPLC2 input, and a failure there would be a finding about the model.

**Negative control:** each test is repeated on a **shuffled connectome**. The shuffle keeps each neuron's in- and out-degree and the sign and weight distribution, but rewires targets (fixed seed). The real effectome passing a test only counts if the shuffled one fails it. If a shuffled control passes, that test is declared uninformative, just as the F-tier miss was handled in Phase 1.

**Causality sanity check, not a validation:** every necessary neuron must have a directed synaptic path to its target. A violation means a bug.

## 6. Linear baseline

Pospisil et al. estimated effects from a linear model of the connectome. The same is done here as a comparison: the signed weight matrix, propagated for 4 hops.

The report gives the Spearman correlation between the linear and spiking effects for each target, and whether the linear map passes V1–V4. This is descriptive and not part of the claim. It shows what the spiking dynamics add.

## 7. Compute

**Measured (CPU float32 engine, 4-vCPU build VM, `results/phase1/fidelity_cpu_float32_exact.json`):** 7.2 s per simulated second.

**Predicted (batched GPU):** the per-replica cost is the neuron-state sweep, about 2.8 MB of traffic per step. On an RTX 4070-class card (504 GB/s) that is about 5.5 µs per replica-step, so roughly 18 simulated seconds per wall-clock second across all replicas. This is a prediction and will be measured first.

| Workload | Simulated seconds | Build VM (CPU) | RTX 4070-class GPU (predicted) |
|---|---|---|---|
| Map A: 138,639 × 3 seeds | 415,917 | about 35 days | about 6.5 h (range 4–13 h depending on card) |
| Map B: 6 conditions × ~400 active × 5 seeds | ~12,000 | about 24 h | about 11 min |
| B+ for V2: 21 × 5 | 105 | about 13 min | seconds |
| Shuffled-connectome controls for V1–V4 | ~10,000 | about 20 h | about 10 min |
| E-P: 20 Brian2 + CPU runs | 40 | about 1 h (mostly network builds) | — |

Map A is the only expensive part, and it only runs on a GPU. It splits into independent shards by source range. Each shard is about 30–60 minutes on one card, so it could be spread across testers' GPUs, with each shard checked by sampled G-P runs. That also extends the call for testers into Phase 2.

Every run over 2 hours waits for the maintainer's go-ahead.

## 8. Outputs

- **Map A** is sparse but large, an estimated 10–100 MB as Parquet. It is over the 1 MB repository limit, so it is published as a versioned external dataset, with its SHA-256 recorded in `data/manifest.json` and fetched by `scripts/fetch_data.py`.
- **Map B and the validation results** go in `results/phase2/` as summary JSON under 1 MB (necessary sets, effects on the readout neurons, pass and fail), each tied to the preregistration.
- `docs/REPORT_PHASE2.md` gives every tier, every validation and every control, misses included.

## 9. Draft preregistration (not registered)

> **Claim under test.** Perturbing each neuron of the FlyWire v783 connectome-constrained model (activation from rest for all 138,639 neurons, and silencing of every active neuron under 6 stimulus conditions) yields an effectome that recovers at least 3 of 4 known pathways (V1–V4), where a degree-preserving shuffled connectome does not.
>
> **Fixed before any run:**
> - perturbation definitions (section 1)
> - conditions, seeds (A: 0–2; B: 0–4) and the 1.0 s window
> - stimulus and readout neurons by FlyWire v783 root ID (generated from `data/stimuli_783.json` and the annotations, committed with the prereg)
> - the "necessary" threshold (all 5 seeds lower, mean ≥ 20%)
> - tests V1–V4 and their thresholds (section 5)
> - the shuffle seed
>
> **Claim rule:**
> 1. The batched engine exactness tests pass in CI.
> 2. E-P is identical in all 20 runs.
> 3. G-P has Jaccard ≥ 0.99 in all 50 runs.
> 4. At least 3 of V1–V4 pass on the real connectome.
> 5. Each passing test fails on the shuffled connectome.
>
> **Predictions:** V1, V2 and V4 pass. V3 is uncertain. The linear baseline correlates positively with the spiking effectome but misses effects that need coincident input.

## Decisions for the maintainer

1. **Approve the two-map design.** The alternative is a background-drive design, where every sensory neuron gets low-rate input and every neuron is perturbed on top of it. That design isn't in Shiu's model, so it would add a modelling assumption.
2. **The four pathways and their thresholds**, and the 3-of-4 pass rule.
3. **Looming set:** right-side LC4 and LPLC2 to DNp01 (right), or a different side or subset.
4. **Hosting for Map A:** an external dataset (for example Zenodo) with its SHA-256 recorded in the manifest.
5. **Who runs Map A:** the maintainer's GPU, or shards spread across testers' GPUs.
