# Phase 1 report: build-VM tiers

**Preregistration:** [`PREREG_PHASE1.md`](PREREG_PHASE1.md), registered in commit `903ec99` before any of the measurements below.
**Runs:** 2026-10-18, on the 4-vCPU build VM (no GPU). Hardware details are recorded in every result file.
**Raw results:** [`results/phase1/`](../results/phase1).

## Summary

| Tier | Result | Preregistered expectation |
|---|---|---|
| **E**: CPU float64 engine vs Brian2, full brain, 1 s, 4 workloads | **PASS**: identical spike for spike in all 4 | identical |
| **F-pos**: CPU float64 engine, fidelity suite | **PASS** (every check) | pass |
| **F-f32**: CPU float32 engine, fidelity suite | **PASS** (every check) | pass |
| **F-neg-weight**: weights × 0.95 | **FAIL** (3 of 4 conditions) | fail ✓ |
| **F-neg-delay**: delay 17 steps instead of 18 | **PASS**: **miss** | fail ✗ |
| **G-pos**: random summation order, seeds 3 and 4 (Amendment 1) | **PASS**: Jaccard 1.0000 in all workloads | pass ✓ |
| **G-neg-delay**: delay 17 steps (Amendment 1) | **FAIL**: Jaccard 0.135 / 0.530 / 0.617 | fail ✓ |
| **G-neg-weight**: weights × 0.95 (Amendment 1) | **FAIL**: Jaccard 0.135 / 0.524 / 0.622 | fail ✓ |
| **S**: GPU speed | not yet run (needs testers) | RTF ≈ 7–15 |

**What happened.** A negative control passed the original statistical tier, so under the preregistered rule F was declared inadequate. That miss stands. A dated Amendment 1 added a same-input tier G, with its own controls, before any G run. The maintainer delegated the decision. All G controls behaved as required.

**Claim status.** Rules 1 and 2 of the amended claim rule are met on the build VM. Rules 3 and 4 need consumer-GPU submissions that show S1, F and G all true, plus an independent reproduction. **No real-time claim is made yet.**

## 1. E: exact tier

Identical Bernoulli(150 Hz × dt) kicks (NumPy seed 0) were fed to the Brian2 reference network (Cython target) and to the ASCEND CPU engine in float64. The full v783 brain ran for 10,000 steps.

| Workload | Spikes (engine = reference) | Active neurons | Differing neurons | Result |
|---|---|---|---|---|
| sugar (21 GRNs) | 13,220 | 365 | 0 | identical |
| bitter (21 GRNs) | 4,603 | 76 | 0 | identical |
| JONs (146) | 28,334 | 483 | 0 | identical |
| none | 0 | 0 | 0 | identical |

The engine computes the reference model exactly, including the refractory freeze, the dropped refractory input, the 18-step delay and the update order.

## 2. F: statistical tier, positive controls

30 trials × 4 conditions, with seeds 0–29 from the engine's own stimulus stream, compared with `results/phase0/reference_rates.json`. Every check passed for both precisions.

| Condition | Check | float64 | float32 | Reference |
|---|---|---|---|---|
| sugar | mn9_left (Hz) | 82.57 | 83.80 | 84.33 |
| sugar | mn9_right (Hz) | 58.07 | 57.57 | 59.33 |
| sugar | neurons outside tolerance | 0 / 366 | 0 / 369 | — |
| sugar | spikes per trial | 13,667 | 13,578 | 13,649 |
| bitter | neurons outside tolerance | 0 / 76 | 0 / 76 | — |
| JONs | DNg62 / DNge078 (Hz) | 19.63 / 20.33 | 19.70 / 20.13 | 19.77 / 20.40 |
| JONs | neurons outside tolerance | 0 / 476 | 0 / 477 | — |
| none | spikes | 0 | 0 | 0 |

Pearson r was ≥ 0.99 in every condition. Full values are in the result files.

## 3. F: negative controls

| Variant | sugar | bitter | JONs | none | Suite |
|---|---|---|---|---|---|
| weights × 0.95 | FAIL: 192 / 365 neurons out; spikes 12,653 vs 13,649 | FAIL: 38 / 76 out | FAIL: 113 / 474 out; DNg62 17.17 vs 19.77 | pass | **FAIL** ✓ |
| delay 17 steps | pass: 0 / 369 out | pass | pass | pass | **PASS** ✗ |

**What the miss means.** Shortening every synaptic delay by one step (1.8 → 1.7 ms) leaves firing rates within sampling noise everywhere: MN9 83.13 vs 84.33 Hz, and total spikes 13,557 vs 13,649. The F tier checks rates only, so it cannot see a timing error of this size. Weight errors are caught, timing errors are not, and an engine can be wrong in timing while passing F.

## 4. Pilot: same-input agreement (after the miss, exploratory, disclosed)

To design a fix, the CPU float32 engine was run with identical hash-RNG kicks (seed 0, 1 s, full brain) in several variants. Each was compared with the CPU float32 engine as built, using the Jaccard overlap of exact (step, neuron) spike events. This pilot was run after the F results above and is not a preregistered measurement.

| Variant | sugar | JONs |
|---|---|---|
| float32, synaptic input summed in random order (twice) | **1.0000**, 1.0000 | **1.0000**, 1.0000 |
| float64 | 0.297 | 0.751 |
| delay 17 steps | 0.142 | 0.623 |
| weights × 0.95 | 0.138 | 0.629 |

Randomising the order of summation leaves float32 spike trains unchanged. That is the only freedom a correct GPU run has, since GPU atomics add in arbitrary order. A delay or weight error makes the trains diverge massively. Changing precision also changes exact spike timing, so a same-input comparison is only valid float32 against float32.

## 5. Amendment 1 (applied 2026-10-18, commit `97d6427`)

The amendment adds a **G tier: same input, same spikes**. The GPU engine (float32) and the CPU engine (float32) get identical hash-RNG kicks on the full brain for 1.0 s, for sugar, bitter and JONs at 150 Hz with kick seed 1 (the pilot used seed 0). **Pass:** Jaccard of (step, neuron) spike events ≥ 0.99 in all 3 workloads. `bench.py --fidelity` runs it on the tester's machine. The CPU reference takes about 15–25 s per workload.

With G, the fidelity chain becomes:

- **E:** Brian2 = CPU float64, exactly.
- **F-f32:** CPU float32 statistically matches the reference.
- **G:** GPU float32 = CPU float32, same input.

F stays as an end-to-end statistical check on the GPU, with its timing blind spot documented.

It also preregisters **G controls on the build VM**:
- The CPU float32 engine with permuted summation order (seeds 3 and 4, fresh) must pass.
- The delay-17 and weight × 0.95 variants must fail.

The amended claim rule requires E, F-pos, F-f32, F-neg-weight failing, and the G controls behaving as required. It also requires a consumer-GPU submission passing S1, F and G, plus an independent reproduction.

### Formal G controls (fresh seeds: kicks seed 1, permutations 3 and 4)

CPU float32 variants are compared with the CPU float32 engine as built, with identical kicks, on the full brain for 1 s (`scripts/sameinput_check.py`, `results/phase1/sameinput_*.json`).

| Run | sugar | bitter | JONs | Required | Outcome |
|---|---|---|---|---|---|
| G-pos, permutation seed 3 | 1.0000 (13,561 spikes) | 1.0000 (4,730) | 1.0000 (28,628) | ≥ 0.99 all | ✓ |
| G-pos, permutation seed 4 | 1.0000 | 1.0000 | 1.0000 | ≥ 0.99 all | ✓ |
| G-neg-delay | 0.135 | 0.530 | 0.617 | < 0.99 all | ✓ |
| G-neg-weight | 0.135 | 0.524 | 0.622 | < 0.99 all | ✓ |

Under bitter, the delay error leaves the spike count unchanged (4,730 vs 4,730) but moves spike times. That is exactly the error F could not see.

## 6. Bugs found and fixed during these runs (disclosed)

1. **`fidelity_check.py` crashed when saving its first condition**, because three checks returned `numpy.bool_`, which JSON cannot serialise (`9d38252`). Nothing had been printed or saved, so no results were seen. The criteria were unchanged, and a regression test now covers it.
2. **The `exact_check.py` no-stimulus workload crashed**, because the Brian2 harness built a spike generator of size 0. It now leaves the generator out when there is no stimulus (`d29f14f`). E-none was rerun and gave identical results (0 spikes on both sides).

## 7. Reproduce

```bash
python scripts/exact_check.py --stimulus sugar --out exact_sugar.json            # also bitter, jon_all, none
python scripts/fidelity_check.py --device cpu --dtype float64 --variant exact    # also float32; delay17, weight95
python scripts/sameinput_check.py --variant shuffle3                              # also shuffle4, delay17, weight95
python scripts/bench.py --engine ascend --device cuda --fidelity --out bench.json # GPU submission: S, F and G
```
