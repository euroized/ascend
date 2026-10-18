# Phase 0 report: reference baseline

**Preregistration:** [`PREREG_PHASE0.md`](PREREG_PHASE0.md), registered in commit `a72e34f` before any measurement.
**Runs:** 2026-10-18, on the cloud build VM described below. No GPU was used.
**Raw results:** [`results/phase0/`](../results/phase0).

## Summary

| Question | Answer |
|---|---|
| Does our harness run the Shiu et al. 2024 model faithfully? | **Yes.** On FlyWire v630 it matches Shiu's published MN9 rates within 0.5–2.5 Hz (R1, 4 of 4 checks). |
| Does sugar GRN → MN9 reproduce on FlyWire v783? | **Yes, REPRODUCED.** All preregistered rules pass: R1 4/4, R2 5/5, R3 4/4, R4 1/1. |
| Reference real-time factor on the build VM | **Median RTF 0.189** (range 0.149–0.211), with Brian2 Cython on a single process. |
| Gap to real time | **5.3×** (1 / 0.189). |

Nothing in this report is a claim about Phase 1, 2 or 3.

## 1. Environment

| | |
|---|---|
| Machine | Cloud VM (KVM), 4 vCPUs, Intel Xeon (family 6, model 173, AVX-512 / AMX), 15.7 GB RAM, no GPU |
| OS | Linux 6.18.44, glibc 2.39 |
| Python and packages | Python 3.13.16, Brian2 2.10.1 (Cython 3.1.3 code generation), NumPy 2.5.3, pandas 3.0.5, PyArrow 26.0.0 |
| Simulator setup | Brian2 runtime mode, `cython` target, one process per network. Brian2 runtime is single-threaded. |

`ascend.hardware.info()` records the same details inside every result file.

## 2. Data

All inputs are fetched by `scripts/fetch_data.py` from pinned upstream commits and verified against SHA-256 values in [`data/manifest.json`](../data/manifest.json). None of them is committed.

| File | Source | Verified |
|---|---|---|
| `Completeness_783.csv`, `Connectivity_783.parquet` | philshiu/Drosophila_brain_model @ `91bdd1e` | ✓ |
| `Completeness_630.csv`, `Connectivity_630.parquet` | same | ✓ |
| `flywire_neuron_annotations.tsv` | flyconnectome/flywire_annotations @ `a83b277` | ✓ |

The v783 connectome matches the project brief exactly: **138,639 neurons, 15,091,983 connections and 54,492,922 synapses**. The synapse total is the sum of `Connectivity`, which equals the sum of |`Excitatory x Connectivity`|.

## 3. Stimulus neurons on v783

Shiu et al.'s stimulus sets use FlyWire v630 root IDs. Most of those IDs are unchanged in v783. A neuron that was edited between the two releases gets a new ID. `scripts/map_stimuli.py` found the v783 counterpart of each changed neuron by comparing synaptic partners (weighted Ruzicka similarity). A match was accepted only if it scored ≥ 0.5 and at least 2× the runner-up.

| Set | v630 ID | v783 ID | Score | Runner-up | v783 cell type |
|---|---|---|---|---|---|
| sugar | 720575940620900446 | 720575940639259967 | 0.938 | 0.112 | LB3c (sugar GRN) |
| bitter | 720575940618600651 | 720575940619072513 | 0.764 | 0.087 | LB1c (bitter GRN) |
| JO-CE | 720575940626307902 | 720575940625054647 | 0.886 | 0.160 | JO-EV3 |
| MN9 (Shiu "right") | 720575940645521262 | 720575940618238523 | 0.688 | 0.065 | CB0701 (ingestion MN) |

Every mapped set keeps Shiu's size and has the expected cell types. For example, sugar = 10 LB3c, 5 LB3d, 4 LB4b and 2 LB3b. Shiu's "right hemisphere" sets are annotated `side = left` in v783, because FlyWire corrected the mirror convention after v630. We keep Shiu's names.

## 4. Harness fidelity

- **Unmodified model.** `ascend/reference/shiu_model.py` is byte-identical to upstream `model.py` (SHA-256 `fc45837d…`, pinned by `tests/test_vendored.py`). The wrapper only calls its functions.
- **Seeding.** `brian2.seed(seed)` is called immediately before each trial. The same seed gives identical spikes and a different seed gives different spikes (tested).
- **Build once, restore per trial.** On the full v783 network, restoring a stored initial state and running seed 3 gave the same spikes, bit for bit, as building a fresh network and running seed 3. `tests/test_reference.py` checks the same on a synthetic network.
- **No-stimulus control.** The model has no noise source, so an unstimulated network must stay silent. It did: zero spikes in all 30 trials (R2).

## 5. Results: sugar GRN → MN9

The comparison values are Shiu et al.'s published model output: `results/example/sugarR*.parquet`, FlyWire v630, 30 trials each. Our runs use seeds 0–29. All rates are the mean ± SD (ddof = 0) over 30 trials.

### R1: harness check on v630 (tolerance ±5 Hz)

| Sugar rate | Readout | Ours | Published | Difference | SE of difference | z |
|---|---|---|---|---|---|---|
| 200 Hz | mn9_left | 92.77 | 93.27 | −0.50 | 1.17 | −0.43 |
| 200 Hz | mn9_right | 62.30 | 61.80 | +0.50 | 1.12 | +0.45 |
| 100 Hz | mn9_left | 69.57 | 67.03 | +2.53 | 1.40 | +1.81 |
| 100 Hz | mn9_right | 51.13 | 48.63 | +2.50 | 1.18 | +2.12 |

All 4 pass. At 200 Hz the agreement is within sampling noise. At 100 Hz both MN9s run about 2.5 Hz (≈ 4–5 %) above the published values, roughly 2 SE each, in the same direction. That is well inside the preregistered tolerance, but it is worth noting.

We cannot pin down the cause from this data. Candidates are the Brian2 version (2.10.1 here; upstream's published environment pins 2.5.1) or the random streams, since the published runs' seeds are unknown. Phase 1 fidelity criteria will be set against our own Phase 0 reference output, which is fully seeded, rather than against the published files.

### R2: qualitative behaviour on v783

| Check | Result | Pass |
|---|---|---|
| Sugar 150 Hz drives mn9_left ≥ 10 Hz | 84.33 ± 4.58 Hz | ✓ |
| Sugar 150 Hz drives mn9_right ≥ 10 Hz | 59.33 ± 4.25 Hz | ✓ |
| No stimulus gives zero spikes in every trial | 0 spikes in 30/30 trials | ✓ |
| Bitter 150 Hz keeps mn9_left < 1 Hz | 0.00 Hz | ✓ |
| Bitter 150 Hz keeps mn9_right < 1 Hz | 0.00 Hz | ✓ |

### R3: v783 against published v630 (tolerance ±25 %)

| Sugar rate | Readout | v783 (ours) | v630 (published) | Relative |
|---|---|---|---|---|
| 200 Hz | mn9_left | 93.27 ± 5.25 | 93.27 | +0.0 % |
| 200 Hz | mn9_right | 61.60 ± 5.08 | 61.80 | −0.3 % |
| 100 Hz | mn9_left | 68.70 ± 4.82 | 67.03 | +2.5 % |
| 100 Hz | mn9_right | 50.73 ± 4.93 | 48.63 | +4.3 % |

All 4 pass. The exact match of mn9_left at 200 Hz (2,798 spikes over 30 trials in both cases) is a coincidence. The per-trial counts differ (ours range from 79, the published from 86), and our own v630 run at the same setting gives 92.77 Hz.

### R4: dose response on v783

mn9_left rises from 68.70 Hz (100 Hz sugar) to 84.33 Hz (150 Hz) to 93.27 Hz (200 Hz). ✓

**Verdict: REPRODUCED** ([`sugar_mn9.json`](../results/phase0/sugar_mn9.json)).

## 6. Reference firing rates (all conditions)

[`reference_rates.json`](../results/phase0/reference_rates.json) holds, for every condition, the mean and SD of every neuron with a nonzero rate, plus per-trial spike counts for the readouts. It is the target Phase 1 must match.

| Condition | Stimulated | Active neurons | Spikes per trial (mean) | mn9_left | mn9_right | DNg62 | DNge078 |
|---|---|---|---|---|---|---|---|
| v630 sugar 100 Hz | 21 | 405 | 9,852 | 69.57 | 51.13 | 0 | 0 |
| v630 sugar 200 Hz | 21 | 450 | 16,980 | 92.77 | 62.30 | 0 | 0 |
| v783 sugar 100 Hz | 21 | 403 | 9,706 | 68.70 | 50.73 | 0 | 0 |
| v783 sugar 150 Hz | 21 | 424 | 13,649 | 84.33 | 59.33 | 0 | 0 |
| v783 sugar 200 Hz | 21 | 444 | 16,920 | 93.27 | 61.60 | 0 | 0 |
| v783 bitter 150 Hz | 21 | 84 | 4,660 | 0 | 0 | 0 | 0 |
| v783 JONs 150 Hz | 146 | 641 | 28,468 | 0 | 0 | 19.77 | 20.40 |
| v783 none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Rates are in Hz. DNg62 and DNge078 are the descending neurons Shiu et al. used as Johnston's organ readouts (`jon_dn1`, `jon_dn2`).

**Descriptive observations (not preregistered, no claims attached):**
- **Sugar:** 403 downstream neurons respond: 280 central, 77 descending and 29 motor. The most active include CB0248, CB0192 and the ingestion motor neurons CB0700.
- **Bitter:** activity spreads to 63 downstream neurons, mostly central (CB0159, CB0219), and reaches neither MN9.
- **JONs:** 495 downstream neurons respond, including Shiu's two descending readouts at about 20 Hz each and DNg84 and DNg29. The JON pathway is a candidate second validation pathway for Phase 2, alongside sugar → MN9.

## 7. Real-time factor

Measured with `scripts/bench.py` per the prereg ([`bench_vm.json`](../results/phase0/bench_vm.json)). The workload was v783, 21 sugar GRNs at 150 Hz, 5 trials of 1.0 s, seeds 0–4, with nothing else heavy running.

| | |
|---|---|
| Network build | 13.2 s |
| Untimed warm-up | 0.7 s |
| `net.run()` per 1 s trial | 5.39, 6.71, 4.91, 4.73, 5.29 s |
| RTF per trial | 0.186, 0.149, 0.204, 0.211, 0.189 |
| **Median RTF** | **0.189** |
| Wall time per 0.1 ms step | about 529 µs (real time needs ≤ 100 µs) |
| **Gap to RTF 1.0** | **5.3×** |

As predicted in the prereg, the RTF is well below 1.0 and of order 0.1 (0.189).

**Cost depends on activity.** During the reference-rate runs, with two networks sharing the VM (so these are not RTF measurements), a trial with no stimulus took 3.8 s, sugar about 6.2 s and JONs (28k spikes) 12.8 s. Brian2's fixed per-step cost is already about 380 µs before any spike is delivered. That is why Phase 1 avoids per-step interpreter overhead entirely. See [`PHASE1_ENGINE_PROPOSAL.md`](PHASE1_ENGINE_PROPOSAL.md).

## 8. Deviations, incidents and disclosures

1. **First reference run lost to a time limit.** The first full `reference_rates.py` run was stopped by the cloud session's background-task time limit (about 30 min) before it wrote any output. Its progress messages were held in a pipe buffer, so no results from it were seen. The script was changed to save each condition when it finishes and to resume from saved conditions (`a9bf441`). The conditions, seeds and trial length are unchanged; the conditions list was checked to be identical. The complete rerun took 25 min.
2. **Benchmark recorded with `repo.dirty = true`.** At bench time, `README.md` and `CLAUDE.md` had uncommitted edits. The engine and benchmark code were at `a72e34f`, exactly as registered.
3. **Pilots before registration.** These are disclosed in the prereg. Short 0.1 s pilot runs had already shown MN9 firing under sugar, so the direction of R2's first check was not blind. R1, R3, R4 and the bitter and no-stimulus predictions were untested.
4. **Code formatting after registration.** `ruff format` and two `lambda → def` refactors were applied after registration (`734ca3e`). We checked that `reference_rates.py` is AST-identical, and the criteria in `reproduce_sugar_mn9.py` are unchanged.
5. **CPU name.** The VM reports a generic "Intel(R) Xeon(R) Processor". Family and model are given in section 1.

## 9. Reproduce

```bash
pip install -e ".[test]"
python scripts/fetch_data.py && python scripts/fetch_data.py --group provenance
python scripts/bench.py --out bench.json            # RTF, about 1 min after the build
python scripts/reference_rates.py --workers 2       # 8 conditions x 30 trials, about 25 min on 4 vCPUs
python scripts/reproduce_sugar_mn9.py               # R1-R4 and the verdict
python scripts/map_stimuli.py                       # optional: re-derive data/stimuli_783.json (about 8 min)
```

## 10. What's next

Phase 1 needs a 5.3× speedup over this VM's CPU reference to reach real time on any hardware, and the target is a consumer GPU. The proposed design (event-driven sparse propagation, a spike delay ring, fused kernels, CUDA Graphs) is in [`PHASE1_ENGINE_PROPOSAL.md`](PHASE1_ENGINE_PROPOSAL.md). It estimates RTF of about 7–15 on a mid-range consumer GPU. **No engine code will be written until the maintainer approves the design.**
