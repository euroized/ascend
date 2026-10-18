# ASCEND technical report

**Version:** 2026-10-18. This report covers Phase 0, the build-VM part of Phase 1, and the null-network comparison. Every number here comes from a result file in [`results/`](../results). Each file records the hardware it ran on and answers to a preregistration in this folder. GPU numbers are not in yet: they come from community submissions ([results board](../results/community/README.md)).

## Summary

ASCEND runs the whole-brain connectome-constrained model of the adult fruit fly from Shiu et al. 2024. It has 138,639 leaky integrate-and-fire neurons wired by the 15,091,983 connections of the FlyWire v783 connectome. ASCEND is a fresh, verified engine for this model, aimed at real time on one consumer GPU.

What is measured so far:
1. **Exact reproduction.** In double precision, the ASCEND CPU engine emits the same spikes as the published Brian2 model, spike for spike, on the full brain, for every stimulus tested.
2. **Statistical fidelity, with controls.** Both precisions match the reference statistically. A 5% weight error is detected. A one-step delay error was not detected by that statistical tier; this miss is published, and a same-input tier that does detect it was added.
3. **The function comes from the wiring.** The model has no trained parameters. With its real wiring it reproduces the selective sensorimotor readouts. Nine matched null networks with the same neurons, connection count and synaptic weights reproduce none of them (section 7).

The real-time GPU claim is preregistered and waiting on testers' hardware.

## 1. Model

Each neuron *i* is a leaky integrate-and-fire unit with an exponentially decaying synaptic variable, exactly as in Shiu et al. 2024 (`ascend/reference/shiu_model.py`, vendored verbatim and hash-pinned):

```
dv/dt = (v_0 − v + g) / t_mbr        (frozen while refractory)
dg/dt = −g / tau                     (frozen while refractory)
spike when v > v_th;  then v ← v_rst, g ← 0, refractory for t_rfc
each spike adds w = w_syn × (signed synapse count) to g of every postsynaptic neuron, t_dly later
```

| Constant | Value | Source (as cited by Shiu et al.) |
|---|---|---|
| v_0 = v_rst | −52 mV | Kakaria & de Bivort 2017 |
| v_th | −45 mV | Kakaria & de Bivort 2017 |
| t_mbr | 20 ms | Kakaria & de Bivort 2017 |
| tau | 5 ms | Jürgensen et al. 2021 |
| t_rfc | 2.2 ms | Lazar et al. 2021 |
| t_dly | 1.8 ms | Paul et al. 2015 |
| w_syn | 0.275 mV per synapse | the model's one free parameter, set by Shiu et al. |
| stimulus | Poisson kicks of 250 × w_syn = 68.75 mV, at 150 Hz | Shiu et al. |
| dt | 0.1 ms (10,000 steps per simulated second) | Brian2 default |

Stimulated neurons have no refractory period, as upstream sets. A stimulated neuron therefore fires at about the stimulus rate.

## 2. Connectome mapping

- **Source:** the FlyWire v783 public release, as packaged by Shiu et al. for their model: `Completeness_783.csv` (neurons) and `Connectivity_783.parquet` (connections). `scripts/fetch_data.py` downloads them from pinned commits and checks them against SHA-256 in [`data/manifest.json`](../data/manifest.json).
- **Neurons:** 138,639. One model neuron per FlyWire neuron, indexed by its FlyWire root ID.
- **Connections:** 15,091,983 pre → post pairs carrying 54,492,922 synapses. That is a median of 2 synapses per pair and a maximum of 2,405.
- **Weights:** w = w_syn × (synapse count) × (+1 or −1). The sign comes from the presynaptic neuron's machine-predicted neurotransmitter (Eckstein et al. 2024), as packaged by Shiu et al.: acetylcholine is excitatory, GABA and glutamate are inhibitory. 40.0% of connections are inhibitory.
- **Stimulus and readout neurons:** Shiu et al.'s sets use FlyWire v630 IDs. `scripts/map_stimuli.py` maps each to v783 by synaptic-partner similarity (weighted Ruzicka, score ≥ 0.5 and ≥ 2× the runner-up). Every mapped set keeps its size and expected cell types ([Phase 0 report](REPORT_PHASE0.md)).

## 3. Parameters and training

There is **no training**:

| What | Count | Set by |
|---|---|---|
| Synaptic weights | 15,091,983 | the connectome (synapse counts and transmitter signs) times one constant |
| Neuron and synapse constants | 7 | literature (table above) |
| Free scalar | 1 (w_syn) | Shiu et al. 2024 |
| Parameters fitted by ASCEND | **0** | — |

Phase 3 is where learning enters. The wiring stays frozen, and only plasticity rules are evolved, against frozen-brain and random-rule controls (see the README).

## 4. Engine architecture

The engine is in `ascend/engine/`. The design is in [`PHASE1_ENGINE_PROPOSAL.md`](PHASE1_ENGINE_PROPOSAL.md).

- **Event-driven delivery.** Connectivity is held in CSR form by presynaptic neuron. Only rows of neurons that spiked are touched, about 1.4 spikes and 150 synaptic updates per step under sugar input. A dense or sparse matrix product per step would move about 121 MB.
- **One delay ring.** Every connection has the same 18-step delay, so spikes are delayed rather than currents: a ring of D + 1 spike lists.
- **Brian2's schedule, reproduced:**
  1. state update;
  2. threshold;
  3. delayed delivery, then stimulus, both applied only to non-refractory targets;
  4. reset.

  This was derived from Brian2's generated code. It is why the float64 engine matches spike for spike.
- **Counter-based stimulus.** Kicks come from a hash of (seed, step, neuron), so the CPU and GPU see identical input. This makes the same-input tier possible.
- **GPU path.** Four CUDA kernels (`update`, `deliver`, `clear`, `advance` in `kernels.cu`) are compiled at run time by NVRTC through CuPy, so no `nvcc` is needed. Each 1,000-step chunk is captured as one CUDA Graph, so there are no host round trips inside a chunk.
- **Testable without a GPU.** The same `kernels.cu` compiles as C++ (`emulate.py`) and runs in CI against the NumPy engine.

## 5. Verification

The engine is only as useful as its fidelity, so speed counts only after these tiers pass ([preregistration](PREREG_PHASE1.md), [report](REPORT_PHASE1.md)).

| Tier | What is compared | Result |
|---|---|---|
| E | CPU float64 vs Brian2, identical replayed input, full brain, 1 s, 4 workloads | **Identical spikes** (13,220 / 4,603 / 28,334 / 0) |
| F-pos, F-f32 | firing rates vs the Phase 0 reference, 4 conditions × 30 trials, float64 and float32 | **Pass** every check |
| F-neg-weight | weights × 0.95 | **Fails**, as required |
| F-neg-delay | delay 17 steps instead of 18 | **Passes: a miss.** Rates cannot see a 0.1 ms timing error. Published, and fixed by tier G |
| G (Amendment 1) | GPU float32 vs CPU float32, identical input; Jaccard of (step, neuron) spike events ≥ 0.99 | Controls on the build VM: random summation order 1.0000; delay error 0.135–0.617 and weight error 0.135–0.622 (fail, as required). GPU runs come from testers |

## 6. Benchmarks

Speed is the real-time factor (RTF): simulated seconds per wall-clock second. The workload is the full brain with 21 sugar GRNs at 150 Hz, 5 trials of 1 s, seeds 0–4. Hardware for the build VM: 4-vCPU Intel Xeon VM, 15.7 GB RAM, Linux, no GPU.

| Engine | Hardware | Precision | Fidelity | RTF, median (range) | Wall time per 0.1 ms step | Peak host memory | Device memory | Fitted parameters |
|---|---|---|---|---|---|---|---|---|
| Brian2 reference (Shiu et al.), Cython target | build VM, CPU | float64 | is the reference | 0.189 (0.149–0.211) | 529 µs | not recorded | — | 0 |
| ASCEND CPU engine | build VM, CPU | float64 | E: identical spikes; F: pass | 0.091 (0.077–0.097) | 1,097 µs | 1,627 MiB | — | 0 |
| ASCEND CPU engine | build VM, CPU | float32 | F: pass; reference for G | 0.149 (0.139–0.153) | 673 µs | 1,648 MiB | — | 0 |
| ASCEND CUDA engine | consumer NVIDIA GPU | float32 | S, F and G run on the tester's machine | **pending submissions**. Preregistered prediction: 7–15 | 7–14 µs (predicted) | recorded per submission | 177 MiB of engine buffers (computed from the allocation sizes), plus the CUDA context | 0 |

Sources:
- [`results/phase0/bench_vm.json`](../results/phase0/bench_vm.json)
- [`results/benchmarks/cpu_float64_vm.json`](../results/benchmarks/cpu_float64_vm.json)
- [`results/benchmarks/cpu_float32_vm.json`](../results/benchmarks/cpu_float32_vm.json)
- GPU submissions go to the [results board](../results/community/README.md), which records `gpu_used_mib` and `host_peak_rss_mib` from `bench.py`.

**How to read this table:**
- The CPU engines are slower than Brian2's compiled code. They are single-threaded NumPy, built for exactness and for running the tests, not for speed.
- Real time needs 100 µs per step. The CUDA engine is designed for that: event-driven delivery, a 3.3 MB neuron-state sweep per step, and no host round trips inside a 1,000-step CUDA Graph. Whether it gets there is what the submissions will show.
- **Task performance** is in section 7. The model is not trained on a task. Its "accuracy" is whether the published sensorimotor readouts appear, and how far its firing rates are from the reference (tiers E and F).

## 7. Real wiring against matched null networks

[Preregistration](PREREG_BASELINES.md), registered in commit `4a9f87b` before any null network was simulated. Raw results: [`results/baselines/null_networks.json`](../results/baselines/null_networks.json).

**Question.** The model has no trained parameters. Do its selective responses come from which neuron connects to which, or would any network with the same size and weight statistics do?

**Networks.** All have the same 138,639 neurons, 15,091,983 connections and multiset of signed weights, and the stimulus and readout neurons keep their identities.
- **weights:** real connections, weights shuffled across them.
- **rewired:** every neuron keeps its in-degree, out-degree and outgoing weights (Dale's law), but its connections' targets are shuffled.
- **random:** only the totals are kept.

There are 3 shuffle seeds each, and every network gets identical input. The engine is CPU float32, with 10 trials of 1 s per condition.

**Score.** There are 6 cells. With sugar input, MN9 is on and the Johnston's organ descending neurons (DNg62 and DNge078) are off. With bitter input, both groups are off. With Johnston's organ input, MN9 is off and DNg62/DNge078 are on. "On" means a mean rate ≥ 10 Hz and "off" means below 1 Hz.

| Network | Shuffle seeds | Score (each seed) | MN9 under sugar (Hz) | DNg62/DNge078 under JONs (Hz) | Spikes per trial: sugar / bitter / JONs | Neurons active under sugar |
|---|---|---|---|---|---|---|
| **real** | — | **6/6** | **70.4** | **19.7** | 13,400 / 4,671 / 28,502 | 412 |
| weights | 0, 1, 2 | 4/6, 4/6, 4/6 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 | about 3,650 / 3,570 / 24,500 | 60–66 |
| rewired | 0, 1, 2 | 4/6, 4/6, 4/6 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 | about 3,980 / 3,490 / 22,870 | 132–171 |
| random | 0, 1, 2 | 4/6, 4/6, 4/6 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 | about 3,670 / 3,730 / 25,340 | 80–94 |

**Outcome.** C1 (real scores 6/6) and C2 (every null network scores below 6/6) both hold, so the preregistered claim holds. No network ran away.

**What the result means:**
- **Every null network scores 4/6, and that 4 is the floor.** A network that never activates its readouts gets exactly the four "off" cells. All nine nulls did exactly that: in no condition did any null network drive MN9 or DNg62/DNge078 at all, while the real network drove them at 70 and 20 Hz.
- **Activity barely propagates without the real wiring.** A stimulated neuron fires at about the stimulus rate in every network, so the 21 sugar GRNs alone account for about 3,150 spikes per trial. The real network adds about 10,000 spikes downstream of them. The null networks add only about 500–850.
- **Keeping the real topology does not help.** The `weights` null keeps every real connection but shuffles which weight sits on which. It does no better than the others, so signed synapse counts on the right connections matter, not topology alone.
- **The Pearson r of whole-brain rates against the reference is high even for the nulls** (0.64–0.95, descriptive). The stimulated neurons fire at about 150 Hz in every network and dominate that correlation, so it is a weak measure of wiring. The preregistered score does not use it.

The [demo](demo/) shows one of these runs (`rewired`, seed 0, with sugar input): the input spreads into olfactory and visual neurons instead of reaching MN9.

## 8. What the connectome constraint gives that a generic model does not

A generic network (an RNN, a transformer, or a spiking network with random or learned weights) can be trained to produce almost any input-output map. Wiring it from the connectome gives four things it cannot.

1. **Function with nothing fitted.** The selective responses appear without any training: sugar drives the feeding motor neuron, antennal mechanosensory input drives its descending neurons, and nothing crosses over. The same 15.1 million weights wired at random produce none of it (section 7). Here the parameters were measured by electron microscopy, not learned.
2. **Every unit is a real, named cell.** Each model neuron is a FlyWire neuron, with a root ID, a position, a cell type and a predicted transmitter. A model result such as "silencing neuron X lowers MN9 firing" is a concrete experiment for a fly lab with genetic tools. Hidden unit 4,182 of a generic model has no counterpart to test.
3. **Interpretation becomes measurement.** Because the units are real cells, perturbing each one in turn (Phase 2, the effectome) gives a map that can be checked against known pathways and new experiments. It is not an analysis of one arbitrary learned solution among many.
4. **A controlled test of learning.** With the wiring frozen at the measured connectome, Phase 3 can ask what plasticity adds on top of the wiring, against frozen-brain and random-rule controls, without the confound of architecture search.

What ASCEND adds beyond the prior work it builds on (Shiu et al.'s model; laptop-GPU and embodied runs of the same connectome by others):

- **Verified exactness.** It is an independent engine that matches the published model spike for spike. Its fidelity tiers include negative controls that must fail, and their one miss is published (done).
- **Real time on a consumer GPU,** checked on testers' own hardware against the CPU engine. This is preregistered and open to anyone (pending).
- **Every neuron perturbed** at full scale, validated against known pathways (Phase 2, proposed).
- **Evolved learning rules** over frozen wiring, behind a preregistered gate (Phase 3, planned).

## 9. Limitations

- **The model is simple.** It uses point neurons with one shared set of constants, no neuron-specific biophysics, no gap junctions, no neuromodulation and no plasticity. Transmitter signs are predictions, and glutamate is treated as inhibitory everywhere. Its successes and failures are those of Shiu et al.'s model, which ASCEND reproduces exactly and does not try to improve in Phases 1 and 2.
- **No body and no closed loop.** Inputs are Poisson drives to sensory neurons, and outputs are firing rates. There is no NeuroMechFly/FlyGym body yet.
- **The scored readouts are few.** The null-network test scores 2 readout groups under 3 inputs, the readouts Shiu et al. validated. It shows that the wiring is necessary for these responses. It does not show that the model is right about the rest of the brain.
- **No GPU numbers yet.** The real-time claim needs consumer-GPU submissions that pass the S, F and G tiers, plus an independent reproduction. Until then, the GPU row of the benchmark table is a preregistered prediction, not a result.
- **Build-VM speed is not a target.** The CPU engines exist for exactness and testing. Their speed is reported for completeness.

## 10. Reproduce

```bash
pip install -e ".[test]" && pytest -q
python scripts/fetch_data.py                                        # connectome, SHA-256 checked
python scripts/exact_check.py --stimulus sugar --out exact.json     # tier E
python scripts/fidelity_check.py --device cpu --dtype float32       # tier F
python scripts/baseline_check.py --variant rewired --seed 0         # one null network (section 7)
python scripts/bench.py --engine ascend --device cpu --dtype float32 --out bench.json
python scripts/bench.py --engine ascend --device cuda --fidelity --out bench.json   # GPU: S, F and G
python scripts/make_demo.py                                         # the demo's recorded runs
```

## Credits

- Shiu et al. 2024 (*Nature* 634) for the model.
- FlyWire (Dorkenwald et al. 2024; Schlegel et al. 2024) for the connectome and annotations.
- Pospisil et al. 2024 for the effectome concept used in Phase 2.
- NeuroMechFly/FlyGym, Eon Systems and erojasoficial-byte/fly-brain for related work on running and embodying the connectome.
