# Phase 1 engine: design proposal

**Status:** proposal, awaiting maintainer approval. No engine code exists yet. The Phase 1 criteria will be set in `docs/PREREG_PHASE1.md` before any engine measurement.

## Goal

Simulate the full FlyWire v783 model (138,639 neurons, 15,091,983 connections, dt = 0.1 ms) at real-time factor ≥ 1.0 on one consumer NVIDIA GPU, with dynamics matching the Shiu et al. 2024 reference model.

## Starting point (Phase 0)

The reference model (Brian2 runtime, Cython target, single process) reaches a median RTF of 0.189 on the 4-vCPU build VM (`results/phase0/bench_vm.json`). That is about 530 µs of wall time per 0.1 ms step, 5.3× too slow.

Real time means 10,000 steps per simulated second, so **the budget is 100 µs per step**, including everything.

## Where the work is

Per step, the model does three things:

1. **Neuron update.** Each of the 138,639 neurons updates two state variables (v, g) by exact exponential integration. Neurons in their refractory period are frozen.
2. **Spike delivery.** Every spike adds `w = 0.275 mV × signed synapse count` to the `g` of each postsynaptic neuron, 1.8 ms (18 steps) later.
3. **Poisson stimulus.** Each stimulated neuron receives a Bernoulli(rate × dt) kick of 68.75 mV to v.

Activity is sparse. In Phase 0 the sugar stimulus produced about 14,000 spikes per simulated second across the whole brain (`spikes_per_trial` in `bench_vm.json`), which is about 1.4 spikes per step. The mean out-degree is 15.09 M / 138,639 ≈ 109 connections. So delivery costs about 150 synaptic updates per step.

| Approach | Memory traffic per step | At 10,000 steps/s |
|---|---|---|
| Dense or sparse matrix product per step (W · s) | 15.09 M × 8 B ≈ 121 MB | ≈ 1.2 TB/s, more than any consumer GPU has |
| Event-driven: only spiking rows of the CSR | ≈ 150 × 8 B ≈ 1.2 kB | negligible |
| Neuron state (v, g, refractory counter, read + write) | 138,639 × 24 B ≈ 3.3 MB | ≈ 33 GB/s |

The connectivity should therefore never be touched densely. The cost floor is the neuron state sweep plus fixed per-step overheads: kernel launch, synchronisation and any host round trip.

## Proposed design

**Data layout (device-resident for the whole run):**
- Connectivity in CSR by presynaptic neuron: `row_ptr` int64 (138,640), `post` int32 (15.09 M) and `w` float32 (15.09 M), about 121 MB in total.
- Neuron state in structure-of-arrays float32 form: `v`, `g`, a refractory countdown (int16), and a per-neuron refractory length (0 for stimulated neurons, as upstream sets).
- **Delay buffer:** the synaptic delay is the same for every connection (18 steps), so we delay spikes rather than currents. A ring of 19 spike lists (indices plus a count per slot), each sized for the worst case of N entries, takes 19 × 138,639 × 4 B ≈ 10.5 MB. At step t, spikes emitted at step t − 18 are delivered.

**Per step, two fused kernels:**
1. `deliver`: one warp per delayed spike walks its CSR row and does `atomicAdd(g[post], w)`. About 150 atomics per step for the sugar stimulus. Launch-bound (about 3 µs).
2. `update`: one thread per neuron. It runs the exact integration (two precomputed exponential factors), the refractory freeze, the Poisson kick (counter-based Philox RNG keyed by seed, step and neuron, so runs are reproducible and order-independent), the threshold test, reset, and warp-aggregated append to the current spike list. Memory-bound on 3.3 MB, which fits in L2 on Ada GPUs.

**No host round trips:** K steps (e.g. 1,000) are captured into a single CUDA Graph and replayed. Spikes are appended to a device-side record and copied back asynchronously once per chunk. The Python loop runs once per K steps, not once per step.

**Order of operations** within a step follows Brian2's schedule (state update → threshold → synaptic delivery → Poisson input → reset), so the engine computes the same model rather than an approximation of it.

**Device-agnostic:** one engine interface with two backends:
- `cpu`: NumPy/Numba, float32 or float64. Runs in CI and in the cloud dev sessions.
- `cuda`: CuPy `RawModule` kernels compiled by NVRTC at runtime, so testers need only `pip install cupy-cuda12x` and no `nvcc`.

`scripts/bench.py --engine ascend --device cuda` reuses the existing workload, timing and JSON schema, so Phase 0 and Phase 1 numbers are directly comparable.

## Expected GPU real-time factor

Estimated time per step on a mid-range consumer GPU (RTX 3060 / 4060 / 4070 class):

| Item | Estimate | Basis |
|---|---|---|
| `deliver` kernel | 2–4 µs | Launch-bound inside a CUDA Graph. About 150 atomics. |
| `update` kernel | 3–10 µs | 3.3 MB read + write: about 3 µs if L2-resident (Ada, 24–36 MB L2), about 9 µs from DRAM at 360 GB/s (RTX 3060, 3 MB L2). |
| Graph and chunk overhead | about 1 µs | One graph launch and one async copy per 1,000 steps. |
| **Total** | **about 6–15 µs** | |

**Expected RTF ≈ 100 / (6–15) ≈ 7–15.** I would preregister a conservative expectation of RTF ≥ 3 on an RTX 3060 class GPU. That leaves 3× margin over 1.0 for per-step costs I have underestimated, such as launch latency on Windows/WDDM, clocks on laptop GPUs, or synchronisation.

**Activity sensitivity.** Delivery cost scales with spikes per step × fan-out. Even a global storm, with 10 % of neurons at 50 Hz, gives about 70 spikes per step × 109 ≈ 7,600 atomics per step, which is still a few µs. Real time is lost only if the per-step fixed cost exceeds 100 µs, which is why the design never returns to Python per step.

**What would miss the target:** a per-step Python loop in PyTorch or CuPy eager mode. That means about 8–12 kernel launches per step at about 5–10 µs each, or 50–120 µs per step, which gives RTF of about 0.8–2 and depends heavily on the platform. This is the main design risk we are avoiding.

## Fidelity plan (to be preregistered)

Two tiers, each checked against Phase 0 outputs:

- **Exact tier (CPU, float64).** Feed the engine the same Poisson input events Brian2 drew and require spike-for-spike identical output for 1 s on the full brain. This proves the engine computes the same model, including the refractory freeze, the reset, the delay and the update order.
- **Statistical tier (GPU, float32, its own RNG).** Over 30 trials with seeds 0–29, compare against `results/phase0/reference_rates.json` for every neuron the reference marks active: per-neuron mean rates, the readout neurons (MN9, JON targets) within preregistered tolerances, and the correlation of the full rate vector. Also require zero spikes with no stimulus.

The real-time claim then needs a consumer-GPU `bench.py` run that passes both the speed and fidelity criteria, plus an independent reproduction on different hardware.

## Build order (after approval)

1. CPU backend and the exact-tier test against Brian2 on synthetic networks (CI), then on the full brain (dev VM).
2. `docs/PREREG_PHASE1.md` with speed and fidelity criteria.
3. CUDA backend, the same tests run by testers, and the bench integration.
4. Open the call for GPU benchmark runs.
