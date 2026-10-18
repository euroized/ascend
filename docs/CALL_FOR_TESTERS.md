# Call for GPU testers: can a whole fly brain model run in real time on a gaming GPU?

ASCEND simulates a connectome-constrained model of the entire adult fruit fly brain: **138,639 neurons and 15 million connections**. The wiring comes from the [FlyWire](https://flywire.ai) connectome, and the dynamics from the [Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9) model. We want to know whether one consumer NVIDIA GPU can simulate it at least as fast as real time.

The engine is built and checked. The answer now depends on your hardware.

## Where things stand

- **The CPU engine is exact.** Fed the same input as the published reference model (Brian2), our engine emits the **same spikes, spike for spike**, on the full brain over 1 s, for every stimulus tested.
- **The reference is slow.** On a 4-core cloud VM, the reference runs at **0.19× real time**: one simulated second takes about 5 seconds of wall time.
- **The GPU engine has never run on a GPU.** It compiles for every NVIDIA architecture from Pascal (GTX 10-series) on, and its kernels pass every test in emulation on CPU. Your run may be the first on real hardware.
- **Our prediction is roughly 7–15× real time** on a mid-range card such as an RTX 3060, 4060 or 4070. It is preregistered, which means written down before anyone measured it. We might be wrong, and that is worth knowing too.

Everything here is preregistered and every result is published, including misses. See [`PREREG_PHASE1.md`](PREREG_PHASE1.md) and [`REPORT_PHASE1.md`](REPORT_PHASE1.md).

## What you need

- An NVIDIA GPU, GTX 10-series or newer, desktop or laptop
- An NVIDIA driver with CUDA 12 support (driver 525 or newer)
- Python 3.10 or newer, about 8 GB of RAM and about 1 GB of disk
- 15–30 minutes, most of it unattended

## Run it

```bash
git clone https://github.com/recurse-root/ascend && cd ascend
pip install -e ".[cuda]"
python scripts/fetch_data.py
python scripts/bench.py --engine ascend --device cuda --fidelity --out bench.json
```

`fetch_data.py` downloads the connectome (about 104 MB) and checks it against SHA-256 checksums. `bench.py` then runs four stages:

1. **Self-test:** compares your GPU against the CPU engine on a small network.
2. **Speed:** 5 timed one-second runs of the full brain.
3. **Statistical fidelity:** 4 stimuli × 30 trials, compared with the published reference rates.
4. **Same-input check:** your GPU and your CPU get identical input and must produce the same spikes.

## Submit it

Open a **[Benchmark result](../../../issues/new?template=benchmark_result.yml)** issue and paste `bench.json` unedited. Please send every result:

- **Slow:** still a data point.
- **Crashed:** paste the error. A first-run crash on your GPU is one of the most useful reports we can get.
- **Laptop:** tell us the power limit if you know it.
- **Data-centre or workstation card** (A100, H100, L4, T4, RTX A-series and so on): published, but it doesn't count toward the consumer-GPU claim.

`bench.json` records your hardware (GPU, CPU, RAM, OS) and software versions so that results can be compared. It contains nothing else about you.

## What counts

The real-time claim is earned only when all of this holds:

1. A consumer-GPU run on an unmodified checkout shows `S1_rtf_median_ge_1`, `F_fidelity_suite` and `G_same_input` all `true`.
2. A second person on a different GPU model reproduces it.

Until then, the README says nothing about real time. Each submission is credited in the results.

## About the model

This is a **connectome-constrained model**: leaky integrate-and-fire neurons wired according to measured synapses. It is not a brain upload, and it makes no claim to be conscious or alive. It builds on Shiu et al. 2024; FlyWire (Dorkenwald et al. 2024, Schlegel et al. 2024); NeuroMechFly and FlyGym; Eon Systems; and [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain).
