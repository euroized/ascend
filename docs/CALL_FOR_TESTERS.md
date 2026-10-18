# Run a whole fly brain model on your own GPU

ASCEND simulates a connectome-constrained model of the entire adult fruit fly brain: **138,639 neurons and 15 million connections**. The wiring comes from the [FlyWire](https://flywire.ai) connectome, and the dynamics from the [Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9) model. It is built to run on a single consumer NVIDIA GPU with one command, and you're invited to try it on yours and add your result to the public record.

## Where things stand

- **The model is reproduced exactly.** In double precision on the CPU, the ASCEND engine fed the same input as the published reference model (Brian2) emits the **same spikes, spike for spike**, on the full brain, for every stimulus tested. The GPU engine is checked against it on your own machine.
- **Reference speed:** the original model runs at **0.19× real time** on a 4-core machine. One simulated second takes about 5 seconds.
- **The ASCEND GPU engine** runs each 1,000-step chunk of simulation as a single CUDA Graph. We predict roughly **7–15× real time** on a mid-range card such as an RTX 3060, 4060 or 4070. The prediction is preregistered, meaning it was written down before any GPU measurement.
- **Results board:** no GPU results are published yet. The first submissions will set the numbers.

Everything is preregistered and every result is published, including misses. See [`PREREG_PHASE1.md`](PREREG_PHASE1.md) and [`REPORT_PHASE1.md`](REPORT_PHASE1.md).

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
2. **Speed:** 5 timed one-second runs of the full brain. You get your real-time factor.
3. **Statistical fidelity:** 4 stimuli × 30 trials, compared with the published reference rates.
4. **Same-input check:** your GPU and your CPU get identical input and must produce the same spikes.

## Share your result

Open a **[Benchmark result](../../../issues/new?template=benchmark_result.yml)** issue and paste `bench.json` unedited. Every result goes on the record:

- **Any speed:** fast or slow, every GPU adds a data point.
- **Errors:** paste the output. A report from new hardware is just as useful as a number.
- **Laptop:** tell us the power limit if you know it.
- **Data-centre or workstation card** (A100, H100, L4, T4, RTX A-series and so on): welcome and published, but listed separately from consumer GPUs.

`bench.json` records your hardware (GPU, CPU, RAM, OS) and software versions so that results can be compared. It contains nothing else about you. Each submission is credited in the results.

## What counts

The real-time claim is earned only when all of this holds:

1. A consumer-GPU run on an unmodified checkout shows `S1_rtf_median_ge_1`, `F_fidelity_suite` and `G_same_input` all `true`.
2. Someone else reproduces it on a different GPU model.

The README states only what those runs show.

## About the model

This is a **connectome-constrained model**: leaky integrate-and-fire neurons wired according to measured synapses. It is not a brain upload, and it makes no claim to be conscious or alive. It builds on Shiu et al. 2024; FlyWire (Dorkenwald et al. 2024, Schlegel et al. 2024); NeuroMechFly and FlyGym; Eon Systems; and [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain).
