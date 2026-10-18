# Results

Every file here comes from a script in `scripts/` and answers to a preregistration in `docs/`. Each one records the hardware and software it ran on. Files stay under 1 MB, and misses are kept alongside hits.

## Phase 0: reference baseline

Preregistration: [`docs/PREREG_PHASE0.md`](../docs/PREREG_PHASE0.md). Report: [`docs/REPORT_PHASE0.md`](../docs/REPORT_PHASE0.md).

| File | Produced by | Contents |
|---|---|---|
| [`phase0/bench_vm.json`](phase0/bench_vm.json) | `scripts/bench.py` | Reference model's real-time factor on the 4-vCPU build VM: median 0.189 |
| [`phase0/reference_rates.json`](phase0/reference_rates.json) | `scripts/reference_rates.py` | 8 conditions × 30 trials: per-neuron rates and readout spike counts |
| [`phase0/sugar_mn9.json`](phase0/sugar_mn9.json) | `scripts/reproduce_sugar_mn9.py` | Preregistered sugar → MN9 checks R1–R4. Verdict: REPRODUCED |

## Phase 1: engine (build-VM tiers)

Preregistration: [`docs/PREREG_PHASE1.md`](../docs/PREREG_PHASE1.md). Report: [`docs/REPORT_PHASE1.md`](../docs/REPORT_PHASE1.md).

| File | Produced by | Contents |
|---|---|---|
| [`phase1/exact_*.json`](phase1) | `scripts/exact_check.py` | E: CPU float64 engine vs Brian2 on the full brain, 1 s. Identical in all 4 workloads |
| [`phase1/fidelity_cpu_float64_exact.json`](phase1/fidelity_cpu_float64_exact.json) | `scripts/fidelity_check.py` | F-pos: pass |
| [`phase1/fidelity_cpu_float32_exact.json`](phase1/fidelity_cpu_float32_exact.json) | `scripts/fidelity_check.py` | F-f32: pass |
| [`phase1/fidelity_cpu_float64_weight95.json`](phase1/fidelity_cpu_float64_weight95.json) | `scripts/fidelity_check.py` | Negative control, weights × 0.95: fails (as required) |
| [`phase1/fidelity_cpu_float64_delay17.json`](phase1/fidelity_cpu_float64_delay17.json) | `scripts/fidelity_check.py` | Negative control, delay 17 steps: **passes (miss)**, so the F tier is inadequate as registered |
| [`phase1/sameinput_shuffle{3,4}.json`](phase1) | `scripts/sameinput_check.py` | G-pos (Amendment 1): Jaccard 1.0000, pass |
| [`phase1/sameinput_delay17.json`](phase1/sameinput_delay17.json) | `scripts/sameinput_check.py` | G-neg-delay: fails (as required) |
| [`phase1/sameinput_weight95.json`](phase1/sameinput_weight95.json) | `scripts/sameinput_check.py` | G-neg-weight: fails (as required) |

## Real wiring against matched null networks

Preregistration: [`docs/PREREG_BASELINES.md`](../docs/PREREG_BASELINES.md). Write-up: [`docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md), section 7.

| File | Produced by | Contents |
|---|---|---|
| [`baselines/null_networks.json`](baselines/null_networks.json) | `scripts/baseline_check.py --summarize` | Real network and 9 null networks (weight-shuffled, rewired, random; 3 seeds each), 3 conditions × 10 trials. Real 6/6; every null 4/6. Claim holds |

## Build-VM benchmarks (descriptive)

| File | Produced by | Contents |
|---|---|---|
| [`benchmarks/cpu_float64_vm.json`](benchmarks/cpu_float64_vm.json) | `scripts/bench.py --engine ascend --device cpu --dtype float64` | RTF median 0.091, peak host memory 1,627 MiB |
| [`benchmarks/cpu_float32_vm.json`](benchmarks/cpu_float32_vm.json) | `scripts/bench.py --engine ascend --device cpu --dtype float32` | RTF median 0.149, peak host memory 1,648 MiB |

## Community benchmarks

Benchmark runs submitted through the [Benchmark result](../../../issues/new?template=benchmark_result.yml) issue template are stored in [`community/`](community) and listed on the [results board](community/README.md). Every submission is listed, whatever its outcome.

`scripts/check_submission.py` checks each one against the Phase 1 preregistration. It recomputes the real-time factor from the raw run times and does not trust the file's own `criteria` block. It also confirms that the checkout was unmodified, that the commit is in this repository, that the connectome matches the manifest, and that the GPU is a consumer card.

```bash
python scripts/check_submission.py bench.json                                   # check only
python scripts/check_submission.py bench.json --issue 12 --submitter octocat --add  # store it and rebuild the board
```
