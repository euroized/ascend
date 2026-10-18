# Results

Every file here comes from a script in `scripts/` and answers to a preregistration in `docs/`. Each one records the hardware and software it ran on. Files stay under 1 MB, and misses are kept alongside hits.

## Phase 0: reference baseline

Preregistration: [`docs/PREREG_PHASE0.md`](../docs/PREREG_PHASE0.md). Report: [`docs/REPORT_PHASE0.md`](../docs/REPORT_PHASE0.md).

| File | Produced by | Contents |
|---|---|---|
| [`phase0/bench_vm.json`](phase0/bench_vm.json) | `scripts/bench.py` | Reference model's real-time factor on the 4-vCPU build VM: median 0.189 |
| [`phase0/reference_rates.json`](phase0/reference_rates.json) | `scripts/reference_rates.py` | 8 conditions × 30 trials: per-neuron rates and readout spike counts |
| [`phase0/sugar_mn9.json`](phase0/sugar_mn9.json) | `scripts/reproduce_sugar_mn9.py` | Preregistered sugar → MN9 checks R1–R4. Verdict: REPRODUCED |

## Community benchmarks

Benchmark runs submitted through the [Benchmark result](../../../issues/new?template=benchmark_result.yml) issue template will be collected here once Phase 1 starts.
