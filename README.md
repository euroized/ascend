# ascend

[![CI](https://github.com/recurse-root/ascend/actions/workflows/ci.yml/badge.svg)](https://github.com/recurse-root/ascend/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Connectome-constrained fly brain, real-time on one GPU, reverse engineered, then learning to improve how it learns.

ASCEND is an open, preregistered project in three phases. Each phase is earned separately, and nothing below is claimed until its preregistered criteria pass on real hardware.

| Phase | Goal | Status |
|---|---|---|
| 0. Baseline | Wrap the Shiu et al. 2024 reference model, reproduce sugar → MN9, measure its speed | **Complete.** Sugar → MN9 reproduced (14/14 preregistered checks). Reference RTF 0.189 on a 4-vCPU VM ([report](docs/REPORT_PHASE0.md), [prereg](docs/PREREG_PHASE0.md)) |
| 1. Boot | Whole FlyWire v783 brain (138,639 neurons, 54.5 M synapses) at real-time factor ≥ 1.0 on one consumer NVIDIA GPU, fidelity-matched to the reference | In progress: engine built, awaiting GPU runs ([prereg](docs/PREREG_PHASE1.md), [design](docs/PHASE1_ENGINE_PROPOSAL.md)) |
| 2. Decompile | Perturb every neuron and map its effect on every other (an in-silico effectome), validated against known pathways | Not started |
| 3. Ascend | Wiring frozen; evolve the learning rules, gated and tested against frozen and random-rule controls | Not started |

This is a connectome-constrained model: a network of leaky integrate-and-fire neurons wired according to the FlyWire connectome. It is not a brain upload, and we make no claims about consciousness or sentience.

## Quick start

Requires Python ≥ 3.10 and a C++ compiler (Brian2 compiles generated code with Cython).

```bash
git clone https://github.com/recurse-root/ascend && cd ascend
pip install -e ".[test]"
pytest -q                       # CPU only, no downloads
python scripts/fetch_data.py    # FlyWire v783 connectome, ~104 MB, SHA-256 verified
python scripts/bench.py --out bench.json
```

`bench.py` builds the full 138,639-neuron network and stimulates the 21 sugar-sensing gustatory neurons at 150 Hz. It runs 5 one-second trials and writes the real-time factor (simulated seconds per wall-clock second) together with your hardware and software, recorded automatically. Building the network takes a few minutes and about 4 GB of RAM.

The default engine is the Brian2 reference model. To benchmark the ASCEND engine on an NVIDIA GPU (CUDA 12, compute capability 6.0 or newer):

```bash
pip install -e ".[cuda]"
python scripts/bench.py --engine ascend --device cuda --fidelity --out bench.json
```

`--fidelity` adds the preregistered fidelity suite: 4 conditions × 30 trials, compared with the Phase 0 reference rates. It takes a few minutes on a GPU.

## Run it on your GPU

Try it on your own GPU and add your result to the public record. Every card and every result counts: fast, slow or failed. [Run it on your GPU](docs/CALL_FOR_TESTERS.md) explains what the run does and what it takes.

1. Run the GPU command above on an unmodified checkout, with nothing else heavy running.
2. Open a [Benchmark result](../../issues/new?template=benchmark_result.yml) issue and paste `bench.json` unedited.

The real-time claim (Phase 1) needs at least one consumer-GPU run that passes every preregistered criterion, plus an independent reproduction on different hardware.

## How we make claims

- Every claim is preregistered in `docs/PREREG_*.md` before it is measured. Preregistrations are never edited after their runs; changes go in dated amendments.
- Fixed seeds, matched controls, and misses published alongside hits.
- Hardware disclosed with every number.
- No data in the repository: inputs are fetched from pinned commits and checked against [`data/manifest.json`](data/manifest.json). Committed results stay under 1 MB.
- CI runs the test suite on CPU for every push.

## Repository

| Path | Contents |
|---|---|
| [`ascend/engine/`](ascend/engine) | The ASCEND engine: exact CPU engine (NumPy), CUDA kernels and CUDA Graph host code, CPU emulator of the kernels |
| [`ascend/reference/`](ascend/reference) | Shiu et al.'s `model.py`, vendored verbatim (hash-pinned), plus a seeded, timed wrapper |
| [`ascend/`](ascend) | Data fetching and verification, stimulus sets, rates, hardware capture |
| [`scripts/`](scripts) | `fetch_data.py`, `bench.py`, `exact_check.py`, `fidelity_check.py`, `reference_rates.py`, `reproduce_sugar_mn9.py`, `map_stimuli.py` |
| [`data/`](data) | Data manifest and stimulus neuron IDs (FlyWire v630 and v783) |
| [`docs/`](docs) | Preregistrations and design notes |
| [`results/`](results) | Small JSON results, each tied to a preregistration |

## Prior art and credits

ASCEND builds directly on others' work:

- **Shiu et al. 2024**, *A Drosophila computational brain model reveals sensorimotor processing*, Nature 634, 210–219. The reference model ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), MIT) that every ASCEND engine must match.
- **FlyWire**: Dorkenwald et al. 2024 and Schlegel et al. 2024, Nature 634. The whole-brain connectome and its annotations ([flywire.ai](https://flywire.ai), [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)).
- **NeuroMechFly / FlyGym 2.x**: the biomechanical fly body and simulation environment.
- **Eon Systems** (March 2026): ran the full FlyWire connectome in a NeuroMechFly body (walking, grooming, feeding), without learning.
- **[erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain)**: runs the connectome on a laptop GPU with a Hebbian learning rule.

If you use ASCEND, please cite these works as well.

## License

MIT, see [LICENSE](LICENSE). The vendored reference model keeps its own MIT license ([`ascend/reference/LICENSE.shiu`](ascend/reference/LICENSE.shiu)). FlyWire data are subject to their own terms.
