# ASCEND: working rules

ASCEND builds a connectome-constrained model of the whole adult *Drosophila* brain in three separately earned phases:

1. **Boot.** Run the full FlyWire v783 brain (138,639 neurons, 15,091,983 connections, 54,492,922 synapses) at real-time factor ≥ 1.0 on one consumer NVIDIA GPU. It must be fidelity-matched to the Shiu et al. 2024 reference model (philshiu/Drosophila_brain_model, Brian2, dt 0.1 ms = 10,000 steps per simulated second).
2. **Decompile.** Perturb every neuron and record its effect on every other, producing an in-silico effectome. Validate it against at least 3 known pathways (sugar → feeding/MN9, looming → escape, and so on).
3. **Ascend.** Keep the wiring frozen. A self-improvement loop evolves the learning rules (dopamine-gated or neuromodulated plasticity), and every change passes a gate. The hit rule: evolved rules beat both the frozen brain and a random-rule control on held-out behavioural tests, and ablating them drops performance.

None of these is achieved until its preregistered criteria pass. The README only states what has been measured.

## Rules

- **Fidelity before speed.** Every engine is checked against the reference model before its speed counts.
- **Preregister** every claim in `docs/PREREG_*.md` before measuring. Never edit a prereg after its run; add dated amendments instead. Use matched controls and fixed seeds, and publish misses.
- **Disclose hardware** with every number. `scripts/bench.py` and `ascend.hardware.info()` record it automatically.
- **Size limit.** Never commit data or results over 1 MB. Inputs are fetched by `scripts/fetch_data.py` and verified against SHA-256 in `data/manifest.json`.
- **Device-agnostic engine code.** Development sessions are CPU-only. Testers run CUDA.
- **CI** runs pytest on CPU for every push. Tests must not need downloaded data.
- **Language.** Say "connectome-constrained model". Never claim consciousness or sentience, never call the model alive, and never call it a brain upload.
- **Credit prior art:** Shiu et al. 2024, FlyWire (Dorkenwald et al. 2024; Schlegel et al. 2024), NeuroMechFly/FlyGym 2.x, Eon Systems, erojasoficial-byte/fly-brain.
- **Private module.** The maintainer's self-improvement module is private and never enters this repo. Phase 3 will reach it only through `ascend/gate/adapter.py`, which will ship with a public stub so the repo runs without it. Do not add private code, records or results.
- **Workflow.** Work on a branch and open PRs into `main`; the maintainer merges. Stop and ask before changing a prereg, adding a README claim, or starting any run over 2 hours.

## Layout

| Path | Contents |
|---|---|
| `ascend/reference/shiu_model.py` | Verbatim upstream reference model. Never edit it; a test pins its hash. |
| `ascend/reference/brian2_ref.py` | Seeded, timed wrapper around the reference model, and the event-replay runner for exact checks. |
| `ascend/engine/` | Phase 1 engine: `cpu.py` (exact NumPy engine), `kernels.cu` (GPU kernels), `cuda.py` and `kernel_engine.py` (CUDA host code), `emulate.py` (kernels on the CPU for tests). |
| `ascend/` | Data fetching, stimuli, analysis, hardware capture, synthetic test networks. |
| `scripts/` | `fetch_data.py`, `bench.py`, `exact_check.py`, `fidelity_check.py`, `reference_rates.py`, `reproduce_sugar_mn9.py`, `map_stimuli.py`, `check_submission.py`, `baseline_check.py`, `make_demo.py`. |
| `data/` | Manifest and stimulus ID tables (committed). Fetched data (ignored). |
| `results/` | Small JSON results, each tied to a prereg. |
| `docs/` | Preregistrations and design notes. |

## Commands

```bash
pip install -e ".[test]"
pytest -q
python scripts/fetch_data.py
python scripts/bench.py --out bench.json
```
