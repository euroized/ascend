"""Statistical fidelity tier: an engine's firing rates against the Phase 0 reference rates.

The engine runs each reference condition for 30 one-second trials (seeds 0-29) with its own
random stimulus, and its rates are compared with results/phase0/reference_rates.json. Criteria
and thresholds are fixed in docs/PREREG_PHASE1.md; this module implements them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np

from . import data, stimuli
from .engine import PoissonInput

REFERENCE = data.REPO_ROOT / "results" / "phase0" / "reference_rates.json"
CONDITIONS = {  # reference condition -> (stimulus set, Poisson rate)
    "v783_sugar_150hz": ("sugar", 150.0),
    "v783_bitter_150hz": ("bitter", 150.0),
    "v783_jon_all_150hz": ("jon_all", 150.0),
    "v783_none": ("none", 0.0),
}
SEEDS = list(range(30))
T_RUN_S = 1.0
Z = 4.0  # per-neuron tolerance: |difference| <= Z * SE + FLOOR_HZ
FLOOR_HZ = 0.5
MIN_RATE_HZ = 1.0  # neurons compared individually: mean >= 1 Hz in the reference or the engine
MAX_FAIL_FRAC = 0.01
MIN_PEARSON = 0.99


def run_condition(make_engine: Callable, stimulus: str, rate_hz: float, n_neurons: int, dt: float) -> dict:
    """Spike counts of every neuron in 30 trials. make_engine(stimulated_indices) -> engine."""
    eng = make_engine(stimulus)
    n_steps = int(round(T_RUN_S / dt))
    if hasattr(eng, "warmup"):
        eng.warmup(n_steps)
    counts = np.zeros((len(SEEDS), n_neurons), dtype=np.int64)
    wall = []
    for t, seed in enumerate(SEEDS):
        eng.reset()
        t0 = time.perf_counter()
        s = eng.run(n_steps, PoissonInput(rate_hz, seed) if rate_hz > 0 else None)
        wall.append(time.perf_counter() - t0)
        counts[t] = np.bincount(s.neurons, minlength=n_neurons)
    return {"counts": counts, "run_s": wall}


def compare(counts: np.ndarray, ref: dict, fly_ids: np.ndarray) -> dict:
    """Evaluate the preregistered criteria for one condition."""
    n_trials = counts.shape[0]
    rate = counts / T_RUN_S
    mean_e, sd_e = rate.mean(0), rate.std(0)
    mean_r, sd_r = np.zeros(len(fly_ids)), np.zeros(len(fly_ids))
    pos = {int(f): i for i, f in enumerate(fly_ids)}
    for f, (m, s) in ref["rates_hz"].items():
        mean_r[pos[int(f)]], sd_r[pos[int(f)]] = m, s
    se = np.sqrt(sd_e**2 / n_trials + sd_r**2 / len(ref["seeds"]))
    ok = np.abs(mean_e - mean_r) <= Z * se + FLOOR_HZ

    spikes_e, spikes_r = counts.sum(1), np.asarray(ref["spikes_per_trial"])
    out = {"spikes_per_trial_engine": float(spikes_e.mean()), "spikes_per_trial_reference": float(spikes_r.mean())}
    checks = []

    if spikes_r.max() == 0:
        checks.append(
            {
                "check": "F1 no stimulus: zero spikes in every trial",
                "value": int(spikes_e.max()),
                "pass": bool(spikes_e.max() == 0),
            }
        )
        return {**out, "checks": checks}

    for name, r in ref["readouts"].items():
        if r["rate_hz_mean"] < MIN_RATE_HZ:
            continue
        i = pos[r["flywire_id"]]
        checks.append(
            {
                "check": f"F2 readout {name}: {mean_e[i]:.2f} vs {mean_r[i]:.2f} Hz (tol {Z * se[i] + FLOOR_HZ:.2f})",
                "value": round(float(mean_e[i] - mean_r[i]), 3),
                "pass": bool(ok[i]),
            }
        )

    compared = (mean_r >= MIN_RATE_HZ) | (mean_e >= MIN_RATE_HZ)
    frac = float((~ok[compared]).mean()) if compared.any() else 0.0
    checks.append(
        {
            "check": f"F3 neurons outside tolerance: {int((~ok[compared]).sum())} of {int(compared.sum())} "
            f"(max {MAX_FAIL_FRAC:.0%})",
            "value": round(frac, 5),
            "pass": frac <= MAX_FAIL_FRAC,
        }
    )

    active = (mean_r > 0) | (mean_e > 0)
    r_p = float(np.corrcoef(mean_e[active], mean_r[active])[0, 1])
    checks.append(
        {
            "check": f"F4 Pearson r of rates over {int(active.sum())} active neurons >= {MIN_PEARSON}",
            "value": round(r_p, 6),
            "pass": r_p >= MIN_PEARSON,
        }
    )

    se_tot = np.sqrt(spikes_e.var() / n_trials + spikes_r.var() / len(spikes_r))
    d_tot = float(spikes_e.mean() - spikes_r.mean())
    checks.append(
        {
            "check": f"F5 total spikes per trial: {spikes_e.mean():.0f} vs {spikes_r.mean():.0f} "
            f"(tol {Z * se_tot + 0.01 * spikes_r.mean():.0f})",
            "value": round(d_tot, 1),
            "pass": abs(d_tot) <= Z * se_tot + 0.01 * spikes_r.mean(),
        }
    )
    return {**out, "checks": checks}


def suite(make_engine: Callable, n_neurons: int, dt: float, fly_ids: np.ndarray, log=print) -> dict:
    """Run and evaluate every condition. make_engine(stimulus_name) -> engine with that stimulus set."""
    ref = json.loads(REFERENCE.read_text())["conditions"]
    results = {}
    for cond, (stim, rate) in CONDITIONS.items():
        t0 = time.time()
        r = run_condition(make_engine, stim, rate, n_neurons, dt)
        results[cond] = {
            **compare(r["counts"], ref[cond], fly_ids),
            "run_s_median": round(float(np.median(r["run_s"])), 4),
        }
        status = "PASS" if all(c["pass"] for c in results[cond]["checks"]) else "FAIL"
        log(f"{cond}: {status} ({time.time() - t0:.0f}s)")
    passed = all(c["pass"] for r in results.values() for c in r["checks"])
    return {
        "prereg": "docs/PREREG_PHASE1.md",
        "seeds": SEEDS,
        "t_run_s": T_RUN_S,
        "pass": passed,
        "conditions": results,
    }


def stimulated(stimulus: str, path_comp) -> list[int]:
    idx = stimuli.id_to_index(path_comp)
    return sorted(idx[i] for i in stimuli.stimulus(stimulus))
