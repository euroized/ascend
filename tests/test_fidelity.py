"""The statistical fidelity criteria, on hand-made reference data."""

import numpy as np

from ascend import fidelity

IDS = np.arange(720575940600000000, 720575940600000000 + 6)


def reference(counts: np.ndarray, readout: int = 2) -> dict:
    rate = counts / fidelity.T_RUN_S
    m, s = rate.mean(0), rate.std(0)
    return {
        "seeds": list(range(len(counts))),
        "spikes_per_trial": counts.sum(1).tolist(),
        "rates_hz": {str(IDS[i]): [float(m[i]), float(s[i])] for i in np.flatnonzero(m > 0)},
        "readouts": {"r": {"flywire_id": int(IDS[readout]), "rate_hz_mean": float(m[readout])}},
    }


def trials(lam, seed):
    return np.random.default_rng(seed).poisson(lam, size=(30, len(lam)))


LAM = np.array([0.0, 5.0, 40.0, 90.0, 12.0, 150.0])


def test_same_process_passes():
    out = fidelity.compare(trials(LAM, 1), reference(trials(LAM, 2)), IDS)
    assert all(c["pass"] for c in out["checks"]), out


def test_shifted_rates_fail():
    out = fidelity.compare(trials(LAM * 1.3, 1), reference(trials(LAM, 2)), IDS)
    assert not all(c["pass"] for c in out["checks"])


def test_silent_reference_needs_silent_engine():
    silent = np.zeros((30, len(IDS)), dtype=int)
    assert fidelity.compare(silent, reference(silent), IDS)["checks"][0]["pass"]
    one = silent.copy()
    one[3, 1] = 1
    assert not fidelity.compare(one, reference(silent), IDS)["checks"][0]["pass"]


def test_thresholds_match_preregistration():
    """docs/PREREG_PHASE1.md fixes these; changing them needs a dated amendment."""
    assert (fidelity.Z, fidelity.FLOOR_HZ, fidelity.MIN_RATE_HZ) == (4.0, 0.5, 1.0)
    assert (fidelity.MAX_FAIL_FRAC, fidelity.MIN_PEARSON) == (0.01, 0.99)
    assert fidelity.SEEDS == list(range(30)) and fidelity.T_RUN_S == 1.0
    assert list(fidelity.CONDITIONS) == ["v783_sugar_150hz", "v783_bitter_150hz", "v783_jon_all_150hz", "v783_none"]
