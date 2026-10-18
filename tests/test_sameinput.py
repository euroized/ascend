"""Same-input tier G (Amendment 1 to docs/PREREG_PHASE1.md)."""

import dataclasses

import numpy as np
import pytest

from ascend import sameinput
from ascend.engine import CPUEngine, Network, Params, PoissonInput, Spikes
from ascend.engine import emulate as emu

P = Params.reference()


def test_thresholds_match_amendment():
    assert (sameinput.KICK_SEED, sameinput.RATE_HZ, sameinput.T_RUN_S) == (1, 150.0, 1.0)
    assert sameinput.WORKLOADS == ("sugar", "bitter", "jon_all") and sameinput.MIN_JACCARD == 0.99


def test_jaccard():
    a = Spikes(np.array([1, 2, 3]), np.array([5, 6, 7]), P.dt)
    b = Spikes(np.array([1, 2, 4]), np.array([5, 6, 7]), P.dt)
    assert sameinput.jaccard(a, a) == 1.0 and sameinput.jaccard(a, b) == pytest.approx(0.5)
    empty = Spikes(np.array([], int), np.array([], int), P.dt)
    assert sameinput.jaccard(empty, empty) == 1.0


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    from ascend import synthetic

    return Network.from_files(*synthetic.write(tmp_path_factory.mktemp("g"), 500, 40000, seed=3), P.w_syn)


def run(engine, n=2000):
    return engine.run(n, PoissonInput(150, 1))


def test_shuffled_summation_order_keeps_float32_spikes(net):
    stim = list(range(0, 500, 20))
    base = run(CPUEngine(net, P, stim, dtype=np.float32))
    shuffled = run(CPUEngine(net, P, stim, dtype=np.float32, shuffle_seed=3))
    assert sameinput.jaccard(base, shuffled) >= sameinput.MIN_JACCARD


def test_delay_error_breaks_same_input_agreement(net):
    stim = list(range(0, 500, 20))
    base = run(CPUEngine(net, P, stim, dtype=np.float32))
    wrong = run(CPUEngine(net, dataclasses.replace(P, t_dly=17 * P.dt), stim, dtype=np.float32))
    assert sameinput.jaccard(base, wrong) < sameinput.MIN_JACCARD


@pytest.mark.skipif(emu.compiler() is None, reason="no C++ compiler")
def test_compare_passes_for_emulated_kernels(net, monkeypatch):
    monkeypatch.setattr(sameinput, "T_RUN_S", 0.05)
    res = sameinput.compare(
        lambda st: emu.EmulatedEngine(net, P, st, dtype=np.float32),
        net,
        P,
        lambda w: list(range(0, 500, 25)),
        log=lambda *_: None,
    )
    assert res["pass"] and set(res["workloads"]) == set(sameinput.WORKLOADS)
