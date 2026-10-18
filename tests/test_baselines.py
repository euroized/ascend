"""Matched null networks (docs/PREREG_BASELINES.md)."""

import numpy as np
import pytest

from ascend import baselines

N = 500


@pytest.fixture(scope="module")
def edges():
    rng = np.random.default_rng(7)
    m = 20_000
    pre, post = rng.integers(0, N, m), rng.integers(0, N, m)
    sign = np.where(rng.random(N) < 0.7, 1, -1)  # Dale's law: sign fixed by presynaptic neuron
    return pre, post, sign[pre] * rng.integers(1, 30, m)


def test_preregistered_constants():
    assert baselines.VARIANTS == ("real", "weights", "rewired", "random")
    assert baselines.SHUFFLE_SEEDS == (0, 1, 2) and baselines.SEEDS == list(range(10))
    assert (baselines.ON_HZ, baselines.OFF_HZ, baselines.RUNAWAY_SPIKES) == (10.0, 1.0, 1_000_000)


def test_real_is_unchanged(edges):
    out = baselines.null_arrays("real", *edges, N, 0)
    assert all(a is b for a, b in zip(out, edges, strict=True))


def test_weights_keeps_connections(edges):
    pre, post, signed = edges
    p, q, s = baselines.null_arrays("weights", *edges, N, 0)
    assert (p == pre).all() and (q == post).all()
    assert sorted(s) == sorted(signed) and not (s == signed).all()


def test_rewired_keeps_degrees_and_dale(edges):
    pre, post, signed = edges
    p, q, s = baselines.null_arrays("rewired", *edges, N, 0)
    assert (p == pre).all() and (s == signed).all()
    assert (np.bincount(q, minlength=N) == np.bincount(post, minlength=N)).all()
    assert (q != post).mean() > 0.9
    for i in range(N):  # every presynaptic neuron keeps a single sign
        assert len(set(np.sign(s[p == i]))) <= 1


def test_random_keeps_only_totals(edges):
    pre, post, signed = edges
    p, q, s = baselines.null_arrays("random", *edges, N, 0)
    assert len(p) == len(q) == len(pre) and sorted(s) == sorted(signed)
    assert p.min() >= 0 and p.max() < N and q.min() >= 0 and q.max() < N
    assert not (np.bincount(p, minlength=N) == np.bincount(pre, minlength=N)).all()


@pytest.mark.parametrize("variant", ["weights", "rewired", "random"])
def test_seeded(edges, variant):
    a = baselines.null_arrays(variant, *edges, N, 0)
    b = baselines.null_arrays(variant, *edges, N, 0)
    c = baselines.null_arrays(variant, *edges, N, 1)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))
    assert any((x != y).any() for x, y in zip(a, c, strict=True))


def test_network_builds(edges):
    net = baselines.network("rewired", 0, N, *edges, w_syn=0.275e-3)
    assert net.n_neurons == N and net.n_connections == len(edges[0])


def test_score():
    real = {"sugar": {"MN9": 70, "JON_DN": 0}, "bitter": {"MN9": 0, "JON_DN": 0}, "jon_all": {"MN9": 0, "JON_DN": 20}}
    assert baselines.score(real, {})["score"] == 6
    weak = {**real, "sugar": {"MN9": 5, "JON_DN": 0}}  # between 1 and 10 Hz: neither on nor off
    assert baselines.score(weak, {})["score"] == 5
    assert baselines.score(real, {"bitter": True})["score"] == 4


def test_chunked_trial_matches_single_run(tmp_path):
    from ascend import synthetic
    from ascend.engine import CPUEngine, Network, Params, PoissonInput

    p = Params.reference()
    comp, con = synthetic.write(tmp_path, n_neurons=300, n_connections=6000, seed=3)
    net = Network.from_files(comp, con, p.w_syn)
    eng = CPUEngine(net, p, range(10), dtype=np.float32)
    counts, runaway = baselines.run_trial(eng, 150.0, 4, 2_000)
    eng.reset()
    ref = np.bincount(eng.run(2_000, PoissonInput(150.0, 4)).neurons, minlength=net.n_neurons)
    assert not runaway and counts.sum() > 0 and (counts == ref).all()
