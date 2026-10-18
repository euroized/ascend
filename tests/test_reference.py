"""The reference wrapper on a tiny synthetic network (Brian2 numpy target, no compiler needed)."""

import numpy as np
import pytest

from ascend.reference import brian2_ref

T = 0.1
EXC = list(range(10))


def same(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def session(tiny_connectome):
    s = brian2_ref.Session(*tiny_connectome, EXC, t_run_s=T, target="numpy")
    s.warmup()
    return s


def test_same_seed_same_spikes(session):
    assert same(session.trial(5).spikes, session.trial(5).spikes)


def test_different_seed_different_spikes(session):
    assert not same(session.trial(5).spikes, session.trial(6).spikes)


def test_session_matches_fresh_build(session, tiny_connectome):
    fresh = brian2_ref.run_trial(*tiny_connectome, EXC, seed=7, t_run_s=T, target="numpy")
    assert same(session.trial(7).spikes, fresh.spikes)


def test_no_stimulus_is_silent(tiny_connectome):
    s = brian2_ref.Session(*tiny_connectome, [], t_run_s=T, target="numpy")
    assert s.trial(0).n_spikes == 0


def test_stimulated_neurons_follow_poisson_rate(session):
    # Each Poisson event (250 x 0.275 mV) crosses threshold on its own, so a stimulated
    # neuron fires at roughly the 150 Hz input rate: 15 spikes expected in 0.1 s.
    t = session.trial(0)
    n = [len(t.spikes.get(i, [])) for i in EXC]
    assert 5 <= np.mean(n) <= 25
    assert t.rtf > 0
