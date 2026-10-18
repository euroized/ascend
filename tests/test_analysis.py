import numpy as np

from ascend import analysis


def test_rates_match_upstream_definition():
    trials = [{0: np.array([0.1, 0.2]), 2: np.array([0.5])}, {0: np.array([0.3, 0.4, 0.6, 0.9])}]
    mean, std = analysis.rates(trials, n_neurons=3, t_run_s=0.5)
    np.testing.assert_allclose(mean, [6.0, 0.0, 1.0])
    np.testing.assert_allclose(std, [2.0, 0.0, 1.0])
