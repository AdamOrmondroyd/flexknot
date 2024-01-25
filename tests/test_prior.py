"""
Test that FlexKnotPrior sorts the x nodes.
"""

import numpy as np
from flexknot import AdaptiveKnotPrior, FlexKnotPrior
from flexknot.utils import get_x_nodes_from_theta

rng = np.random.default_rng()
x_min = 0
x_max = 1
y_min = -2
y_max = 2
N_min = 1
N_max = 10


def test_flexknotprior_x_nodes_are_sorted():
    """
    Test that the prior for the x_nodes is sorted.
    """
    hypercube = rng.random(2 * N_max - 2)
    prior = FlexKnotPrior(x_min, x_max, y_min, y_max)(hypercube)

    assert np.all(np.diff(get_x_nodes_from_theta(prior, adaptive=False)) >= 0)


def test_adaptiveknotprior_x_nodes_are_sorted():
    """
    Test that the first N-2 x nodes of the adaptive prior are sorted.
    """

    hypercube = rng.random(2 * N_max - 1)
    prior = AdaptiveKnotPrior(
        x_min, x_max, y_min, y_max, N_min, N_max
    )(hypercube)

    assert np.all(np.diff(get_x_nodes_from_theta(prior, adaptive=True)) >= 0)
