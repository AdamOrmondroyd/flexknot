"""
Test that LinfPrior sorts the x nodes.
"""

import numpy as np
from linf import LinfPrior
from linf.helper_functions import get_x_nodes_from_theta

rng = np.random.default_rng()
x_min = 0
x_max = 1
y_min = -2
y_max = 2


def test_linfprior_y_nodes_are_sorted():
    """
    Test that the prior for the y_nodes is sorted.
    """
    n = 10
    hypercube = rng.random(2 * n + 2)
    prior = LinfPrior(x_min, x_max, y_min, y_max)(hypercube)

    assert np.all(np.diff(get_x_nodes_from_theta(prior)) >= 0)
