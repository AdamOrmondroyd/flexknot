"""
Test that linf returns the same results as np.interp, and that
adaptive_linf() does the same as linf().
"""

import numpy as np
from linf import AdaptiveLinf, Linf


def test_linf():
    """
    Test that linf gives the same results as np.interp.
    """
    x_min = 0
    x_max = 1
    n = 8
    rng = np.random.default_rng()
    x_nodes = np.sort(rng.uniform(x_min, x_max, n))
    y_nodes = rng.uniform(-10, 10, n + 2)
    theta = np.zeros(len(x_nodes) + len(y_nodes))
    theta[1 : 2 * n + 1 : 2] = x_nodes
    theta[0 : 2 * n + 2 : 2] = y_nodes[:-1]
    theta[-1] = y_nodes[-1]
    xs = np.linspace(x_min, x_max, 100)
    assert np.all(
        np.interp(xs, np.concatenate(([x_min], x_nodes, [x_max])), y_nodes)
        == Linf(x_min, x_max)(xs, theta)
    )


def test_adaptive_linf():
    """
    Test that adaptive_linf returns the same results as linf with appropriate arguments.
    """
    x_min = 0.0
    x_max = 6.0
    theta = np.array([2.5, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    theta_n = np.array([0, 1, 1, 2, 2, 3, 3, 6])
    xs = np.linspace(x_min, x_max, 100)
    assert np.all(
        Linf(x_min, x_max)(xs, theta_n) == AdaptiveLinf(x_min, x_max)(xs, theta)
    )
