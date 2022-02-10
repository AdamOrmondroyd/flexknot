"""
Check that adaptive_linf() does the same as linf().
"""

import numpy as np
from linf import get_linf, get_adaptive_linf


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
        get_linf(x_min, x_max)(xs, theta_n)
        == get_adaptive_linf(x_min, x_max)(xs, theta)
    )
