import numpy as np
from src.linf import linf, adaptive_linf, wavelength


def test_adaptive_linf():
    theta = np.array([2.5, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    theta_n = np.array([0, 1, 1, 2, 2, 3, 3, 6])
    xs = np.linspace(0, wavelength)
    assert np.all(linf(xs, theta_n) == adaptive_linf(xs, theta))
