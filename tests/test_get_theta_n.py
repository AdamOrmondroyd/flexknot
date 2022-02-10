import numpy as np
from linf.helper_functions import get_theta_n


def test_get_theta_n():
    """
    Check that get_theta_n() extracts the correct elements from [3, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6].
    """

    theta = np.array([3, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    theta_n = get_theta_n(theta)
    assert np.all(np.array([0, 1, 1, 2, 2, 3, 3, 6]) == theta_n)
