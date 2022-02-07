import numpy as np
from linf.linfs import create_theta


def test_create_theta():
    x_nodes = np.array([0.25, 0.75])
    y_nodes = np.array([0, 1, -1, 2])
    theta = np.array([0, 0.25, 1, 0.75, -1, 2])

    assert np.all(create_theta(x_nodes, y_nodes) == theta)
