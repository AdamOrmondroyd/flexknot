import numpy as np
from linf.helper_functions import (
    create_theta,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)

x_nodes = np.array([0.25, 0.75])
y_nodes = np.array([0, 1, -1, 2])
theta = np.array([0, 0.25, 1, 0.75, -1, 2])


def test_create_theta():
    assert np.all(create_theta(x_nodes, y_nodes) == theta)


def test_get_x_nodes_from_theta():
    assert np.all(x_nodes == get_x_nodes_from_theta(theta))


def test_get_y_nodes_from_theta():
    assert np.all(y_nodes == get_y_nodes_from_theta(theta))
