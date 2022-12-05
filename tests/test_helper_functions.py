"""
Test each of the functions from helper_functions.py
"""

import numpy as np
from flexknot.helper_functions import (
    create_theta,
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)

x_nodes = np.array([0.25, 0.75])
y_nodes = np.array([0, 1, -1, 2])
theta = np.array([0, 0.25, 1, 0.75, -1, 2])


def test_get_theta_n():
    """
    Check that get_theta_n() extracts the correct elements from [3, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6].
    """

    theta = np.array([5, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    theta_n = get_theta_n(theta)
    assert np.all(np.array([0, 1, 1, 2, 2, 3, 3, 6]) == theta_n)


def test_get_theta_one():
    theta = np.array([1.5, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    theta_n = get_theta_n(theta)
    assert theta_n == 6


def test_get_theta_zero():
    theta = np.array([0.5, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    theta_n = get_theta_n(theta)
    assert 0 == len(theta_n)


def test_create_theta():
    """
    Test that create_theta() combines x_nodes and y_nodes correctly.
    """
    assert np.all(create_theta(x_nodes, y_nodes) == theta)


def test_get_x_nodes_from_theta():
    """
    Test that get_x_nodes_from_theta() extracts the x_nodes correcly.
    """
    assert np.all(x_nodes == get_x_nodes_from_theta(theta, adaptive=False))


def test_get_y_nodes_from_theta():
    """
    Test that get_y_nodes_from_theta() extracts the y_nodes correctly.
    """
    assert np.all(y_nodes == get_y_nodes_from_theta(theta, adaptive=False))
