"""
Helpful functions for dealing with the interleaved arrangement of theta.
"""

import numpy as np


def validate_theta(theta, adaptive=True):
    """
    Check that theta contains an odd/even number of elements for an adaptive/
    non-adaptive linf, and in the adaptive case that floor(theta[0])-2 isn't
    greater than the provided number of internal nodes.
    """
    if adaptive:
        if len(theta) % 2 != 1:
            raise ValueError(
                "theta must contain an odd number of elements for an adaptive linf."
            )
        if np.floor(theta[0]) > (len(theta) + 1) / 2:
            raise ValueError("n = ceil(theta[0]) exceeds the number of internal nodes.")

    if not adaptive:
        if len(theta) > 1 and len(theta) % 2 != 0:
            raise ValueError(
                "theta must contain an even number of elements for a non-adaptive linf."
            )


def get_theta_n(theta):
    """
    Extracts the first n parameters from

    theta = [N, y0, x1, y1, x2, y2, ..., x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)]

    where Nmax is the maximum value of floor(N).

    returns theta_n = [y0, x1, y1, x2, y2, ..., x_floor(N)-2, y_floor(N)-2, y_(Nmax-1)]
    """
    validate_theta(theta, adaptive=True)

    n = np.floor(theta[0]).astype(int) - 2
    # w = -1 case, return empty
    if -2 == n:
        return np.array([])
    theta_n = np.concatenate(
        (
            theta[1 : 2 * n + 2],  # y0 and internal x and y
            theta[-1:],  # y end node
        )
    )
    return theta_n


def create_theta(x_nodes, y_nodes):
    """
    Takes x_nodes = [x1, ... x_n] and y_nodes = [y0, y1, ..., yn, y(n+1)] to
    return theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1], where n is the number
    of internal nodes.
    """
    if len(x_nodes) + 2 != len(y_nodes):
        raise ValueError("y_nodes must have exactly two more elements than x_nodes")
    n = len(x_nodes)
    theta = np.zeros(len(x_nodes) + len(y_nodes))
    theta[1 : 2 * n + 1 : 2] = x_nodes
    theta[0 : 2 * n + 2 : 2] = y_nodes[:-1]
    theta[-1] = y_nodes[-1]
    return theta


def get_x_nodes_from_theta(theta, adaptive=False):
    """
    Takes theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] to return
    x_nodes = [x1, x2, ..., xn], where n is the number of internal nodes.
    """
    validate_theta(theta, adaptive)
    if adaptive:
        theta = theta[1:]
    n = len(theta) // 2 - 1
    return theta[1 : 2 * n + 1 : 2]


def get_y_nodes_from_theta(theta, adaptive=False):
    """
    Takes theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] to return
    y_nodes = [x0, x1, ..., yn, yn+1], where n is the number of internal nodes.
    """
    validate_theta(theta, adaptive)
    if adaptive:
        theta = theta[1:]
    n = len(theta) // 2 - 1
    return np.concatenate((theta[0 : 2 * n + 2 : 2], theta[-1:]))
