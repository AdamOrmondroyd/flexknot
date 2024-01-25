"""Utilities for dealing with the interleaved arrangement of theta."""

import numpy as np


def validate_theta(theta, adaptive):
    """Check theta for mistakes.

    theta must contain an odd/even number of elements for an adaptive/
    non-adaptive flex-knot,

    In the adaptive case, floor(theta[0])-2 must not be greater than the
    provided number of internal nodes.

    Parameters
    ----------
    theta : array-like

    adaptive : bool

    """
    if adaptive:
        if len(theta) % 2 != 1:
            raise ValueError(
                "theta must contain an odd number of elements "
                "for an adaptive flex-knot."
            )
        if np.floor(theta[0]) > (len(theta) + 1) / 2:
            raise ValueError("n = ceil(theta[0]) exceeds the "
                             "number of internal nodes.")

    if not adaptive:
        if len(theta) > 1 and len(theta) % 2 != 0:
            raise ValueError("theta must contain an even number of elements "
                             "for a non-adaptive flex-knot.")


def get_theta_n(theta):
    """Extract the first N parameters from theta.

    Parameters
    ----------
    theta : array-like
    [N, y0, x1, y1, x2, y2, ..., x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
    where Nmax is the maximum value of floor(N).

    Returns
    -------
    theta_n : array-like
    [y0, x1, y1, x2, y2, ..., x_floor(N)-2, y_floor(N)-2, y_(Nmax-1)]

    """
    validate_theta(theta, adaptive=True)

    n = np.floor(theta[0]).astype(int) - 2
    # w = -1 case, return empty
    if -2 == n:
        return np.array([])
    theta_n = np.concatenate(
        (
            theta[1:2*n+2],  # y0 and internal x and y
            theta[-1:],  # y end node
        )
    )
    return theta_n


def create_theta(x_nodes, y_nodes):
    """Interleave x and y nodes to create theta.

    Parameters
    ----------
    x_nodes : array-like
    [x1, ... x_(N-2)]

    y_nodes : array-like
    [y0, y1, ..., y_(N-2), y_(N-1)]

    Returns
    -------
    theta : array-like
    [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)]

    """
    if len(y_nodes) == 1:
        return y_nodes
    if len(x_nodes) + 2 != len(y_nodes):
        raise ValueError("y_nodes must have exactly two "
                         "more elements than x_nodes")
    n = len(x_nodes)
    theta = np.zeros(len(x_nodes) + len(y_nodes))
    theta[1:2*n+1:2] = x_nodes
    theta[0:2*n+2:2] = y_nodes[:-1]
    theta[-1] = y_nodes[-1]
    return theta


def get_x_nodes_from_theta(theta, adaptive):
    """Get the x nodes from theta.

    Parameters
    ----------
    theta : array-like
    [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)]

    adaptive : bool
    Whether theta is for an adaptive or non-adaptive flex-knot.

    Returns
    -------
    x_nodes : array-like
    [x1, ... x_(N-2)]

    """
    validate_theta(theta, adaptive)
    if adaptive:
        theta = get_theta_n(theta)
    n = len(theta) // 2 - 1
    return theta[1:2*n+1:2]


def get_y_nodes_from_theta(theta, adaptive):
    """Get the y nodes from theta.

    Parameters
    ----------
    theta : array-like
    [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)]

    adaptive : bool
    Whether theta is for an adaptive or non-adaptive flex-knot.

    Returns
    -------
    y_nodes : array-like
    [y0, y1, ..., y_(N-2), y_(N-1)]

    """
    validate_theta(theta, adaptive)
    if adaptive:
        theta = get_theta_n(theta)
    n = len(theta) // 2 - 1
    return np.concatenate((theta[0:2*n+2:2], theta[-1:]))
