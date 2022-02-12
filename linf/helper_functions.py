import numpy as np


def get_theta_n(theta):
    """
    Extracts the first n parameters from

    theta = [n, y0, x1, y1, x2, y2, ..., x_nmax, y_nmax, y_nmax+1]

    where nmax is the maximum value of ceil(n).

    returns theta_n = [y0, x1, y1, x2, y2, ..., x_ceil(n), y_ceil(n), y_nmax+1]
    """
    n = np.ceil(theta[0]).astype(int)
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
    return theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1]
    """
    n = len(x_nodes)
    theta = np.zeros(len(x_nodes) + len(y_nodes))
    theta[1 : 2 * n + 1 : 2] = x_nodes
    theta[0 : 2 * n + 2 : 2] = y_nodes[:-1]
    theta[-1] = y_nodes[-1]
    return theta


def get_x_nodes_from_theta(theta, adaptive=False):
    """
    Takes theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] to return
    x_nodes = [x1, x2, ..., xn]
    """
    if adaptive:
        theta = theta[1:]
    N = len(theta) // 2 - 1
    return theta[1 : 2 * N + 1 : 2]


def get_y_nodes_from_theta(theta, adaptive=False):
    """
    Takes theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] to return
    y_nodes = [x0, x1, ..., yn, yn+1]
    """
    if adaptive:
        theta = theta[1:]
    N = len(theta) // 2 - 1
    return np.concatenate((theta[0 : 2 * N + 2 : 2], theta[-1:]))
