"""
Linear INterpolation Functions.

The tricky part here is going to be working out how the "wavelength" (and equivalently 0) should be provided.

theta refers to the full set of parameters for an adaptive linear interpolation model,
[n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
where N is the greatest value that n is allowed to take.
"""
import numpy as np


def linf(x, theta):
    """
    Vectorised linf using n nodes.

    params in format  [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.
    """
    n = len(theta) // 2 - 1
    return np.interp(
        x,
        np.concatenate(([0], theta[1 : 2 * n + 1 : 2], [wavelength])),
        np.concatenate((theta[0 : 2 * n + 2 : 2], theta[-1:])),
    )


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


def super_model(x, theta):
    """
    Super model which allows the number of parameters being used to vary.

    The first element of params is n, the number of interior nodes used in
    the linear interpolation model. This is then used to select the
    appropriate other elements of params to pass to f_end_nodes()

    params = [n, [θ1], [θ2], ..., [θN], y0, y_N+1], since the end points can
    be shared between the models with varying n (I think)
    """
    theta_n = get_theta_n(theta)
    return linf(x, theta_n)
