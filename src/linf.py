"""
Linear INterpolation Functions.

The tricky part here is going to be working out how the "wavelength" (and equivalently 0) should be provided.

theta refers to the full set of parameters for an adaptive linear interpolation model,
[n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
where N is the greatest allowed value of ceil(n).
"""
import numpy as np

wavelength = (
    1.0  # temporary while I think about how I'm going to deal with the boundaries.
)


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


def adaptive_linf(x, theta):
    """
    Adaptive linf which allows the number of parameters being used to vary.

    The first element of params is n; ceil(n) is number of interior nodes used in
    the linear interpolation model. This is then used to select the
    appropriate other elements of params to pass to linf()

    theta = [n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
    where N is the greatest allowed value of ceil(n).
    """
    theta_n = get_theta_n(theta)
    return linf(x, theta_n)
