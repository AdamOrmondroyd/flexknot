"""
Linear INterpolation Functions.

The tricky part here is going to be working out how the "wavelength" (and equivalently 0) should be provided.

theta refers to the full set of parameters for an adaptive linear interpolation model,
[n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
where N is the greatest allowed value of ceil(n).
"""
import numpy as np


def linf(x_min, x_max):
    """
    Returns a linf, with end nodes at x_min and x_max.

    x_min: float
    x_max: float > x_min

    Returns:
    linf(x, theta)

    theta in format  [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.
    """

    def linf_function(x, theta):
        """
        Vectorised linf using n nodes.

        theta in format  [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.

        y0 and yn+1 are the y values corresponding to x_min and x_max respecively.
        """
        n = len(theta) // 2 - 1
        return np.interp(
            x,
            np.concatenate(([x_min], theta[1 : 2 * n + 1 : 2], [x_max])),
            np.concatenate((theta[0 : 2 * n + 2 : 2], theta[-1:])),
        )

    return linf_function


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


def adaptive_linf(x_min, x_max):
    """
    Adaptive linf which allows the number of parameters being used to vary.

    x_min: float
    x_max: float > x_min

    Returns:
    adaptive_linf(x, theta)

    The first element of theta is n; ceil(n) is number of interior nodes used in
    the linear interpolation model.

    theta = [n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
    where N is the greatest allowed value of ceil(n).
    """
    linf_function = linf(x_min, x_max)

    def adaptive_linf_function(x, theta):
        """
        Adaptive linf which allows the number of parameters being used to vary.

        The first element of theta is n; ceil(n) is number of interior nodes used in
        the linear interpolation model. This is then used to select the
        appropriate other elements of params to pass to linf()

        theta = [n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
        where N is the greatest allowed value of ceil(n).
        """
        theta_n = get_theta_n(theta)
        return linf_function(x, get_theta_n(theta))

    return adaptive_linf_function