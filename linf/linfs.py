"""
Linear INterpolation Functions.

theta refers to the full set of parameters for an adaptive linear interpolation model,
[n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
where N is the greatest allowed value of ceil(n).

The reason for the interleaving of x and y is it avoids the need to know N.
"""
import numpy as np

from linf.helper_functions import (
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)


def get_linf(x_min, x_max):
    """
    Returns a linf, with end nodes at x_min and x_max.

    x_min: float
    x_max: float > x_min

    Returns:
    linf(x, theta)

    theta in format [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.
    """

    def linf_function(x, theta):
        """
        Vectorised linf using n nodes.

        theta in format  [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.

        y0 and yn+1 are the y values corresponding to x_min and x_max respecively.
        """
        return np.interp(
            x,
            np.concatenate(([x_min], get_x_nodes_from_theta(theta), [x_max])),
            get_y_nodes_from_theta(theta),
        )

    return linf_function


def get_adaptive_linf(x_min, x_max):
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
    linf_function = get_linf(x_min, x_max)

    def adaptive_linf_function(x, theta):
        """
        Adaptive linf which allows the number of parameters being used to vary.

        The first element of theta is n; ceil(n) is number of interior nodes used in
        the linear interpolation model. This is then used to select the
        appropriate other elements of params to pass to linf()

        theta = [n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_N+1],
        where N is the greatest allowed value of ceil(n).
        """
        return linf_function(x, get_theta_n(theta))

    return adaptive_linf_function
