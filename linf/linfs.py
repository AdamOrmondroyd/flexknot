"""
Linear INterpolation Functions.

theta refers to the full set of parameters for an adaptive linear interpolation model,
[n, y0, x1, y1, x2, y2, ..., x_n, y_n, y_n+1],
where n is the greatest allowed value of ceil(n).

The reason for the interleaving of x and y is it avoids the need to know n.
"""
import numpy as np

from linf.helper_functions import (
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)


class Linf:
    """
    linf with end nodes at x_min and x_max.

    x_min: float
    x_max: float > x_min

    Returns:
    linf(x, theta)

    theta in format [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.
    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x, theta):
        """
        theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.

        y0 and yn+1 are the y values corresponding to x_min and x_max respecively.
        """
        return np.interp(
            x,
            np.concatenate(([self.x_min], get_x_nodes_from_theta(theta), [self.x_max])),
            get_y_nodes_from_theta(theta),
        )


class AdaptiveLinf(Linf):
    """
    Adaptive linf which allows the number of parameters being used to vary.

    x_min: float
    x_max: float > x_min

    Returns:
    adaptive_linf(x, theta)

    The first element of theta is n; ceil(n) is number of interior nodes used in
    the linear interpolation model.

    theta = [n, y0, x1, y1, x2, y2, ..., x_n, y_n, y_n+1],
    where n is the greatest allowed value of ceil(n).
    """

    def __call__(self, x, theta):
        """
        The first element of theta is n; ceil(n) is number of interior nodes used in
        the linear interpolation model. This is then used to select the
        appropriate other elements of params to pass to linf()

        theta = [n, y0, x1, y1, x2, y2, ..., x_n, y_n, y_n+1],
        where n is the greatest allowed value of ceil(n).
        """
        return super().__call__(x, get_theta_n(theta))
