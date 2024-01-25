"""
Flex-knot.

x and y are interleaved so that N does not need to be provided for
a non-adaptive flex-knot.
"""

import numpy as np
from scipy.integrate import quad

from flexknot.utils import (
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)


class FlexKnot:
    """
    Flex-knot with end nodes at x_min and x_max.

    x_min: float
    x_max: float > x_min

    Returns
    -------
    flexknot(x, theta): callable

    theta has format
    [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)]
    for N nodes.

    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x, theta):
        """
        Flex-knot with end nodes at x_min and x_max.

        For N nodes:
        theta = [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)].

        y0 and y_(N-1) are the y values at x_min and x_max respecively.

        If theta only contains a single element, the flex-knot is constant.
        If theta is empty, the flex-knot is constant at -1 (cosmology!)

        Parameters
        ----------
        x : float or array-like

        theta : array-like

        Returns
        -------
        float or array-like

        """
        if 0 == len(theta):
            return np.full_like(x, -1)
        if 1 == len(theta):
            return np.full_like(x, theta[-1])
        return np.interp(
            x,
            np.concatenate(
                (
                    [self.x_min],
                    get_x_nodes_from_theta(theta, adaptive=False),
                    [self.x_max],
                )
            ),
            get_y_nodes_from_theta(theta, adaptive=False),
        )

    def area(self, theta0, theta1):
        """
        Area between two flex-knots.

        Parameters
        ----------
        theta0 : array-like

        theta1 : array-like
        """
        return quad(lambda x: np.abs(self(x, theta0)-self(x, theta1)),
                    self.x_min, self.x_max)[0] / (self.x_max - self.x_min)


class AdaptiveKnot(FlexKnot):
    """
    Adaptive flex-knot which allows the number of knots to vary.

    x_min: float
    x_max: float > x_min

    Returns
    -------
    adaptive_flexknot(x, theta): callable

    The first element of theta is N; floor(N)-2 is number of interior nodes
    used by the flexknot.

    theta = [N, y0, x1, y1, x2, y2, ..., x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
    where Nmax is the greatest allowed value of floor(N).

    if floor(N) = 1, the flex-knot is constant at theta[-1] = y_(Nmax-1).
    if floor(N) = 0, the flex-knot is constant at -1 (cosmology!)
    """

    def __call__(self, x, theta):
        """
        Adaptive flex-knot with end nodes at x_min and x_max.

        The first element of theta is N; floor(N)-2 is number of interior nodes
        used in the flexknot. This is then used to select the
        appropriate other elements of params to pass to flexknot()

        For a maximum of Nmax nodes:
        theta = [N, y0, x1, y1, x2, y2, ...,
                 x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
        where Nmax is the greatest allowed value of floor(N).

        if floor(N) = 1, the flex-knot is constant at theta[-1] = y_(Nmax-1).
        if floor(N) = 0, the flex-knot is constant at -1 (cosmology!)

        Parameters
        ----------
        x : float or array-like

        theta : array-like

        Returns
        -------
        float or array-like

        """
        return super().__call__(x, get_theta_n(theta))
