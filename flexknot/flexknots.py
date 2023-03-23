"""
Linear INterpolation Functions.

theta refers to the full set of parameters for an adaptive linear interpolation model,
[n, y0, x1, y1, x2, y2, ..., x_n, y_n, y_n+1],
where n is the greatest allowed value of ceil(n).

The reason for the interleaving of x and y is it avoids the need to know n.
"""
import numpy as np

from flexknot.helper_functions import (
    intersection,
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)


class FlexKnot:
    """
    Flex-knot with end nodes at x_min and x_max.

    x_min: float
    x_max: float > x_min

    Returns:
    flexknot(x, theta)

    theta in format [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)] for N nodes.
    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x, theta):
        """
        Flex-knot with end nodes at x_min and x_max

        theta = [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)] for N nodes.

        y0 and y_(N-1) are the y values corresponding to x_min and x_max respecively.

        If theta only contains a single element, the flex-knot is constant at that value.
        If theta is empty, the flex-knot if constant at -1 (cosmology!)
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

    def intersections(self, theta0, theta1):
        """
        Find the x-coordinates of intersections between
        the flex-knot with parameters theta0 and theta1.
        """
        if len(theta0) == 0:
            theta0 = np.array([-1, -1])
        elif len(theta0) == 1:
            theta0 = np.full(2, theta0[0])
        if len(theta1) == 0:
            theta1 = np.array([-1, -1])
        elif len(theta1) == 1:
            theta1 = np.full(2, theta1[0])
        theta0 = np.concatenate(([self.x_min], theta0[:-1],
                                 [self.x_max], theta0[[-1]]))
        theta1 = np.concatenate(([self.x_min], theta1[:-1],
                                 [self.x_max], theta1[[-1]]))
        intersections = []
        for ii in range(0, len(theta1) - 2, 2):
            for i in range(0, len(theta0) - 2, 2):
                x = intersection(theta0[i:i+2], theta0[i+2:i+4],
                                 theta1[ii:ii+2], theta1[ii+2:ii+4])[0]
                if (theta0[i] < x < theta0[i+2] and
                   theta1[ii] < x < theta1[ii+2]):
                    intersections.append(x)
        intersections = np.array(intersections)
        if intersections.size > 0:
            intersections = np.sort(intersections)
        return intersections

    def area(self, theta0, theta1):
        """
        Calculate the area between the flex-knot with parameters
        theta_0 and theta_1.
        """
        print(theta0)
        x = self.intersections(theta0, theta1)
        print(theta0)
        if x.size > 0:
            print(f"x: {x}")
            print(self(x, theta0) - self(x, theta1))
            # assert np.all(np.isclose(self(x, theta0), self(x, theta1)))

        x = np.concatenate((x, [self.x_min, self.x_max],
                            get_x_nodes_from_theta(theta0, adaptive=False),
                            get_x_nodes_from_theta(theta1, adaptive=False)))
        x = np.sort(x)
        y = np.abs(self(x, theta0) - self(x, theta1))
        return np.trapz(y, x=x)


class AdaptiveKnot(FlexKnot):
    """
    Adaptive flex-knot which allows the number of parameters being used to vary.

    x_min: float
    x_max: float > x_min

    Returns:
    adaptive_flexknot(x, theta)

    The first element of theta is N; floor(N)-2 is number of interior nodes used in
    the linear interpolation model.

    theta = [N, y0, x1, y1, x2, y2, ..., x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
    where Nmax is the greatest allowed value of floor(N).

    if floor(N) = 1, the flex-knot is constant at theta[-1] = y_(Nmax-1).
    if floor(N) = 0, the flex-knot is constant at -1 (cosmology!)
    """

    def __call__(self, x, theta):
        """
        The first element of theta is N; floor(N)-2 is number of interior nodes used in
        the linear interpolation model. This is then used to select the
        appropriate other elements of params to pass to flexknot()

        theta = [N, y0, x1, y1, x2, y2, ..., x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
        where Nmax is the greatest allowed value of floor(N).

        if floor(N) = 1, the flex-knot is constant at theta[-1] = y_(Nmax-1).
        if floor(N) = 0, the flex-knot is constant at -1 (cosmology!)
        """
        return super().__call__(x, get_theta_n(theta))

    def intersections(self, theta0, theta1):
        return super().intersections(get_theta_n(theta0), get_theta_n(theta1))

    def area(self, theta0, theta1):
        """
        Calculate the area between the flex-knot with parameters
        theta_0 and theta_1.
        """
        print(theta0)
        x = self.intersections(theta0, theta1)
        print(theta0)
        if x.size > 0:
            print(f"x: {x}")
            print(self(x, theta0) - self(x, theta1))
            # assert np.all(np.isclose(self(x, theta0), self(x, theta1)))

        x = np.concatenate((x, [self.x_min, self.x_max],
                            get_x_nodes_from_theta(theta0, adaptive=True),
                            get_x_nodes_from_theta(theta1, adaptive=True)))
        x = np.sort(x)
        y = np.abs(self(x, theta0) - self(x, theta1))
        return np.trapz(y, x=x)
