"""
Likelihoods using linfs.
"""

import numpy as np
from scipy.special import erf
from linf.helper_functions import (
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)
from linf.linfs import AdaptiveLinf, Linf, get_theta_n


class LinfLikelihood:
    """
    Likelihood for a linf, relative to data described by xs, ys, and sigma.

    sigma is either sigma_y, [sigma_x, sigma_y], [sigma_ys] or [[sigma_xs], [sigma_ys]].

    (obviously the middle two are degenerate when len(sigma) = 2, in which case
    [sigma_x, sigma_y] is assumed.)

    Returns likelihood(theta) -> log(L), [] where [] is the (lack of) derived parameters.
    """

    def __init__(self, x_min, x_max, xs, ys, sigma, adaptive=True):
        self._likelihood_function = create_likelihood_function(
            x_min, x_max, xs, ys, sigma, adaptive
        )

    def __call__(self, theta):
        """
        Likelihood relative to a linf with parameters theta.

        If using the adaptive linf, the first element of theta is n; ceil(n) is the number of
        interior nodes used to calculate the likelihood.

        theta = [n, y0, x1, y1, x2, y2, ..., x_n, y_n, y_n+1].

        Otherwise, theta is the same but without n.

        theta = [y0, x1, y1, x2, y2, ..., x_n, y_n, y_n+1].
        """
        return self._likelihood_function(theta)


def create_likelihood_function(x_min, x_max, xs, ys, sigma, adaptive=True):
    """
    Creates a likelihood function for a linf, relative to data descrived by xs, ys, and sigma.

    sigma is either sigma_y, [sigma_x, sigma_y], [sigma_ys] or [[sigma_xs, sigma_ys]].

    (obviously the middle two are degenerate when len(sigma) = 2, in which case
    [sigma_x, sigma_y] is assumed.)
    """
    LOG_2_SQRT_2πλ = np.log(2) + 0.5 * np.log(2 * np.pi * (x_max - x_min))

    # check for sigma_x
    has_sigma_x = False
    if hasattr(sigma, "__len__"):
        if len(sigma.shape) == 2:
            has_sigma_x = True
        if 2 == len(sigma):
            has_sigma_x = True

    if has_sigma_x:
        # sigma balls
        sigma_x = sigma[0]
        sigma_y = sigma[1]
        var_x = sigma_x**2
        var_y = sigma_y**2

        def xy_errors_likelihood(theta):
            x_nodes = np.concatenate(
                ([x_min], get_x_nodes_from_theta(theta, adaptive), [x_max])
            )

            y_nodes = get_y_nodes_from_theta(theta, adaptive)

            ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
            cs = y_nodes[:-1] - ms * x_nodes[:-1]

            # save recalculating things
            # indices in order [data point, relevant m and c]
            q = (np.outer(var_x, ms**2).T + var_y).T
            delta = np.subtract.outer(ys, cs)
            beta = (xs * var_x + (delta * ms).T * var_y).T / q
            gamma = (np.outer(xs, ms) - delta) ** 2 / 2 / q

            t_minus = (np.sqrt(q / 2).T / (sigma_x * sigma_y)).T * (x_nodes[:-1] - beta)
            t_plus = (np.sqrt(q / 2).T / (sigma_x * sigma_y)).T * (x_nodes[1:] - beta)

            logL = -len(xs) * LOG_2_SQRT_2πλ
            logL += np.sum(
                np.log(
                    np.sum(
                        np.exp(-gamma) * q**-0.5 * (erf(t_plus) - erf(t_minus)),
                        axis=-1,
                    )
                )
            )

            return logL, []

        # if adaptive:

        #     def super_likelihood(theta):

        #         theta_n = get_theta_n(theta)
        #         return xy_errors_likelihood(theta_n)

        #     return super_likelihood

        return xy_errors_likelihood

    # sigma_y only

    var_y = sigma**2

    if adaptive:
        linf = AdaptiveLinf(x_min, x_max)
    else:
        linf = Linf(x_min, x_max)

    def y_errors_likelihood(theta):
        if hasattr(var_y, "__len__"):
            logL = -0.5 * np.sum(np.log(2 * np.pi * var_y))
        else:
            logL = -0.5 * len(ys) * np.log(2 * np.pi * var_y)

        logL += np.sum(-((ys - linf(xs, theta)) ** 2) / 2 / var_y)
        return logL, []

    return y_errors_likelihood
