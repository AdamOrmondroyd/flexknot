"""
Likelihoods using flex-knots.
"""

import numpy as np
from scipy.special import erf, logsumexp
from flexknot.utils import get_x_nodes_from_theta
from flexknot.core import AdaptiveKnot, FlexKnot


class FlexKnotLikelihood:
    """
    Likelihood for a flex-knot, with respect to data
    described by xs, ys, and sigma.

    sigma is either sigma_y, [sigma_x, sigma_y], [sigma_ys]
    or [[sigma_xs], [sigma_ys]].

    (obviously the middle two are degenerate when len(sigma) = 2, in which case
    [sigma_x, sigma_y] is assumed.)

    Returns likelihood(theta) -> log(L), [] where []
    is the (lack of) derived parameters.
    """

    def __init__(self, x_min, x_max, xs, ys, sigma, adaptive):
        self._likelihood_function = create_likelihood_function(
            x_min, x_max, xs, ys, sigma, adaptive
        )

    def __call__(self, theta):
        """
        Likelihood relative to a flex-knot with parameters theta.

        If self.adaptive = True, the first element of theta is N; floor(N) is
        the number of nodes used to calculate the likelihood.

        theta = [N, y0, x1, y1, x2, y2, ...,
                 x_(N-2), y_(N-2), y_(N-1)] for N nodes.

        Otherwise, theta is the same but without N.

        theta = [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)].
        """
        return self._likelihood_function(theta)


def create_likelihood_function(x_min, x_max, xs, ys, sigma, adaptive):
    """
    Creates a likelihood function for a flex-knot, relative to data
    described by xs, ys, and sigma.

    sigma is either sigma_y, [sigma_x, sigma_y], [sigma_ys] or
    [[sigma_xs], [sigma_ys]].

    (obviously the middle two are degenerate when len(sigma) = 2, in which case
    [sigma_x, sigma_y] is assumed.)
    """
    LOG_2_SQRT_2πλ = np.log(2) + 0.5 * np.log(2 * np.pi * (x_max - x_min))

    if adaptive:
        flexknot = AdaptiveKnot(x_min, x_max)
    else:
        flexknot = FlexKnot(x_min, x_max)

    # check for sigma_x
    has_sigma_x = False
    if hasattr(sigma, "__len__"):
        if len(sigma.shape) == 2:
            has_sigma_x = True
        if 2 == len(sigma):
            has_sigma_x = True

    if has_sigma_x:
        sigma_x = sigma[0]
        sigma_y = sigma[1]
        var_x = sigma_x**2
        var_y = sigma_y**2

        def xy_errors_likelihood(theta):

            x_nodes = np.concatenate(
                ([x_min], get_x_nodes_from_theta(theta, adaptive), [x_max])
            )
            # use flex-knots to get y nodes,
            # as this is simplest way of dealing with N=0 or 1
            y_nodes = flexknot(x_nodes, theta)

            ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
            cs = y_nodes[:-1] - ms * x_nodes[:-1]

            # save recalculating things
            # indices in order [data point, relevant m and c]
            q = (np.outer(var_x, ms**2).T + var_y).T
            delta = np.subtract.outer(ys, cs)
            beta = (xs * var_y + (delta * ms).T * var_x).T / q
            gamma = (np.outer(xs, ms) - delta) ** 2 / 2 / q

            t_minus = ((np.sqrt(q / 2).T / (sigma_x * sigma_y)).T
                       * (x_nodes[:-1] - beta))
            t_plus = ((np.sqrt(q / 2).T / (sigma_x * sigma_y)).T
                      * (x_nodes[1:] - beta))

            logL = -len(xs) * LOG_2_SQRT_2πλ
            logL += np.sum(
                logsumexp(-gamma, b=q**-0.5 * (erf(t_plus) - erf(t_minus)),
                          axis=-1))
            return logL, []

        return xy_errors_likelihood

    # sigma_y only

    var_y = sigma**2

    def y_errors_likelihood(theta):
        if hasattr(var_y, "__len__"):
            logL = -0.5 * np.sum(np.log(2 * np.pi * var_y))
        else:
            logL = -0.5 * len(ys) * np.log(2 * np.pi * var_y)

        logL += np.sum(-((ys - flexknot(xs, theta)) ** 2) / 2 / var_y)
        return logL, []

    return y_errors_likelihood
