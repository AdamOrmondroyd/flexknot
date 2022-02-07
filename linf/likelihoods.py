"""
Likelihoods using linfs. 

I think by making a start on writing/copying this over from toy_sine it should become clear how to
proceed with specifying the boundaries.
"""

import numpy as np
from scipy.special import erf, logsumexp
from linf import adaptive_linf, get_theta_n, linf


def get_likelihood(x_min, x_max, xs, ys, sigma, adaptive=True):
    """
    sigma is either sigma_y, [sigma_x, sigma_y], [[sigma_ys]] or [[sigma_xs], [sigma_ys]]

    (obviously the middle two are degenerate if there is only one data point, in which case
    [sigma_x, sigma_y] is assumed.)
    """
    LOG_2_SQRT_2PIλ = np.log(2) + 0.5 * np.log(2 * np.pi * (x_max - x_min))

    ## is sorting actually necessary? Will test in toy sine
    xs_sorted_index = np.argsort(xs)
    xs, ys = xs[xs_sorted_index], ys[xs_sorted_index]

    # check for sigma_x
    has_sigma_x = False
    if hasattr(sigma, "__len__"):
        if len(sigma.shape) == 2:
            has_sigma_x = True
        if 2 == len(sigma) and len(ys) > 1:
            has_sigma_x = True

    if has_sigma_x:
        # sigma balls
        sigma_x = sigma[0]
        sigma_y = sigma[1]
        var_x = sigma_x**2
        var_y = sigma_y**2

        def xy_errors_likelihood(theta):
            n = len(theta) // 2 - 1
            x_nodes = np.concatenate(([x_min], theta[1 : 2 * n + 1 : 2], [x_max]))
            y_nodes = np.concatenate((theta[0 : 2 * n + 2 : 2], theta[-1:]))

            ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
            cs = y_nodes[:-1] - ms * x_nodes[:-1]

            # save recalculating things

            q = (np.outer(var_x, ms**2).T + var_y).T
            delta = np.subtract.outer(ys, cs)
            beta = (xs * var_x + (delta * ms).T * var_y).T / q
            gamma = (np.outer(xs, ms) - delta) ** 2 / 2 / q

            t_minus = (np.sqrt(q / 2).T / (sigma_x * sigma_y)).T * (x_nodes[:-1] - beta)
            t_plus = (np.sqrt(q / 2).T / (sigma_x * sigma_y)).T * (x_nodes[1:] - beta)

            logL = -len(xs) * LOG_2_SQRT_2PIλ
            logL = np.sum(
                logsumexp(
                    -gamma + np.log(q**-0.5 * (erf(t_plus) - erf(t_minus))), axis=-1
                )
            )

            return logL, []

        if adaptive:

            def super_likelihood(theta):

                theta_n = get_theta_n(theta)
                return xy_errors_likelihood(theta_n)

            return super_likelihood

        else:
            return xy_errors_likelihood

    else:  # sigma_y only

        var_y = sigma**2

        if adaptive:
            f = adaptive_linf(x_min, x_max)
        else:
            f = linf(x_min, x_max)

        def y_errors_likelihood(theta):
            if hasattr(var_y, "__len__"):
                logL = -0.5 * np.sum(np.log(2 * np.pi * var_y))
            else:
                logL = -0.5 * len(ys) * np.log(2 * np.pi * var_y)

            logL += np.sum(-((ys - f(xs, theta)) ** 2) / 2 / var_y)
            return logL, []

        return y_errors_likelihood
