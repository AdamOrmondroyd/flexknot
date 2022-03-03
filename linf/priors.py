"""
Priors using linfs.

I need to think carefully how best to abstract away all the weird slicing that
goes on to make a linf work.

Currently going for n then interleaving x_nodes and y_nodes.
"""
import numpy as np

from linf.helper_functions import (
    create_theta,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)
from pypolychord.priors import UniformPrior, SortedUniformPrior


class LinfPrior(UniformPrior):
    """
    Interleaved uniform and sorted uniform priors appropriate for a linf.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, theta):
        """
        Prior for linf.

        theta = [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes.
        """
        return create_theta(
            SortedUniformPrior(self.x_min, self.x_max)(get_x_nodes_from_theta(theta)),
            UniformPrior(self.y_min, self.y_max)(get_y_nodes_from_theta(theta)),
        )


class AdaptiveLinfPrior(LinfPrior):
    """
    Interleaved uniform and sorted uniform priors appropriate for a linf.

    n_max: int is the maximum number of nodes to use with an interactive linf.
    """

    def __init__(self, x_min, x_max, y_min, y_max, n_max):
        self.n_max = n_max
        super().__init__(x_min, x_max, y_min, y_max)

    def __call__(self, theta):
        """
        Prior for adaptive linf.

        theta = [n, y0, x1, y1, x2, y2, ..., x_N, y_N, y_(n_max+1)],
        where n_max is the greatest allowed value of ceil(n).
        """
        return np.concatenate(
            (
                UniformPrior(-1, self.n_max)(theta[0:1]),
                super().__call__(theta[1:]),
            )
        )
