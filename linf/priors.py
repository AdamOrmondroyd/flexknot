"""
Priors using linfs.

I need to think carefully how best to abstract away all the weird slicing that
goes on to make a linf work.

Currently going for interleaving x_nodes and y_nodes.
"""
import numpy as np
from pypolychord.priors import UniformPrior, SortedUniformPrior
from linf.helper_functions import (
    create_theta,
    get_theta_n,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)


class LinfPrior(UniformPrior):
    """
    Interleaved uniform and sorted uniform priors appropriate for a linf.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self._x_prior = SortedUniformPrior(x_min, x_max)
        self._y_prior = UniformPrior(y_min, y_max)

    def __call__(self, hypercube):
        """
        Prior for linf.

        hypercube = [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)] for N nodes.
        """
        if len(hypercube) > 2:
            _x_prior = self._x_prior(get_x_nodes_from_theta(hypercube, adaptive=False))
        else:
            _x_prior = np.array([])
        return create_theta(
            _x_prior,
            self._y_prior(get_y_nodes_from_theta(hypercube, adaptive=False)),
        )


class AdaptiveLinfPrior(LinfPrior):
    """
    Interleaved uniform and sorted uniform priors appropriate for a linf.

    N_max: int is the maximum number of nodes to use with an interactive linf.
    """

    def __init__(self, x_min, x_max, y_min, y_max, N_min, N_max):
        self._N_prior = UniformPrior(N_min, N_max)
        super().__init__(x_min, x_max, y_min, y_max)

        # redefine self._x_prior
        self.__used_x_prior = self._x_prior
        self.__unused_x_prior = UniformPrior(x_min, x_max)
        self.__n_x_nodes = N_max
        self._x_prior = (
            lambda hypercube_x: np.concatenate(
                (
                    self.__used_x_prior(hypercube_x[: self.__n_x_nodes]),
                    self.__unused_x_prior(hypercube_x[self.__n_x_nodes :]),
                )
            )
            if self.__n_x_nodes > 0
            else self.__unused_x_prior(hypercube_x)
        )

    def __call__(self, hypercube):
        """
        Prior for adaptive linf.

        hypercube = [N, y0, x1, y1, x2, y2, ..., x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
        where Nmax is the greatest allowed value of floor(N).
        """
        prior = np.empty(hypercube.shape)
        prior[0] = self._N_prior(hypercube[0:1])
        self.__n_x_nodes = int(prior[0])
        prior[1:] = super().__call__(hypercube[1:])
        return prior
