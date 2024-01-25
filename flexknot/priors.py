"""
Priors using flex-knots.

I need to think carefully how best to abstract away all the weird slicing that
goes on to make a flex-knots work.

Currently going for interleaving x_nodes and y_nodes.
"""
import numpy as np
from pypolychord.priors import UniformPrior, SortedUniformPrior
from flexknot.utils import (
    create_theta,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)


class FlexKnotPrior(UniformPrior):
    """Interleaved uniform and sorted uniform priors for a flex-knot."""

    def __init__(self, x_min, x_max, y_min, y_max):
        self._x_prior = SortedUniformPrior(x_min, x_max)
        self._y_prior = UniformPrior(y_min, y_max)

    def __call__(self, hypercube):
        """
        Prior for flex-knot.

        For N nodes:
        hypercube -> [y0, x1, y1, x2, y2, ..., x_(N-2), y_(N-2), y_(N-1)].

        Parameters
        ----------
        hypercube : array-like of Uniform(0, 1).

        Returns
        -------
        theta : array-like of SortedUniform(x_min, x_max)
        and Uniform(y_min, y_max).

        """
        if len(hypercube) > 2:
            _x_prior = self._x_prior(get_x_nodes_from_theta(hypercube,
                                                            adaptive=False))
        else:
            _x_prior = np.array([])
        return create_theta(
            _x_prior,
            self._y_prior(get_y_nodes_from_theta(hypercube, adaptive=False)),
        )


class AdaptiveKnotPrior(FlexKnotPrior):
    """
    Interleaved uniform and sorted uniform priors appropriate for a flex-knot.

    N_max: int
    The maximum number of nodes to use with an adaptive flex-knot.
    """

    def __init__(self, x_min, x_max, y_min, y_max, N_min, N_max):
        self._N_prior = UniformPrior(N_min, N_max + 1)
        super().__init__(x_min, x_max, y_min, y_max)

        # redefine self._x_prior
        self.__used_x_prior = self._x_prior
        self.__unused_x_prior = UniformPrior(x_min, x_max)
        self._x_prior = (
            lambda hypercube_x: np.concatenate(
                (
                    self.__used_x_prior(hypercube_x[: self.__n_x_nodes]),
                    self.__unused_x_prior(hypercube_x[self.__n_x_nodes:]),
                )
            )
            if self.__n_x_nodes > 0
            else self.__unused_x_prior(hypercube_x)
        )

    def __call__(self, hypercube):
        """
        Prior for adaptive flex-knot.

        hypercube = [N, y0, x1, y1, x2, y2, ...,
                     x_(Nmax-2), y_(Nmax-2), y_(Nmax-1)],
        where Nmax is the greatest allowed value of floor(N), i.e.
        the maximum number of nodes.

        """
        prior = np.empty(hypercube.shape)
        prior[[0]] = self._N_prior(hypercube[0:1])
        self.__n_x_nodes = int(prior[0])
        prior[1:] = super().__call__(hypercube[1:])
        return prior
