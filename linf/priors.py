"""
Priors using linfs.

I need to think carefully how best to abstract away all the weird slicing that
goes on to make a linf work.

Currently going for n then interleaving x_nodes and y_nodes.
"""
import numpy as np

from helper_functions import (
    create_theta,
    get_x_nodes_from_theta,
    get_y_nodes_from_theta,
)
from pypolychord.priors import UniformPrior, SortedUniformPrior


def get_prior(x_min, x_max, y_min, y_max, N_max=None):
    """
    Returns a uniform + sorted uniform prior appropriate for a linf.

    N_max = int is the maximum number of nodes to use with an adaptive linf.

    N_max=None will return the prior for a non-adaptive model.
    """
    # non-adaptive prior
    def prior(theta):
        # TODO: how best to combine the priors? Interleave or separate?
        # for now we're going with separate
        # and now I've decided to go back to interleaving
        return create_theta(
            SortedUniformPrior(x_min, x_max)(get_x_nodes_from_theta(theta)),
            UniformPrior(y_min, y_max)(get_y_nodes_from_theta(theta)),
        )

    if N_max is None:

        return prior

    def adaptive_prior(theta):
        n_prior = UniformPrior(0, N_max)(theta[0:1])
        return np.concatenate((n_prior, prior(theta[1:])))

    return adaptive_prior
