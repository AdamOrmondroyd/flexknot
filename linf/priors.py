"""
Priors using linfs.

I need to think carefully how best to abstract away all the weird slicing that
goes on to make a linf work.

Currently going for n then x_nodes then y_nodes.

Contains UnformPrior, SortedUniformPrior and forced_identifiability_transform()
which are blatantly stolen from Will Handley's PolyChord
"""
import numpy as np

from linf.helper_functions import get_x_nodes_from_theta, get_y_nodes_from_theta

## copied from polychord, not sure what best way to "steal" this is
## TODO: figure out better way of stealingß
class UniformPrior:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a + (self.b - self.a) * x


def forced_identifiability_transform(x):
    N = len(x)
    t = np.zeros(N)
    t[N - 1] = x[N - 1] ** (1.0 / N)
    for n in range(N - 2, -1, -1):
        t[n] = x[n] ** (1.0 / (n + 1)) * t[n + 1]
    return t


class SortedUniformPrior(UniformPrior):
    def __call__(self, x):
        t = forced_identifiability_transform(x)
        return super(SortedUniformPrior, self).__call__(t)


## end blatant stealing from polychord


def get_prior(x_min, x_max, y_min, y_max, N_max=None):
    """
    Returns a uniform + sorted uniform prior appropriate for a linf.

    N_max = int is the maximum number of nodes to use with an adaptive linf.

    N_max=None will return the prior for a non-adaptive model.
    """
    # non-adaptive prior
    def prior(theta):
        x_prior = SortedUniformPrior(x_min, x_max)(get_x_nodes_from_theta(theta))
        y_prior = UniformPrior(y_min, y_max)(get_y_nodes_from_theta(theta))

        # TODO: how best to combine the priors? Interleave or separate?
        # for now we're going with separate
        return np.concatenate((x_prior, y_prior))

    if N_max is None:

        return prior

    def adaptive_prior(theta):
        n_prior = UniformPrior(0, N_max)(theta[0:1])
        return np.concatenate((n_prior, prior(theta[1:])))

    return adaptive_prior
