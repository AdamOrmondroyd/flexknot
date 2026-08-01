"""
Test that flexknot.Prior sorts the x nodes.
"""

import numpy as np
from flexknot import AdaptivePrior, Prior
from flexknot.utils import get_x_nodes_from_theta

rng = np.random.default_rng()
x_min = 0
x_max = 1
y_min = -2
y_max = 2
N_min = 1
N_max = 10


def test_flexknotprior_x_nodes_are_sorted():
    """
    Test that the prior for the x_nodes is sorted.
    """
    hypercube = rng.random(2 * N_max - 2)
    prior = Prior(x_min, x_max, y_min, y_max)(hypercube)

    assert np.all(np.diff(get_x_nodes_from_theta(prior, adaptive=False)) >= 0)


def test_adaptiveknotprior_x_nodes_are_sorted():
    """
    Test that the first N-2 x nodes of the adaptive prior are sorted.
    """

    hypercube = rng.random(2 * N_max - 1)
    prior = AdaptivePrior(
        x_min, x_max, y_min, y_max, N_min, N_max
    )(hypercube)

    assert np.all(np.diff(get_x_nodes_from_theta(prior, adaptive=True)) >= 0)


def test_adaptiveknotprior_sorts_only_the_used_x_nodes():
    """
    Regression test for the n_x_nodes off-by-2 (int(N) vs int(N) - 2).

    AdaptivePrior must apply SortedUniformPrior to exactly the floor(N) - 2
    *used* interior x nodes. Sorting floor(N) of them (the old bug) pulls 2
    unused "phantom" nodes into the forced-identifiability nested product, so
    the used node positions come out wrong -- silently distorting the prior and
    breaking mirror symmetry in adaptive-N runs. So it is not enough for the
    used nodes to be sorted (they always are): their positions must match
    SortedUniformPrior applied to just those floor(N) - 2 hypercube values.
    """
    from pypolychord.priors import SortedUniformPrior

    prior = AdaptivePrior(x_min, x_max, y_min, y_max, N_min, N_max)
    for n in range(3, N_max + 1):  # floor(N) = n, i.e. n - 2 >= 1 interior knots
        hypercube = rng.random(2 * N_max - 1)
        hypercube[0] = (n + 0.5 - N_min) / ((N_max + 1) - N_min)  # force floor(N) = n

        used = get_x_nodes_from_theta(prior(hypercube), adaptive=True)
        x_hypercube = get_x_nodes_from_theta(hypercube[1:], adaptive=False)
        expected = SortedUniformPrior(x_min, x_max)(x_hypercube[: n - 2])

        assert np.allclose(used, expected), f"floor(N)={n}: {used} != {expected}"
