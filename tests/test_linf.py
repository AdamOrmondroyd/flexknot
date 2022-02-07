import numpy as np
from linf.linfs import linf


def test_linf():
    """
    Test that linf gives the same results as np.interp
    """
    x_min = 0
    x_max = 1
    N = 8
    rng = np.random.default_rng()
    x_nodes = np.sort(rng.uniform(x_min, x_max, N))
    y_nodes = rng.uniform(-10, 10, N + 2)
    theta = np.zeros(len(x_nodes) + len(y_nodes))
    theta[1 : 2 * N + 1 : 2] = x_nodes
    theta[0 : 2 * N + 2 : 2] = y_nodes[:-1]
    theta[-1] = y_nodes[-1]
    xs = np.linspace(x_min, x_max, 100)
    assert np.all(
        np.interp(xs, np.concatenate(([x_min], x_nodes, [x_max])), y_nodes)
        == linf(x_min, x_max)(xs, theta)
    )
