import numpy as np
from flexknot import FlexKnot


def test_area():
    x_min = 1
    x_max = 2

    fk = FlexKnot(x_min, x_max)

    # rectangle
    theta0 = np.array([0])
    theta1 = np.array([1])
    assert fk.area(theta0, theta1) == 1

    # crossover
    theta0 = np.array([0, 1])
    theta1 = np.array([1, 0])
    assert fk.area(theta0, theta1) == 0.5

    # Check same result both ways around
    theta0 = np.random.rand(10)
    theta1 = np.random.rand(10)
    assert fk.area(theta0, theta1) == fk.area(theta1, theta0)
